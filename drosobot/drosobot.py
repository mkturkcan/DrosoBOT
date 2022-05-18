
import pathlib

import networkx as nx
from haystack.reader.farm import FARMReader
from haystack.document_stores.memory import InMemoryDocumentStore
from haystack.retriever.sparse import TfidfRetriever
from haystack.pipelines import ExtractiveQAPipeline

class QueryEngine:
    def __init__(self, **eargs):
        self.prediction = None
        self.last_query = None
        self.send_message = print

        # Define important data structures
        ## Antennal lobe glomeruli
        self.gloms = ['VP5', 'VP4', 'VP3', 'VP2', 'VP1m', 'VP1l', 'VP1d', 'VM7v', 'VM7d', 'VM5v', 'VM5d', 'VM4', 'VM3', 'VM2', 'VM1', 'VL2p', 'VL2a', 'VL1', 'VC5', 'VC4', 'VC3m', 'VC3l', 'VC2', 'VC1', 'VA7m', 'VA7l', 'VA6', 'VA5', 'VA4', 'VA3', 'VA2', 'VA1v', 'VA1d', 'V', 'DP1m', 'DP1l', 'DM6', 'DM5', 'DM4', 'DM3', 'DM2', 'DM1', 'DL5', 'DL4', 'DL3', 'DL2v', 'DL2d', 'DL1', 'DC4', 'DC3', 'DC2', 'DC1', 'DA4m', 'DA4l', 'DA3', 'DA2', 'DA1', 'D']
        ## Mushroom Body compartments
        self.compartments = []
        ## Cell types in the mushroom body
        self.mb_cell_types = ['PPL1','PPL101','PPL102','PPL103','PPL104','PPL105','PPL106','PPL107','PPL108']
        ## Well-known standard keywords for the Drosophila domain; if the user query contains any of these keywords, direct keyword matches are included in the search results
        self.domain_keywords = self.gloms + self.mb_cell_types + ['Global Feedback', 'Feedback Loop', 'feedback loop', 'Feedback', 'patchy'] + ['antennal lobe local neuron']
        ## Some search terms are not relevant to the adult Drosophila domain in the context of
        ## the hemibrain dataset
        self.labels_to_avoid = ['thoracic', 'primordium', 'adult mushroom body/', 'columnar neuron', 'centrifugal neuron C', 'lamina', 'medulla ','anastomosis', 'GC', 'larva', 'glia', 'abdom', 'embry', 'blast', 'fiber']

        self.data_path = '../data'
        self.retriever_top_k = 30
        self.reader_top_k = 30
        self.query_threshold = 200 # Maximum number of neurons to return for a query
        self.to_debug = False
        for extra_arg in eargs:
            if extra_arg == 'data_path':
                self.data_path = eargs[extra_arg]
            if extra_arg == 'labels_to_avoid':
                self.labels_to_avoid = eargs[extra_arg]


    def prepare(self, num_processes = None):
        path = pathlib.Path(__file__).parent.resolve()
        G = nx.read_gexf(pathlib.PurePath(path, self.data_path, 'GD_augmented.gexf'))
        Gnodes = list(G.nodes())
        if self.to_debug:
            print('First few ontology nodes:', Gnodes[:5])
            print('Sample node details:', G.nodes()['http://flybase.org/reports/FBgn0000137'])
        self.Gnodes = Gnodes
        self.G = G

        label_dict = {}
        for i in G.nodes():
            label_dict[G.nodes()[i]['label']] = i
        self.label_dict = label_dict

        sentences = []
        vnodes = []
        data_dicts = []

        Gvis = nx.read_gexf(pathlib.PurePath(path, self.data_path, 'GV_augmented.gexf'))
        Gvisnodes = list(Gvis.nodes())
        self.Gvisnodes = Gvisnodes
        self.Gvis = Gvis


        for i in Gnodes:
            if 'definition' in G.nodes()[i] and i in Gvisnodes:
                if len(list(nx.descendants(Gvis,i)))>0:
                    full_label = G.nodes()[i]['label']
                    if True:
                        if len(G.nodes()[i]['definition'])>0:
                            labels = G.nodes()[i]['label'].replace('/',' or ')
                            defsents = [labels + ': ' + G.nodes()[i]['definition']]
                            data_dicts.append({'content': defsents[0], 'content_type': 'text', 'meta': {'link': i}})
                            for j in defsents:
                                if len(j)>0:
                                    jk = j
                                    if not j[-1] == '.':
                                        jk = j + '.'
                                    sentences.append(jk)
                                    vnodes.append(i)
                        elif len(G.nodes()[i]['comments'])>0:
                            labels = G.nodes()[i]['label'].replace('/',' or ')
                            defsents = [labels+': ' + G.nodes()[i]['comments']]
                            data_dicts.append({'content': defsents[0], 'content_type': 'text', 'meta': {'link': i}})
                            for j in defsents:
                                if len(j)>0:
                                    jk = j
                                    if not j[-1] == '.':
                                        jk = j + '.'
                                    sentences.append(jk)
                                    vnodes.append(i)
                    
        document_store = InMemoryDocumentStore()
        document_store.write_documents(data_dicts)


        retriever = TfidfRetriever(document_store=document_store)
        reader = FARMReader(model_name_or_path="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                            use_gpu=True, num_processes = num_processes)
        self.pipe = ExtractiveQAPipeline(reader, retriever)

    def query_interpreter(self, x):
        return_struct = {'message': [], 'query': '', 'warning': ''}
        if self.to_debug:
            print('Received the following query:', x)
        G = self.G

        x = x.replace('slash','/') # data transfer does not always allow for the / key; simple robust hack since word is not a term used in this domain
        if x.startswith("!quit"): # Quit command for the search engine
            return True
        if x.startswith("!query "):
            q = x.replace("!query ","")
        if x.startswith("!ask "):
            q = x.replace("!ask ","")
            # Run the model predictions 
            prediction = self.pipe.run(query=q, params = {"Retriever": {"top_k": self.retriever_top_k}, "Reader": {"top_k": self.reader_top_k}})
            uj = 1
            message = []

            
            found_answers = []
            keyword_matches = {}

            # Antennal Lobe-specific keyword correction routine
            glom_found = False
            for i in self.gloms:
                if i in q:
                    glom_found = True
            if 'feedback loop' in q and glom_found == False:
                q = q.replace('feedback loop', 'Global Feedback ')
            if 'antennal lobe' in q and 'local neuron' in q:
                q = q + ' ' + 'antennal lobe local neuron'

            # Keyword Match Parser
            for domain_keyword in self.domain_keywords:
                if domain_keyword+' ' in q or domain_keyword+'/' in q or q.endswith(domain_keyword):
                    if self.to_debug:
                        print('Found Domain Keyword:', domain_keyword)
                    for name in G.nodes():
                        if domain_keyword in G.nodes()[name]['label']:
                            if name not in found_answers: # prevent duplicates
                                if len(list(nx.descendants(self.Gvis,name)))>0: # only if this node has corresponding neurons in the dataset
                                    full_label = G.nodes()[name]['label']
                                    if all(n not in full_label for n in self.labels_to_avoid): # check against labels to avoid
                                        found_answers.append(name)
                                        if self.to_debug:
                                            print('Keyword Match Response:', G.nodes()[name]['label'])
                                        ujk = uj
                                        if 'patchy' in full_label:
                                            ujk = 100
                                        mes = {'label': G.nodes()[name]['label'],
                                               'name': name,
                                               'link_id': ujk,
                                               'entry': uj,
                                               'definition': G.nodes()[name]['definition']}
                                        message.append(mes)
                                        uj += 1
            # Neural Match Parser
            for i in prediction['answers']:
                if i.score>0.00:
                    name = i.meta['link']
                    if name not in found_answers:
                        if len(list(nx.descendants(self.Gvis,name)))>0:
                            full_label = G.nodes()[name]['label']
                            if all(n not in full_label for n in self.labels_to_avoid):
                                found_answers.append(name)
                                ujk = uj
                                ## Patch for demo capability: We use the id 100 for the application demo
                                if 'patchy' in full_label:
                                    ujk = 100
                                mes = {'label': G.nodes()[name]['label'],
                                       'name': name,
                                       'link_id': ujk,
                                       'entry': uj,
                                       'definition': G.nodes()[name]['definition']}
                                message.append(mes)
                                uj += 1
            self.prediction = prediction
            return_struct['message'] = message
            return return_struct
        if x.startswith("!visualize ") or x.startswith("!pin ") or x.startswith("!unpin "):
            if x.startswith("!pin "):
                mode = 'pin'
            elif x.startswith("!unpin "):
                mode = 'unpin'
            else:
                mode = 'add'
            q = x.replace("!visualize ","")
            q = q.replace("!pin ","")
            q = q.replace("!unpin ","")
            if self.to_debug:
                print('Visualization Request:', q)
            if q in self.Gvisnodes:
                if self.to_debug:
                    print('Visualization Request was in the graph.')
                nodes = [i for i in list(nx.descendants(self.Gvis,q)) if i.isnumeric()][:100]
                if len(nodes)>0:
                    return_struct['query'] = 'add /:referenceId:[' + ', '.join(nodes) + '])'
                    if mode == 'pin':
                        return_struct['query'] = return_struct['query'].replace('add ','pin ')
                    if mode == 'unpin':
                        return_struct['query'] = return_struct['query'].replace('add ','unpin ')
                    return return_struct
                else:
                    answer = 'Drosobot Response: '
                    answer += '\nNothing found to visualize.'
                    return_struct['warning'] = answer
                    return return_struct
            else:
                nodes = []
                for i in self.label_dict.keys():
                    if q in i:
                        nodes = nodes + [i for i in list(nx.descendants(self.Gvis,self.label_dict[i])) if i.isnumeric()]
                        nodes = list(set(nodes))
                if len(nodes)>0:
                    nodes = nodes[:self.query_threshold] # We want to implement a filter in case the query is too large
                    return_struct['query'] = 'add /:referenceId:[' + ', '.join(nodes) + '])'
                    if mode == 'pin':
                        return_struct['query'] = return_struct['query'].replace('add ','pin ')
                    return return_struct
                else:
                    answer = 'Drosobot Response: '
                    answer += '\nNo previous query was found.'
                    return_struct['warning'] = answer
                    return return_struct
        elif x.startswith("!visualize"):
            if self.prediction is not None:
                q = self.prediction['answers'][0].meta['link']
                if q in self.Gvisnodes:
                    nodes = [i for i in list(nx.descendants(self.Gvis,q)) if i.isnumeric()][:self.query_threshold]
                    if len(nodes)>0:
                        return_struct['query'] = 'add /:referenceId:[' + ', '.join(nodes) + '])'
                        return return_struct
                    else:
                        answer = 'Drosobot Response: '
                        answer += '\nNothing found to visualize.'
                        return_struct['warning'] = answer
                        return return_struct
                else:
                    nodes = []
                    for i in self.label_dict:
                        if q in i:
                            nodes = nodes + [i for i in list(nx.descendants(self.Gvis,q)) if i.isnumeric()]
                            nodes = list(set(nodes))
                    if len(nodes)>0:
                        nodes = nodes[:200]
                        return_struct['query'] = 'add /:referenceId:[' + ', '.join(nodes) + '])'
                        return return_struct
                    else:
                        answer = 'Drosobot Response: '
                        answer += '\nThis node was not found in the database.'
                        return_struct['warning'] = answer
                        return return_struct
            else:
                answer = 'Drosobot Response: '
                answer += '\nNo previous query was found.'
                return_struct['warning'] = answer
                return return_struct

