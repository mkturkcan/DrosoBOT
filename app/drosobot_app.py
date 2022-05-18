# Flask imports
from flask import Flask
from markupsafe import escape
from flask.ext.jsonpify import jsonify

# General imports
import json
import numpy as np
import networkx as nx

# Haystack imports
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.retriever.dense import DensePassageRetriever
from haystack.retriever.sparse import TfidfRetriever
from haystack.pipeline import ExtractiveQAPipeline
from haystack.retriever import ElasticsearchRetriever

TO_DEBUG = False # Debug mode flag
RETRIEVER = 'elasticsearch' # Retriever type: either 'tfidf', 'elasticsearch' or 'dpr'


G = nx.read_gexf('GD_augmented.gexf')
Gnodes = list(G.nodes())

if TO_DEBUG:
    print('First few ontology nodes:', Gnodes[:5])
    print('Sample node details:', G.nodes()['http://flybase.org/reports/FBgn0000137'])

label_dict = {}
for i in G.nodes():
    label_dict[G.nodes()[i]['label']] = i
    
sentences = []
vnodes = []
data_dicts = []

Gvis = nx.read_gexf('GV_augmented.gexf')
Gvisnodes = list(Gvis.nodes())


for i in Gnodes:
    if 'definition' in G.nodes()[i] and i in Gvisnodes:
        if len(list(nx.descendants(Gvis,i)))>0:
            """
            sent = G.nodes()[i]['definition'] + ' ' + G.nodes()[i]['comments']
            if len(sent)==0:
                sent = G.nodes()[i]['label']
            sentences.append(sent)
            vnodes.append(i)
            """
            full_label = G.nodes()[i]['label']
            #if 'neuromere' not in full_label and 'thorax' not in full_label and 'larva' not in full_label and 'glia' not in full_label and 'abdom' not in full_label and 'embry' not in full_label and 'blast' not in full_label and 'fiber' not in full_label:
            #if 'anastomosis' not in full_label and 'GC' not in full_label and 'larva' not in full_label and 'glia' not in full_label and 'abdom' not in full_label and 'embry' not in full_label and 'blast' not in full_label and 'fiber' not in full_label:
            if True:
                if len(G.nodes()[i]['definition'])>0:
                    # defsents = G.nodes()[i]['definition'].split('. ')
                    labels = G.nodes()[i]['label'].replace('/',' or ')
                    defsents = [labels + ': ' + G.nodes()[i]['definition']]
                    # if len(G.nodes()[i]['comments'])>0:
                    #     defsents[0] = defsents[0] + G.nodes()[i]['comments']
                    data_dicts.append({'text': defsents[0], 'meta': i})
                    for j in defsents:
                        if len(j)>0:
                            jk = j
                            if not j[-1] == '.':
                                jk = j + '.'
                            sentences.append(jk)
                            vnodes.append(i)
                elif len(G.nodes()[i]['comments'])>0:
                    # defsents = G.nodes()[i]['comments'].split('. ')
                    labels = G.nodes()[i]['label'].replace('/',' or ')
                    defsents = [labels+': ' + G.nodes()[i]['comments']]
                    data_dicts.append({'text': defsents[0], 'meta': i})
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
# retriever = ElasticsearchRetriever(document_store=document_store)

"""
retriever = DensePassageRetriever(document_store=document_store,
                                  query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                  passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                  max_seq_len_query=64,
                                  max_seq_len_passage=256,
                                  batch_size=16,
                                  use_gpu=True,
                                  embed_title=True,
                                  use_fast_tokenizers=True)
document_store.update_embeddings(retriever)
"""




reader = FARMReader(model_name_or_path="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", use_gpu=True)

pipe = ExtractiveQAPipeline(reader, retriever)

# prediction = pipe.run(query="what are adult Drosophila descending neuron subtypes", top_k_retriever=10, top_k_reader=10)
gloms = ['VP5', 'VP4', 'VP3', 'VP2', 'VP1m', 'VP1l', 'VP1d', 'VM7v', 'VM7d', 'VM5v', 'VM5d', 'VM4', 'VM3', 'VM2', 'VM1', 'VL2p', 'VL2a', 'VL1', 'VC5', 'VC4', 'VC3m', 'VC3l', 'VC2', 'VC1', 'VA7m', 'VA7l', 'VA6', 'VA5', 'VA4', 'VA3', 'VA2', 'VA1v', 'VA1d', 'V', 'DP1m', 'DP1l', 'DM6', 'DM5', 'DM4', 'DM3', 'DM2', 'DM1', 'DL5', 'DL4', 'DL3', 'DL2v', 'DL2d', 'DL1', 'DC4', 'DC3', 'DC2', 'DC1', 'DA4m', 'DA4l', 'DA3', 'DA2', 'DA1', 'D']
compartments = []
mb_cell_types = ['PPL1','PPL101','PPL102','PPL103','PPL104','PPL105','PPL106','PPL107','PPL108']
domain_keywords = gloms + mb_cell_types + ['Global Feedback', 'Feedback Loop', 'feedback loop', 'Feedback', 'patchy'] + ['antennal lobe local neuron']




class QueryEngine:
    def __init__(self):
        self.prediction = None
        self.last_query = None
        self.send_message = print
    def query_interpreter(self, x):
        return_struct = {'message': '', 'query': '', 'warning': ''}
        print('Received the following query:', x)
        x = x.replace('slash','/')
        if x.startswith("!quit"):
            return True
        if x.startswith("!query "):
            q = x.replace("!query ","")
        if x.startswith("!ask "):
            
            q = x.replace("!ask ","")
            
            # Neural query finder:
            prediction = pipe.run(query=q, top_k_retriever=30, top_k_reader=30)
            uj = 1
            mes = ''
            # mes = 'Drosobot Response: '
            # self.send_message(mes)

            ## Keyword Match Parser
            found_answers = []
            keyword_matches = {}
            glom_found = False
            for i in gloms:
                if i in q:
                    glom_found = True
            if 'feedback loop' in q and glom_found == False:
                q = q.replace('feedback loop', 'Global Feedback ')
            if 'antennal lobe' in q and 'local neuron' in q:
                q = q + ' ' + 'antennal lobe local neuron'

            for domain_keyword in domain_keywords:
                if domain_keyword+' ' in q or domain_keyword+'/' in q or q.endswith(domain_keyword):
                    print('Found Domain Keyword:', domain_keyword)
                    for name in G.nodes():
                        if domain_keyword in G.nodes()[name]['label']:
                            # keyword_matches.append(name)
                            if name not in found_answers:
                                if len(list(nx.descendants(Gvis,name)))>0:
                                    full_label = G.nodes()[name]['label']
                                    if 'thoracic' not in full_label and 'primordium' not in full_label and 'adult mushroom body/' not in full_label and 'columnar neuron' not in full_label and 'centrifugal neuron C' not in full_label and 'lamina' not in full_label and 'medulla ' not in full_label and 'anastomosis' not in full_label and 'GC' not in full_label and 'larva' not in full_label and 'glia' not in full_label and 'abdom' not in full_label and 'embry' not in full_label and 'blast' not in full_label and 'fiber' not in full_label:
                                        found_answers.append(name)
                                        print('Keyword Match Response:', G.nodes()[name]['label'])
                                        mes += '\n **' + str(uj)+'.' + G.nodes()[name]['label'] + """** (<a target="_blank" href='""" + name + """'>""" + name + '</a>)'
                                        # self.send_message(mes)
                                        ujk = uj
                                        if 'patchy' in full_label:
                                            ujk = 100
                                        mes += """ <p></p><a id='plusplusresult"""+ str(ujk) + """' onclick="window.plusplusSearch('!visualize """ + name + """')" class="info-try btn btn-xs"><i class="fa fa-angle-double-right" aria-hidden="true"></i> Add to Workspace</a>"""
                                        mes += """<a id='plusplusbresult"""+ str(ujk) + """' onclick="window.plusplusSearch('!pin """ + name + """')" class="info-try btn btn-xs"><i class="fa fa-angle-double-right" aria-hidden="true"></i> Pin</a>"""
                                        mes += """<a id='pluspluscresult"""+ str(ujk) + """' onclick="window.plusplusSearch('!unpin """ + name + """')" class="info-try btn btn-xs"><i class="fa fa-angle-double-right" aria-hidden="true"></i> Unpin</a>"""
                                        mes += '\n *' + G.nodes()[name]['definition'] + '*'
                                        # self.send_message(mes)
                                        uj += 1
            # Neural Match Parser
            for i in prediction['answers']:
                if i['score']>0.00:
                    name = i['meta']
                    if name not in found_answers:
                        if len(list(nx.descendants(Gvis,name)))>0:
                            full_label = G.nodes()[name]['label']
                            if 'thoracic' not in full_label and 'primordium' not in full_label and 'adult mushroom body/' not in full_label and 'columnar neuron' not in full_label and 'centrifugal neuron C' not in full_label and 'lamina' not in full_label and 'medulla ' not in full_label and 'anastomosis' not in full_label and 'GC' not in full_label and 'larva' not in full_label and 'glia' not in full_label and 'abdom' not in full_label and 'embry' not in full_label and 'blast' not in full_label and 'fiber' not in full_label:
                                found_answers.append(name)
                                mes += '\n **' + str(uj)+'.' + G.nodes()[name]['label'] + """** (<a target="_blank" href='""" + name + """'>""" + name + '</a>)'
                                # self.send_message(mes)
                                ujk = uj
                                if 'patchy' in full_label:
                                    ujk = 100
                                mes += """ <p></p><a id='plusplusresult"""+ str(ujk) + """' onclick="window.plusplusSearch('!visualize """ + name + """')" class="info-try btn btn-xs"><i class="fa fa-angle-double-right" aria-hidden="true"></i> Add to Workspace</a>"""
                                mes += """<a id='plusplusbresult"""+ str(ujk) + """' onclick="window.plusplusSearch('!pin """ + name + """')" class="info-try btn btn-xs"><i class="fa fa-angle-double-right" aria-hidden="true"></i> Pin</a>"""
                                mes += """<a id='pluspluscresult"""+ str(ujk) + """' onclick="window.plusplusSearch('!unpin """ + name + """')" class="info-try btn btn-xs"><i class="fa fa-angle-double-right" aria-hidden="true"></i> Unpin</a>"""
                                mes += '\n *' + G.nodes()[name]['definition'] + '*'
                                # self.send_message(mes)
                                uj += 1
            # self.send_message(mes)
            self.prediction = prediction
            print('Prediction Details:')
            print(prediction)
            return_struct['message'] = mes
            
            q = self.prediction['answers'][0]['meta']
            if q in Gvisnodes:
                # print(i, q)
                # print([G.nodes()[i]['label'] for i in list(nx.descendants(Gvis,q)) if i.isnumeric()])
                nodes = [i for i in list(nx.descendants(Gvis,q)) if i.isnumeric()][:200]
                if len(nodes)>0:
                    answer = 'Drosobot Response: '
                    answer += '\n[Generated NeuroNLP Tag](https://hemibrain12.neuronlp.fruitflybrain.org/?query=add%20/:referenceId:[' + ',%20'.join(nodes) + '])'
                    # self.send_message(answer)
                    # return_struct['query'] = 'add /:referenceId:[' + ', '.join(nodes) + '])'
                    return return_struct
                else:
                    return return_struct
            else:
                nodes = []
                for i in label_dict:
                    if q in i:
                        nodes = nodes + [i for i in list(nx.descendants(Gvis,q)) if i.isnumeric()]
                        nodes = list(set(nodes))
                if len(nodes)>0:
                    nodes = nodes[:200]
                    answer = 'Drosobot Response: '
                    answer += '\n[Generated NeuroNLP Tag](https://hemibrain12.neuronlp.fruitflybrain.org/?query=add%20/:referenceId:[' + ',%20'.join(nodes) + '])'
                    # return_struct['query'] = 'add /:referenceId:[' + ', '.join(nodes) + '])'
                    return return_struct
                else:
                    return return_struct
            
            return return_struct
        if x.startswith("!visualize ") or x.startswith("!pin ") or x.startswith("!unpin "):
            if x.startswith("!pin "):
                mode = 'pin'
            elif x.startswith("!unpin "):
                mode = 'unpin'
            else:
                mode = 'show'
            q = x.replace("!visualize ","")
            q = q.replace("!pin ","")
            q = q.replace("!unpin ","")
            print('Visualization Request:', q)
            if q in Gvisnodes:
                print('Visualization Request was in the graph.')
                nodes = [i for i in list(nx.descendants(Gvis,q)) if i.isnumeric()][:100]
                if len(nodes)>0:
                    answer = 'Drosobot Response: '
                    answer += '\n[Generated NeuroNLP Tag](https://hemibrain12.neuronlp.fruitflybrain.org/?query=add%20/:referenceId:[' + ',%20'.join(nodes) + '])'
                    # self.send_message(answer)
                    return_struct['query'] = 'add /:referenceId:[' + ', '.join(nodes) + '])'
                    if mode == 'pin':
                        return_struct['query'] = return_struct['query'].replace('add ','pin ')
                    if mode == 'unpin':
                        return_struct['query'] = return_struct['query'].replace('add ','unpin ')
                    return return_struct
                else:
                    answer = 'Drosobot Response: '
                    answer += '\nNothing found to visualize.'
                    # self.send_message(answer)
                    return_struct['warning'] = answer
                    return return_struct
            else:
                nodes = []
                for i in label_dict.keys():
                    if q in i:
                        # print(i,q)
                        # print([G.nodes()[i]['label'] for i in list(nx.successors(Gvis, label_dict[i])) if 'http' in i])
                        nodes = nodes + [i for i in list(nx.descendants(Gvis,label_dict[i])) if i.isnumeric()]
                        nodes = list(set(nodes))
                if len(nodes)>0:
                    nodes = nodes[:200]
                    answer = 'Drosobot Response: '
                    answer += '\n[Generated NeuroNLP Tag](https://hemibrain12.neuronlp.fruitflybrain.org/?query=add%20/:referenceId:[' + ',%20'.join(nodes) + '])'
                    # self.send_message(answer)
                    return_struct['query'] = 'add /:referenceId:[' + ', '.join(nodes) + '])'
                    if mode == 'pin':
                        return_struct['query'] = return_struct['query'].replace('add ','pin ')
                    return return_struct
                else:
                    # self.send_message('This node was not found in the database.')
                    answer = 'Drosobot Response: '
                    answer += '\nNo previous query was found.'
                    return_struct['warning'] = answer
                    return return_struct
        elif x.startswith("!visualize"):
            if self.prediction is not None:
                q = self.prediction['answers'][0]['meta']
                if q in Gvisnodes:
                    nodes = [i for i in list(nx.descendants(Gvis,q)) if i.isnumeric()][:200]
                    if len(nodes)>0:
                        answer = 'Drosobot Response: '
                        answer += '\n[Generated NeuroNLP Tag](https://hemibrain12.neuronlp.fruitflybrain.org/?query=add%20/:referenceId:[' + ',%20'.join(nodes) + '])'
                        return_struct['query'] = 'add /:referenceId:[' + ', '.join(nodes) + '])'
                        return return_struct
                    else:
                        answer = 'Drosobot Response: '
                        answer += '\nNothing found to visualize.'
                        return_struct['warning'] = answer
                        return return_struct
                else:
                    nodes = []
                    for i in label_dict:
                        if q in i:
                            nodes = nodes + [i for i in list(nx.descendants(Gvis,q)) if i.isnumeric()]
                            nodes = list(set(nodes))
                    if len(nodes)>0:
                        nodes = nodes[:200]
                        answer = 'Drosobot Response: '
                        answer += '\n[Generated NeuroNLP Tag](https://hemibrain12.neuronlp.fruitflybrain.org/?query=add%20/:referenceId:[' + ',%20'.join(nodes) + '])'
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
                    
engine = QueryEngine()

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, welcome to Drosobot!</p>"




@app.route('/send_message/<message>')
def send_message(message):
    output = engine.query_interpreter(str(escape(message)))
    if output is None:
        output = {'message': '', 'query': '', 'warning': ''}
    return jsonify(kind='drosobot', message=output['message'], query=output['query'], warning=output['warning'])