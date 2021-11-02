
from flask import Flask
from markupsafe import escape
from flask.ext.jsonpify import jsonify


import networkx as nx
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

import numpy as np

G = nx.read_gexf('GD_augmented.gexf')
Gnodes = list(G.nodes())
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
            if 'larva' not in full_label and 'glia' not in full_label and 'abdom' not in full_label and 'embry' not in full_label and 'blast' not in full_label and 'fiber' not in full_label:
                #if True:
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


# retriever = TfidfRetriever(document_store=document_store)
# retriever = ElasticsearchRetriever(document_store)


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





reader = FARMReader(model_name_or_path="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", use_gpu=True)

pipe = ExtractiveQAPipeline(reader, retriever)

# prediction = pipe.run(query="what are adult Drosophila descending neuron subtypes", top_k_retriever=10, top_k_reader=10)



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
            prediction = pipe.run(query=q, top_k_retriever=10, top_k_reader=8)
            uj = 1
            mes = ''
            # mes = 'Drosobot Response: '
            # self.send_message(mes)
            for i in prediction['answers']:
                if i['score']>0.7:
                    name = i['meta']
                    mes += '\n **' + str(uj)+'.' + G.nodes()[name]['label'] + """** (<a class="drosobotTag" onclick="window.plusplusSearch('!visualize """ + name + """')">""" + name + '</a>)'
                    # self.send_message(mes)
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
                    return_struct['query'] = 'add /:referenceId:[' + ', '.join(nodes) + '])'
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
                    return_struct['query'] = 'add /:referenceId:[' + ', '.join(nodes) + '])'
                    return return_struct
                else:
                    return return_struct
            
            return return_struct
        if x.startswith("!visualize "):
            
            q = x.replace("!visualize ","")
            if q in Gvisnodes:
                nodes = [i for i in list(nx.descendants(Gvis,q)) if i.isnumeric()][:100]
                if len(nodes)>0:
                    answer = 'Drosobot Response: '
                    answer += '\n[Generated NeuroNLP Tag](https://hemibrain12.neuronlp.fruitflybrain.org/?query=add%20/:referenceId:[' + ',%20'.join(nodes) + '])'
                    # self.send_message(answer)
                    return_struct['query'] = 'add /:referenceId:[' + ', '.join(nodes) + '])'
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
                    # print(i, q)
                    # print([G.nodes()[i]['label'] for i in list(nx.descendants(Gvis,q)) if i.isnumeric()])
                    nodes = [i for i in list(nx.descendants(Gvis,q)) if i.isnumeric()][:200]
                    if len(nodes)>0:
                        answer = 'Drosobot Response: '
                        answer += '\n[Generated NeuroNLP Tag](https://hemibrain12.neuronlp.fruitflybrain.org/?query=add%20/:referenceId:[' + ',%20'.join(nodes) + '])'
                        # self.send_message(answer)
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
"""
def send_gitter_message(*x):
    x = ' '.join([str(i) for i in x])
    print('Sending message:', x)
    gitter.messages.send('mkturkcan/chatbot', x)
engine.send_message = send_gitter_message
print('Query Engine Setup complete.')
import time
try:
    while True:
        try:
            last_message = gitter.messages.list('mkturkcan/chatbot')[-1]['text']
            a = engine.query_interpreter(last_message)
        except:
            gitter = GitterClient('92694476870ad4cefcea8a75ce69d2d1432b13b2')

            # Check_my id
            print('id:', gitter.auth.get_my_id)

            gitter.rooms.join('mkturkcan/chatbot')
            pass
        time.sleep(5)
except KeyboardInterrupt:
    pass
"""
app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, welcome to Drosobot!</p>"


import json


@app.route('/send_message/<message>')
def send_message(message):
    # show the user profile for that user
    output = engine.query_interpreter(str(escape(message)))
    if output is None:
        output = {'message': '', 'query': '', 'warning': ''}
    # output = {}
    # output['response'] = str(a)
    # output['kind'] = 'drosobot'
    return jsonify(kind='drosobot', message=output['message'], query=output['query'], warning=output['warning'])
    # return json.dumps(output)

# !ask What are descending neurons?
# !visualize descending neuron
# !visualize vpoEN