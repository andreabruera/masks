import argparse
import matplotlib
import random
import numpy
import os
import sklearn
import re
import torch

from matplotlib import pyplot
from sklearn.cluster import MeanShift, KMeans, DBSCAN
from sklearn.manifold import TSNE
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, AutoModelWithLMHead

parser = argparse.ArgumentParser()
parser.add_argument('--layer', choices=['middle_four', 'top_four',
                                        'high_four', 'first_four',
                                        'top_six', 'top_twelve',
                                        ],
                    required=True, help='Which layer?')
parser.add_argument('--model', choices=['ITBERT', 'MBERT', 'GILBERTO',
                                        'ITGPT2small', 'ITGPT2medium',
                                        'geppetto', 
                                        'xlm-roberta-large',
                                        ],
                    required=True, help='Which model?')
parser.add_argument('--tokens', choices=['single_words', 'sentence_averaged'],
                    required=True, help='How to use words?')
parser.add_argument('--cuda', \
                    choices=['0', '1', '2'], \
                    required=True, help='Indicates which \
                    CUDA device to use')
parser.add_argument('--corpus', \
                    choices=['all', 'wiki', 'itwac', 'gutenberg', 'opensubtitles'], \
                    required=True, help='Indicates which corpus to use')
args = parser.parse_args()

cluster_folder = os.path.join('plots', 'cluster_visualization') 
os.makedirs(cluster_folder, exist_ok=True)

if args.model == 'ITBERT':
    model_name = 'dbmdz/bert-base-italian-xxl-cased'
if args.model == 'GILBERTO':
    model_name = 'idb-ita/gilberto-uncased-from-camembert'
if args.model == 'ITGPT2small':
    model_name = 'GroNLP/gpt2-small-italian'
if args.model == 'ITGPT2medium':
    model_name = 'GroNLP/gpt2-medium-italian-embeddings'
if args.model == 'geppetto':
    model_name = 'LorenzoDeMattei/GePpeTto'
if args.model == 'MBERT':
    model_name = 'bert-base-multilingual-cased'
if args.model == 'xlm-roberta-large':
    model_name = 'xlm-roberta-large'

cuda_device = 'cuda:{}'.format(args.cuda)
sep_token = '[SEP]'
re_sep_token = '\[SEP\]'
if 'GeP' in model_name:
    model = AutoModelWithLMHead.from_pretrained("LorenzoDeMattei/GePpeTto").to(cuda_device)
    required_shape = model.config.n_embd
    max_len = model.config.n_positions
    n_layers = model.config.n_layer
elif 'gpt' in model_name or 'GPT' in model_name:
    model = AutoModel.from_pretrained(model_name).to(cuda_device)
    required_shape = model.embed_dim
    max_len = model.config.n_positions
    n_layers = model.config.n_layer
else:
    if 'bert' in model_name:
        model = AutoModelForMaskedLM.from_pretrained(model_name).to(cuda_device)
    else:
        ### luke is a bit of a special boy...
        sep_token = '<ent>'
        re_sep_token = '<ent>'
        model = AutoModel.from_pretrained(model_name).to(cuda_device)
    required_shape = model.config.hidden_size
    max_len = model.config.max_position_embeddings
    n_layers = model.config.num_hidden_layers
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_len, 
                                          truncation=True, sep_token=sep_token,
                                          )

entity_vectors = dict()
cluster_vectors = dict()

sentences_folder = 'stimuli_sentences_from_all_corpora'

with tqdm() as pbar:
    for f in os.listdir(sentences_folder):
        stimulus = f.replace('.sentences', '')
        entity_vectors[stimulus] = list()
        cluster_vectors[stimulus] = list()
        with open(os.path.join(sentences_folder, f)) as i:
            lines = [l.strip().split('\t') for l in i]
        if args.corpus != 'all':
            lines = [l for l in lines if l[0]==args.corpus]
        lines = [l[1] for l in lines]
        assert len(lines) >= 1

        counter = 0
        for l in lines:
            line_collector = list()
            if counter > 10:
                continue

            inputs = tokenizer(l, return_tensors="pt")

            if args.tokens == 'single_words':
                spans = [i_i for i_i, i in enumerate(inputs['input_ids'].numpy().reshape(-1)) if 
                        i==tokenizer.convert_tokens_to_ids([sep_token])[0]]
                if 'bert' in model_name.lower() and len(spans)%2==1:
                    spans = spans[:-1]
                        
                if len(spans) > 1:
                    try:
                        assert len(spans) % 2 == 0
                    except AssertionError:
                        #print(l)
                        continue
                    ### Correcting spans
                    l = re.sub(r'{}'.format(re_sep_token), '', l)
                    correction = list(range(1, len(spans)+1))
                    spans = [max(0, s-c) for s,c in zip(spans, correction)]
                    split_spans = list()
                    for i in list(range(len(spans)))[::2]:
                        if 'gpt' in model_name.lower():
                            current_span = (spans[i]+1, spans[i+1])
                            #current_span = (spans[i]-1, spans[i+1])
                        else:
                            current_span = (spans[i], spans[i+1])
                        split_spans.append(current_span)
                    assert len(split_spans) > 0

                    if len(tokenizer.tokenize(l)) > max_len-10:
                        continue
                    #outputs = model(**inputs, output_attentions=False, \
                    #                output_hidden_states=True, return_dict=True)
                    try:
                        inputs = tokenizer(l, return_tensors="pt").to(cuda_device)
                    except RuntimeError:
                        print('error')
                        continue
                    #try:
                    outputs = model(**inputs, output_attentions=False, \
                                    output_hidden_states=True, return_dict=True)
                    #except RuntimeError:
                    #    print(l)
                    #    continue

                    hidden_states = numpy.array([s[0].cpu().detach().numpy() for s in outputs['hidden_states']])
                    #last_hidden_states = numpy.array([k.detach().numpy() for k in outputs['hidden_states']])[2:6, 0, :]
                    for beg, end in split_spans:
                        print(tokenizer.tokenize(l)[beg:end])
                        ### If there are less than one token that must be a mistake
                        if len(tokenizer.tokenize(l)[beg:end]) < 1:
                            print(l)
                            continue
                        mention = hidden_states[:, beg:end, :]
                        mention = numpy.average(mention, axis=1)
                        if args.layer == 'first_four':
                            layer_start = 1
                            ### outputs has at dimension 0 the final output
                            layer_end = 5
                        if args.layer == 'middle_four':
                            layer_start = int(n_layers / 2)-2
                            layer_end = int(n_layers/2)+3
                        if args.layer == 'top_four':
                            layer_start = -4
                            ### outputs has at dimension 0 the final output
                            layer_end = n_layers+1
                        if args.layer == 'top_six':
                            layer_start = -6
                            ### outputs has at dimension 0 the final output
                            layer_end = n_layers+1
                        if args.layer == 'top_twelve':
                            layer_start = -12
                            ### outputs has at dimension 0 the final output
                            layer_end = n_layers+1
                        if args.layer == 'high_four':
                            layer_start = -5
                            ### outputs has at dimension 0 the final output
                            layer_end = n_layers-2
                        mention = mention[layer_start:layer_end, :]

                        mention = numpy.average(mention, axis=0)
                        assert mention.shape == (required_shape, )
                        entity_vectors[stimulus].append(mention)
                        cluster_vectors[stimulus].append(mention)
                        entity_vectors[stimulus] = [numpy.average(entity_vectors[stimulus].copy(), axis=0)]
                        pbar.update(1)
                        counter += 1

            if args.tokens == 'sentence_average':
                inputs = tokenizer(l, return_tensors="pt").to(cuda_device)
                if len(tokenizer.tokenize(l)) > max_len:
                    continue
                outputs = model(**inputs, output_attentions=False, \
                                output_hidden_states=True, return_dict=True)

                hidden_states = numpy.array([k.cpu().detach().numpy() for k in outputs['hidden_states']])
                ### We leave out the CLS
                hidden_states = hidden_states[:, 0, 1:-1, :]
                if args.layer == 'middle_four':
                    layer_start = int(n_layers / 2)-2
                    layer_end = int(n_layers/2)+2
                if args.layer == 'top_four':
                    layer_start = -4
                    ### outputs has at dimension 0 the final output
                    layer_end = n_layers +1
                if args.layer == 'high_four':
                    layer_start = -5
                    ### outputs has at dimension 0 the final output
                    layer_end = n_layers-2
                ### Averaging tokens
                mention = numpy.average(hidden_states, axis=1)
                assert mention.shape == (n_layers+1, required_shape)
                ### Averaging layers
                mention = numpy.average(mention[layer_start:layer_end, :], axis=0)
                assert mention.shape == (required_shape, )
                entity_vectors[stimulus].append(mention)
                pbar.update(1)

'''
clustered_vectors = dict()

for k, v in cluster_vectors.items():
    clustered_vectors[k] = list()
    #cluster_labels = DBSCAN().fit_predict(v)
    cluster_labels = KMeans(n_clusters=min(len(v), 50)).fit_predict(v)
    fig, ax = pyplot.subplots(constrained_layout=True)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    res = tsne.fit_transform(v)
    for label in set(cluster_labels):
        indices = [r_i for r_i, r in enumerate(cluster_labels) if r==label] 
        clust_vec = numpy.average([v[i] for i in indices], axis=0)
        clustered_vectors[k].append(clust_vec)
        ax.scatter(res[indices, 0], res[indices, 1])
    ax.set_title(k)
    pyplot.savefig(os.path.join(cluster_folder, '{}.jpg'.format(k)))
    pyplot.clf()
    pyplot.close()

'''
out_folder = os.path.join('word_vectors_11_2022', 
                      args.model,
                      args.layer, 
                      args.tokens,
                      #'average'
                      )
os.makedirs(out_folder, exist_ok=True)
for k, v in entity_vectors.items():
    with open(os.path.join(out_folder, '{}.vector'.format(k)), 'w') as o:
        if len(v) > 0:
            for vec in v:
                assert vec.shape[-1] == required_shape
                for dim in vec:
                    assert not numpy.isnan(dim)
                    o.write('{}\t'.format(float(dim)))
                o.write('\n')

'''
### Clustered vecs
out_folder = os.path.join('clustered_word_vectors_10_2022', 
                      args.experiment_id,
                      args.model,
                      args.layer, 
                      args.tokens,
                      #'average'
                      )
os.makedirs(out_folder, exist_ok=True)
for k, v in clustered_vectors.items():
    with open(os.path.join(out_folder, '{}.vector'.format(k)), 'w') as o:
        if len(v) > 0:
            for vec in v:
                if args.tokens == 'span_averaged':
                    assert vec.shape[-1] == required_shape
                elif args.tokens == 'span_concatenated':
                    assert vec.shape[-1] == required_shape*2
                for dim in vec:
                    assert not numpy.isnan(dim)
                    o.write('{}\t'.format(float(dim)))
                o.write('\n')
'''
