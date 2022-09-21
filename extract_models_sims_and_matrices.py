import argparse
import gensim
import itertools
import logging
import math
import numpy
import os
import random
import re
import scipy
import sys

from matplotlib import image
from nltk.corpus import wordnet
from skimage import metrics
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel

from extraction_utils import levenshtein, read_words

sys.path.append('/import/cogsci/andrea/github/')

from grano.plot_utils import confusion_matrix_simple

parser = argparse.ArgumentParser()

parser.add_argument('--experiment_id', required=True, \
                    choices=['one', 'two'], \
                    help='Which experiment?')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

plot_path = os.path.join('..', 'confusion_matrices', args.experiment_id)
os.makedirs(plot_path, exist_ok=True)

it_words, en_words = read_words(args)

assert len(it_words) in [32, 40]
#word_combs = [k for k in itertools.combinations(it_words, r=2)]

output_folder = os.path.join('..', 'similarities', args.experiment_id)
os.makedirs(output_folder, exist_ok=True)

wiki_folder = os.path.join('..', '..', 'resources', 'it_wiki_pages')

logging.info ('Extracting orthography')

### Loooking at orthography similarities

orthography_sims = list()

with open(os.path.join(output_folder, 'orthography.sims'), 'w') as o:
    for word_one in it_words:
        word_one_list = list()
        for word_two in it_words:
            if word_one != word_two:
                current_sim = levenshtein(word_one, word_two)
                current_corr = current_sim
            else:
                current_sim = 0.
                current_corr = 0.
            word_one_list.append(current_sim)
            o.write('{}\t{}\t{}\t{}\n'.format(word_one, word_two, current_sim, current_corr))
        orthography_sims.append(word_one_list)

### Plotting the matrix

title = 'Confusion matrix for orthography distance'

file_path = os.path.join(plot_path, 'orthography_confusion_exp_{}.png'.format(args.experiment_id))
confusion_matrix_simple(numpy.array(orthography_sims), it_words, title, file_path, text=False, \
                        vmin=numpy.amin(orthography_sims), vmax=numpy.amax(orthography_sims))

CORnet_layers = list()

layers = ['V1', 'V2', 'V4', 'decoder']
layers = ['V1']
for layer in layers:

    logging.info ('Extracting CORnet {}'.format(layer))

    ### Loooking at orthography similarities
    features_folder = os.path.join('..', '..', 'resources', 'CORnet_{}'.format(layer)) 
    feat_dict = dict()
    for w in it_words:
        with open(os.path.join(features_folder, '{}.layer'.format(w))) as i:
            feats = [l.strip().split('\t') for l in i.readlines()][0]
        feats = numpy.array(feats, dtype=numpy.double)
        feat_dict[w] = feats

    V1_sims = list()

    with open(os.path.join(output_folder, 'CORnet_{}.sims'.format(layer)), 'w') as o:
        for word_one in it_words:
            img_one = feat_dict[word_one]
            word_one_list = list()
            for word_two in it_words:
                if word_one != word_two:
                    img_two = feat_dict[word_two]
                    current_sim = 1. - scipy.stats.spearmanr(img_one, img_two)[0]
                    current_corr = 1. - scipy.stats.pearsonr(img_one, img_two)[0]
                else:
                    current_sim = 1.
                    current_corr = 1.
                word_one_list.append(current_sim)
                o.write('{}\t{}\t{}\t{}\n'.format(word_one, word_two, current_sim, current_corr))
            V1_sims.append(word_one_list)

    ### Plotting the matrix

    title = 'Confusion matrix for CORnet {}'.format(layer)

    file_path = os.path.join(plot_path, 'CORnet_{}_confusion_exp_{}.png'.format(layer, args.experiment_id))
    confusion_matrix_simple(numpy.array(V1_sims), it_words, title, file_path, text=False, \
                            vmin=numpy.amin(V1_sims), vmax=numpy.amax(V1_sims))

    CORnet_layers.append(V1_sims)

logging.info('Extracting images')

images_path = os.path.join('..', '..', 'resources', 'stimuli_images') 
word_dict = dict()
for w in it_words:
    im = image.imread(os.path.join(images_path, '{}.png'.format(w)))
    word_dict[w] = im

print('Now computing pairwise similarities...')

pixelwise_sims = list()

with open(os.path.join(output_folder, 'pixelwise.sims'), 'w') as o:
    for word_one in it_words:
        word_one_list = list()
        im_one = word_dict[word_one]
        for word_two in it_words:
            if word_one != word_two:
                im_two = word_dict[word_two]
                total = 0
                same = 0

                for i in range(im_one.shape[0]):
                    for j in range(im_one.shape[1]):
                        if im_one[i,j].tolist() == im_two[i,j].tolist():
                            same += 1
                        total += 1

                current_sim = 1. - metrics.structural_similarity(im_one, im_two, multichannel=True)
                current_corr = same/total

            else:
                current_sim = 0.
                current_corr = 0.

            word_one_list.append(current_sim)
            o.write('{}\t{}\t{}\t{}\n'.format(word_one, word_two, current_sim, current_corr))
        pixelwise_sims.append(word_one_list)

'''
print('Now unit normalizing...')
norm_sims = collections.defaultdict(list)
for i in range(2):
    
    std = numpy.nanstd([v[i] for k, v in sims.items()])
    mean = numpy.nanmean([v[i] for k, v in sims.items()])
    #l1_norm = sum([v[i] for k, v in sims.items()])
    #max_value = max([v[i] for k, v in sims.items()])
    #min_value = min([v[i] for k, v in sims.items()])
    for c, res in sims.items():
        #norm_sims[c].append(res[i]/l1_norm)
        norm_sims[c].append((res[i]-mean)/std)
        #norm_sims[c].append((res[i]-min_value)/(max_value-min_value))

print(stats.pearsonr([k[0] for i, k in norm_sims.items()], [k[1] for i, k in norm_sims.items()]))

with open('../rsa_analyses/computational_models/visual/visual.sims', 'w') as o:
    o.write('Word 1\tWord 2\tPixel overlap\tStructural similarity\n')
    for c, res in norm_sims.items():
        o.write('{}\t{}\t{}\t{}\n'.format(c[0], c[1], res[0], res[1]))
'''
### Plotting the matrix

title = 'Confusion matrix for pixelwise'

file_path = os.path.join(plot_path, 'pixelwise_confusion_exp_{}.png'.format(args.experiment_id))
confusion_matrix_simple(numpy.array(pixelwise_sims), it_words, title, file_path, text=False, \
                        vmin=numpy.amin(pixelwise_sims), vmax=numpy.amax(pixelwise_sims))

logging.info ('Extracting Word2Vec')

### Loooking at Word2Vec similarities

w2v_sims = list()

with open(os.path.join(output_folder, 'w2v.sims'), 'w') as o:
    w2v_file = os.path.join('..', '..', \
                            'resources', \
                            'w2v_it_combined_corpora', \
                            #'w2v_itwac_vocab_150000', \
                            'w2v_it_combined_corpora_vocab_150000'
                            #'w2v_itwac_vocab_150000'\
                            )
    w2v = gensim.models.Word2Vec.load(w2v_file)
    #for word_one, word_two in word_combs:
    for word_one in it_words:
        word_one_list = list()
        for word_two in it_words:
            if word_one != word_two:
                current_sim = w2v.wv.similarity(word_one, word_two)
                current_corr = scipy.stats.pearsonr(w2v.wv[word_one], w2v.wv[word_two])[0]
            else:
                current_sim = 0.
                current_corr = 0.
            word_one_list.append(current_sim)
            o.write('{}\t{}\t{}\t{}\n'.format(word_one, word_two, current_sim, current_corr))
        w2v_sims.append(word_one_list)

### Plotting the matrix

title = 'Confusion matrix for Word2Vec'

file_path = os.path.join(plot_path, 'w2v_confusion_exp_{}.png'.format(args.experiment_id))
confusion_matrix_simple(numpy.array(w2v_sims), it_words, title, file_path, text=False, \
                        vmin=numpy.amin(w2v_sims), vmax=numpy.amax(w2v_sims))


logging.info ('Extracting co-occurrences and PPMIs')
### Collecting co-occurrences and PPMIs

coocs = list()
log_coocs = list()
ppmis = list()

cooc_folder = os.path.join('..', '..', 'resources', \
                           'cooc_combined_corpora_window_5')
                           #'co_occurrences_itwac_vocab_150000_window_5_subsampling_False')
for word_one in it_words:
    #relative_freq_one = word_freqs[word_one]

    word_one_path = os.path.join(cooc_folder, word_one[:3], '{}.coocs'.format(word_one))
    assert os.path.exists(word_one_path)
    with open(word_one_path, encoding='utf-8') as i:
        lines = [l.strip().split('\t') for l in i.readlines()][1:]

    word_coocs = {l[0] : float(l[1]) for l in lines}
    word_log_coocs = {l[0] : math.log(float(l[1]), 2) for l in lines}
    word_ppmis = {l[0] : float(l[3]) for l in lines}

    for w in it_words:
        if w not in word_coocs.keys():
            word_coocs[w] = 0.
            word_log_coocs[w] = 0.
            word_ppmis[w] = 0.
 
    word_one_coocs = [word_coocs[w] if w!=word_one else 0. for w in it_words]
    word_one_log_coocs = [word_log_coocs[w] if w!=word_one else 0. for w in it_words]
    word_one_ppmis = [word_ppmis[w] if w!=word_one else 0. for w in it_words]

    coocs.append(word_one_coocs)
    log_coocs.append(word_one_log_coocs)
    ppmis.append(word_one_ppmis)

## Writing pure co-ocs
with open(os.path.join(output_folder, 'cooc.sims'), 'w') as o:
    for w_i_one, word_one in enumerate(it_words):
        for w_i_two, word_two in enumerate(it_words):
            o.write('{}\t{}\t{}\n'.format(word_one, word_two, coocs[w_i_one][w_i_two]))

### Plotting the matrix for co-ocs

title = 'Confusion matrix for ItWac co-occurrences'

file_path = os.path.join(plot_path, 'cooc_confusion_exp_{}.png'.format(args.experiment_id))
confusion_matrix_simple(numpy.array(coocs), it_words, title, file_path, text=False, \
                        vmin=numpy.amin(coocs), vmax=numpy.amax(coocs))

### Writing log co-ocs
with open(os.path.join(output_folder, 'log_cooc.sims'), 'w') as o:
    for w_i_one, word_one in enumerate(it_words):
        for w_i_two, word_two in enumerate(it_words):
            o.write('{}\t{}\t{}\n'.format(word_one, word_two, log_coocs[w_i_one][w_i_two]))

### Plotting the matrix for the log co-occurrences

title = 'Confusion matrix for log co-occurrence'

file_path = os.path.join(plot_path, 'log_cooc_confusion_exp_{}.png'.format(args.experiment_id))
confusion_matrix_simple(log_coocs, it_words, title, file_path, text=False, \
                        vmin=numpy.amin(log_coocs), vmax=numpy.amax(log_coocs))

### Writing ppmis
with open(os.path.join(output_folder, 'ppmi.sims'), 'w') as o:
    for w_i_one, word_one in enumerate(it_words):
        for w_i_two, word_two in enumerate(it_words):
            ppmi_score = ppmis[w_i_one][w_i_two]
            o.write('{}\t{}\t{}\n'.format(word_one, word_two, ppmi_score))

### Plotting the matrix for PPMI

title = 'Confusion matrix for PPMI'

file_path = os.path.join(plot_path, 'ppmi_confusion_exp_{}.png'.format(args.experiment_id))
confusion_matrix_simple(numpy.array(ppmis), it_words, title, file_path, text=False, \
                        vmin=numpy.amin(ppmis), vmax=numpy.amax(ppmis))

logging.info ('Extracting BERT')
### Looking at BERT similarities

#model_name = "dbmdz/bert-base-italian-xxl-cased"
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained("bert-base-multilingual-uncased")

bert_vectors = dict()

for w in tqdm(it_words):
    
    w_vectors = list()
    wiki_file = os.path.join(wiki_folder, '{}.wiki'.format(w))
    with open(wiki_file, encoding='utf-8') as i:
        final_lines = [l.strip() for l in i.readlines()]
    
    final_lines = final_lines[:6]


    for l in final_lines:

        ### Encoding the sentence
        encoded_input = tokenizer(l, return_tensors='pt')
        
        ### Getting the model output
        output = model(**encoded_input, output_hidden_states=True, \
                       output_attentions=False, return_dict=True)
    
        ### Averaging all layers and all sentences

        mention = list()
        #for layer in range(1, 13):
        for layer in [1]:
            layer_vec = output['hidden_states'][layer][0, 1:-1, :].detach().numpy()
            mention.append(numpy.average(layer_vec, axis=0))
        mention = numpy.average(mention, axis=0)
        assert mention.shape == (768, )
        w_vectors.append(mention)

    ### Averaging across mentions
    assert len(w_vectors) >= 1
    bert_vectors[w] = numpy.average(w_vectors, axis=0)
     
    '''
    # Encoding individual mentions instead of sentences

    bert_tokens = tokenizer.tokenize(w)
    bert_tokens = bert_tokens if isinstance(bert_tokens, list) else [bert_tokens]
    token_id = tokenizer.convert_tokens_to_ids(bert_tokens)
    if token_id == [100]:
        import pdb; pdb.set_trace()
        ### Finding the relevant indices
        tokenized_input = '_'.join(tokenizer.tokenize(l))
        joined_id = '_'.join(bert_tokens)
        marked_id = re.sub(r'[a-z]+', 'MARK', joined_id)
        tokenized_input = tokenized_input.replace(joined_id, marked_id)
        split_marked = tokenized_input.split('_')
        relevant_indices = [t_i for t_i, t in enumerate(split_marked) if 'MARK' in t]

        encoded_input = tokenizer(l, return_tensors='pt')
        #relevant_indices = [t_i for t_i, t in enumerate(encoded_input['input_ids'][0, :].detach().tolist()) \
                                                           #if t in token_id]
        #if len(relevant_indices) >= 1:
        for rel_i in relevant_indices:

            mention = list()
            ### Extracting layers 3-8
            for layer in range(3, 8):
                try:
                    layer_vec = output['hidden_states'][layer][0, rel_i, :].detach().numpy()
                except IndexError:
                    import pdb; pdb.set_trace()
                mention.append(layer_vec)
            ### Averaging across layers
            mention = numpy.average(mention, axis=0)
            assert mention.shape == (768, )
            w_vectors.append(mention)
    '''
    
### Loooking at BERT similarities

berts = list()

with open(os.path.join(output_folder, 'bert.sims'), 'w') as o:
    for word_one in it_words:
        word_one_list = list()
        for word_two in it_words:
            current_corr = scipy.stats.pearsonr(bert_vectors[word_one], bert_vectors[word_two])[0]
            word_one_list.append(current_corr)
            o.write('{}\t{}\t{}\n'.format(word_one, word_two, current_corr))
        berts.append(word_one_list)

### Plotting the matrix

title = 'Confusion matrix for BERT'

file_path = os.path.join(plot_path, 'bert_confusion_exp_{}.png'.format(args.experiment_id))
confusion_matrix_simple(numpy.array(berts), it_words, title, file_path, text=False, \
                        vmin=numpy.amin(berts), vmax=numpy.amax(berts))

logging.info('Extracting Wordnet')
### Collecting Wordnet similarities

### Using the list of it_words in English

en_mapping = {'pork' : 'pig', \
              'cow' : 'cattle', \
              'glass' : 'drinking_glass', \
              'plate' : 'dish'}

en_words = [w if w not in en_mapping.keys() else en_mapping[w] for w in en_words]

if args.experiment_id == 'one':
    synsets = [wordnet.synset(w) for w in en_words]
elif args.experiment_id == 'two':
    synsets = [wordnet.synsets(w)[0] for w in en_words]
     
wordnets = list()

with open(os.path.join(output_folder, 'wordnet.sims'), 'w') as o:
    for w_one_i, w_syns_one in enumerate(synsets):
        w_sims = list()
        for w_two_i, w_syns_two in enumerate(synsets):
            if w_one_i != w_two_i:
                sim = wordnet.path_similarity(w_syns_one, w_syns_two)
            else:
                sim = 0.
            w_sims.append(sim)
            o.write('{}\t{}\t{}\n'.format(it_words[w_one_i], it_words[w_two_i], sim))
        wordnets.append(w_sims)

wordnets = numpy.array(wordnets)

### Plotting the matrix for WordNet

title = 'Confusion matrix for WordNet'

file_path = os.path.join(plot_path, 'wordnet_confusion_exp_{}.png'.format(args.experiment_id))
confusion_matrix_simple(wordnets, it_words, title, file_path, text=False, \
                        vmin=numpy.amin(wordnets), vmax=numpy.amax(wordnets))

logging.info('Plotting the confusion matrix comparing all the models')

### Plotting the models correlations

models = [coocs, log_coocs, ppmis, w2v_sims, berts, wordnets, pixelwise_sims, orthography_sims] + CORnet_layers
#models = [w2v_sims, berts, wordnets]
names = ['raw co-occurrence', 'log co-occurrence', 'PPMI', 'Word2Vec', 'mBERT', 'WordNet', 'Pixels', 'Orthography'] + layers
#names = ['Word2Vec', 'BERT', 'WordNet']

flattened = list()
for m_i, m in enumerate(models):
    new_m = numpy.array([[w[i] for i in range(len(it_words)) if i!=w_i] for w_i, w in enumerate(m)]).flatten()
    flattened.append(new_m)

models_corr = list()
for m_one in flattened:
    m_list = list()
    for m_two in flattened:
        corr = scipy.stats.pearsonr(m_one, m_two)[0]
        m_list.append(corr)
    models_corr.append(m_list)

models_corr = numpy.array(models_corr)
title = 'Confusion matrix comparing the models'

file_path = os.path.join(plot_path, 'all_models_confusion_exp_{}.png'.format(args.experiment_id))
confusion_matrix_simple(models_corr, names, title, file_path, text=True)
