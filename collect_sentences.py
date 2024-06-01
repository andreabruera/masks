import argparse
import multiprocessing
import os
import re

from qwikidata.linked_data_interface import get_entity_dict_from_api
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--corpora_folder', type=str, required=True)
args = parser.parse_args()

def process_entity(all_args):
    corpus = all_args[0]
    words = all_args[1]
    args = all_args[2]
    folder = args.corpora_folder
    ent_sentences = {e : list() for e in words}
    with tqdm() as counter:
        phrase_counter = 0
        with open(os.path.join(folder, '{}_it.corpus'.format(corpus))) as i:
            for l in i.readlines():
                phrase_counter += 1
                if phrase_counter >= 100000:
                    continue
                for e in words:
                    finder = re.findall('(?<!\w){}(?!\w)'.format(e), l)
                    if len(finder) >= 1:
                        new_sent = re.sub('(?<![a-zA-Z#]){}(?![a-zA-Z#])'.format(e), '[SEP]#{}#[SEP]'.format(e), l)
                        new_sent = re.sub(r'#', r' ', new_sent)
                        new_sent = re.sub('\s+', r' ', new_sent)
                        ent_sentences[e].append('{}\t{}'.format(corpus, new_sent.strip()))
                        counter.update(1)
    return ent_sentences

### Reading words
sample_file = '../../dataset/neuroscience/unaware_semantics_bids/derivatives/sub-01/sub-01_task-namereadingimagery_events.tsv'
with open(sample_file) as i:
    lines = [l.split('\t')[2] for l in i.readlines()][1:]
words = list(set([l for l in lines if len(l)>2]))
print(words)
assert len(words) == 32

ent_sentences = dict()

corpora = ['wiki', 'opensubtitles', 'itwac', 'gutenberg']

with multiprocessing.Pool(processes=len(corpora)) as pool:
   results = pool.map(process_entity, [(corpus, words, args) for corpus in corpora])
   pool.terminate()
   pool.join()

out_folder = 'stimuli_sentences_from_all_corpora'
os.makedirs(out_folder, exist_ok=True)
final_sents = {k : list() for k in words}
for ent_dict in results:
    for k, v in ent_dict.items():
        final_sents[k].extend(v)

for stimulus, ent_sentences in final_sents.items():
    with open(os.path.join(out_folder, '{}.sentences'.format(stimulus)), 'w') as o:
        for sent in ent_sentences:
            o.write('{}\n'.format(sent.strip()))
