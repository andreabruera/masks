import mne
import os
import collections
import itertools
import numpy
import scipy
import argparse
import random
from multiprocessing import Process

from io_utils import ComputationalModels, EvokedResponses
from rsa_utils import run_rsa

parser = argparse.ArgumentParser()
parser.add_argument('--permutation', \
                    action='store_true', default=False, \
                    help='Indicates whether to run a permutation analysis or not')
parser.add_argument('--searchlight', action='store_true', \
                    default=False, help='Indicates whether to run a searchlight analysis or not')
parser.add_argument('--analysis', default='both_worlds', \
                    choices=['objective_accuracy', 'subjective_judgments', 'both_worlds'], \
                    help='Indicates which pairwise similarities to compare, \
                          whether by considering objective accuracy or subjective judgments')
parser.add_argument('--word_selection', default='targets_only', \
                    choices=['all_words', 'targets_only'], \
                    help='Indicates whether to use for the analyses only the targets or all the words')
parser.add_argument('--minimum_ERPs', type=int, default=4, \
                    choices=[1,2,4], help='Indicates how many \
                    erps will be required for keeping the words for the analyses \
                    and how many will be used for averaging')
parser.add_argument('--computational_model', default='w2v', \
                    choices=['cslb', 'CORnet', 'visual', 'orthography', 'w2v', 'original_cooc', 'ppmi', 'new_cooc', 'wordnet'], \
                    help='Indicates which similarities to use for comparison to the eeg similarities')
parser.add_argument('--hop', default=3, type=int, \
                    help='Indicates which similarities to use for comparison to the eeg similarities')
parser.add_argument('--temporal_window_size', default=7, \
                    type=int, help='Indicates which similarities to use for comparison to the eeg similarities')
args = parser.parse_args()

        ''' 
### RSA

if __name__ == '__main__':
    
    ### Loading the computational model
    if args.computational_model == 'w2v':
        computational_model = ComputationalModels().w2v
    elif args.computational_model == 'original_cooc':
        computational_model = ComputationalModels().original_cooc
    elif args.computational_model == 'new_cooc':
        computational_model = ComputationalModels().new_cooc
    elif args.computational_model == 'ppmi':
        computational_model = ComputationalModels().ppmi
    elif args.computational_model == 'wordnet':
        computational_model = ComputationalModels().wordnet
    elif args.computational_model == 'orthography':
        computational_model = ComputationalModels().orthography
    elif args.computational_model == 'visual':
        computational_model = ComputationalModels().visual
    elif args.computational_model == 'CORnet':
        computational_model = ComputationalModels().CORnet
    elif args.computational_model == 'cslb':
        computational_model = ComputationalModels().cslb

    #rsa_per_subject(args, 3, computational_model)
    if args.permutation:

        ### Preparing the batches
        batches = list()
        workers = os.cpu_count() - 1
        separators = [i for i in range(1, 301, workers)]
        for i, v in enumerate(separators):
            if v != separators[-1]:
                batches.append([i for i in range(v, separators[i+1])])
            else:
                batches.append([i for i in range(v, 301)])
        for b in batches:
            assert len(b) <= workers


        #for s in range(6, 17): 
        for s in range(3, 18): 
        #for s in [17]: 

            evoked_responses = EvokedResponses(s)
            all_time_points = evoked_responses.time_points

            for batch in batches:
                processes = list()
                for permutation in batch:

                    #run_rsa(args, s, evoked_responses, computational_model, all_time_points, permutation)
                    
                    proc = Process(target=run_rsa, args=(args, s, evoked_responses, computational_model, all_time_points, permutation, ))
                    processes.append(proc)
                    proc.start()

                for proc in processes:
                    proc.join()
    
        pass
    else:
        processes = list()

        for s in range(1, 17): 

            evoked_responses = EvokedResponses(s)
            all_time_points = evoked_responses.time_points

            proc = Process(target=run_rsa, args=(args, s, evoked_responses, computational_model, all_time_points))
            processes.append(proc)
            proc.start()

        for proc in processes:
            proc.join()
        '''
