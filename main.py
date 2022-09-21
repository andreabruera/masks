import argparse
import itertools
import logging
import multiprocessing
import os
import random
import scipy

from tqdm import tqdm

from io_utils import ComputationalModel, ExperimentInfo, EEGData
from classification.time_resolved_classification import run_classification, run_searchlight_classification
from plot_scripts.plot_classification import plot_classification
from rsa.group_searchlight import run_group_searchlight
from rsa.rsa_searchlight import finalize_rsa_searchlight, run_searchlight
from searchlight.searchlight_utils import SearchlightClusters

### Logging utility
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

### Parsing input arguments
parser = argparse.ArgumentParser()

parser.add_argument('--analysis', required=True, \
                    choices=['classification', \
                             'classification_searchlight', \
                             'rsa_searchlight', \
                             'rsa_searchlight', \
                             'classification_searchlight'], \
                    help='Indicates which analysis to perform')

parser.add_argument('--computational_model', required=False, \
                    choices=['cooc', 'log_cooc', 'ppmi', \
                             'w2v', 'bert', 'wordnet', \
                             'orthography', 'pixelwise', 'CORnet_V1'], \
                    help='Which model?')

parser.add_argument('--data_split', required=True, \
                    choices=['objective_accuracy', \
                             'perceptual_awareness', \
                             ], \
                    help='Indicates which pairwise similarities \
                          to compare, whether by considering \
                          objective accuracy or subjective judgments')

parser.add_argument('--bids_folder', type=str, required=True, \
                    help='Folder where to find the BIDS dataset')

parser.add_argument('--plot', action='store_true', required=False, \
                    help='Plot?')
parser.add_argument('--debugging', action='store_true', \
                    required=False, \
                    help='Debugging time?')

parser.add_argument('--data_kind', choices=['erp'], \
                    default='erp', help='ERP analyses?')

args = parser.parse_args()

general_output_folder = os.path.join('results', args.analysis, args.data_split)
os.makedirs(general_output_folder, exist_ok=True)

experiment = ExperimentInfo(args)

### Behavioural analyses
if args.analysis == 'behavioural':
    pass

### Time-resolved classification with all electrodes
elif args.analysis == 'classification':

    ### Just plotting
    if args.plot:
        plot_classification(args)

    ### Computing the classification scores
    else:
        accuracies = list()
        ### One subject at a time
        if args.debugging:
            for n in tqdm(range(1, experiment.n_subjects+1)):
                accuracies.append(run_classification([experiment, n, args, general_output_folder]))
        ### All subjects together
        else:
            with multiprocessing.Pool() as p:
                accuracies = p.map(run_classification, [[experiment, n, args, general_output_folder] for n in range(1, experiment.n_subjects+1)])
            p.terminate()
            p.join()


### From now on only searchlight
else:

    ### Collecting electrode clusters
    searchlight_clusters = SearchlightClusters()
    electrode_indices = [searchlight_clusters.neighbors[center] \
                                        for center in range(128)]


    ### RSA check
    if args.analysis in [\
                         'group_rsa_searchlight', \
                         'rsa_searchlight'] \
                                       and not args.computational_model:
        raise RuntimeError('You need to specify a computational model!')

    ### Group searchlight
    if args.plot:
        run_group_searchlight(args, experiment, searchlight_clusters, \
                                          general_output_folder)

    ### RSA searchlight preparation
    else:


        ### Within-subject multiprocessing loop
        for n in tqdm(range(1, experiment.n_subjects+1)):

            eeg = EEGData(experiment, n, args)

            data = eeg.data_dict
            mapper = {'1' : 'low', '2' : 'medium', '3' : 'high'}
            data = {mapper[k] : v for k, v in data.items()}

            ### Extracting actual clusters
            times = eeg.times
            relevant_times = [t_i for t_i, t in enumerate(times) if t_i+16<len(times)][::8]
            explicit_times = [times[t] for t in relevant_times]
            clusters = [(e_s, t_s) for e_s in electrode_indices for t_s in relevant_times]

            ### Looping over the various awareness levels
            for awareness, vecs in data.items():

                ### Searchlight-based classification
                if args.analysis == 'classification_searchlight':
                    os.makedirs(general_output_folder, exist_ok=True)

                    logging.info('Now running analysis on '
                                 'subject {}, awareness level {}'\
                                 .format(n, awareness))

                    animals = [k for k in vecs.keys() if experiment.trigger_to_info[k][1]=='animal']
                    objects = [k for k in vecs.keys() if experiment.trigger_to_info[k][1]=='object']
                    min_len = min([len(animals), len(objects)])
                    if min_len <= 7:
                        print('not enough data for subject {}, {}'.format(n, awareness))
                        continue

                    combs = list(itertools.product(animals, objects, repeat=1))
                    combs = random.sample(combs, k=100)


                    if args.debugging:
                        results = list()
                        for cluster in tqdm(clusters):
                            results.append(\
                                      run_searchlight_classification([\
                                      experiment, \
                                      n, args, data[awareness], \
                                      cluster,\
                                      combs]))
                    else:
                        with multiprocessing.Pool() as p:

                            results = p.map(\
                                      run_searchlight_classification, \
                                      [[experiment, n, args, \
                                        data[awareness], cluster, \
                                        combs] \
                                        for cluster in clusters])
                            p.terminate()
                            p.join()

                ### Searchlight-based RSA
                elif args.analysis == 'rsa_searchlight':

                    general_output_folder = os.path.join(general_output_folder, args.computational_model)
                    os.makedirs(general_output_folder, exist_ok=True)
                    comp_model = ComputationalModel(args, experiment)
                    words = [k for k in vecs.keys() if k<33]
                    ### Only employing conditions with at
                    ### least 5 words
                    #if len(words) >= 5:
                    ### Only employing conditions with
                    ### at least 16 words
                    if len(words) >= 16:
                        ordered_words, combs, pairwise_similarities = comp_model.compute_pairwise(words)

                        with multiprocessing.Pool() as p:

                            results = p.map(run_searchlight, \
                                        [[vecs, \
                                        comp_model, \
                                        cluster, \
                                        combs, \
                                        pairwise_similarities] \
                                        for cluster in clusters])
                            p.terminate()
                            p.join()

                if len(results) > 1:
                    finalize_rsa_searchlight(results, \
                                         relevant_times, \
                                         explicit_times, \
                                         general_output_folder, \
                                         awareness,
                                         n)
