import collections
import itertools
import argparse
import os
import scipy
import random
import mne
import numpy
import scipy
import sklearn

from sklearn import svm
from tqdm import tqdm
from matplotlib import pyplot
from scipy import stats
from tqdm import tqdm
from io_utils import ExperimentInfo, EEGData, ComputationalModel

def classify(data, test_splits):

    results_list = list()
    ### Running on test split for each time point
    for s in test_splits:
    
        train_samples = list()
        train_labels = list()
        test_samples = list()
        test_labels = list()
        for cat_vecs_i, cat_vecs in enumerate(data.items()):
            current_s = s[cat_vecs_i]
            for vec_i, vec in enumerate(cat_vecs[1]):
                if vec_i not in current_s:
                    '''
                    ### Time-resolved
                    if len(vec.shape) > 1:
                        train_samples.append(vec[:, time_i])
                    ### Searchlight
                    else:
                        train_samples.append(vec)
                    '''
                    train_samples.append(vec)

                    train_labels.append(cat_vecs[0])
                else:
                    '''
                    ### Time-resolved
                    if len(vec.shape) > 1:
                        test_samples.append(vec[:, time_i])
                    ### Searchlight
                    else:
                        test_samples.append(vec)
                    '''
                    test_samples.append(vec)
                    test_labels.append(cat_vecs[0])

        svm_model = svm.SVC().fit(train_samples, train_labels)
        accuracy = svm_model.score(test_samples, test_labels)
        results_list.append(accuracy)

    ### Averaging performances at a given time
    average_score = numpy.average(results_list)-.5
    #median_score = numpy.median(results_list)
            
    return average_score

def run_searchlight_classification(all_args):

    exp = all_args[0]
    n = all_args[1]
    args = all_args[2]
    eeg = all_args[3]
    cluster = all_args[4]
    test_splits = all_args[5]

    places = list(cluster[0])
    start_time = cluster[1]

    eeg_scores = list()

    animals = [k for k in eeg.keys() if exp.trigger_to_info[k][1]=='animal']
    objects = [k for k in eeg.keys() if exp.trigger_to_info[k][1]=='object']
    min_len = min([len(animals), len(objects)])

    ### Reducing eeg to the relevant cluster
    eeg = {k : [vec[places, start_time:start_time+16].flatten() for vec in v] for k, v in eeg.items()}
    results_list = list()
    for c in test_splits:
        ### Train
        train_animals = [k for k in animals if k not in c]
        train_objects = [k for k in objects if k not in c]
        train_animals = random.sample(train_animals, k=min_len-1)
        train_objects = random.sample(train_objects, k=min_len-1)
        train_idxs = random.sample(train_animals+train_objects, k=(min_len-1)*2)
        train_input = [vecs[k][:, time_i] for k in train_idxs]
        for t in train_input:
            assert t.shape[0] == 128
        train_target = [exp.trigger_to_info[k][1] for k in train_idxs]

        ### Test
        test_input = [vecs[k][:, time_i] for k in c]
        for t in test_input:
            assert t.shape[0] == 128
        test_target = [exp.trigger_to_info[k][1] for k in c]

        ridge_model = sklearn.linear_model.RidgeClassifier()
        ridge_model.fit(train_input, train_target)
        accuracy = ridge_model.score(test_input, test_target)
        results_list.append(accuracy)
    accuracy_score = numpy.average(results_list)

    return [(places[0], start_time), accuracy_score]

def run_classification(all_args):

    exp = all_args[0]
    n = all_args[1]
    args = all_args[2]
    general_output_folder = all_args[3]
    
    eeg = EEGData(exp, n, args)
    data = eeg.data_dict
    times = list(eeg.times)

    mapper = {'1' : 'low', '2': 'medium', '3' : 'high'}

    for awareness, vecs in data.items():    

        if args.data_split == 'perceptual_awareness':
            awareness = mapper[awareness]

        animals = [k for k in vecs.keys() if exp.trigger_to_info[k][1]=='animal']
        objects = [k for k in vecs.keys() if exp.trigger_to_info[k][1]=='object']
        min_len = min([len(animals), len(objects)])
        if min_len <= 7:
            print('not enough data for subject {}, {}'.format(n, awareness))
            continue

        combs = list(itertools.product(animals, objects, repeat=1))
        combs = random.sample(combs, k=100)

        ### Time-resolved classification

        ### Classifying each time point
        scores_times = list()
        for time_i, time in tqdm(enumerate(times)):
            results_list = list()
            for c in combs:
                #time_t_data = {k : [vec[:, time_i] for vec in v] for k, v in vecs.items()}
                #scores_times.append(classify(time_t_data, test_splits))

                ### Train
                train_animals = [k for k in animals if k not in c]
                train_objects = [k for k in objects if k not in c]
                train_animals = random.sample(train_animals, k=min_len-1)
                train_objects = random.sample(train_objects, k=min_len-1)
                train_idxs = random.sample(train_animals+train_objects, k=(min_len-1)*2)
                train_input = [vecs[k][:, time_i] for k in train_idxs]
                for t in train_input:
                    assert t.shape[0] == 128
                train_target = [exp.trigger_to_info[k][1] for k in train_idxs]

                ### Test
                test_input = [vecs[k][:, time_i] for k in c]
                for t in test_input:
                    assert t.shape[0] == 128
                test_target = [exp.trigger_to_info[k][1] for k in c]

                ridge_model = sklearn.linear_model.RidgeClassifier()
                ridge_model.fit(train_input, train_target)
                accuracy = ridge_model.score(test_input, test_target)
                results_list.append(accuracy)
            results_list = numpy.average(results_list)
            scores_times.append(results_list)
        print(numpy.average(scores_times))

        ### Writing to file
        with open(os.path.join(general_output_folder, \
                     'sub_{:02}_{}_scores.txt'.\
                     format(n+1, awareness)), 'w') as o:
            for t in times:
                o.write('{}\t'.format(t))
            o.write('\n')
            for d in scores_times:
                o.write('{}\t'.format(d))

def run_time_resolved_rsa(all_args):

    exp = all_args[0]
    n = all_args[1]
    args = all_args[2]
    general_output_folder = all_args[3]
    
    eeg = EEGData(exp, n, args)
    data = eeg.data_dict
    times = list(eeg.times)

    mapper = {'1' : 'low', '2': 'medium', '3' : 'high'}

    comp_model = ComputationalModel(args, exp)

    for awareness, vecs in data.items():    
        if args.data_split == 'perceptual_awareness':
            awareness = mapper[awareness]

        words = [k for k in vecs.keys() if k<33]
        if len(words) <= 14:
            print('not enough data for subject {}, {}'.format(n, awareness))
            continue
        ordered_words, combs, pairwise_similarities = comp_model.compute_pairwise(words)

        ### Time-resolved RSA

        ### RSA-ing each time point
        scores_times = list()
        for time_i, time in tqdm(enumerate(times)):
            eeg_similarities = list()
            for word_one, word_two in combs:

                eeg_one = vecs[word_one][:, time_i].flatten()
                eeg_two = vecs[word_two][:, time_i].flatten()

                eeg_score = stats.pearsonr(eeg_one, eeg_two)[0]
                eeg_similarities.append(eeg_score)
            rho_score = stats.pearsonr(eeg_similarities, pairwise_similarities)[0]
            scores_times.append(rho_score)

        ### Writing to file
        with open(os.path.join(general_output_folder, \
                     'sub_{:02}_{}_scores.txt'.\
                     format(n+1, awareness)), 'w') as o:
            for t in times:
                o.write('{}\t'.format(t))
            o.write('\n')
            for d in scores_times:
                o.write('{}\t'.format(d))
