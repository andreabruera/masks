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
from sklearn import decomposition
from tqdm import tqdm
from io_utils import ExperimentInfo, EEGData, ComputationalModel

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
        #min_len = len(animals) + len(objects)
        if min_len < 7:
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
                ### Balancing animals and objects
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
                     format(n, awareness)), 'w') as o:
            for t in times:
                o.write('{}\t'.format(t))
            o.write('\n')
            for d in scores_times:
                o.write('{}\t'.format(d))

def run_time_resolved_decoding(all_args):

    exp = all_args[0]
    n = all_args[1]
    args = all_args[2]
    general_output_folder = all_args[3]
    mapper = {'1' : 'low', '2': 'medium', '3' : 'high'}

    comp_model = ComputationalModel(args, exp)
    
    ### Averaging 50 runs
    results = dict()
    repetitions = int(args.n_iterations)
    for _ in tqdm(range(repetitions)):
    
        eeg = EEGData(exp, n, args)
        data = eeg.data_dict
        data = {k : v for k, v in data.items() if len(v)>0}
        all_times = list(eeg.times)

        for awareness, aw_vecs in data.items():    
            '''
            ### PCA
            original_shape = list(set([v.shape for v in aw_vecs.values()]))[0]
            pca = sklearn.decomposition.PCA(n_components=0.99)
            pcas = pca.fit_transform([v.flatten() for k, v in aw_vecs.items()])
            pcas = pca.inverse_transform(pcas)
            assert len(pcas) == len(aw_vecs.keys())
            aw_vecs = {k[0] : pcas[k_i].reshape(original_shape) for k_i, k in enumerate(aw_vecs.items())}
            '''

            ### Subsampling
            times = [all_times[min(i, i+1)] for i in list(range(len(all_times)))[::3]]
            vecs = {k : [numpy.average(v[:, i:i+3], axis=1) for i in list(range(len(v))[::3])] for k, v in aw_vecs.items()}
            vecs = {k : numpy.array(v) for k, v in vecs.items()}
            ### Not subsampling
            #times = all_times.copy()
            #vecs = aw_vecs.copy()
            if args.data_split == 'perceptual_awareness':
                awareness = mapper[awareness]
            if awareness not in results.keys():
                results[awareness] = list()

            if args.balance_semantic_domains:
                ### Balancing animals and objects
                animals = [k for k in vecs.keys() if exp.trigger_to_info[k][1]=='animal']
                objects = [k for k in vecs.keys() if exp.trigger_to_info[k][1]=='object']
                min_len = min([len(animals), len(objects)])

                train_animals = random.sample(animals, k=min_len)
                train_objects = random.sample(objects, k=min_len)
                words = train_animals + train_objects
            else:
                ### not balancing, just checking number of items
                words = [k for k in vecs.keys() if k<33]
            ### Checking if there's enough words
            if len(words) < int(args.minimum_number_words):
                print('not enough data for subject {}, {}'.format(n, awareness))
                continue
            ### If that's the case using only that amount of words
            else:
                words = random.sample(words, k=int(args.minimum_number_words))

            ### Computing pairwise word correlations
            ordered_words, combs, pairwise_similarities = comp_model.compute_pairwise(words)
            model_vectors = {k : [comp_model.word_sims[exp.trigger_to_info[k][0], exp.trigger_to_info[w][0]] for w in ordered_words if k!=w] for k in ordered_words}

            ### Time-resolved RSA

            ### RSA-ing each time point
            scores_times = list()
            for time_i, time in enumerate(times):
                time_vecs = {k : v[:, time_i].flatten() for k, v in vecs.items() if k in words}
                ridge = sklearn.linear_model.Ridge()
                scores = list()
                for c in combs:
                    train_input = [v for k, v in time_vecs.items() if k not in c]
                    train_target = [model_vectors[k] for k, v in time_vecs.items() if k not in c]
                    test_input = [time_vecs[c[0]], time_vecs[c[1]]]
                    test_target = [model_vectors[c[0]], model_vectors[c[1]]]
                    ridge.fit(train_input, train_target)
                    pred = ridge.predict(test_input)
                    assert pred.shape[0] == 2
                    correct = 0.
                    wrong = 0.
                    for i_one in range(2):
                        for i_two in range(2):
                            corr = scipy.stats.pearsonr(pred[i_one], test_target[i_two])[0]
                            if i_one == i_two:
                                correct += corr
                            else:
                                wrong += corr
                    if correct > wrong:
                        scores.append(1.)
                    else:
                        scores.append(0.)

                score = numpy.average(scores) - 0.5

                scores_times.append(score)
            results[awareness].append(scores_times)

    results = {k : numpy.average(v, axis=0) for k, v in results.items() if len(v)==repetitions}
    for awareness, scores_times in results.items():
        ### Writing to file
        with open(os.path.join(general_output_folder, \
                     'sub_{:02}_{}_scores.txt'.\
                     format(n, awareness)), 'w') as o:
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
    mapper = {'1' : 'low', '2': 'medium', '3' : 'high'}

    comp_model = ComputationalModel(args, exp)
    
    ### Averaging 50 runs
    results = dict()
    repetitions = int(args.n_iterations)
    for _ in tqdm(range(repetitions)):
    
        eeg = EEGData(exp, n, args)
        data = eeg.data_dict
        data = {k : v for k, v in data.items() if len(v)>0}
        all_times = list(eeg.times)

        for awareness, aw_vecs in data.items():    
            '''
            ### PCA
            original_shape = list(set([v.shape for v in aw_vecs.values()]))[0]
            pca = sklearn.decomposition.PCA(n_components=0.99)
            pcas = pca.fit_transform([v.flatten() for k, v in aw_vecs.items()])
            pcas = pca.inverse_transform(pcas)
            assert len(pcas) == len(aw_vecs.keys())
            aw_vecs = {k[0] : pcas[k_i].reshape(original_shape) for k_i, k in enumerate(aw_vecs.items())}
            '''

            ### Subsampling
            #times = [all_times[min(i, i)] for i in list(range(len(all_times)))[::2]]
            #vecs = {k : [numpy.average(v[:, i:i+2], axis=1) for i in list(range(len(v))[::2])] for k, v in aw_vecs.items()}
            #vecs = {k : numpy.array(v) for k, v in vecs.items()}
            times = all_times.copy()
            vecs = aw_vecs.copy()
            if args.data_split == 'perceptual_awareness':
                awareness = mapper[awareness]
            if awareness not in results.keys():
                results[awareness] = list()

            if args.balance_semantic_domains:
                ### Balancing animals and objects
                animals = [k for k in vecs.keys() if exp.trigger_to_info[k][1]=='animal']
                objects = [k for k in vecs.keys() if exp.trigger_to_info[k][1]=='object']
                min_len = min([len(animals), len(objects)])

                train_animals = random.sample(animals, k=min_len)
                train_objects = random.sample(objects, k=min_len)
                words = train_animals + train_objects
            else:
                ### not balancing, just checking number of items
                words = [k for k in vecs.keys() if k<33]
            ### Checking if there's enough words
            if len(words) < int(args.minimum_number_words):
                print('not enough data for subject {}, {}'.format(n, awareness))
                continue
            ### If that's the case using only that amount of words
            else:
                words = random.sample(words, k=int(args.minimum_number_words))

            ### Computing pairwise word correlations
            ordered_words, combs, pairwise_similarities = comp_model.compute_pairwise(words)

            ### Time-resolved RSA

            ### RSA-ing each time point
            scores_times = list()
            for time_i, time in enumerate(times):
                eeg_similarities = list()
                for word_one, word_two in combs:

                    eeg_one = vecs[word_one][:, time_i].flatten()
                    eeg_two = vecs[word_two][:, time_i].flatten()

                    eeg_score = 1. - stats.pearsonr(eeg_one, eeg_two)[0]
                    eeg_similarities.append(eeg_score)
                rho_score = stats.pearsonr(eeg_similarities, pairwise_similarities)[0]
                scores_times.append(rho_score)
            results[awareness].append(scores_times)

    results = {k : numpy.average(v, axis=0) for k, v in results.items() if len(v)==repetitions}
    for awareness, scores_times in results.items():
        ### Writing to file
        with open(os.path.join(general_output_folder, \
                     'sub_{:02}_{}_scores.txt'.\
                     format(n, awareness)), 'w') as o:
            for t in times:
                o.write('{}\t'.format(t))
            o.write('\n')
            for d in scores_times:
                o.write('{}\t'.format(d))

def run_grand_average(all_args):

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
        #min_len = len(animals) + len(objects)
        if min_len < 4:
            print('not enough data for subject {}, {}'.format(n, awareness))
            continue

        ### Balancing animals and objects
        train_animals = random.sample(animals, k=min_len)
        train_objects = random.sample(objects, k=min_len)

        animals_avg = numpy.average([vecs[k] for k in train_animals], axis=0)
        objects_avg = numpy.average([vecs[k] for k in train_objects], axis=0)

        for obj_type, avg in zip(['animals', 'objects'], [animals_avg, objects_avg]):
            ### Writing to file
            with open(os.path.join(general_output_folder, \
                         'sub_{:02}_{}_{}_scores.txt'.\
                         format(n, awareness, obj_type)), 'w') as o:
                for t in times:
                    o.write('{}\t'.format(t))
                o.write('\n')
                for elec in avg:
                    for val in elec:
                        o.write('{}\t'.format(val))
                    o.write('\n')
