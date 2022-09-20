import collections
import itertools
import os
import scipy
import random
import mne
import numpy
import scipy

from tqdm import tqdm


from io_utils import EvokedResponses
from searchlight import SearchlightClusters, run_searchlight

def restrict_evoked_responses(args, evoked_responses):

    events = evoked_responses.events
    epochs = evoked_responses.original_epochs

    assert len(events) == len(epochs)

    if args.word_selection == 'targets_only':
        target_indices = [ind for ind, ev in enumerate(events.items()) if ev[1][1]=='target']
        events_infos = [v for k, v in events.items() if v[1]=='target']
        del events
        events = {i : v for i, v in enumerate(events_infos)}
        epochs = [epochs[ind] for ind in target_indices]

    assert len(events) == len(epochs)

    '''
    if args.permutation:
        event_keys = [k for k in events.keys()]
        events_infos = [v for k, v in events.items()]
        ### Randomizing the relevant experimental condition
        if args.analysis == 'objective_accuracy':
            random_conditions = random.sample([v[2] for v in events_infos], k=len(event_keys))
            random_event_values = [[events_infos[i][0], events_infos[i][1], random_conditions[i], events_infos[i][3]] for i in range(len(events_infos))]
        elif args.analysis == 'subjective_judgments':
            random_conditions = random.sample([v[3] for v in events_infos], k=len(event_keys))
            random_event_values = [[events_infos[i][0], events_infos[i][1], events_infos[i][2], random_conditions[i]] for i in range(len(events_infos))]
        events = {k : v for k, v in zip(event_keys, random_event_values)}
    '''

    restricted_evoked_responses = collections.defaultdict(lambda : collections.defaultdict(list))

    for current_epoch_index in range(len(events)):

        current_event = events[current_epoch_index]
        # 0 = word, 1 = target/filler, 2 = accuracy, 3 = awareness 
        word = current_event[0]
        accuracy = current_event[2]
        awareness = current_event[3]

        ### Creating 2 conditions: correct/wrong
        if args.analysis == 'objective_accuracy':
            restricted_evoked_responses[accuracy][word].append(epochs[current_epoch_index])
        ### Creating 3 conditions: 1/2/3
        elif args.analysis == 'subjective_judgments':
            restricted_evoked_responses[awareness][word].append(epochs[current_epoch_index])
        elif args.analysis == 'both_worlds':
            if awareness == 'medium' and accuracy == 'correct':
                awareness = 'aware'
            elif awareness == 'medium' and accuracy == 'wrong':
                awareness = 'unaware'
            else:
                awareness = 'aware' if awareness == 'high' else 'unaware'
            restricted_evoked_responses[awareness][word].append(epochs[current_epoch_index])

    return restricted_evoked_responses

def run_all_electrodes_rsa(evoked_dict, word_combs, computational_scores, time_points): 

    current_condition_rho = collections.defaultdict(list)

    for relevant_time in tqdm(time_points):

        eeg_similarities = list()

        for word_one, word_two in word_combs:

            eeg_one = evoked_dict[word_one][:, relevant_time]
            eeg_two = evoked_dict[word_two][:, relevant_time]

            word_comb_score = scipy.stats.spearmanr(eeg_one, eeg_two)[0]
            eeg_similarities.append(word_comb_score)

        rho_score = scipy.stats.spearmanr(eeg_similarities, computational_scores)[0]
        current_condition_rho['all electrodes'].append(rho_score)

    return current_condition_rho

def prepare_folder(args, s, permutation=0):

    brain_approach = 'searchlight' if args.searchlight else 'all_electrodes'
    analysis = args.analysis
    words = args.word_selection
    computational_model = args.computational_model
    permutation = 'permutation_{:03}'.format(permutation) if permutation != 0 else 'true'

    if args.searchlight:
        searchlight_params = 'window_{}_hop_{}'.format(args.temporal_window_size, args.hop)
    else:
        searchlight_params = ''

    folder_path = os.path.join('rsa_maps', 'minimum_{}'.format(args.minimum_ERPs), brain_approach, searchlight_params, \
                                analysis, words, \
                                computational_model, \
                                'sub-{:02}'.format(s), permutation)

    os.makedirs(folder_path, exist_ok=True)

    return folder_path
    

def run_rsa(args, s, evoked_responses, computational_model, all_time_points, permutation=0):

    subject_folder = prepare_folder(args, s, permutation)
    
    ### Selecting evoked responses for the current pairwise similarity computations
    selected_evoked = restrict_evoked_responses(args, evoked_responses)

    subject_results = collections.defaultdict(dict)
    subject_info = collections.defaultdict(lambda : collections.defaultdict(list))

    for condition, evoked_dict in selected_evoked.items():

        #print('Current condition: {}'.format(condition))

        evoked_dict = {k : numpy.average(v, axis=0) for k, v in evoked_dict.items() if len(v) >= args.minimum_ERPs}
        
        if args.permutation:
            keys = random.sample(evoked_dict.keys(), k=len(evoked_dict.keys()))
            evoked_dict = {k : v[1] for k, v in zip(keys, evoked_dict.items())}

        present_words = [k for k in evoked_dict.keys()]

        subject_info[condition]['words used'] = present_words

        word_combs = [k for k in itertools.combinations(present_words, r=2)]

        computational_scores = list()

        for word_one, word_two in word_combs:

            computational_scores.append(computational_model[word_one][word_two])

        ### Differentiating among searchlight and full-cap RSA
        if args.searchlight:
            time_points = [t for t in range(0, len(all_time_points), args.hop)]
            print(len(time_points))
            current_condition_rho = run_searchlight(evoked_dict, word_combs, computational_scores, time_points, args.temporal_window_size) 
        else:
            time_points = [k for k in all_time_points.keys()]
            current_condition_rho = run_all_electrodes_rsa(evoked_dict, word_combs, computational_scores, time_points) 

        subject_results[condition] = current_condition_rho

    ### Writing to file
    for condition, condition_dict in subject_results.items():

        print('Now writing results for {}'.format(condition))
        ### Writing the Spearman rho maps
        with open(os.path.join(subject_folder, '{}.map'.format(condition)), 'w') as o:
            if args.searchlight:
                o.write('Searchlight cluster index\tSpearman Rho per time window\n')
            else:
                o.write('Electrode type\tSpearman Rho per time window\n')
            for cluster_index, rho_map in condition_dict.items():
                o.write('{}\t'.format(cluster_index))
                for rho in rho_map:
                    o.write('{}\t'.format(rho))
                o.write('\n')

    if not args.permutation:
        ### Writing the words actually used
        with open(os.path.join(subject_folder, 'words_used_info.txt'), 'w') as o:
            for condition, condition_info in subject_info.items():
                words_used = condition_info['words used']
                o.write('Condition:\t{}\nNumber of words used:\t{}\n\n'.format(condition, len(words_used)))
