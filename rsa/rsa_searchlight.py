import numpy
import os

from scipy import stats

def finalize_rsa_searchlight(results, relevant_times, explicit_times, output_folder, awareness, n):

    ### Turning results into a single array

    results_array = list()
    results_dict = {r[0] : r[1] for r in results}

    for e in range(128):
        e_row = list()
        for t in relevant_times:
            e_row.append(results_dict[(e, t)])
        results_array.append(e_row)

    results_array = numpy.array(results_array)

    ### Writing to file

    with open(os.path.join(output_folder, '{}_sub-{:02}.rsa'.format(awareness, n+1)), 'w') as o:
        for t in explicit_times:
            o.write('{}\t'.format(t))
        o.write('\n')
        for e in results_array:
            for t in e:
                o.write('{}\t'.format(t))
            o.write('\n')

def run_searchlight(all_args): 

    eeg = all_args[0]
    comp_model = all_args[1] 
    cluster = all_args[2]
    word_combs = all_args[3]
    pairwise_similarities = all_args[4]

    places = list(cluster[0])
    start_time = cluster[1]

    eeg_similarities = list()

    for word_one, word_two in word_combs:

        eeg_one = eeg[word_one][places, start_time:start_time+16].flatten()
        eeg_two = eeg[word_two][places, start_time:start_time+16].flatten()

        word_comb_score = stats.spearmanr(eeg_one, eeg_two)[0]
        eeg_similarities.append(word_comb_score)

    rho_score = stats.spearmanr(eeg_similarities, pairwise_similarities)[0]
    #print('done with {} {}'.format(places[0], start_time))

    return [(places[0], start_time), rho_score]

