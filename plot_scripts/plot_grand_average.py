import matplotlib
import mne
import numpy
import os
import re

from matplotlib import pyplot
from tqdm import tqdm

from searchlight.searchlight_utils import SearchlightClusters

def read_results(args):
    general_output_folder = os.path.join('results', args.analysis, args.data_split)
    assert os.path.exists(general_output_folder)
    files = os.listdir(general_output_folder)
    animal_fs = [f for f in files if 'animal' in f]
    object_fs = [f for f in files if 'objects' in f]
    results = dict()
    #for k, fs in zip(['animals', 'objects'], [animal_fs, object_fs]):
    #    for f in fs:
    for data_type in ['low', 'medium', 'high']:
        for sub in range(1, 46):
            for k in ['animals', 'objects']:
                f_name = 'sub_{:02}_{}_{}_scores.txt'.format(sub, data_type, k)
                f_path = os.path.join(general_output_folder, f_name)
                if os.path.exists(f_path):
                    with open(f_path) as i:
                        lines = [l.strip().split('\t') for l in i.readlines()]
                        times = numpy.array(lines[0], dtype=numpy.float32)
                        elecs = numpy.array(lines[1:], dtype=numpy.float32)
                        assert elecs.shape[0] == 128
                        if data_type not in results.keys():
                            results[data_type] = dict()
                        if k not in results[data_type].keys():
                            results[data_type][k] = list()
                        results[data_type][k].append(elecs)
    ### Diff
    results = {k : [v_one - v_two for v_one, v_two in zip(v['animals'], v['objects'])] for k, v in results.items()}
    results = {k : [vec.T for vec in v] for k, v in results.items()}
    return results, times

def plot_grand_average(args):
    results, times = read_results(args)
    searchlight_clusters = SearchlightClusters(max_distance=30)
    electrode_indices = [searchlight_clusters.neighbors[center] \
                                        for center in range(128)]
    mne_adj_matrix = searchlight_clusters.mne_adjacency_matrix
    electrode_index_to_code = searchlight_clusters.index_to_code

    for k, v in tqdm(results.items()):
        #full_adj = mne.stats.combine_adjacency(282, mne_adj_matrix)
        v = numpy.array(v, dtype=numpy.float64)

        ### LAN (200-500ms), N400 (300-600), P600 (500-700)
        for beg, end in [(200, 500), (300, 600), (500, 700)]:
            current_indices = [t_i for t_i, t in enumerate(times) if t>=beg*0.0001 and t<=end*0.0001]
            current_diffs = numpy.average(v[:, current_indices, :], axis=1)
            promising = [26, 40, 24, 27, 31, 36, 90, 25, 28, 30, 35, 37, 38, 39, 43]
            import pdb; pdb.set_trace()

            t_stats, _, \
            p_values, _ = mne.stats.spatio_temporal_cluster_1samp_test(current_diffs,
                                                               tail=0, 
                                                               #adjacency=full_adj,
                                                               adjacency=mne_adj_matrix,
                                                               threshold=dict(start=0, step=0.2), 
                                                               n_jobs=os.cpu_count()-1, 
                                                               #n_permutations=4096, 
                                                               )
            print([k, beg, end])
            print(numpy.amin(p_values))
            original_shape = t_stats.shape
            tmin = 0.
            info = mne.create_info(\
                   ch_names=[v for k, v in \
                   electrode_index_to_code.items()], \
                   sfreq=256, \
                   ch_types='eeg')
            montage=mne.channels.make_standard_montage('biosemi128')

            ### Plotting only statistically significant time points

            ### Plotting the results

            log_p = -numpy.log(p_values)
            #log_p[log_p<=-numpy.log(0.05)] = 0.0
            log_p[log_p<=-numpy.log(0.05)] = 0.0

            log_p = log_p.reshape(original_shape).T
            evoked = mne.EvokedArray(log_p.reshape(-1, 1), info=info, tmin=tmin)

            evoked.set_montage(montage)

            title='ERP difference time points for awareness {} - {}-{}ms'.format(k, beg, end)

            evoked.plot_topomap(ch_type='eeg', \
                                time_unit='s', \
                                #times=significant_times, \
                                #times = times[::12],
                                #units='-log(p)\nif\np<=.05', \
                                #ncols=7, nrows='auto', \
                                vmin=0., \
                                scalings={'eeg':1.}, \
                                #cmap=cmap, 
                                title=title
                                )

            plot_path = os.path.join(\
                                     'plots', 
                                     args.analysis,
                                     args.data_split,
                                     )
            os.makedirs(plot_path, exist_ok=True)
            pyplot.savefig(os.path.join(plot_path, 
                            '{}_{}_{}.jpg'.
                            format(k, beg, end)), 
                            dpi=600)
            pyplot.clf()

    '''
    for k, v in tqdm(results.items()):
        for elec in range(128):
            fig,ax = pyplot.subplots(constrained_layout=True, figsize=(20, 10))
            for cat, data in v.items():
                ax.plot(times, data[elec, :], label=cat)
            ax.legend()
            ax.hlines(y=0., xmin=min(times), xmax=max(times))
            pyplot.savefig(os.path.join(plot_path, 'elec_{}.jpg'.format(elec)))
            pyplot.clf()
            pyplot.close()
    '''

