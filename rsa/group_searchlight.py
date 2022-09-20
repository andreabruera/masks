import argparse
import os
import collections
import numpy
import re
import logging
import itertools
import functools
import mne
import pickle
import scipy
import multiprocessing

from scipy import stats
from matplotlib import pyplot
from tqdm import tqdm

def run_group_searchlight(args, exp, clusters, input_folder):

    pyplot.rcParams['figure.constrained_layout.use'] = True
    if args.analysis == 'group_classification_searchlight':
        input_folder = input_folder.replace(\
                       'group_classification', 'classification')
        args.computational_model = ''
        marker = 'classification'
    else:
        input_folder = input_folder.replace('group_rsa', 'rsa')
        marker = '{}_rsa'.format(args.computational_model)
    input_folder = os.path.join(input_folder, args.computational_model)

    missing_per_condition = dict()
    present_per_condition = dict()

    electrode_index_to_code = clusters.index_to_code
    mne_adj_matrix = clusters.mne_adjacency_matrix

    plot_path = os.path.join('plots', args.analysis, \
                             args.experiment_id, args.data_split)
    os.makedirs(plot_path, exist_ok=True)

    if args.data_split == 'subjective_judgments':
        awareness_levels = ['low', 'medium', 'high']
        cmap = 'PuBu'
    elif args.data_split == 'objective_accuracy':
        awareness_levels = ['correct', 'wrong']
        cmap = 'YlOrBr'

    for awareness in tqdm(awareness_levels):

        all_subjects = list()

        for n in range(1, exp.n_subjects+1):

            file_path = os.path.join(input_folder, '{}_sub-{:02}.rsa'.format(awareness, n))

            if os.path.exists(file_path):
                with open(file_path, 'r') as i:
                    lines = [l.strip().split('\t') for l in i.readlines()]
                times = [float(w) for w in lines[0]]
                electrodes = numpy.array([[float(v) for v in l] for l in lines[1:]]).T
                all_subjects.append(electrodes)
            else:
                if awareness not in missing_per_condition.keys():
                    missing_per_condition[awareness] = [n]
                else:
                    missing_per_condition[awareness].append(n)

        all_subjects = numpy.array(all_subjects)
        present_per_condition[awareness] = all_subjects.shape[0]

        t_stats, _, \
        p_values, _ = mne.stats.spatio_temporal_cluster_1samp_test(all_subjects, \
                                                           tail=1, \
                                                           adjacency=mne_adj_matrix, \
                                                           threshold=dict(start=0, step=0.2), \
                                                           n_jobs=os.cpu_count()-1, \
                                                           #n_permutations=4096, \
                                                           n_permutations='all', \
                                                           )

        original_shape = t_stats.shape
        tmin = times[0]
        info = mne.create_info(\
               ch_names=[v for k, v in \
               electrode_index_to_code.items()], \
               #the step is 8 samples, so we divide the original one by 7
               sfreq=256/8, \
               ch_types='eeg')
        montage=mne.channels.make_standard_montage('biosemi128')

        ### Plotting only statistically significant time points

        print(numpy.amin(p_values))
        ### Plotting the results

        log_p = -numpy.log(p_values)
        log_p[log_p<=-numpy.log(0.05)] = 0.0
        #log_p[log_p<=-numpy.log(0.005)] = 0.0

        log_p = log_p.reshape(original_shape).T

        #significant_points = res[2].reshape(res[0].shape).T
        #significant_points = -numpy.log(significant_points)
        #significant_points[significant_points<=-numpy.log(0.05)] = 0.0

        significant_indices = [i[0] for i in enumerate(numpy.nansum(log_p.T, axis=1)>0.) if i[1]==True]
        significant_times = [times[i] for i in significant_indices]
        print(significant_times)
        #print(significant_times)
        #relevant_times
        if len(significant_times) >= 1:

            evoked = mne.EvokedArray(log_p, info=info, tmin=tmin)

            evoked.set_montage(montage)

            title='Significant time points for {} '\
                  '- awareness: {}'.format(\
                  marker, awareness)

            evoked.plot_topomap(ch_type='eeg', \
                                time_unit='s', \
                                times=significant_times, \
                                units='-log(p)\nif\np<=.05', \
                                ncols=7, nrows='auto', \
                                vmin=0., \
                                scalings={'eeg':1.}, \
                                cmap=cmap, title=title)

            pyplot.savefig(os.path.join(plot_path, \
                            '{}_{}_significant_points.png'.\
                            format(awareness, \
                            marker)), dpi=600)
            pyplot.clf()

        ### Plotting all the rest
        all_p = p_values.reshape(original_shape).T
        #all_t = t_stats.reshape(original_shape).T
        evoked = mne.EvokedArray(all_p, info=info, tmin=tmin)

        evoked.set_montage(montage)

        title='Visualization of the time points for {}'\
              ' - awareness: {}'.format(\
              marker, awareness)

        evoked.plot_topomap(ch_type='eeg', time_unit='s', \
                            times=times, \
                            units='permutation\np-values', \
                            ncols=7, 
                            nrows='auto', \
                            vmax=1., vmin=0., \
                            scalings={'eeg':1.}, \
                            cmap= '{}_r'.format(cmap), \
                            title=title)
        pyplot.savefig(os.path.join(plot_path, \
                        '{}_{}_all_points.png'.format(\
                          awareness, marker)), dpi=600)
        pyplot.clf()

    with open(os.path.join(plot_path, 'missing_present_subjects_log.txt'), 'w') as o:
        for k, v in missing_per_condition.items():
            o.write('Condition\t{}\tmissing subjects\t'.format(k))
            for sub in v:
                o.write('{} '.format(sub))
            o.write('\n')
            o.write('Condition\t{}\tpresent subjects\t'.format(k))
            o.write('{} '.format(present_per_condition[k]))
            o.write('\n')
