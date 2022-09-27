import matplotlib
import mne
import numpy
import os
import scipy

from matplotlib import pyplot
from scipy import stats

def check_statistical_significance(args, setup_data, significance):

    if not type(setup_data) == numpy.array:
        setup_data = numpy.array(setup_data)
    ### Checking for statistical significance
    if args.analysis == 'time_resolved_rsa':
        random_baseline = 0.
    else:
        random_baseline = .5
    ### T-test
    '''
    original_p_values = stats.ttest_1samp(setup_data, \
                         popmean=random_baseline, \
                         alternative='greater').pvalue
    '''
    significance_data = setup_data - random_baseline
    ### Wilcoxon
    '''
    original_p_values = list()
    for t in significance_data.T:
        p = stats.wilcoxon(t, alternative='greater')[1]
        original_p_values.append(p)
    '''
    adj = numpy.zeros((setup_data.shape[-1], setup_data.shape[-1]))
    for i in range(setup_data.shape[-1]):
        #if args.subsample == 'subsample_2' or args.data_kind != 'erp':
        #if args.subsample == 'subsample_2':
        #    win = range(1, 2)
        #else:
        win = range(1, 4)
        for window in win:
            adj[i, max(0, i-window)] = 1
            adj[i, min(setup_data.shape[-1]-1, i+window)] = 1
    adj = scipy.sparse.coo_matrix(adj)
    t_stats, _, \
    original_p_values,\
    _ = mne.stats.spatio_temporal_cluster_1samp_test(\
                               significance_data, \
                               tail=1, \
                               adjacency=adj, \
                               threshold=dict(start=0, step=0.2), \
                               n_jobs=os.cpu_count()-1, \
                               n_permutations=4096, \
                               #n_permutations='all', \
                               )
    assert len(original_p_values) == setup_data.shape[-1]

    ### FDR correction
    corrected_p_values = original_p_values.copy()
    #corrected_p_values = mne.stats.fdr_correction(original_p_values)[1]
    print(min(corrected_p_values))
    significant_indices = [i for i, v in enumerate(corrected_p_values) if v<=significance]

    return significant_indices

def possibilities(args):

    if args.data_split == 'perceptual_awareness':
        categories = ['low', 'medium', 'high'] 
    elif args.data_split == 'best_case':
        categories = ['best_case']
    elif args.data_split == 'worst_case':
        categories = ['worst_case']
    elif args.data_split == 'all_cases':
        categories = list()
        for pas_label in ['1', '2', '3']:
            for acc_label in ['correct', 'wrong']:
                label = '{}_{}'.format(pas_label, acc_label)
                categories.append(label)
    elif args.data_split == 'grand_average':
        categories = ['grand_average']
    else:
        categories = ['correct', 'wrong']
    
    data_dict = {c : list() for c in categories}

    return data_dict

def read_files(args):

    subjects = list(range(1, 46))
    bads = [8, 22, 25, 28, 37]
    subjects = [s for s in subjects if s not in bads]

    data_dict = possibilities(args)

    for cat, values in data_dict.items():
        path = os.path.join(\
                            'results', 
                            args.analysis, \
                            args.data_split,
                            )
        if args.analysis == 'time_resolved_rsa':
            path = os.path.join(path, args.computational_model)
        assert os.path.exists(path)

        file_lambda = lambda arg_list : os.path.join(arg_list[0], \
                      'sub_{:02}_{}_scores.txt'.format(\
                      arg_list[1], 
                      arg_list[2])
                      )

        for sub in subjects:
            file_path = file_lambda([path, sub, cat])
            try:
                assert os.path.exists(file_path)
                with open(file_path) as i:
                    lines = [l.strip().split('\t') \
                             for l in i.readlines()]
                assert len(lines) == 2
                times = [float(v) for v in lines[0]]
                lines = [float(v) for v in lines[1]]

                ### Shortening
                lines = [lines[t_i] for t_i, t in enumerate(times) if t>-.05 and t<0.9]
                times = [t for t in times if t>-.05 and t<0.9]

                data_dict[cat].append(lines)
            except AssertionError:
                #print('missing file: {}'.format(file_path))
                pass

    return data_dict, times

def plot_classification(args):

    colors = {'low' : (86., 180., 233.),
              'medium' : (240., 228., 66.),
              'high' : (230., 159., 0.),
              'wrong' : (0., 114., 178.),
              'correct' : (213., 94., 0.),
              }

    colors = {k : numpy.array(v)/255 for k, v in colors.items()}
    significance = 0.08

    data_dict, times = read_files(args)

    plot_path = os.path.join(\
                             'plots', 
                             args.analysis,
                             args.data_split
                             )
    os.makedirs(plot_path, exist_ok=True)

    ### Preparing a double plot
    fig, ax = pyplot.subplots(\
                              nrows=2, ncols=1,
                              gridspec_kw={'height_ratios': [4, 1]}, 
                              figsize=(20,9),
                              constrained_layout=True,
                              )
    ### Set limits
    ax[0].set_xlim(min(times), max(times))
    ax[1].set_xlim(min(times), max(times))
    ax[0].set_ylim(-.05, .135)

    ### Main plot properties

    title = 'Classification scores '\
            'for 128 electrodes\n'\
            'data split according to: '\
            '{}\n'.format(args.data_split)
    title = title.replace('_', ' ')
    if args.analysis == 'time_resolved_rsa':
        title = '{} - model: {}'.format(title, args.computational_model)
        title = title.replace('Classification', 'Time-resolved RSA')
    ax[0].set_title(title, pad=20)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].set_ylabel('Pearson correlation (EEG x model)', labelpad=10)

    if args.analysis == 'time_resolved_rsa':
        random_baseline = 0.
    else:
        random_baseline = .5
    ax[0].hlines(y=random_baseline, xmin=times[0], \
                 xmax=times[-1], color='darkgrey', \
                 linestyle='dashed')

    ### Legend properties

    ### Removing all the parts surrounding the plot below
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)

    ### Setting colors for plotting different setups
    #cmap = cm.get_cmap('plasma')
    #colors = [cmap(i) for i in range(32, 220, int(220/len(folders)))]

    number_lines = len([1 for k, v in data_dict.items()])

    line_counter = 0
    ax[1].text(0., 0., 'p<={}'.format(significance), va='center', ha='center')

    for cat, subs in data_dict.items():

        #color = (numpy.random.random(), numpy.random.random(), numpy.random.random())
        color = colors[cat]
        print(color)

        subs = numpy.array(subs)

        if subs.shape[0] < 15:
            continue
        
        sig_indices = check_statistical_significance(args, subs, significance)

        average_data = numpy.average(subs, axis=0)
        sem_data = stats.sem(subs, axis=0)
        ax[0].plot(times, average_data, linewidth=.5, color=color)
        #ax[0].errorbar(x=times, y=numpy.average(subs, axis=0), \
                       #yerr=stats.sem(subs, axis=0), \
                       #color=colors[f_i], \
                       #elinewidth=.5, linewidth=1.)
                       #linewidth=.5)
        ax[0].fill_between(times, average_data-sem_data, average_data+sem_data, \
                           color=color, alpha=0.08)
        #for t_i, t in enumerate(times):
            #ax[0].violinplot(dataset=setup_data[:, t_i], positions=[t_i], showmedians=True)
        
        ax[0].scatter([times[t] for t in sig_indices], \
        #ax[0].scatter(significant_indices, \
                   [numpy.average(subs, axis=0)[t] \
                        for t in sig_indices], \
                        color='white', \
                        edgecolors='black', \
                        s=9., linewidth=.5)

        ### Plot below for significance
        ax[1].scatter([times[t] for t in sig_indices], \
                      [line_counter*0.01 for t in sig_indices], \
                        color=color, \
                        edgecolors='white', \
                        s=9., linewidth=.5)

        ax[0].scatter(\
                      line_counter*0.15, 
                      .13, 
                      marker='H', 
                      color=color, 
                      s=100,
                      )
        ax[0].text(\
                   0.01+(line_counter*0.15), 
                   .13, 
                   '{} - n={}'.format(cat, subs.shape[0]), 
                   va='center', 
                   ha='left',
                   )
        line_counter += 1

    out_file = os.path.join(plot_path,
                   'classification_{}_{}.jpg'.format(\
                    args.analysis,
                    args.data_split,
                    )
                   )
    if args.analysis == 'time_resolved_rsa':
        out_file = out_file.replace('.jpg', '_{}.jpg'.format(args.computational_model))
    pyplot.savefig(out_file)
    pyplot.clf()
    pyplot.close(fig)

def plot_classification_subject_per_subject(args):

    data_dict, times = read_files(args)

    plot_path = os.path.join('plots', 
                             '{}_subject_per_subject'.format(args.analysis), 
                             args.data_split,
                             )
    os.makedirs(plot_path, exist_ok=True)

    for subject in range(1, 46):

        ### Preparing a double plot
        fig, ax = pyplot.subplots(nrows=2, ncols=1, \
                                  gridspec_kw={'height_ratios': [4, 1]}, \
                                  figsize=(20,9),
                                  constrained_layout=True)
        ### Set limits
        ax[0].set_xlim(min(times), max(times))
        ax[1].set_xlim(min(times), max(times))
        ax[0].set_ylim(-.05, .135)

        ### Main plot properties

        title = 'Classification scores for 128 electrodes\n'\
                'data split according to: {}'.format(\
                        args.data_split)
        title = title.replace('_', ' ')
        if args.analysis == 'time_resolved_rsa':
            title = '{}\nmodel: {}'.format(title, args.computational_model)
            title = title.replace('Classification', 'Time-resolved RSA')
        ax[0].set_title(title, pad=20)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)

        if args.analysis == 'time_resolved_rsa':
            random_baseline = 0.
        else:
            random_baseline = .5
        ax[0].hlines(y=random_baseline, xmin=times[0], \
                     xmax=times[-1], color='darkgrey', \
                     linestyle='dashed')

        ### Legend properties

        ### Removing all the parts surrounding the plot below
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['left'].set_visible(False)
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)

        ### Setting colors for plotting different setups
        #cmap = cm.get_cmap('plasma')
        #colors = [cmap(i) for i in range(32, 220, int(220/len(folders)))]

        number_lines = len([1 for k, v in data_dict.items()])

        line_counter = 0
        ax[1].text(0., 0., 'p<=0.05', va='center', ha='center')
        marker = True

        for cat, subs in data_dict.items():

            color = (numpy.random.random(), numpy.random.random(), numpy.random.random())

            subs = numpy.array(subs)
            
            try:
                ax[0].plot(times, subs[subject], linewidth=.5, color=color)
            except IndexError:
                marker = False
                continue
            line_counter += 1
        if marker:

            out_file = os.path.join(plot_path,
                           'classification_subject_{}_{}_{}.jpg'.format(subject, args.analysis, \
                            args.data_split))
            if args.analysis == 'time_resolved_rsa':
                out_file = out_file.replace('.jpg', '{}.jpg'.format(args.computational_model))
            pyplot.savefig(out_file)
            pyplot.clf()
            pyplot.close(fig)
