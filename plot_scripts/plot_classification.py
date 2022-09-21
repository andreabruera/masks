import matplotlib
import mne
import numpy
import os

from matplotlib import pyplot
from scipy import stats

def check_statistical_significance(args, setup_data):

    if not type(setup_data) == numpy.array:
        setup_data = numpy.array(setup_data)
    ### Checking for statistical significance
    random_baseline = .5
    ### T-test
    '''
    original_p_values = stats.ttest_1samp(setup_data, \
                         popmean=random_baseline, \
                         alternative='greater').pvalue
    '''
    ### Wilcoxon
    significance_data = setup_data.T - random_baseline
    original_p_values = list()
    for t in significance_data:
        p = stats.wilcoxon(t, alternative='greater')[1]
        original_p_values.append(p)

    assert len(original_p_values) == setup_data.shape[-1]

    ### FDR correction
    corrected_p_values = mne.stats.fdr_correction(original_p_values)[1]
    significant_indices = [i for i, v in enumerate(corrected_p_values) if v<=0.05]

    return significant_indices

def possibilities(args):

    categories = ['low', 'medium', 'high'] if args.data_split == \
                  'perceptual_awareness' else ['correct', 'wrong']
    
    ### ERP
    if args.data_kind == 'erp':

        data_dict = {'' : {c : list() for c in categories}}

    ### Time-frequency
    elif args.data_kind == 'time_frequency':

        frequencies = numpy.arange(1, 40, 3)
        data_dict = {'{}_hz_'.format(hz) : {c : list() for c in categories}}

    return data_dict

def read_files(args):

    subjects = [s for s in range(1, 46) if s not in []]

    data_dict = possibilities(args)

    for hz, v in data_dict.items():

        for cat, values in v.items():
            path = os.path.join('results', 
                                args.analysis, \
                                args.data_split)
            assert os.path.exists(path)

            file_lambda = lambda arg_list : os.path.join(arg_list[0], \
                          '{}sub_{:02}_{}_scores.txt'.format(\
                          arg_list[1], arg_list[2], arg_list[3]))

            for sub in subjects:
                file_path = file_lambda([path, hz, sub, cat])
                try:
                    assert os.path.exists(file_path)
                    with open(file_path) as i:
                        lines = [l.strip().split('\t') \
                                 for l in i.readlines()]
                    assert len(lines) == 2
                    times = [float(v) for v in lines[0]]
                    lines = [float(v) for v in lines[1]]
                    data_dict[hz][cat].append(lines)
                except AssertionError:
                    print('missing file: {}'.format(file_path))

    return data_dict, times

def plot_classification(args):

    data_dict, times = read_files(args)

    data_dict = {k : {k_two : [vec for vec in v_two if len(vec)==282] for k_two, v_two in v.items()} for k, v in data_dict.items()}

    plot_path = os.path.join('plots', args.analysis, \
                             args.data_split)
    os.makedirs(plot_path, exist_ok=True)

    for hz, v in data_dict.items():

        ### Preparing a double plot
        fig, ax = pyplot.subplots(nrows=2, ncols=1, \
                                  gridspec_kw={'height_ratios': [4, 1]}, \
                                  figsize=(12,5))

        ### Main plot properties

        title = 'Classification scores for {} data\n'\
                        'type of analysis {} - {}'.format(\
                        args.analysis, args.data_split, hz)
        title = title.replace('_', ' ')
        ax[0].set_title(title)

        random_baseline = .5
        ax[0].hlines(y=random_baseline, xmin=times[0], \
                     xmax=times[-1], color='darkgrey', \
                     linestyle='dashed')

        ### Legend properties

        ### Removing all the parts surrounding the plot below
        ax[1].set_xlim(left=0., right=1.)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['left'].set_visible(False)
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)

        ### Setting colors for plotting different setups
        #cmap = cm.get_cmap('plasma')
        #colors = [cmap(i) for i in range(32, 220, int(220/len(folders)))]

        number_lines = len([1 for k, v in data_dict.items() \
                           for d in v.keys()])

        line_counter = 0

        for cat, subs in v.items():

            subs = numpy.array(subs)
            
            sig_indices = check_statistical_significance(args, subs)

            average_data = numpy.average(subs, axis=0)
            sem_data = stats.sem(subs, axis=0)
            ax[0].plot(times, average_data, linewidth=.5)
            #ax[0].errorbar(x=times, y=numpy.average(subs, axis=0), \
                           #yerr=stats.sem(subs, axis=0), \
                           #color=colors[f_i], \
                           #elinewidth=.5, linewidth=1.)
                           #linewidth=.5)
            ax[0].fill_between(times, average_data-sem_data, average_data+sem_data, \
                               alpha=0.08)
            #for t_i, t in enumerate(times):
                #ax[0].violinplot(dataset=setup_data[:, t_i], positions=[t_i], showmedians=True)
            
            ax[0].scatter([times[t] for t in sig_indices], \
            #ax[0].scatter(significant_indices, \
                       [numpy.average(subs, axis=0)[t] \
                            for t in sig_indices], \
                            color='white', \
                            edgecolors='black', \
                            s=9., linewidth=.5)

            ### Plotting the legend in 
            ### a separate figure below
            line_counter += 1
            if line_counter <= number_lines/3:
                x_text = .1
                y_text = 0.+line_counter*.1
            elif line_counter <= number_lines/3:
                x_text = .4
                y_text = 0.+line_counter*.066
            elif line_counter > number_lines/3:
                x_text = .7
                y_text = 0.+line_counter*.033

            label = '{} {}'.format(hz, cat)
            ax[1].scatter(x_text, y_text, \
                          #color=colors[f_i], \
                          label=label, alpha=1.)
            ax[1].text(x_text+.05, y_text, label)


        pyplot.savefig(os.path.join(plot_path,\
                       'classification_{}_{}_{}.jpg'.format(args.analysis, \
                        args.data_split, hz)),\
                       dpi=600)
        pyplot.clf()
        pyplot.close(fig)
