import matplotlib
import numpy
import os

from matplotlib import pyplot
from scipy import stats

from utils_two import read_words_and_triggers

numpy.seterr(all='raise')

### Creating out folder

plot_path = os.path.join('plots', 'behavioural')
os.makedirs(plot_path, exist_ok=True)

### Reading questions
_, questions = read_words_and_triggers(return_questions=True)

### Reading files

bids_folder = os.path.join('unaware_semantics_bids', 'sourcedata')
assert os.path.exists(bids_folder)

subjects = list(range(1, 46))
res_dict = dict()

for s in subjects:
    for r in range(1, 24):
        if s == 14 and r == 3:
            continue
        elif s == 26 and r == 10:
            continue
        elif s == 28 and r == 4:
            continue
        f_path = os.path.join(bids_folder, 'sub-{:02}'.format(s),
                'sub-{:02}_task-namereadingimagery_run-{:02}_events.tsv'.format(s, r))
        print(f_path)
        assert os.path.exists(f_path)
        with open(f_path) as i:
            lines = [l.strip().split('\t') for l in i.readlines()]
        header = lines[0]
        if len(res_dict.keys()) == 0:
            for k in header:
                res_dict[k] = list()
                res_dict['subject'] = list()
                res_dict['required_answer'] = list()
        data = lines[1:]
        for l in data:
            ### Adding expected answer
            if l[2] not in ['_', '']:
                quest = l[-1]
                if quest in questions[l[2]]:
                    ans = 'YES'
                else:
                    ans = 'NO'
            else:
                ans = 'NO'
            res_dict['required_answer'].append(ans)
            ### Adding subject
            res_dict['subject'].append(s)
            for h, v in zip(header, l):
                res_dict[h].append(v)

mapper = {1 : 'low', 2 : 'medium', 3 : 'high'}

### Accuracy scores

all_correct = list()
all_wrong = list()

for s in subjects:
    s_data = [w for w_i, w in enumerate(res_dict['accuracy']) if int(res_dict['subject'][w_i])==s]

    correct = len([w for w in s_data if w == 'correct'])
    wrong = len([w for w in s_data if w == 'wrong'])
    total = correct+wrong

    all_correct.append(correct/total)
    all_wrong.append(wrong/total)

all_correct = numpy.array(all_correct)
all_wrong = numpy.array(all_wrong)

fig, ax = pyplot.subplots(figsize=(16, 9))

ax.bar(subjects, all_correct, label='correct')
ax.bar(subjects, all_wrong, bottom=all_correct, label='wrong')

ax.set_xticks(subjects)

title = 'Overall performances for each subject'
ax.set_title(title, pad=40.)
ax.legend(loc=(0.25, 1.025), ncol=2)
pyplot.tight_layout()
#pyplot.show()
file_path = os.path.join(plot_path, 'overall_performance.png')
pyplot.savefig(file_path)

pyplot.clf()

### Awareness scores

all_low = list()
all_medium = list()
all_high = list()

for s in subjects:
    s_data = [w for w_i, w in enumerate(res_dict['PAS_score']) if int(res_dict['subject'][w_i])==s]

    low = len([w for w in s_data if int(w) == 1])
    medium = len([w for w in s_data if int(w) == 2])
    high = len([w for w in s_data if int(w) == 3])

    total = low + medium + high

    all_low.append(low/total)
    all_medium.append(medium/total)
    all_high.append(high/total)

all_low = numpy.array(all_low)
all_medium = numpy.array(all_medium)
all_high = numpy.array(all_high)

fig, ax = pyplot.subplots(figsize=(16, 9))

ax.bar(subjects, all_low, label='low')
ax.bar(subjects, all_medium, bottom=all_low, label='medium')
ax.bar(subjects, all_high, bottom=all_medium+all_low, label='high')
ax.set_xticks(subjects)

title = 'PAS scores for each subject'
ax.set_title(title, pad=40.)
ax.legend(loc=(0.25, 1.025), ncol=3)
pyplot.tight_layout()
#pyplot.show()
file_path = os.path.join(plot_path, 'pas.png')
pyplot.savefig(file_path)

pyplot.clf()

### Accuracy for PAS

all_low = list()
all_medium = list()
all_high = list()

pas = [1,2,3]
acc = ['correct', 'wrong']

results = {p : list() for p in pas}

for s in subjects:
    s_data = [(int(w), res_dict['accuracy'][w_i]) for w_i, w in enumerate(res_dict['PAS_score']) if int(res_dict['subject'][w_i])==s]
    for p in pas:
        current_corr = len([1 for s_p, s_a in s_data if s_p==p and s_a=='correct'])
        current_wrong = len([1 for s_p, s_a in s_data if s_p==p and s_a=='wrong'])
        if current_corr == 0 and current_wrong == 0:
            s_r = numpy.nan
        else:
            s_r = current_corr/(current_corr+current_wrong)
        results[p].append(s_r)

fig, ax = pyplot.subplots(figsize=(16, 9))

#ax.violinplot([p_r for k, p_r in results.items()])
for k_p_r_i, k_p_r in enumerate(results.items()):
    ax.scatter([i+1+k_p_r_i*0.1 for i in range(len(k_p_r[1]))], k_p_r[1], \
                            label=mapper[k_p_r[0]])
ax.legend()
ax.hlines(y=0.5, xmin=0, xmax=len(subjects)+1, alpha=0.7, color='darkgray', linestyles='dotted')
ax.hlines(y=1., xmin=0, xmax=len(subjects)+1, alpha=0.7, color='darkgray', linestyles='dotted')
ax.set_xticks([s+0.1 for s in subjects])
ax.set_xticklabels(subjects)
title = 'Accuracy scores across subjects'
ax.set_title(title, pad=40.)
ax.legend(loc=(0.25, 1.025), ncol=3)
pyplot.tight_layout()

file_path = os.path.join(plot_path, 'accuracies.png')
pyplot.savefig(file_path)
#pyplot.show()

pyplot.clf()

## D-prime

pas = [1,2,3]
acc = ['correct', 'wrong']

results = {p : list() for p in pas}

for s in subjects:
    s_data = [(int(w), res_dict['accuracy'][w_i], res_dict['required_answer'][w_i]) for w_i, w in enumerate(res_dict['PAS_score']) if int(res_dict['subject'][w_i])==s]
    for p in pas:
        current_corr = len([1 for s_p, s_a, req in s_data if s_p==p and s_a=='correct' and req=='YES']) / len([1 for s_p, s_a, req in s_data if req=='YES'])
        current_wrong = len([1 for s_p, s_a, req in s_data if s_p==p and s_a=='wrong' and req=='NO']) / len([1 for s_p, s_a, req in s_data if req=='NO'])
        z_hit = stats.norm.ppf(current_corr)
        z_fa = stats.norm.ppf(current_wrong)
        try:
            d_prime = z_hit - z_fa
        except FloatingPointError:
            d_prime = numpy.nan
        #print([p, d_prime])
        results[p].append(d_prime)

fig, ax = pyplot.subplots(figsize=(16, 9))

#ax.violinplot([p_r for k, p_r in results.items()])
for k_p_r_i, k_p_r in enumerate(results.items()):
    ax.scatter([i+1+k_p_r_i*0.1 for i in range(len(k_p_r[1]))], k_p_r[1], \
                            label=mapper[k_p_r[0]])
ax.legend()
ax.hlines(y=0.0, xmin=0, xmax=len(subjects)+1, alpha=0.7, color='darkgray', linestyles='dotted')
ax.hlines(y=1.0, xmin=0, xmax=len(subjects)+1, alpha=0.7, color='darkgray', linestyles='dotted')
ax.set_xticks([s+0.1 for s in subjects])
ax.set_xticklabels(subjects)
title = 'D-prime scores across subjects'
ax.set_title(title, pad=40.)
ax.legend(loc=(0.25, 1.025), ncol=3)
pyplot.tight_layout()

file_path = os.path.join(plot_path, 'dprime.png')
pyplot.savefig(file_path)
#pyplot.show()

pyplot.clf()
