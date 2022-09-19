import collections
import itertools
import mne
import numpy
import os
import random

class ExperimentInfo:

    def __init__(self, args):

        self.n_subjects = 45
        self.events_log, self.trigger_to_info = self.read_events_log(args)

    def read_events_log(self, args):

        for s in range(16, self.n_subjects):
            file_path = os.path.join(args.bids_folder,
                   'derivatives',
                   'sub-{:02}'.format(s),
                   'sub-{:02}_task-namereadingimagery_events.tsv'.format(s)
                   )
            assert os.path.exists(file_path)
            with open(file_path) as i:
                lines = [l.strip().split('\t') for l in i.readlines()]
            header = lines[0]
            if s == 16:
                events_collector = {h : list() for h in header}
                events_collector['subject'] = list()
            data = lines[1:]
            for d_line in data:
                for h, d in zip(header, d_line):
                    events_collector[h].append(d)
                    events_collector['subject'].append(s)

        ### Collecting triggers bottom-up

        trigger_to_word = {t : set() for t in range(1, 34)}
        for t, w in zip(events_collector['value'], events_collector['trial_type']):
            trigger_to_word[int(t)].add(w)
        trigger_to_word = {t : list(ws) for t, ws in trigger_to_word.items()}
        for t, ws in trigger_to_word.items():
            if '_' not in ws:
                assert len(ws) == 1
            else:
                assert len(ws) == 2

        ### Collecting categories bottom-up
        trigger_to_cat = {t : set() for t in range(1, 34)}
        for t, w in zip(events_collector['value'], events_collector['category']):
            trigger_to_cat[int(t)].add(w)
        trigger_to_cat = {t : list(ws) for t, ws in trigger_to_cat.items()}
        for t, ws in trigger_to_cat.items():
            if t == 33:
                trigger_to_cat[t] = ['NA']
            else:
                assert len(ws) == 1

        trigger_to_info = {t : [trigger_to_word[t][0], trigger_to_cat[t][0]] for t in range(1, 34)}

        return events_collector, trigger_to_info


class EEGData:

    def __init__(self, experiment_info, n, args):
        self.subject = n
        assert (n > 0 and n < 46)
        self.experiment_info = experiment_info
        self.eeg_data, self.times, self.permutations = self.get_eeg_data(args)

    def get_eeg_data(self, args):

        eeg_path = os.path.join(args.bids_folder,
               'derivatives',
               'sub-{:02}'.format(self.subject),
               'sub-{:02}_task-namereadingimagery_eeg-epo.fif.gz'.format(self.subject)
               )
        assert os.path.exists(eeg_path)

        epochs = mne.read_epochs(eeg_path, verbose=False)
        times = epochs.times

        ### Reading subjects events
        sub_indices = [v_i for v_i, v in enumerate(self.experiment_info.events_log['subject']) if v==self.subject]
        import pdb; pdb.set_trace()
        assert (len(sub_indices) > 100 and len(sub_indices) < 1000)
        current_events = {k : [v[idx] for idx in sub_indices] for k, v in self.experiment_info.events_log.items()}
        for ev_t, eeg_t in zip(current_events['value'], epochs.events[:, 2]):
            assert ev_t == eeg_t

        ### Selecting the key for the label
        if args.data_split == 'objective_accuracy':
            relevant_key = 'accuracy'
        elif args.data_split == 'perceptual_awareness':
            relevant_key = 'PAS_score'

        data_dict = dict()
        for epoch, label, trigger in zip(epochs.get_data(), current_events[relevant_key], epochs.events[:, 2]):
            if label not in data_dict.keys():
                data_dict[label] = dict()
            if trigger not in data_dict[label].keys():
                data_dict[label][trigger] = list()
            data_dict[label][trigger].append(epoch)

        for k, v in data_dict.items():
            for t, vecs in v.items():
                data_dict[k][t] = numpy.average(vecs, axis=0)
                assert data_dict[k][t].shape == epoch.shape

        return final_dict, times

class ComputationalModel:

    def __init__(self, args):

        self.model = args.computational_model
        self.word_sims = self.load_word_sims(args)

    def load_word_sims(self, args):

        path = os.path.join('computational_models', 'similarities', \
                            '{}.sims'.format(self.model))
        assert os.path.exists(path)
        with open(path, encoding='utf-8') as i:
            lines = [l.strip().split('\t') for l in i.readlines()]
        word_sims = {(sim[0], sim[1]) : float(sim[2]) for sim in lines}

        return word_sims

    def compute_pairwise(self, words):

        ordered_words = sorted(words)
        combs = list(itertools.combinations(ordered_words, 2))
        pairwise_similarities = list()
        for c in combs:
            sim = self.word_sims[c]
            pairwise_similarities.append(sim)

        return ordered_words, combs, pairwise_similarities
