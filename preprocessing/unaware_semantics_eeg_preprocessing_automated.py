import argparse
import logging
import mne
import os
import pandas
import tqdm
import matplotlib
import autoreject

import numpy as np
from matplotlib import pyplot
from multiprocessing import Pool

from tqdm import tqdm

from mne.preprocessing import create_ecg_epochs, create_eog_epochs, ICA, read_ica

from autoreject.utils import interpolate_bads

def preprocess_eeg(s):

    sub_list = list()
    sub_events = list()
    problems = list()

    base_folder = os.path.join(args.bids_data_folder, 'sourcedata', 'sub-{:02}'.format(s))
    out_folder = os.path.join(args.bids_data_folder, 'derivatives', 'sub-{:02}'.format(s))
    os.makedirs(out_folder, exist_ok=True)

    for r in range(1, 24+1):
        if s == 14 and r == 3:
            continue
        elif s == 26 and r == 10:
            continue
        elif s == 28 and r == 4:
            continue

        print('\n\tSubject {} - run {}\n'.format(s, r))

        bdf_filename = 'sub-{:02}_task-namereadingimagery_run-{:02}_eeg.bdf'.format(s, r)
        events_filename = 'sub-{:02}_task-namereadingimagery_run-{:02}_events.tsv'.format(s, r)

        bdf_path = os.path.join(base_folder, bdf_filename)
        assert os.path.exists(bdf_path)
        events_path = os.path.join(base_folder, events_filename)
        assert os.path.exists(events_path)
        full_raw = mne.io.read_raw_bdf(bdf_path, \
                                      preload=True, \
                                      eog=eog_channels, \
                                      exclude=excluded_channels, \
                                      verbose=False, \
                                      )
        full_raw.set_montage(montage)

        ### Step 1: low-pass filtering
        ### Low-pass filtering all data at 0.01-80.hz

        logging.info('Step #1a: Low-pass filtering all data at 80hz')

        full_raw.filter(l_freq=None, h_freq=80.,
                        verbose=False)

        run_events_full = mne.find_events(full_raw,
                                 initial_event=False,
                                 verbose=False,
                                 stim_channel='Status',
                                 #min_duration=0.5
                                 )
        ### Fixing wrong events
        mask = run_events_full[:, 2] < 34
        run_events = run_events_full[mask]

        if s not in [14, 26, 28]:
            n_ev = 33
        else:
            if s == 14 and r == 2:
                n_ev = 66
            elif s == 26 and r == 9:
                n_ev = 66
            elif s == 28 and r == 3:
                n_ev = 66
            else:
                n_ev = 33
        assert run_events.shape[0] <= n_ev
        if run_events.shape[0] != n_ev:
            problems.append('subject {}, run {}, number of trials: {}'.format(s, r, run_events.shape[0]))

        if run_events.shape[0] > 3:

            ### Step #1b: Filtering EOG between 1-50hz
            logging.info('Step #1b: Filtering EOG between 1-50hz')
            ### Filtering EOG at 1.-50. hz so as to avoid
            ### problems with
            ### autoreject (following the
            ### reproducible pipeline on Wakeman & Henson
            picks_eog = mne.pick_types(full_raw.info, eeg=False, eog=True)
            full_raw.filter(
                           l_freq=1.,
                           h_freq=50.,
                           picks=picks_eog,
                           verbose=False)
            print('\n\tICA preparation\n')
            logging.info('Step #1c: Filtering for ICA')
            ica_raw = full_raw.copy().filter(
                                             l_freq=2.,
                                             h_freq=None,
                                             verbose=False
                                             )
            logging.info('Step #2: Epoching for ICA')
            ### Finding the events and reducting the data to epochs
            # Baseline correction is NOT applied
            ### see https://sccn.ucsd.edu/pipermail/eeglablist/2018/013943.html
            ica_epochs = mne.Epochs(
                                    ica_raw,
                                    run_events,
                                    tmin=-0.1,
                                    tmax=1.,
                                    baseline=None,
                                    preload=True,
                                    verbose=False,
                                    )

            logging.info('Step #3: Subsampling to 256hz')
            ### Reducing to a sample rate of 256
            ica_epochs.decimate(8)

            ### Autoreject EEG channels and epochs
            logging.info('Step #4: Cleaning channels and epochs for ICA')
            picks_eeg = mne.pick_types(ica_epochs.info, eeg=True, eog=False)
            ar = autoreject.AutoReject(n_jobs=os.cpu_count(), \
                                       random_state=2, \
                                       #verbose='tqdm', \
                                       verbose=False,
                                       picks=picks_eeg,
                                       cv=min(ica_epochs.events.shape[0], 10),
                                       )
            ica_epochs, autoreject_log = ar.fit_transform(ica_epochs,
                                                          return_log=True)

            ### Setting the reference to the average of the channels
            logging.info('Step #5: Setting the reference to the average of the 128 EEG channels for ICA')
            ica_epochs.set_eeg_reference(
                                         ref_channels='average',
                                         ch_type='eeg',
                                         verbose=False
                                         )

            assert not ica_epochs.baseline
            ### Computing and applying ICA to correct EOG
            ### ICA is computed on non-epoched filtered data,
            ### and applied on unfiltered data
            logging.info('Step #6: Finding eye-related components based on ICA')
            #ica_raw = full_raw.copy().filter(l_freq=1., h_freq=None, verbose=False)
            ica = ICA(
                      n_components=15,
                      random_state=1,
                      verbose=False
                      )
            ica.fit(ica_epochs)
            ### We create the EOG epochs, and then look for the correlation
            ### between them and the individual components
            eog_epochs = create_eog_epochs(
                                           ica_raw,
                                           tmin=-.5,
                                           tmax=.5,
                                           verbose=False
                                           )
            ### number of standard deviations to find outliers in EOG zscored
            ### correlation scores, from 3. to 2. SDs
            thresholds = [3., 2.8, 2.6, 2.4, 2.2, 2.]
            #thresholds = [3.]
            eog_inds = list()
            for threshold in thresholds:
                if len(eog_inds) == 0:
                    eog_inds, scores_eog = ica.find_bads_eog(eog_epochs, threshold=threshold)
            ica.exclude = eog_inds

            print('\n\tData preprocessing\n')
            logging.info('Step #7: Epoching data')
            ### Finding the events and reducting the data to epochs
            # Baseline correction is NOT applied
            ### see https://sccn.ucsd.edu/pipermail/eeglablist/2018/013943.html
            run_epochs = mne.Epochs(full_raw, \
                                    run_events, \
                                    tmin=-0.1, \
                                    tmax=1., \
                                    baseline=None,
                                    preload=True,
                                    verbose=False,
                                    )
            logging.info('Step #8: Subsampling to 256hz')
            ### Reducing to a sample rate of 256
            run_epochs.decimate(8)
            logging.info('Step #9: Removing eye-related ICs from epochs')
            ica.apply(run_epochs)
            assert not run_epochs.baseline
            ### Autoreject EEG channels and epochs
            logging.info('Step #11: Interpolating bad channels and dropping bad epochs using autoreject')
            picks_eeg = mne.pick_types(run_epochs.info, eeg=True, eog=False)
            ar = autoreject.AutoReject(n_jobs=os.cpu_count(), \
                                       random_state=2, \
                                       #verbose='tqdm', \
                                       verbose=True,
                                       picks=picks_eeg,
                                       cv=min(run_epochs.events.shape[0], 10),
                                       )
            run_epochs, autoreject_log = ar.fit_transform(run_epochs,
                                                          return_log=True)

            ### Setting the reference to the average of the channels
            logging.info('Step #12: Setting the reference to the average of the 128 EEG channels')
            run_epochs.set_eeg_reference(ref_channels='average', \
                                         ch_type='eeg')

            ### Dropping the EOG
            logging.info('Step #13: Dropping the EOG channels')
            run_epochs.drop_channels(eog_channels)

            sub_list.append(run_epochs)

            ### Reducing the list of events to the events actually remaining
            logging.info('Step #7: Updating events list, removing discarded events')
            with open(events_path) as i:
                lines = [l.strip().split('\t') for l in i.readlines()]
            header = lines[0]
            current_lines = lines[1:]
            current_lines = [l for l in current_lines if int(l[3]) in run_epochs.events[:, 2]]

            ### Checking all's good
            try:
                assert len(current_lines) == run_epochs.events[:, 2].shape[0]
            ### Correcting for longer runs
            except AssertionError:
                correct_current_lines = list()
                used_lines = list()
                used_trigs = list()
                for ev in run_epochs.events:
                    marker = False
                    for l_i, wrong_l in enumerate(current_lines):
                        if int(wrong_l[3]) == ev[2] and l_i not in used_lines and len(used_trigs)<run_epochs.events.shape[0] and marker == False:
                            new_line = wrong_l.copy()
                            new_line[0] = str(float(ev[0]/2048))
                            used_lines.append(l_i)
                            used_trigs.append(ev)
                            correct_current_lines.append(new_line)
                            marker = True
                del current_lines
                current_lines = correct_current_lines.copy()
                assert len(current_lines) == run_epochs.events[:, 2].shape[0]

            for l, e in zip(current_lines, run_epochs.events[:, 2]):
                assert int(l[3]) == e

            if len(sub_events) == 0:
                sub_events.append(header)
            sub_events.extend(current_lines)

    bdf_out = 'sub-{:02}_task-namereadingimagery_eeg-epo.fif.gz'.format(s, r)
    events_out = 'sub-{:02}_task-namereadingimagery_events.tsv'.format(s, r)

    bdf_out_path = os.path.join(out_folder, bdf_out)
    events_out_path = os.path.join(out_folder, events_out)

    sub_epochs = mne.concatenate_epochs(sub_list)
    logging.info('Step #10: Applying baseline correction')
    assert not sub_epochs.baseline
    sub_epochs.apply_baseline(baseline=(None, 0))

    ### Checking lengths match, excluding header
    assert len(sub_events)-1 == sub_epochs.events.shape[0]

    sub_epochs.save(bdf_out_path,
                   overwrite=True)

    with open(events_out_path, 'w') as o:
        for l_i, l in enumerate(sub_events):
            if l_i > 0:
                l[0] = '0.'
            o.write('{}\n'.format('\t'.join(l)))

    return problems

parser = argparse.ArgumentParser()
parser.add_argument('--bids_data_folder', required=True,
                    help='Where is the BIDS folder?')
parser.add_argument('--debugging', action='store_true',
                    help='Running one subject at a time')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)

### Channel naming

eog_channels = ['EXG1', 'EXG2']
excluded_channels = ['EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7',
                     'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2',
                     'Resp', 'Plet', 'Temp']

montage = mne.channels.make_standard_montage(kind='biosemi128')
subjects = [i for i in range(1, 46)]

if args.debugging:
    for s in subjects:
        preprocess_eeg(s)
else:
    with Pool(processes=os.cpu_count()-1) as pool:
        problems = pool.map(preprocess_eeg, subjects)
        pool.close()
        pool.join()

print('Could not get correctly events for:')
print(problems)
