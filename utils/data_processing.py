"""
Created on Sat September 16 21:15:06 2023

Description: Data preprocessing functions for:
Paper url: https://www.biorxiv.org/content/10.1101/2023.09.11.557230v2

@author: Ekaterina Kuzmina, ekaterina.kuzmina@skoltech.ru
@author: Dmitrii Kriukov, dmitrii.kriukov@skoltech.ru
"""

import numpy as np
import h5py
import mat73
from scipy.ndimage import gaussian_filter1d


def load_h5_file(path, filename):
    with h5py.File(path+filename, 'r') as h5file:
        ans = {}
        for key, item in h5file.items():
            ans[key] = item[()] 
    return ans

def save_h5_file(path, filename, data):
    with h5py.File(path + filename, 'w') as h5file:
        for key, item in data.items():
            h5file[key] = item
    print('File {} successfully saved!'.format(filename))


### Grasping Data Processing ###

# Divide functions for Reachings and Graspings
## Reachings
def extract_and_resample(spikes, fs=0.01, max_len=2800):
    # 1. Extract data excluding None
    # from data_dict['spikes'] into list (n_units x 1)
    xspikes = []
    for s_arr in spikes:
        if s_arr[0] is not None:
            for i in range(len(s_arr[0])):
                if s_arr[0][i] is not None:
                    xspikes.append(s_arr[0][i])
                    
    # 2. Resample with fs sampling rate
    time = np.arange(0., max_len, fs)
    data = np.zeros((len(xspikes), time.shape[0]))
    for n_idx, neuron in enumerate(xspikes):
        hist = np.zeros(time.shape)
        for spike_time in neuron:
            t_strt = int(spike_time // fs)
            t_end = t_strt + 1

            in_range = time[t_strt] <= spike_time < time[t_end]
            if in_range:
                hist[t_strt] += 1
            else:
                hist[t_end] += 1
        data[n_idx] = hist
    return data


def cut_condition_wise(data, go_cue, instruction_cue, move_start,
                       move_end, fs=0.01, max_len=2800):
    
    time = np.arange(0., max_len, fs)
    cond_num = len(instruction_cue)
    for_each_cond = [instruction_cue[i].shape[0] for i in range(cond_num)]
    
    cond_dict = {}
    for cc in range(cond_num):
        one_cond_trials = []
        for c_idx in range(for_each_cond[cc]):
            gc = go_cue[cc][c_idx]
            gc_idx = int(gc // fs)

            tstrt, tend = instruction_cue[cc][c_idx], move_end[cc][c_idx]
            # +1 we round the top border; 
            # +1 in extract_and_resample [t_strt,t_end), so we shifted spikes one step ahead
            tstrt_idx, tend_idx = int(tstrt // fs), int(tend // fs) + 2

            crop = data[:, tstrt_idx : tend_idx]
            new_gc_idx = gc_idx - tstrt_idx
            one_cond_trials.append([crop, new_gc_idx])

        cond_dict[str(cc)] = one_cond_trials
        
    return cond_dict


def align_pad_smooth(cond_dict, sigma=10, signal_len='min'):
    from scipy.ndimage import gaussian_filter1d
    
    aligned_cond_dict = {} # dict of aligned data
    cond_gc_dict = {} # dict of idx of aligment point
    
    for cond_num, trials_list in cond_dict.items():
        # collect all go cue idxs
        gc_start = [trials_list[i][1] for i in range(len(trials_list))]
        
        # the most frequent go cue idx
        values, counts = np.unique(gc_start, return_counts=True)
        gc_align = values[np.argmax(counts)] 

        # find the idx of signals where go cue is not the most frequent one
        idx_to_align = np.argwhere(gc_start != gc_align)
        idx_to_align = idx_to_align.reshape((idx_to_align.shape[0]))
        
        # calculate difference and shift signal to align go cue
        aligned_trials = [trials_list[i][0] for i in range(len(trials_list))]
        for idx in idx_to_align:
            signal = trials_list[idx][0]
            diff = gc_start[idx] - gc_align

            if np.sign(diff) == 1:
                # remove diff number of elems from beginning
                aligned_signal = signal[:, diff:]

            elif np.sign(diff) == -1:
                # add diff number of zeros to beginning
                aligned_signal = np.concatenate((np.zeros((signal.shape[0], np.abs(diff))),
                                                 signal), axis=1)
            aligned_trials[idx] = aligned_signal

        # calc max and min lens of signals
        lens = [aligned_trials[i].shape[1] for i in range(len(aligned_trials))]
        signal_max_len, signal_min_len = max(lens), min(lens)
        
        trials_list_padded = []
       # pad to maximal len
        if signal_len == 'max':
            for trial in aligned_trials:
                to_pad = np.zeros((trial.shape[0],
                                   signal_max_len - trial.shape[1]))
                trials_list_padded.append(np.concatenate((trial, to_pad),
                                                         axis=1))
        # OR cut to the shortest len
        elif signal_len == 'min':
            for trial in aligned_trials:
                trials_list_padded.append(trial[:, :signal_min_len])

        # average by trials
        averaged_trials = np.mean(np.asarray(trials_list_padded), axis=0)

        # smooth with gauss 25 ms s.d.
        smthd_avrg_trial = []
        for ntrial in averaged_trials:
            smthd_avrg_trial.append(gaussian_filter1d(ntrial, sigma=sigma,
                                                      mode='constant'))
        aligned_cond_dict[str(cond_num)] = np.asarray(smthd_avrg_trial)
        cond_gc_dict[str(cond_num)] = gc_align
        
    return aligned_cond_dict, cond_gc_dict


def align_mean_conds(cond_dictq, gc_dict):

    cond_num = len(gc_dict.keys())
    neuron_num = cond_dictq['2'].shape[0]
    
    # collect all go cue idxs
    gc_start = [gc_dict[str(i)] for i in range(cond_num)]

    # the most frequent go cue idx
    values, counts = np.unique(gc_start, return_counts=True)
    gc_align = values[np.argmax(counts)]

    # find the idx of signals where go cue is not the most frequent one
    idx_to_align = np.argwhere(gc_start != gc_align)
    idx_to_align = idx_to_align.reshape((idx_to_align.shape[0]))

    avrg_aligned_conds = [cond_dictq[str(i)] for i in range(cond_num)]
    for cond_idx in idx_to_align:
        cond_signal = cond_dictq[str(cond_idx)]
        diff = gc_start[cond_idx] - gc_align

        if np.sign(diff) == 1:
            # remove diff number of elems from beginning
            aligned_signal = cond_signal[:, diff:]

        elif np.sign(diff) == -1:
            # add diff number of zeros to beginning
            aligned_signal = np.concatenate((np.zeros((cond_signal.shape[0],
                                                       np.abs(diff))),
                                             cond_signal), axis=1)
        avrg_aligned_conds[cond_idx] = aligned_signal
    
    min_len = min([avrg_aligned_conds[i].shape[1] for i in range(cond_num)])
    processed_data = np.empty((cond_num, min_len, neuron_num))
    for condn in range(cond_num):
        signal = avrg_aligned_conds[condn]
        processed_data[condn] = signal[:, :min_len].T
        
    return processed_data, gc_align


# For 6th dataset
def sort_cues(cond_times_dict, instr_cue, go_cue, ms_cue, me_cue):
    
    # Which instruction cue for which cond
    cond_num = len(cond_times_dict.keys())
    sort_instr_cue = {x:[] for x in range(cond_num)}
    cond_lens = {x: cond_times_dict[x].shape[0] for x in range(cond_num)}
    
    for instr_time in instr_cue:
        cond_mins = []
        for cond_idx in range(cond_num):
            diff = np.abs(np.asarray(cond_times_dict[cond_idx] - instr_time))
            cond_mins.append(np.amin(diff))

        cond_min_idx = np.where(cond_mins == np.amin(cond_mins))[0][0]
        sort_instr_cue[cond_min_idx].append(instr_time)
    
    # Sort go_cue, ms_cue, me_cue according to instr_cue
    go_cell = [[] for i in range(cond_num)]
    instr_cell = [[] for i in range(cond_num)]
    stmv_cell = [[] for i in range(cond_num)]
    endmv_cell = [[] for i in range(cond_num)]
    
    for gc_time, ms_time, me_time in zip(go_cue, ms_cue, me_cue):
        cond_mins = []
        in_cond_mins_idx = []

        for cond_idx in range(cond_num):
            diff = np.abs(np.asarray(sort_instr_cue[cond_idx] - gc_time))
            cond_mins.append(np.amin(diff))
            in_cond_mins_idx.append(np.where(diff == np.amin(diff))[0][0])

        cond_min_idx = np.where(cond_mins == np.amin(cond_mins))[0][0]
        instr_idx = in_cond_mins_idx[cond_min_idx]
        instr = sort_instr_cue[cond_min_idx][instr_idx]

        go_cell[cond_min_idx].append(gc_time)
        instr_cell[cond_min_idx].append(instr)
        stmv_cell[cond_min_idx].append(ms_time)
        endmv_cell[cond_min_idx].append(me_time)
        
    go_cell_f = [np.asarray(j) for j in go_cell]
    instr_cell_f = [np.asarray(j) for j in instr_cell]
    stmv_cell_f = [np.asarray(j) for j in stmv_cell]
    endmv_cell_f = [np.asarray(j) for j in endmv_cell]
    
    return go_cell_f, instr_cell_f, stmv_cell_f, endmv_cell_f


def extract_and_process_grasp(neural_data, data_dict, align_cue,
                              fixed_align_cue=None,
                              signal_len='max', sigma=10):
    """
    neural_data: list of [times x neurons] arrays
    """
    cond_arange = np.unique(data_dict['ObjectInd'])
    cond_seq = data_dict['ObjectInd']
    all_conds_prep = {} 
    aligning_cue = {}

    for cond_idx in cond_arange:
        idxs = np.argwhere(cond_seq == cond_idx)
        idxs = idxs.reshape((idxs.shape[0]))
        
        trials, time = [], []
        wsm_idx, fg_idx = [], []

        for i in idxs:
            trials.append(neural_data[i])
            time.append(data_dict['TimeBins'][i][0])
            
            # WristStartMove -> MaxApp -> FingerGrasp
            wsm = data_dict['EventTimes']['WristStartMove'][i]
            fg = data_dict['EventTimes']['FingerGrasp'][i]
        
            diff = np.abs(data_dict['TimeBins'][i][0] - wsm)
            wsm_idx.append(np.where(diff == np.amin(diff))[0][0])
            diff = np.abs(data_dict['TimeBins'][i][0] - fg)
            fg_idx.append(np.where(diff == np.amin(diff))[0][0])
            
        # choose according to which cue align    
        if align_cue == 'wsm':
            al_idx = wsm_idx
        elif align_cue == 'fg':
            al_idx = fg_idx
        
        v, c = np.unique(al_idx, return_counts=True)
        al_cue = v[np.argmax(c)] if v[np.argmax(c)] != 0 else int(np.mean(v))
        
        # We might hard code al_cue
        if fixed_align_cue:
            al_cue = fixed_align_cue
            
        # find the idx of signals where go cue is NOT aligning
        idx_to_align = np.argwhere(np.asarray(al_idx) != al_cue)
        idx_to_align = idx_to_align.reshape((idx_to_align.shape[0]))

        # calculate difference and shift signal to align go cue
        aligned_trials = [trials[j] for j in range(len(trials))]
        for i2a in idx_to_align:
            diff = al_idx[i2a] - al_cue

            if np.sign(diff) == 1:
                # remove diff number of elems from beginning
                aligned_signal = trials[i2a][diff:, :]

            elif np.sign(diff) == -1:
                # add diff number of zeros to beginning
                aligned_signal = np.concatenate((np.zeros((np.abs(diff),
                                                           trials[i2a].shape[1])),
                                                 trials[i2a]), axis=0)
            aligned_trials[i2a] = aligned_signal
        
        # calc max and min lens of signals
        lens = [aligned_trials[i].shape[0] for i in range(len(aligned_trials))]
        signal_max_len, signal_min_len = max(lens), min(lens)
        
        trials_list_padded = []
       # pad to maximal len
        if signal_len == 'max':
            for trial in aligned_trials:
                trial = trial.T
                to_pad = np.zeros((trial.shape[0],
                                   signal_max_len - trial.shape[1]))
                trials_list_padded.append(np.concatenate((trial, to_pad),
                                                         axis=1))
        # OR cut to the shortest len
        elif signal_len == 'min':
            for trial in aligned_trials:
                trial = trial.T
                trials_list_padded.append(trial[:, :signal_min_len])

        # average by trials
        averaged_trials = np.mean(np.asarray(trials_list_padded), axis=0)
        
        # smooth with gauss 25 ms s.d.
        smthd_avrg_trial = []
        for ntrial in averaged_trials:
            smthd_avrg_trial.append(gaussian_filter1d(ntrial, sigma=sigma,
                                                      mode='constant'))
        all_conds_prep[str(int(cond_idx) - 1)] = np.asarray(smthd_avrg_trial)
        aligning_cue[str(int(cond_idx) - 1)] = al_cue
        
    return all_conds_prep, aligning_cue


def load_grasping_dataset(didx):
    from utils.config import grasp_d_names, grasp_d_params, GRASP_DATA_PATH, DEXT

    d_dict = load_h5_file(GRASP_DATA_PATH, grasp_d_names[didx]+DEXT)
    cut_idx = grasp_d_params[grasp_d_names[didx]]['cutoffs']

    datas = d_dict['data'][:, cut_idx[0]:cut_idx[1], :]
    times = d_dict['time'][:datas.shape[1]]
    go_cue = d_dict['align_cue_idx'] - cut_idx[0]

    return [i for i in datas], list(times), go_cue


### LFP Data Processing ###

def extract_and_process_grasp_LFP(neural_data, dd, signal_len='max', sigma=10):
    """
    neural_data: list of [times x neurons] arrays
    """
    all_conds_prep = {} 
    aligning_cue = {}

    cond_arange = np.unique(dd['tgtDir'])
    cond_seq = dd['tgtDir']
    for cond_idx in cond_arange:
        idxs = np.argwhere(cond_seq == cond_idx)
        idxs = idxs.reshape((idxs.shape[0]))
        
        trials = []
        start, end = [], []
        go_cue, start_move = [], []
        show_target = []
        
        for i in idxs:
            trials.append(neural_data[i])
            
            start.append(dd['idx_trial_start'][i])
            show_target.append(dd['idx_tgtOnTime'][i])
            go_cue.append(dd['idx_goCueTime'][i])
            start_move.append(dd['idx_movement_on'][i])
            end.append(dd['idx_endTime'][i])

        v, c = np.unique(go_cue, return_counts=True)
        al_cue = v[np.argmax(c)] if v[np.argmax(c)] != 0 else int(np.mean(v))

        # find the idx of signals where go cue is NOT aligning
        idx_to_align = np.argwhere(np.asarray(go_cue) != al_cue)
        idx_to_align = idx_to_align.reshape((idx_to_align.shape[0]))

        # calculate difference and shift signal to align go cue
        aligned_trials = [trials[j] for j in range(len(trials))]
        for i2a in idx_to_align:
            diff = int(go_cue[i2a] - al_cue)

            if np.sign(diff) == 1:
                # remove diff number of elems from beginning
                aligned_signal = trials[i2a][diff:, :]

            elif np.sign(diff) == -1:
                # add diff number of zeros to beginning
                aligned_signal = np.concatenate((np.zeros((np.abs(diff),
                                                            trials[i2a].shape[1])),
                                                    trials[i2a]), axis=0)
            aligned_trials[i2a] = aligned_signal
        
        # calc max and min lens of signals
        lens = [aligned_trials[i].shape[0] for i in range(len(aligned_trials))]
        signal_max_len, signal_min_len = max(lens), min(lens)

        trials_list_padded = []
        # pad to maximal len
        if signal_len == 'max':
            for trial in aligned_trials:
                trial = trial.T
                to_pad = np.zeros((trial.shape[0],
                                    signal_max_len - trial.shape[1]))
                trials_list_padded.append(np.concatenate((trial, to_pad),
                                                            axis=1))
        # OR cut to the shortest len
        elif signal_len == 'min':
            for trial in aligned_trials:
                trial = trial.T
                trials_list_padded.append(trial[:, :signal_min_len])

        # average by trials
        averaged_trials = np.mean(np.asarray(trials_list_padded), axis=0)
        
        # smooth with gauss
        smthd_avrg_trial = []
        for ntrial in averaged_trials:
            smthd_avrg_trial.append(gaussian_filter1d(ntrial, sigma=sigma,
                                                        mode='constant'))
        all_conds_prep[cond_idx] = np.asarray(smthd_avrg_trial)
        aligning_cue[cond_idx] = int(al_cue)
            
    return all_conds_prep, aligning_cue, cond_arange


def align_mean_conds_LFP(cond_dictq, gc_dict):

    cond_num = len(gc_dict.keys())
    gc_start_key = list(gc_dict.keys())
    neuron_num = cond_dictq[gc_start_key[0]].shape[0]
    
    # collect all go cue idxs
    gc_start = [gc_dict[i] for i in gc_start_key]

    # the most frequent go cue idx
    values, counts = np.unique(gc_start, return_counts=True)
    gc_align = values[np.argmax(counts)]

    # find the idx of signals where go cue is not the most frequent one
    idx_to_align = np.argwhere(gc_start != gc_align)
    idx_to_align = idx_to_align.reshape((idx_to_align.shape[0]))

    avrg_aligned_conds = [cond_dictq[i] for i in gc_start_key]
    for cond_idx in idx_to_align:
        cond_signal = cond_dictq[gc_start_key[cond_idx]]
        diff = gc_start[cond_idx] - gc_align

        if np.sign(diff) == 1:
            # remove diff number of elems from beginning
            aligned_signal = cond_signal[:, diff:]

        elif np.sign(diff) == -1:
            # add diff number of zeros to beginning
            aligned_signal = np.concatenate((np.zeros((cond_signal.shape[0],
                                                       np.abs(diff))),
                                             cond_signal), axis=1)
        avrg_aligned_conds[cond_idx] = aligned_signal
    
    min_len = min([avrg_aligned_conds[i].shape[1] for i in range(cond_num)])
    processed_data = np.empty((cond_num, min_len, neuron_num))
    for condn in range(cond_num):
        signal = avrg_aligned_conds[condn]
        processed_data[condn] = signal[:, :min_len].T
        
    return processed_data, gc_align


def lfp_dataset_processing(path, plot_all=False):
    print(path)
    data_dict = mat73.loadmat(path)
    dd = data_dict['trial_data']
    k = dd.keys()
    labels = ['M1', 'PMd', 'LeftS1Area2']
    in_labels = []
    for l in labels:
        ll = l + '_spikes'
        if ll in k:
            in_labels.append(l)

    data_to_return = {}
    for l in in_labels:
        print('For ', l)
        
        neural_data = dd[l + '_spikes']
        a, g, _ = extract_and_process_grasp_LFP(neural_data,
                                                dd, signal_len='max',
                                                sigma=12)
        print(a.keys())
        # 2. Выровнять по go cue средние кондишоныh
        proc, gc = align_mean_conds_LFP(a, g)
        print(proc.shape, gc)
        data_to_return[l] = [proc, gc]

    return data_to_return