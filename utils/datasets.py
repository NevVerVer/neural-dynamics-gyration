"""
Created on Sat September 17 21:15:06 2023

Description: Datasets class implementation for
Paper url: https://www.biorxiv.org/content/10.1101/2023.09.11.557230v2

@author: Ekaterina Kuzmina, ekaterina.kuzmina@skoltech.ru
@author: Dmitrii Kriukov, dmitrii.kriukov@skoltech.ru
"""

import mat73
import numpy as np
import jPCA
from jPCA.util import plot_projections
import matplotlib.patches as patches
from matplotlib import pyplot as plt
import glob
import os
import scipy.io
from scipy.ndimage import gaussian_filter1d
from utils.utils import generate_response

from utils.data_processing import (
    extract_and_process_grasp,
    extract_and_resample,
    sort_cues,
    align_pad_smooth,
    align_mean_conds,
    cut_condition_wise,
    save_h5_file,
    load_h5_file,
    align_mean_conds_LFP,
    extract_and_process_grasp_LFP,
    )
from utils.vis import (
    plot_each_cond,
    plot_each_neuron,
    plot_peth,
    build_jPCA,
    build_pca,
    plot_clusterization,
    plot_curvature,
    plot_eigenPCA,
    plot_NTCcov_matrises,
    plot_R2_fitting,
    plot_all_stats
)

from utils.shuffling import (
    shuffle_data,
    shuffling_CMPT
)

from utils.utils import (
    soft_normalize,
    fit_running_wave,
    subtract_cc_mean,
    )
from utils.color_palettes import (
    blue_yellow_cmap,
    journ_color_dict,
    bg_gray_color,
    cmaps,
    )
from jPCA.util import red_green_cmap


class NeuralDataset:
    """
    Parent class. Has basic loading, saving and plotting methods.
    load_data() and preprocess_data() should be defined for 
    each dataset individually.
    Input:
    dataset_dir: folder containing raw files to load
    """
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.preproc_h5 = False
        self.cmap = red_green_cmap()

    def load_data(self):
        raise NotImplementedError("Subclasses must implement this method")

    def preprocess_data(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def load_additional_info(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def load_preprocessed(self, preproc_dir, pfname):
        self.info_d = load_h5_file(preproc_dir, pfname)
        self.data = self.info_d['data']
        self.time, self.go_cue = self.info_d['time'], self.info_d['go_cue']
        self.load_additional_info()
        self.full_data = self.data.copy()
        self.full_time, self.full_go_cue = self.time.copy(), self.go_cue
        self.preproc_h5 = True

    def crop_data(self, tstart, tend):
        if self.full_data.shape != self.data.shape:
            print('Reset data shape before cropping again!')
            self.reset_data()
        self.data = self.data[:, tstart:tend, :]
        self.time = self.time[:self.data.shape[1]]
        self.go_cue = self.go_cue - tstart

    def reset_data(self):
        self.data = self.full_data.copy()
        self.go_cue = self.full_go_cue
        self.time = self.full_time.copy()

    def shuffle_data(self, shuffling_type, keep_var=False):
        if shuffling_type in ['1', '2', '3']:
            self.data = shuffle_data(self.data, self.time, 
                                shuffle_type=shuffling_type,
                                go_cue_time = self.go_cue,
                                keep_var=keep_var)
        elif shuffling_type == 'cmpt':
            self.data = shuffling_CMPT(self.data, threshold=0.95)
        else:
            raise ValueError('Wrong shuffling type!')
        
    def print_dataset_info(self): 
        print(f'Dataset shape: {self.data.shape}')
        print(f'Go cue: {self.go_cue}')
        print(f'Original dataset shape: {self.full_data.shape}')

    def list_dataset_files(self, path, ext='*.h5'):
        d_names = glob.glob(path + ext, recursive=True)
        return d_names

    def save2h5(self, fname, fpath):
        self.create_info_dict()
        if os.path.isdir(fpath) == False:
            os.mkdir(fpath)
        save_h5_file(fpath, fname, self.info_d)

    # Ploting functions
    def plot_neurons(self, subtract_mean=True, n_neurons=25,
                     save_name=None):
        plot_each_neuron(self.data,
                            self.go_cue,
                            subtract_mean=subtract_mean,
                            n=n_neurons,
                            save_name=save_name)

    def plot_conditions(self, subtract_mean=True, n_conds=8,
                        save_name=None):
        plot_each_cond(self.data,
                        self.go_cue,
                        subtract_mean=subtract_mean,
                        n=n_conds,
                        save_name=save_name)

    def plot_peth(self, subtract_mean=True, cond_list=[], sort_individ=False,
                  save_name=None):
        datas_list = [i for i in self.data]
        plot_peth(datas_list, list(self.time),
                    subtract_mean=subtract_mean,
                    go_cue=self.go_cue,
                    cond_list=cond_list,
                    sort_individ=sort_individ, 
                    save_name=save_name)

    def plot_jpca(self, tstart, tend, subtract_mean=True,
                  arrow_size=0.05, save_name=None):
        build_jPCA(self.data, self.go_cue, tstart, tend,
                    subtract_mean=subtract_mean,
                    shuffling=None,
                    siz=arrow_size,
                    save_name=save_name,
                    jpca_cmap=self.cmap)

    def plot_pca(self, tstart, tend, subtract_mean=True,
                 arrow_size=0.001, save_name=None):
        build_pca(self.data, self.time, tstart, tend,
                    sub_mean=subtract_mean,
                     siz=arrow_size, save_name=save_name,
                     pca_cmap=self.cmap)

    def plot_eigenpca(self, npca=4, subtract_mean=False, save_name=None):
        plot_eigenPCA(self.data, npca=npca,
                      sub_mean=subtract_mean, save_name=save_name)

    def plot_clusterization(self, subtract_mean=False, save_name=None):
        plot_clusterization(self.data, self.time,
                            sub_mean=subtract_mean,
                            save_name=save_name)

    def plot_ntc_cov_matrices(self, subtract_mean, save_name=None):
        plot_NTCcov_matrises(self.data, sub_mean=subtract_mean, save_name=save_name)

    def plot_curvature(self, subtract_mean=False, save_name=None):
        plot_curvature(self.data, self.go_cue, sub_mean=subtract_mean, save_name=save_name)

    def plot_r2_fitting(self, save_name=None):
        plot_R2_fitting(self.data, self.time, save_name=save_name)

    def fit_to_synthetic_model(self):
        norm_data = soft_normalize(self.data)
        _, R2, _ = fit_running_wave(norm_data, self.time)
        return R2


class GraspingSuresh(NeuralDataset):
    """
    Dataset from paper:
    "Neural Population Dynamics in Motor Cortex are Different for Reach and Grasp",
    Suresh et al. 2020 [doi.org/10.7554/eLife.58848]

    Comparing neural dynamics during reaching and grasping.  
    Behaviour: isolated grasping task, center-out reaching task
    Brain areas: primary motor cortex (M1), somatosensory cortex (SCx)

    Link to dataset: 
        doi.org/10.5061/dryad.xsj3tx9cm
    Or see the tutorial how to download at:
        github.com/NevVerVer/neural-dynamics-gyration/blob/main/datasets_analysis.ipynb
    """
    def load_data(self, dataset_name):
        self.data_dict = mat73.loadmat(self.dataset_dir + dataset_name)
        print(f'File {dataset_name} successfully loaded!')

    def load_additional_info(self):
        self.align_cue = self.info_d['align_cue_name']
        self.signal_len = self.info_d['signal_len_cropping']
        self.max_len = self.info_d['max_len']
        self.dataset_idx = self.info_d['dataset_idx']
        self.freq_sampling = self.info_d['freq_sampling']
        self.area = self.info_d['area']
        self.sigma = self.info_d['sigma']

        self.cmap_name = self.info_d.get('cmap_name')
        if self.cmap_name is not None:
            self.cmap_name = self.cmap_name.decode("utf-8")
        else:
            self.cmap_name = 'church'
        self.cmap = cmaps[self.cmap_name]

    def preprocess_data(self, sigma, dataset_idx, align_cue='wsm', fixed_align_cue=None,
                        freq_sampling=0.01, signal_len='max', max_len=2800, area='M1'):
        # Write preproc parameters to class vars  
        self.sigma = sigma
        self.dataset_idx = dataset_idx 
        self.align_cue = align_cue
        self.freq_sampling = freq_sampling
        self.area = area
        self.signal_len = signal_len
        self.max_len = max_len

        if dataset_idx == 5 or dataset_idx == 6:
            data_raw = extract_and_resample(self.data_dict['spikes'],
                                        fs=freq_sampling,
                                        max_len=max_len)
            (go_cue, 
             instruction_cue,
             move_start, 
             move_end) = self.extract_cues(dataset_idx)   
            
            # Parse list of all events for 8 conditions and cut signals condition-wise
            dict_cond_gcidx = cut_condition_wise(data_raw, go_cue, instruction_cue,
                            move_start, move_end, fs=0.01, max_len=2800)
            # Align with go cue, pad with zeros, average, smooth
            a, c = align_pad_smooth(dict_cond_gcidx,
                                    sigma=sigma,
                                    signal_len=signal_len)

        elif dataset_idx == 3 or dataset_idx == 4:
            neur_d = self.data_dict2list(dataset_idx, self.area)
            a, c = extract_and_process_grasp(neur_d,
                                             self.data_dict,
                                             align_cue=align_cue,
                                             fixed_align_cue=fixed_align_cue,
                                             signal_len=signal_len,
                                             sigma=sigma)
        else:
            raise ValueError('Wrong index of dataset: {dataset_idx}! Choose from [3, 4, 5, 6]')

        self.data, self.go_cue = align_mean_conds(a, c)
        self.time = np.arange(0, self.data.shape[1])

        self.full_data = self.data.copy()
        self.full_go_cue, self.full_time = self.go_cue, self.time.copy()

        self.cmap_name = self.data_dict.get('cmap_name')
        if self.cmap_name is not None:
            self.cmap_name = self.cmap_name.decode("utf-8")
        else:
            self.cmap_name = 'church'
        self.cmap = cmaps[self.cmap_name]


        print(f'Data pre-processed! Shape: {self.data.shape}, Go cue={self.go_cue}')

    def extract_cues(self, dataset_idx):
        if dataset_idx == 5:
            go_cue = self.data_dict['go_cell']
            instruction_cue = self.data_dict['instr_cell']
            move_start = self.data_dict['stmv_cell']
            move_end = self.data_dict['endmv_cell']
        elif dataset_idx == 6:
            instr_cue = self.data_dict['events']['times'][1]
            go_cue = self.data_dict['events']['times'][2]
            move_start = self.data_dict['events']['times'][3]
            move_end = self.data_dict['events']['times'][4]
            conds = {x: self.data_dict['events']['times'][y] for x, y in zip(range(8), range(6, 14))}
            # Sort cues and dump excessive
            (go_cue,
            instruction_cue,
            move_start,
            move_end) = sort_cues(conds, instr_cue, go_cue, move_start, move_end)
        return go_cue, instruction_cue, move_start, move_end

    def data_dict2list(self, dataset_idx, area):
        if dataset_idx == 3:
            neur_d = [self.data_dict['SpikeCounts'][area][i][0] for i in range(len(self.data_dict['SpikeCounts'][area]))]
        elif dataset_idx == 4:
            neur_d = [self.data_dict['SpikeCounts'][area][i] for i in range(len(self.data_dict['SpikeCounts'][area]))]
        return neur_d

    def create_info_dict(self):
        self.info_d = {'data': self.data,
                       'time': self.time,
                       'go_cue': self.go_cue,
                       'align_cue_name': self.align_cue,
                       'signal_len_cropping': self.signal_len,
                       'max_len': self.max_len,
                       'dataset_idx': self.dataset_idx,
                       'freq_sampling': self.freq_sampling,
                       'area': self.area,
                       'sigma': self.sigma,
                       'dataset_dir': self.dataset_dir,
                       'cmap_name': self.cmap_name}



class ReachingGallego(NeuralDataset):
    """
    Dataset from paper:
    "Local field potentials reflect cortical population dynamics in a
     region-specific and frequency-dependent manner", 
    Gallego et al. 2022 [doi.org/10.7554/eLife.73155]

    Latent dynamics of Local Field Potential. 
    Behaviour: delay centre-out reaching task using a manipulandum
    Brain areas: primary motor cortex (M1), dorsal premotor cortex (PMd),
                 primary somatosensory cortex (area 2)

    Link to dataset: 
        doi.org/10.5061/dryad.xd2547dkt
    Or see the tutorial how to download at:
        github.com/NevVerVer/neural-dynamics-gyration/blob/main/datasets_analysis.ipynb
    """
    def load_data(self, dataset_name):
        self.data_dict = mat73.loadmat(self.dataset_dir + dataset_name)
        print(f'File {dataset_name} successfully loaded!')

    def load_additional_info(self):
        self.signal_len = self.info_d['signal_len_cropping']
        self.area = self.info_d['area']
        self.sigma = self.info_d['sigma']
        self.dataset_dir = self.info_d['dataset_dir']

        self.cmap_name = self.data_dict.get('cmap_name')
        if self.cmap_name is not None:
            self.cmap_name = self.cmap_name.decode("utf-8")
        else:
            self.cmap_name = 'church'
        self.cmap = cmaps[self.cmap_name]

    def preprocess_data(self, sigma, signal_len='max', area='M1'):
        self.sigma = sigma
        self.signal_len = signal_len
        self.area = area
        
        data_raw = self.data_dict['trial_data']
        a, g, _ = extract_and_process_grasp_LFP(data_raw[area + '_spikes'],
                                                data_raw, 
                                                signal_len=signal_len,
                                                sigma=sigma)
        self.data, self.go_cue = align_mean_conds_LFP(a, g)
        self.time = np.arange(0, self.data.shape[1])

        self.full_data = self.data.copy()
        self.full_go_cue, self.full_time = self.go_cue, self.time.copy()

        self.cmap_name = self.data_dict.get('cmap_name')
        if self.cmap_name is not None:
            self.cmap_name = self.cmap_name.decode("utf-8")
        else:
            self.cmap_name = 'church'
        self.cmap = cmaps[self.cmap_name]

        print(f'Data pre-processed! Shape: {self.data.shape}, Go cue={self.go_cue}')

    def create_info_dict(self):
        self.info_d = {'data': self.data,
            'time': self.time,
            'go_cue': self.go_cue,
            'signal_len_cropping': self.signal_len,
            'area': self.area,
            'sigma': self.sigma,
            'dataset_dir': self.dataset_dir,
            'cmap_name': self.cmap_name}



class ReachingChurchland(NeuralDataset):
    """
    Dataset from paper:
    "Neural Population Dynamics During Reaching", Churchland et al. 2012
    [doi.org/10.1038/nature11129]

    Original jPCA paper. No raw data. Data is already preprocessed and sorted. 
    Behaviour: straight hand reaches, curved hand reaches
    Brain areas: primary motor cortex (M1), dorsal premotor cortex (PMd)

    Link to dataset: 
        ...
    Or see the tutorial how to download at:
        github.com/NevVerVer/neural-dynamics-gyration/blob/main/datasets_analysis.ipynb
    """
    def load_data(self, dataset_name):
        from jPCA.util import load_churchland_data
        self.d = load_churchland_data(self.dataset_dir + dataset_name)
        self.data_dict = {'data': self.d[0], 'time': self.d[1]}
        print(f'File {dataset_name} successfully loaded!')

    def load_additional_info(self):
        self.dataset_dir = self.info_d['dataset_dir']
        
        self.cmap_name = self.data_dict.get('cmap_name')
        if self.cmap_name is not None:
            self.cmap_name = self.cmap_name.decode("utf-8")
        else:
            self.cmap_name = 'church'
        self.cmap = cmaps[self.cmap_name]

    def preprocess_data(self):
        self.data = np.asarray(self.data_dict['data'])
        self.time = np.asarray(self.data_dict['time'])
        go_cue_time = 50 
        self.go_cue = list(self.time).index(go_cue_time)

        self.full_data = self.data.copy()
        self.full_go_cue, self.full_time = self.go_cue, self.time.copy()

        self.cmap_name = self.data_dict.get('cmap_name')
        if self.cmap_name is not None:
            self.cmap_name = self.cmap_name.decode("utf-8")
        else:
            self.cmap_name = 'church'
        self.cmap = cmaps[self.cmap_name]

        print('No preprocessing needed! Loaded.')

    def create_info_dict(self):
        self.info_d = {
            'data': self.data,
            'time': self.time,
            'go_cue': self.go_cue,
            'dataset_dir': self.dataset_dir,
            'cmap_name': self.cmap_name}


class ReachingKalidindi(NeuralDataset):
    """
    Dataset from paper:
    "Rotational dynamics in motor cortex are consistent with a feedback controller", 
    Kalidindi et al. 2021 [doi.org/10.7554/eLife.67256]

    Test what NN architecture can produce brain-like signals.  
    Behaviour: delayed center-out reaching task, posture perturbation task
    Brain areas: premotor and primary motor cortex (MC),
                 primary somatosensory cortex (areas 1, 2, 3a, 5)

    Link to datasets: 
        1. https://archive.softwareheritage.org/browse/revision/d61decd3cd750147ef098de1041326fd2be07ab2/?path=monkey_analysis/data_neural
        2. doi.org/10.5061/dryad.nk98sf7q7
    Or see the tutorial how to download at:
        github.com/NevVerVer/neural-dynamics-gyration/blob/main/datasets_analysis.ipynb
    """
    def load_data(self, dataset_name):
        import scipy.io
        self.data_dict = scipy.io.loadmat(self.dataset_dir + dataset_name)
        print(f'File {dataset_name} successfully loaded!')

    def load_additional_info(self):
        self.dataset_dir = self.info_d['dataset_dir']

        self.cmap_name = self.data_dict.get('cmap_name')
        if self.cmap_name is not None:
            self.cmap_name = self.cmap_name.decode("utf-8")
        else:
            self.cmap_name = 'church'
        self.cmap = cmaps[self.cmap_name]
        
    def preprocess_data(self):
        self.data = np.asarray([self.data_dict['Data_struct'][0][i][0] for i in range(8)])
        # np.asarray(self.data_dict['Data_struct'][0][0][1][0])
        self.time = np.arange(0, self.data.shape[1]) 
        self.go_cue = 0

        self.full_data = self.data.copy()
        self.full_go_cue, self.full_time = self.go_cue, self.time.copy()

        self.cmap_name = self.data_dict.get('cmap_name')
        if self.cmap_name is not None:
            self.cmap_name = self.cmap_name.decode("utf-8")
        else:
            self.cmap_name = 'church'
        self.cmap = cmaps[self.cmap_name]

        print('No preprocessing needed! Loaded.')

    def create_info_dict(self):
        self.info_d = {'data': self.data,
            'time': self.time,
            'go_cue': self.go_cue,
            'dataset_dir': self.dataset_dir,
            'cmap_name': self.cmap_name}



class BehaviouralMante(NeuralDataset):
    """
    Dataset from paper: 
    "Context-dependent Computation by Recurrent Dynamics in Prefrontal Cortex", 
    Mante et al. 2013 [doi.org/10.1038/nature12742]

    Unfolding dynamical process in PFC during complex behaviour.
    Behaviour: context-dependent 2-alternative forced-choice
               visual discrimination task (perceptual decision-making)
    Brain areas: prefrontal cortex (PFC)

    Link to dataset:
        https://www.ini.uzh.ch/en/research/groups/mante/data.html
    Or see the tutorial how to download at:
        github.com/NevVerVer/neural-dynamics-gyration/blob/main/datasets_analysis.ipynb
    """
    def load_data(self):
        self.paths = sorted(os.listdir(self.dataset_dir))
        print('This dataset is loaded into preprocessing method!')

    def load_additional_info(self):
        self.cond_sorting = self.info_d['cond_sorting']
        self.sigma = self.info_d['sigma']
        self.dataset_dir = self.info_d['dataset_dir']

        self.cmap_name = self.data_dict.get('cmap_name')
        if self.cmap_name is not None:
            self.cmap_name = self.cmap_name.decode("utf-8")
        else:
            self.cmap_name = 'church'
        self.cmap = cmaps[self.cmap_name]

    def preprocess_data(self, sigma, cond_sorting):
        self.sigma = sigma
        self.cond_sorting = cond_sorting

        if cond_sorting == 1 or cond_sorting == 3:
            nc = 36
        elif cond_sorting == 2 or cond_sorting == 4:
            nc = 6
        nt, nn = 750, len(self.paths)

        self.data = np.zeros((nc, nt+1, nn))
        for neuron in range(nn):
            data_dict = scipy.io.loadmat(self.dataset_dir + self.paths[neuron])

            # Extract 36 main conditions (for one file(unit))
            self.stim_dir_cond = data_dict['unit'][0][0][1][0][0][0][:, 0]
            self.stim_col_cond = data_dict['unit'][0][0][1][0][0][1][:, 0]
            self.stim_dir_labels = np.unique(self.stim_dir_cond)
            self.stim_col_labels = np.unique(self.stim_col_cond)
            # idx=4. context: motion (+1) or color context (-1)
            self.context = data_dict['unit'][0][0][1][0][0][4][:, 0]

            i = 0
            conditions_idxes = self.extract_idxes(cond_sorting=cond_sorting)
            for idx in conditions_idxes:

                    cond_trials = data_dict['unit'][0][0][0][idx, :]

                    averaged_trials = np.mean(np.asarray(cond_trials), axis=0)
                    avrg_smoth_cond = gaussian_filter1d(averaged_trials,
                                                        sigma=sigma, mode='constant')
                    self.data[i, :, neuron] = avrg_smoth_cond
                    i += 1
        self.go_cue = 0
        self.time = np.arange(0, nt)

        self.full_data = self.data.copy()
        self.full_go_cue, self.full_time = self.go_cue, self.time.copy()

        self.cmap_name = self.data_dict.get('cmap_name')
        if self.cmap_name is not None:
            self.cmap_name = self.cmap_name.decode("utf-8")
        else:
            self.cmap_name = 'church'
        self.cmap = cmaps[self.cmap_name]

        print(f'Data pre-processed! Shape: {self.data.shape}, Go cue={self.go_cue}')

    def extract_idxes(self, cond_sorting):
        idx_list = []
        if cond_sorting == 1: # 
            for dir in self.stim_dir_labels:
                for col in self.stim_col_labels:
                    idx = np.argwhere((self.stim_dir_cond==dir) & \
                                    (self.stim_col_cond==col)).flatten()
                    idx_list.append(idx)

        elif cond_sorting == 2: # Color only
            for col in self.stim_col_labels:
                idx = np.argwhere(self.stim_col_cond==col).flatten() 
                idx_list.append(idx)

        elif cond_sorting == 3: # Color_movement_context
            for dir in self.stim_dir_labels:
                idx = np.argwhere((self.stim_dir_cond==dir) & \
                                  (self.context==1)).flatten()
                idx_list.append(idx)
            for col in self.stim_col_labels:
                idx = np.argwhere((self.stim_col_cond==col) & \
                                    (self.context==-1)).flatten()
                idx_list.append(idx)

        elif cond_sorting == 4: # Movement only
            for dir in self.stim_dir_labels:
                idx = np.argwhere(self.stim_dir_cond==dir).flatten()
                idx_list.append(idx)
        else:
            raise ValueError('Inproper condition sorting parameter!')
        
        return idx_list

    def create_info_dict(self):
        self.info_d = {'data': self.data,
                       'time': self.time,
                       'go_cue': self.go_cue,
                       'cond_sorting': self.cond_sorting,
                       'sigma': self.sigma,
                       'dataset_dir': self.dataset_dir,
                       'cmap_name': self.cmap_name}
        

class SyntheticTravelingWave(NeuralDataset):
    def load_data(self, dataset_name):
        self.data_dict = {}

    def load_additional_info(self):
        self.sigma = self.info_d['sigma']
        self.b = self.info_d['wave_speed']
        self.sigma_noise = self.info_d['sigma_noise']
        self.phase_noise = self.info_d['phase_noise']
        self.amp_noise = self.info_d['amp_noise']

        self.cmap_name = self.data_dict.get('cmap_name')
        if self.cmap_name is not None:
            self.cmap_name = self.cmap_name.decode("utf-8")
        else:
            self.cmap_name = 'church'
        self.cmap = cmaps[self.cmap_name]

    def preprocess_data(self, sigma, a, b, ncond, N, t_max, steps,
                        sigma_noise=0.0, phase_noise=0.0, amp_noise=0.0,
                        amp_range=[0.5, 1.0]):
        print('Generating datasets!')
        self.sigma = sigma
        self.steps = steps
        self.a = a
        self.b = b
        self.sigma_noise = sigma_noise
        self.phase_noise = phase_noise
        self.amp_noise = amp_noise
        self.data = [generate_response(t_max=t_max, N=N, 
                                        sigma=sigma, 
                                        a=a, b=b, steps=steps, 
                                        amp_mean=ampi,
                                        sigma_noise=sigma_noise,
                                        phase_noise=phase_noise,
                                        amp_noise=amp_noise)[0].T 
                                        for ampi in np.linspace(*amp_range, ncond)]
        self.data = np.asarray(self.data)
        self.time = np.arange(self.data.shape[1])
        self.go_cue = 0

        self.full_data = self.data.copy()
        self.full_go_cue, self.full_time = self.go_cue, self.time.copy()

        self.cmap_name = self.data_dict.get('cmap_name')
        if self.cmap_name is not None:
            self.cmap_name = self.cmap_name.decode("utf-8")
        else:
            self.cmap_name = 'church'
        self.cmap = cmaps[self.cmap_name]

        print(f'Data pre-processed! Shape: {self.data.shape}, Go cue={self.go_cue}')

    def create_info_dict(self):
        self.info_d = {
            'data': self.data,
            'time': self.time,
            'go_cue': self.go_cue,
            'sigma': self.sigma,
            'wave_speed': self.b,
            'sigma_noise': self.sigma_noise,
            'phase_noise': self.phase_noise,
            'amp_noise': self.amp_noise,
            'cmap_name': self.cmap_name,
        }