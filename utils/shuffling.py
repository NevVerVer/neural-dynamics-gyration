"""
Created on Sat September 16 21:15:06 2023

Description: Implementation of different shuffling approaches for neural datasets.
Paper url: https://www.biorxiv.org/content/10.1101/2023.09.11.557230v1

@author: Ekaterina Kuzmina, ekaterina.kuzmina@skoltech.ru
@author: Dmitrii Kriukov, dmitrii.kriukov@skoltech.ru
"""

import numpy as np
import random
from itertools import product

def shuffle_data(datas, times, shuffle_type, go_cue_time, keep_var=False):
    """
    Shufflings of data from paper:
    "Neural Population Dynamics During Reaching", Churchland et al., 2012

    All 3 base shuffle controls are based on the distinction between preparatory
    activity (which is left intact) and peri-movement activity.
    1 type: 
        The pattern of peri-movement activity was inverted for half of the
        conditions, selected at random. The inversion was performed around
        the dividing time-point, such that continuity with preparatory
        activity was preserved. This procedure was performed separately 
        for each neuron.
    2 type:
        Similar to the 1st, but inverted the peri-movement activity
        pattern for all conditions. 
    3 type: 
        Randomly reassigned the peri-movement activity from one condition
        to the preparatory activity from another. The beginning of the
        peri-movement pattern was simply appended to the final firing rate
        during the preparatory state, such that there was no discontinuity.
        The same reassignment was performed for all neurons.
    """

    inv_start = go_cue_time
    shuffled_data = np.stack(datas)
    n_cond, n_time, n_neuron = shuffled_data.shape
    
    if shuffle_type == 1:
        s = np.random.binomial(1, .5, (n_neuron, n_cond))

        for neuron_idx in range(n_neuron):
            f = s[neuron_idx]
            for cond_idx in range(n_cond):
                if f[cond_idx]:
                    d = shuffled_data[cond_idx, :, neuron_idx]
                    origin = d[inv_start]

                    d[inv_start:] = ((d[inv_start:] - origin) * -1) + origin
                    shuffled_data[cond_idx, :, neuron_idx] = d
#         reversed_neuron_idxs = np.where(s[0]) 

    elif shuffle_type == 2:
        for neuron_idx in range(n_neuron):
            for cond_idx in range(n_cond):
                d = shuffled_data[cond_idx, :, neuron_idx]
                origin = d[inv_start]

                d[inv_start:] = ((d[inv_start:] - origin) * -1) + origin
                shuffled_data[cond_idx, :, neuron_idx] = d
                      
    elif shuffle_type == 3:

        p1 = np.arange(0, n_cond)
        np.random.shuffle(p1)
        p2 = p1[n_cond // 2 : n_cond]
        p1 = p1[0 : n_cond // 2]
        dif_distr = []
        for neuron_idx in range(n_neuron):
            for cond_idx in range(n_cond//2):

                d = shuffled_data[p1[cond_idx], :, neuron_idx]
                d2 = shuffled_data[p2[cond_idx], :, neuron_idx]

                d_ = d.copy()
                d2_ = d2.copy()

                origin, d2_origin = d[inv_start], d2[inv_start]
                dif = np.abs(origin - d2_origin)
                dif_distr.append(dif)
                if d2_origin > origin:
                    d[inv_start:] = d2_[inv_start:] - dif
                    d2[inv_start:] = d_[inv_start:] + dif

                elif d2_origin < origin:
                    d[inv_start:] = d2_[inv_start:] + dif
                    d2[inv_start:] = d_[inv_start:] - dif

                else:
                    d[inv_start:] = d2_[inv_start:]
                    d2[inv_start:] = d_[inv_start:]

                shuffled_data[p1[cond_idx], :, neuron_idx] = d
                shuffled_data[p2[cond_idx], :, neuron_idx] = d2
    else:
        raise ValueError('Incorrect shuffle type')
    
    return shuffled_data

def shuffling_CMPT(datas, threshold=0.95, verbose=0): # TODO
    datas_idx_0 = np.arange(np.prod(datas.shape)).reshape(datas.shape)
    datas_idx_s = datas_idx_0.copy()
    neuron_stack_obs = np.concatenate(datas)
    cov_obs = np.cov(neuron_stack_obs.T)
    
    if verbose > 0: print('Initialize initial permutation.')
    step1 = np.zeros_like(datas)
    permutations = [np.random.permutation(datas.shape[0]) for _ in range(datas.shape[2])]
    for i in range(datas.shape[2]):
        neuron = datas[:, :, i]
        step1[:, :, i] = neuron[permutations[i]]
        neuron_idx = datas_idx_s[:, :, i]
        datas_idx_s[:, :, i] = neuron_idx[permutations[i]]

    step2 = step1.copy()
    neuron_stack_per = np.concatenate(step2)
    cov_per = np.cov(neuron_stack_per.T)
    similarity = 1 - np.sum((cov_per - cov_obs)**2) / np.sum(cov_obs**2)

    comb = []
    for i in range(0, datas.shape[0]):
        for j in range(i+1, datas.shape[0]):
            for k in range(datas.shape[2]):
                comb.append((i, j, k)) 

    if verbose > 0: print(f'Total number of possible permutations: {len(comb)}')
    if verbose > 0: print('Start targeting permutations.')
    j = 0            
    while len(comb) > 0:
        j+=1
        ri = np.random.randint(len(comb))
        *pair, ni = comb.pop(ri)

        tmp = step2.copy()
        tmp[pair[0],:,ni], tmp[pair[1],:,ni] = tmp[pair[1],:,ni], tmp[pair[0],:,ni]
        datas_idx_s[pair[0],:,ni], datas_idx_s[pair[1],:,ni] = datas_idx_s[pair[1],:,ni], datas_idx_s[pair[0],:,ni]
        tmp_cov = np.cov(np.concatenate(tmp).T)
        new_similarity = 1 - np.sum((tmp_cov - cov_obs)**2) / np.sum(cov_obs**2)
        if j % 10000 == 0:
            print(f"{j} iter:", similarity)
        if new_similarity > similarity:
            similarity = new_similarity
            step2 = tmp
        else:
            continue    
        if similarity > threshold:
            break
    
    if verbose > 0:
        print("Percent of indices coincide: {:.4f}%".format((datas_idx_0 == datas_idx_s).sum() / np.prod(datas_idx_s.shape) * 100))
    print("Achieved similarity: {:.4f}% for {} iterations".format(similarity, j))
    print('Done!')
    return step2