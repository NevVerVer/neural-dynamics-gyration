"""
Created on Mon October 23 11:05:36 2023

Description: Visualization functions for
Paper url: https://www.biorxiv.org/content/10.1101/2023.09.11.557230v2

@author: Ekaterina Kuzmina, ekaterina.kuzmina@skoltech.ru
@author: Dmitrii Kriukov, dmitrii.kriukov@skoltech.ru
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

from tqdm import tqdm 
import jPCA
from jPCA.util import plot_projections
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from utils.color_palettes import (
    bg_gray_color,
    jpca_palette_journ1,
    journ_color_dict,
    blue_yellow_cmap)
from utils.utils import (
    subtract_cc_mean,
    soft_normalize,
    compute_curvature,
    pca_rotation)


def plot_curvature(datas, go_cue, ax, fs, sub_mean=False):
    datas = soft_normalize(datas)
    if sub_mean:
        datas = subtract_cc_mean(datas)
    times = np.arange(datas.shape[1]) * fs
    cc_curv = []
    for X in datas:
        curv = compute_curvature(X)
        cc_curv.append(curv)
        ax.plot(times, curv, color='grey', lw=1 )
    ax.axhline(0, color='black', alpha=0.4, ls='--')
    ax.axvline(go_cue, color='black', alpha=0.4, ls='--', 
                label=f'go cue={go_cue}')
    cc_mean = np.mean(np.asarray(cc_curv), axis=0)
    ax.plot(times, cc_mean, c=journ_color_dict['red'], lw=1.4, label='mean')
    ax.legend(prop={'size': 9})
    ax.grid(alpha=0.3)
    
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    ax.set_facecolor(bg_gray_color)
    ax.yaxis.tick_right()


def peth_normalize(x):
    mean = x.mean(axis=0)
    x_ = x - mean
    pmax = np.abs(x_).max(axis=0)
    pmax = np.where(pmax==0, 1, pmax)
    return x_ / (pmax)


def plot_peth(datas_list, go_cue, ax, fs, sub_mean=False):
    """
    Plot PETHs averaged across conditions for a given data
    """
    datas = soft_normalize(datas_list)
    if sub_mean:
        datas = subtract_cc_mean(datas)

    # sort by average wave maximum
    cc_mean = np.mean(datas, axis=0)
    idxs = np.argmax(cc_mean, axis=0)
    cc_mean_sorted = cc_mean[:, np.argsort(idxs)]
    divider = make_axes_locatable(ax)

    # avg condition
    mean_data_sort_norm = peth_normalize(cc_mean_sorted)
    im1 = ax.imshow(mean_data_sort_norm.T, aspect='auto',
                       cmap=blue_yellow_cmap(), interpolation='none')
    ax.axvline(go_cue, c='black', ls='--', alpha=0.4)
    cax1 = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(im1, cax=cax1, orientation='vertical')
    cbar.ax.tick_params(labelsize=8)

    tick_locator = ticker.MaxNLocator(nbins=4)
    cbar.locator = tick_locator
    cbar.update_ticks()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))

    pos = np.arange(0., datas.shape[1]+1, datas.shape[1]//4)
    times = pos * fs
    times = [int(t) for t in times]
    ax.set_xticks(pos, labels=times)

def build_jPCA(data, ax, ts, te, sub_mean=True,
               c_siz=0.001, arr_siz=0.001, cmap=jpca_palette_journ1(),
               fitting=False):
    cond_num = data.shape[0]
    datas_list = [data[i] for i in range(cond_num)]
    times = list(np.arange(datas_list[0].shape[0]))

    data_list = list(soft_normalize(datas_list))
    jpca = jPCA.JPCA(num_jpcs=6)
    (projected, _, _, _) = jpca.fit(data_list, times=times, 
                tstart=ts, tend=te, subtract_cc_mean=sub_mean,
                num_pcs=6, pca=True, soft_normalize=0)
    if fitting:
        projected = [-1.*proj for proj in projected]
    plot_projections(projected, axis=ax, x_idx=0, y_idx=1, circle_size=c_siz, arrow_size=arr_siz,
                     colormap=cmap)


def build_pca(data, ax, ts, te, sub_mean=True,
              c_siz=0.001, arr_siz=0.001, cmap=jpca_palette_journ1()):
    datas_list = [i for i in data]
    datas_list = soft_normalize(datas_list)
    _, data_list, _, _ = pca_rotation(datas_list, ts, te, n_pca=6,
                                      sub_mean=sub_mean, random_state=228)
    plot_projections(data_list, axis=ax, x_idx=0, y_idx=1, circle_size=c_siz, arrow_size=arr_siz,
                     colormap=cmap) 


def func(t, mu, sigma, amp):
    return amp * np.exp(-((t - mu) / sigma)**2)


def fit_running_wave(datas, times, kalid=False):
    datas_cap = np.zeros_like(datas)
    r2_list_across_conds = []
    param_list_across_conds = []
    if kalid:
        a, b = 2., 2.5
    else:
        a, b = 10., 1.5
    for c, cond in tqdm(enumerate(datas)):
        r2_list = []
        param_list = []
        t = times
        for i in range(cond.shape[1]):
            y = cond[:, i]
            popt, pcov = curve_fit(func, t, y, bounds=([-100, a, 0.], [600, 5000, b]))
            y_cap = func(t, *popt)
            datas_cap[c, :, i] = y_cap
            r2 = round(pearsonr(y, y_cap)[0]**2, 3)
            r2_list.append(r2)
            param_list.append(popt)
        r2_list_across_conds.append(r2_list)
        param_list_across_conds.append(param_list)
    return datas_cap, np.asarray(r2_list_across_conds), np.asarray(param_list_across_conds)


def plot_R2_fitting(datas, times, ax, ts, te, 
                    c_siz=0.001, arr_siz=0.001, cmap=jpca_palette_journ1(),
                    pfc=False, kalid=False):
    datas = soft_normalize(datas)
    datas_cap, R2, _ = fit_running_wave(datas, times, kalid=kalid)
    if pfc:
        build_jPCA(datas_cap, ax, times[0], times[-2], sub_mean=True,
                c_siz=c_siz, arr_siz=arr_siz, cmap=cmap, fitting=True)
    else:
        build_jPCA(datas_cap, ax, ts, te, sub_mean=True,
                   c_siz=c_siz, arr_siz=arr_siz, cmap=cmap, fitting=True)
    return R2