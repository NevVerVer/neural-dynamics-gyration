"""
Created on Sat September 16 21:15:06 2023

Description: Controller class implementations for:
Paper url: https://www.biorxiv.org/content/10.1101/2023.09.11.557230v1

@author: Ekaterina Kuzmina, ekaterina.kuzmina@skoltech.ru
@author: Dmitrii Kriukov, dmitrii.kriukov@skoltech.ru
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from utils.color_palettes import blue_yellow_cmap
from utils.utils import subtract_cc_mean, soft_normalize, compute_lambdas
from utils.utils import compute_curvature, pca_rotation
from utils.shuffling import shuffle_data
import jPCA
from jPCA.util import plot_projections
from utils.utils import fit_running_wave

from utils.data_processing import load_h5_file
from utils.color_palettes import GR1, RB1, YP1, YS1, BG1
from jPCA.util import red_green_cmap
import os


cmaps = {'lfp': YP1(),
        'churchland': GR1(),
        'grasping': RB1(),
        'kalid': YS1(),
        'mante': BG1()}

SHOW = True

def plot_each_cond(cond_tensor, gc, subtract_mean=True,
                   n=8, save_name=None):
    datas = soft_normalize(list(cond_tensor))
    if subtract_mean:
        datas = subtract_cc_mean(datas)

    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    print('Each condition for all neurons')
    i, j = 0, 0
    for cond in range(n):
        if cond == 4:
            i, j = 1, 0
        for signal in datas[cond].T:
            ax[i, j].plot(signal)
        ax[i, j].axvline(gc, ls='--', c='black', alpha=0.5)
        ax[i, j].set_title(f'Condition number {cond + 1}')
        j += 1

    if save_name:
        plt.savefig(save_name + f'_8conds.png',
            dpi=300, bbox_inches='tight', pad_inches=0.0)
    if SHOW:
        plt.show()
    plt.close()


def plot_each_neuron(cond_tensor, gc, subtract_mean=True,
                     n=25, save_name=None):
    datas = soft_normalize(list(cond_tensor))
    if subtract_mean:
        datas = subtract_cc_mean(datas)

    fig, ax = plt.subplots(5, 5, figsize=(20, 20))
    print('Each neuron for all conditions')
    i, j = 0, 0
    for neuron in range(n):
        if neuron in [5, 10, 15, 20]:
            j = 0
            i += 1
        for signal in datas[:, :, neuron]:
            ax[i, j].plot(signal)

        ax[i, j].axvline(gc, ls='--', c='black', alpha=0.5)
        ax[i, j].set_title(f'Neuron number {neuron + 1}')
        j += 1

    if save_name:
        plt.savefig(save_name + f'_25neurons.png',
            dpi=300, bbox_inches='tight', pad_inches=0.0)
    if SHOW:
        plt.show()
    plt.close()


def peth_normalize(x):
    mean = x.mean(axis=0)
    x_ = x - mean
    pmax = np.abs(x_).max(axis=0)
    pmax = np.where(pmax==0, 1, pmax)
    return x_ / (pmax)


def plot_peth(datas_list, times, subtract_mean, go_cue=0,
              cond_list=[], sort_individ=False, 
              save_name=None):
    """
    Plot PETHs averaged across conditions for a given data
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    datas = soft_normalize(datas_list)
    if subtract_mean:
        datas = subtract_cc_mean(datas)

    # sort by average wave maximum
    cc_mean = np.mean(datas, axis=0)
    idxs = np.argmax(cc_mean, axis=0)
    cc_mean_sorted = cc_mean[:, np.argsort(idxs)]

    fig, ax = plt.subplots(1, 1 + len(cond_list), figsize=(4 * (len(cond_list) + 1), 8))
    ax = [ax] if not cond_list else ax
    divider = make_axes_locatable(ax[-1])

    # avg condition
    mean_data_sort_norm = peth_normalize(cc_mean_sorted)
    im1 = ax[0].imshow(mean_data_sort_norm.T, aspect='auto',
                       cmap=blue_yellow_cmap(), interpolation='none')
    ax[0].axvline(go_cue, c='white', ls='--', alpha=1)
    ax[0].set_title('Average PETH, Subtract mean={}'.format(subtract_mean))
    cax1 = divider.append_axes('right', size='5%', pad=0.05)
    ax[0].set_yticks([])
        
    datas = np.asarray(datas)
    for cnum, cond in enumerate(cond_list, start=1):
        cond_idxs = np.argmax(datas[cond], axis=0) if sort_individ else idxs
        datas_sort_norm = peth_normalize(datas[cond, :, np.argsort(cond_idxs)])
        ax[cnum].imshow(datas_sort_norm, aspect='auto',
                        cmap=blue_yellow_cmap(), interpolation='none')
        ax[cnum].axvline(go_cue, c='white', ls='--', alpha=1)
        ax[cnum].set_title(f'Normalized PETH of {cond}th condition')
        ax[cnum].set_yticks([])
        
    fig.colorbar(im1, cax=cax1, orientation='vertical')

    if save_name:
        plt.savefig(save_name + f'_peth_{subtract_mean}.png',
         dpi=300, bbox_inches='tight', pad_inches=0.0)

    if SHOW:
        plt.show()
    plt.close()


def build_jPCA(data, gc, ts, te, subtract_mean, shuffling=None,
               siz=0.005, save_name=None, jpca_cmap=red_green_cmap()):
    
    cond_num = data.shape[0]
    datas_list = [data[i] for i in range(cond_num)]
    times = list(np.arange(datas_list[0].shape[0]))

    if shuffling:
        datas_list = shuffle_data(datas_list, times, 
                                  shuffle_type=shuffling,
                                  go_cue_time = gc, keep_var=False)
    data_list = list(soft_normalize(datas_list))

    jpca = jPCA.JPCA(num_jpcs=6)
    (projected, _, _, _) = jpca.fit(data_list, times=times, 
                tstart=ts, tend=te, subtract_cc_mean=subtract_mean,
                num_pcs=6, pca=True, soft_normalize=0)
    
    if save_name:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        plot_projections(projected, axis=axes[0], x_idx=0, y_idx=1, circle_size=siz,
                         arrow_size=siz, colormap=jpca_cmap) 
        plot_projections(projected, axis=axes[1], x_idx=2, y_idx=3, circle_size=siz,
                         arrow_size=siz, colormap=jpca_cmap) 
        axes[0].set_title("jPCA Plane 1")
        axes[1].set_title("jPCA Plane 2")
        plt.tight_layout()
        fig.savefig(save_name + '_jPCA.png')
        if SHOW:
            plt.show()
        plt.close()
    else:
        print('jPCA')
        plot_projections(projected, x_idx=0, y_idx=1, circle_size=siz, arrow_size=siz, 
                         colormap=jpca_cmap)


def build_pca(data, time, ts, te, sub_mean=True, siz=0.001, save_name=None,
              pca_cmap=red_green_cmap()):
    datas_list = [i for i in data]
    times = list(time)

    datas_list = soft_normalize(datas_list)
    _, data_list, _, _ = pca_rotation(datas_list, ts, te, n_pca=6,
                                      sub_mean=sub_mean, random_state=228)
    if save_name:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        plot_projections(data_list, axis=axes[0], x_idx=0, y_idx=1, circle_size=siz,
                         arrow_size=siz, colormap=pca_cmap) 
        plot_projections(data_list, axis=axes[1], x_idx=2, y_idx=3, circle_size=siz,
                         arrow_size=siz, colormap=pca_cmap) 
        axes[0].set_title("PCA Plane 1")
        axes[1].set_title("PCA Plane 2")
        plt.tight_layout()
        fig.savefig(save_name + '_PCA.png')
        if SHOW:
            plt.show()
        plt.close()
    else:
        print('PCA')
        plot_projections(data_list, x_idx=0, y_idx=1, circle_size=siz,
                         arrow_size=siz, colormap=pca_cmap) 


def plot_clusterization(datas_list, times, sub_mean=False, save_name=None):

    start, end = times[0], times[-1]
    data = soft_normalize(datas_list)[:, start:end, :]
    if sub_mean:
        data = subtract_cc_mean(data)

    n_cond = data.shape[0]
    cond_stack = np.transpose(data, (0, 2, 1)).reshape(n_cond, -1)
    clust = sns.clustermap(np.corrcoef(cond_stack), method='complete',
                           vmin=-1, vmax=1, cmap='RdBu_r')
    print('Clusterization of all conditions')
    if save_name:
        clust.savefig(save_name + '_cluster.png', dpi=300, 
                    bbox_inches='tight', pad_inches=0.0)
        if SHOW:
            plt.show() 
        plt.close()


def plot_eigenPCA(datas, npca=4, sub_mean=False, save_name=None):

    datas = soft_normalize(datas)
    if sub_mean:
        datas = subtract_cc_mean(datas)

    variances = []
    for i in range(0, datas.shape[0]):
        cond = datas[i]
        pca = PCA(npca)
        Xp = pca.fit_transform(cond)
        variances.append(pca.explained_variance_ratio_.sum())

        plt.plot(Xp[:, 0], Xp[:, 1])
    plt.title(f'EigenPCA, npca={npca}')
    if save_name:
        plt.savefig(save_name + '_eigenPCA.png', dpi=300,
                    bbox_inches='tight', pad_inches=0.0)
    if SHOW:
        plt.show()
    plt.close()
    # print("Min-Max variance explained:", min(variances), max(variances))
    return Xp


def plot_NTCcov_matrises(datas, sub_mean=False, save_name=None):

    datas = soft_normalize(datas)
    if sub_mean:
        datas = subtract_cc_mean(datas)

    neuro_stack = np.concatenate(datas)
    time_stack = np.concatenate(datas.transpose(0, 2, 1))
    cond_stack = np.transpose(datas, (0, 2, 1)).reshape(datas.shape[0], -1)
    NC = np.cov(neuro_stack.T)
    TC = np.cov(time_stack.T)
    CC = np.cov(cond_stack)
    Neig = np.abs(np.sort(np.linalg.eigh(NC)[0])[::-1])
    Teig = np.abs(np.sort(np.linalg.eigh(TC)[0])[::-1])
    Ceig = np.abs(np.sort(np.linalg.eigh(CC)[0])[::-1])
    ln, lt, lc = Neig[0] / Neig.sum(), Teig[0] / Teig.sum(), Ceig[0] / Ceig.sum()

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    divider = make_axes_locatable(ax[-1])
    
    r = 3
    im1 = ax[0].imshow(NC, aspect='auto',
                    cmap='RdBu_r', interpolation='none')
    ax[0].set_title(f'Neural Covariance, lambda={round(ln, r)}', fontsize=14)
    ax[1].imshow(TC, aspect='auto',
                cmap='RdBu_r', interpolation='none')
    ax[1].set_title(f'Time Covariance, lambda={round(lt, r)}', fontsize=14)
    ax[2].imshow(CC, aspect='auto',
                cmap='RdBu_r', interpolation='none')
    ax[2].set_title(f'Condition Covariance, lambda={round(lc, r)}', fontsize=14)

    cax1 = divider.append_axes('right', size='5%', pad=0.07)
    fig.colorbar(im1, cax=cax1, orientation='vertical')

    if save_name:
        plt.savefig(save_name + f'_NTCcov.png',
            dpi=300, bbox_inches='tight', pad_inches=0.0)
    if SHOW:
        plt.show()
    plt.close()
    return ln, lt, lc 


def plot_curvature(datas, go_cue, sub_mean=False, save_name=None):

    datas = soft_normalize(datas)
    if sub_mean:
        datas = subtract_cc_mean(datas)

    cc_curv = []
    plt.figure(figsize=(8, 5))
    for X in datas:
        curv = compute_curvature(X)
        cc_curv.append(curv)
        plt.plot(curv, color='grey', lw=0.5)
    plt.axhline(0, color='grey', ls='--')
    plt.axvline(go_cue, color='black', alpha=0.6, ls='--',
                label=f'go cue={go_cue}')

    cc_mean = np.mean(np.asarray(cc_curv), axis=0)
    plt.plot(cc_mean, c='r', lw=0.8, label='mean')
    plt.title('Curvature for each condition')
    plt.legend()
    plt.grid(alpha=0.3)
    if save_name:
        plt.savefig(save_name + '_curvature.png', dpi=300,
                    bbox_inches='tight', pad_inches=0.0)
    if SHOW:
        plt.show()
    plt.close()


def plot_R2_fitting(datas, times, save_name=None):
    datas = soft_normalize(datas)
    _, R2, _ = fit_running_wave(datas, times)
    _, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.hist(np.array(R2).ravel(), bins=40)
    plt.title('All conditions fit R2 distribution')
    ax.axvline(np.nanmean(R2), c='red', label='mean') 
    plt.xlim([0., 1.]) 
    plt.legend()
    if save_name:
        plt.savefig(save_name + '_r2_hist.png', dpi=300,
                    bbox_inches='tight', pad_inches=0.0)
    if SHOW:
        plt.show() 
    plt.close()


def plot_all_stats(data, time, go_cue, sort_individ, sub_mean,
                   jpca_ts, jpca_te, jpca_siz,
                   pca_ts, pca_te, pca_siz, mult=1,
                   save_name=None, cond_list=[0,1,2],
                   shuffling=None, crop_signal=None):
    """
    data: [cond, time, neurons]
    """
    print('Subtract mean = ', sub_mean)
    if crop_signal:
        data = data[:, crop_signal[0]:crop_signal[1], :]
        new_len = data.shape[1]
        time = time[:new_len]
        go_cue = go_cue - crop_signal[0]
        jpca_ts, jpca_te = jpca_ts - crop_signal[0] + 1, jpca_te - crop_signal[0] - 1
        pca_ts, pca_te = pca_ts - crop_signal[0] + 1, pca_te - crop_signal[0] - 1

    if shuffling:
        data = shuffle_data(data, time, 
                            shuffle_type=shuffling,
                            go_cue_time = go_cue)
    # Example conditions
    plot_each_cond(data, go_cue, subtract_mean=sub_mean, n=4,
                   save_name=save_name)
    # Example neurons
    # plot_each_neuron(data, go_cue, subtract_mean=sub_mean,
    #                  save_name=save_name)

    datas_list = [i for i in data]
    times = list(time)     
    # With subtract_mean=True                  
    plot_peth(datas_list, times, subtract_mean=True, go_cue=go_cue,
             cond_list=cond_list, sort_individ=sort_individ, 
             save_name=save_name)
    # Without subtract_mean=False
    plot_peth(datas_list, times, subtract_mean=False, go_cue=go_cue,
            cond_list=cond_list, sort_individ=sort_individ, 
            save_name=save_name)

    plot_curvature(data, go_cue, sub_mean=sub_mean, save_name=save_name)
    plot_clusterization(data, times, sub_mean=sub_mean, save_name=save_name)
    ln, lt, lc = plot_NTCcov_matrises(data, sub_mean=sub_mean, save_name=save_name)

    _ = plot_eigenPCA(data, sub_mean=sub_mean, save_name=save_name)
    build_jPCA(data*mult, go_cue, jpca_ts, jpca_te, sub_mean,
               siz=jpca_siz, save_name=save_name)
    build_pca(data*mult, time, pca_ts, pca_te, sub_mean=sub_mean,
              siz=pca_siz, save_name=save_name)

    plot_R2_fitting(data, times, save_name=save_name)

    return ln, lt, lc


def plot_dataset_gyration_plane(cond_n,
                                h5_save_dir, siz=0,
                                Q = 1., W = 1., ins_size=0.07, ins_borders = 0.3,
                                save_name=None): 
    d_names = os.listdir(h5_save_dir)

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    for dname in d_names:

        info_d = load_h5_file(h5_save_dir, dname)

        datas = info_d['data']
        cond_n = np.min([datas.shape[0], cond_n])
        datas = datas[:cond_n, :, :]

        start_pad = info_d['start_pad'] if info_d.get('start_pad') is not None else 0
        end_pad = info_d['end_pad'] if info_d.get('end_pad') is not None else datas.shape[1]
        go_cue = info_d['go_cue'] if info_d.get('go_cue') is not None else 0

        time = np.arange(0, datas.shape[1]) # info_d['time']
        tstart = np.max([0, go_cue-start_pad])
        tend = np.min([len(time)-1, go_cue+end_pad])

        # fit jPCA
        datas_list = [datas[i] for i in range(datas.shape[0])]
        jpca = jPCA.JPCA(num_jpcs=6)
        (projected, _, _, _) = jpca.fit(datas_list, times=list(time), 
                tstart=tstart+1, tend=tend, subtract_cc_mean=True, num_pcs=6, pca=True)
    
        datas = subtract_cc_mean(soft_normalize(datas))
        datas = datas[:, tstart:tend, :]
        time = time[:tend]
        
        Xeig = compute_lambdas(datas)
        xnom = np.abs(np.real(Xeig[0])) + np.abs(np.real(Xeig[1]))
        ynom = np.abs(np.imag(Xeig[0])) + np.abs(np.imag(Xeig[1]))
        x = xnom / (np.abs(Xeig).sum()) * Q
        y = ynom / (np.abs(Xeig).sum()) * W
    
        ins = ax.inset_axes([x - ins_size/2, y - ins_size/2, ins_size, ins_size], transform=ax.transData)
        ins.set_aspect('equal', adjustable='datalim')
        for axis in ['top', 'bottom', 'left', 'right']:
            ins.spines[axis].set_linewidth(ins_borders)
        
        if info_d.get('cmap') is not None:
            cmap = info_d['cmap']
            color_pal = cmap.decode("utf-8")
        else:
            color_pal = 'grasping'

        plot_projections(projected, axis=ins, x_idx=0, y_idx=1, circle_size=siz, arrow_size=siz,
                        colormap=cmaps[color_pal]) 
        ins.set_xticks([])
        ins.set_yticks([])
    ax.plot([0,0], [0,1], lw=1, ls='--', color='grey') 
    ax.plot([0,1], [0,0], lw=1, ls='--', color='grey') 
    ax.plot([0,1], [1,1], lw=1, ls='--', color='grey') 
    ax.plot([1,1], [0,1], lw=1, ls='--', color='grey') 
    ax.plot([0,1], [0,1], lw=1, ls='--', color='grey', alpha=0.4)

    ax.grid(alpha=0.3)
    circle2 = plt.Circle((0, 0), 1, color='grey',lw=1, ls='--', fill=False, alpha=0.4)
    ax.add_patch(circle2)

    ax.set_ylabel('Rotation axis', fontsize=10)
    ax.set_xlabel('Decay axis', fontsize=10)
    ax.set_facecolor('#fafafa')
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.tick_params(axis="y",direction="in", labelsize=7)
    ax.tick_params(axis="x",direction="in", labelsize=7)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    if save_name:
        plt.savefig(save_name, dpi=300, format='pdf',
                bbox_inches='tight', pad_inches=0.0)
    if SHOW:
        plt.show()
    plt.close()