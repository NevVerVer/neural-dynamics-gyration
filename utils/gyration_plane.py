"""
Created on Sat September 16 21:15:06 2023
Description: Controller class implementations for:
TODO: paper url here
@author: Ekaterina Kuzmina, ekaterina.kuzmina@skoltech.ru
@author: Dmitrii Kriukov, dmitrii.kriukov@skoltech.ru
"""

from matplotlib import pyplot as plt
import numpy as np
from utils.data_processing import load_h5_file
from utils.utils import subtract_cc_mean, soft_normalize
import jPCA
from jPCA.util import plot_projections
from utils.color_palettes import GR1, RB1, YP1, YS1, BG1
import os


cmaps = {'lfp': YP1(),
        'churchland': GR1(),
        'grasping': RB1(),
        'kalid': YS1(),
        'mante': BG1()}


def compute_lambdas(datas):
    """
    Return: lambda_n, lambda_t, lambda_c
    """
    #nc, nt, nn = datas.shape
    neuro_stack = np.concatenate(datas)
    time_stack = np.concatenate(datas.transpose(0, 2, 1))
    cond_stack = np.transpose(datas, (0, 2, 1)).reshape(datas.shape[0], -1)
    NC = np.cov(neuro_stack.T)
    TC = np.cov(time_stack.T)
    CC = np.cov(cond_stack)
    Neig = np.abs(np.sort(np.linalg.eigh(NC)[0])[::-1])
    Teig = np.abs(np.sort(np.linalg.eigh(TC)[0])[::-1])
    Ceig = np.abs(np.sort(np.linalg.eigh(CC)[0])[::-1])

    #compute differential covariance
    X = np.concatenate([x[:-1] for x in datas])
    X_dot = np.concatenate([np.diff(x, axis=0) for x in datas])

    Xeig = np.linalg.eigvals(X_dot.T @ X,)
    Xeig_real = np.abs(np.real(Xeig)[0])
    Xeig_imag = np.abs(np.imag(Xeig)[0])
    Xeig_sum = np.abs(Xeig.sum())

    return Neig[0] / Neig.sum(), Teig[0] / Teig.sum(), Ceig[0] / Ceig.sum(), (Xeig_real, Xeig_imag, Xeig_sum, Xeig)

def plot_dataset_gyration_plane(cond_n,
                                h5_save_dir, siz=0,
                                Q = 1., W = 1., ins_size=0.07, ins_borders = 0.3,
                                save_name=None): 
    d_names = os.listdir(h5_save_dir)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for dname in d_names:

        info_d = load_h5_file(h5_save_dir, dname)

        start_pad = info_d['start_pad']
        end_pad = info_d['end_pad']


        datas = info_d['data']
        cond_n = np.min([datas.shape[0], cond_n])
        datas = datas[:cond_n, :, :]

        go_cue = info_d['go_cue']
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
        _, _, _, tup = compute_lambdas(datas)

        xnom = np.abs(np.real(tup[3][0])) + np.abs(np.real(tup[3][1]))
        ynom = np.abs(np.imag(tup[3][0])) + np.abs(np.imag(tup[3][1]))
        x = xnom / (np.abs(tup[3]).sum()) * Q
        y = ynom / (np.abs(tup[3]).sum()) * W
    
        ins = ax.inset_axes([x - ins_size/2, y - ins_size/2, ins_size, ins_size], transform=ax.transData)
        ins.set_aspect('equal', adjustable='datalim')
        for axis in ['top', 'bottom', 'left', 'right']:
            ins.spines[axis].set_linewidth(ins_borders)
        color_pal = info_d['cmap'].decode("utf-8")

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
    plt.show()