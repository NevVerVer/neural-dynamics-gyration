"""
Created on Sat September 16 21:15:06 2023

Description: Additional utilities functions for
Paper url: https://www.biorxiv.org/content/10.1101/2023.09.11.557230v2

@author: Ekaterina Kuzmina, ekaterina.kuzmina@skoltech.ru
@author: Dmitrii Kriukov, dmitrii.kriukov@skoltech.ru
"""

import numpy as np
from sklearn.decomposition import PCA
from numpy.linalg import norm
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def soft_normalize(datas, thtrsh = 1e-2):
    fr_range = np.max(datas, axis=(0,1)) - np.min(datas, axis=(0,1))
    fr_range = np.where(fr_range > thtrsh, fr_range, 1.)
    datas /= fr_range
    return datas


def subtract_cc_mean(datas):
    return datas - np.mean(datas, axis=0)
    

def pca_rotation(datas, start, end, n_pca=6, sub_mean=True,
                 random_state=None):
    """
    PCA on stacked conditions data. Stacked data is a 2D array 
    that typically have size of [Time*Conditions x Neurons].
    
    Returns: Principal components, List of principal components splitted by 
    conditions, explained variance ratio, pca class instance.
    """
    if type(datas) != np.ndarray:
        datas = np.stack(datas)

    n_cond = datas.shape[0]
    cc_mean = np.mean(datas, axis=0, keepdims=False)
    fullstack = np.concatenate([datas[i, start:end, :] for i in range(n_cond)],)
    fullmean = np.concatenate([cc_mean[start:end, :]] * n_cond)
    
    centered = fullstack - fullmean if sub_mean else fullstack

    pca = PCA(n_pca, random_state=random_state)
    pca.fit(centered)
    Xp = pca.transform(centered)
    Xp_s = [Xp[i * (end - start):(i + 1) * (end - start)] for i in range(Xp.shape[0] // (end - start))] 

    return Xp, Xp_s, pca.explained_variance_ratio_, pca


def eliminate_difference(d, d2, inv_start):
    """
    removes a shelf in time series after swapping
    """
    from copy import deepcopy
    d_ = deepcopy(d)
    d2_ = deepcopy(d2)
    
    origin, d2_origin = d[inv_start], d2[inv_start]
    dif = np.abs(origin - d2_origin)
    if d2_origin > origin:
        d[inv_start:] = d2_[inv_start:] - dif
        d2[inv_start:] = d_[inv_start:] + dif
        
    elif d2_origin < origin:
        d[inv_start:] = d2_[inv_start:] + dif
        d2[inv_start:] = d_[inv_start:] - dif
        
    else:
        d[inv_start:] = d2_[inv_start:]
        d2[inv_start:] = d_[inv_start:]
    return d, d2


def interp_timepoints(data, scale):
    """
    Input:
        data - numpy 3D array [cond x time x neurons]
    Output
        interp_data - numpy 3D array [cond x time x neurons]
    """
    from scipy import interpolate
    
    t_end = data.shape[1]
    t = np.arange(0, t_end)
    f = interpolate.interp1d(t, data, kind='cubic', axis=1)
    
    step = (t_end - 1) / (t_end * scale)
    t_new = np.arange(0, t_end-1, step)
     # use interpolation function returned by `interp1d`
    interp_data = f(t_new)  

    return interp_data, step


# we parametrize data with k = bi under known b parameter
def generate_response(t_max=600, N=100, a=0., b=1., steps=61, 
                        phase_noise=0, phases=None, 
                        sigma=1800, sigma_noise=0.,
                        amp_noise=0, amp_mean=1.
                        ):
    """
    Generate neuron sequence-like firing with Gaussians.
    Input:
    sigma: width of a wave
    N:     number of neurons
    a:     wave start point
    b:     wave speed
    steps: number of time points for a generated wave
    phase_noise: std of noise level of a wave phase
    phases: manually given phases (initial positions) of waves
    sigma_noise: std of wave widths
    amp_noise: std of noise level for a wave amplitude 
    amp_mean: mean of wave amplitudes

    Returns: 
    np.ndarray of size [num_of_neurons, time_steps]
    """
    t = np.linspace(0, t_max, steps)
    if phases is None:
        i = np.arange(N)
        t_i = (b*i + a)[:, None]
    else:
        t_i = phases[:, None]
        
    ts = np.tile(t, (t_i.shape[0], 1))
    eps_a = (np.random.randn(N) * amp_noise)[:, None]
    eps_p = (np.random.randn(N) * phase_noise)[:, None]
    eps_s = (np.random.uniform(0, sigma_noise, N))[:, None]
    x = (amp_mean + eps_a) * np.exp(-(ts - t_i + eps_p)**2 / (sigma + eps_s)**2)
    
    return x, t, t_i


def compute_curvature(X, clip=False):
    vel = np.array([np.gradient(X[:, i]) for i in range(X.shape[1])]).T
    acc = np.array([np.gradient(vel[:, i]) for i in range(vel.shape[1])]).T
    speed = norm(vel, axis=1)
    cross = np.sqrt((norm(vel, axis=1) * norm(acc, axis=1))**2 - (np.sum(acc * vel, axis=1))**2)
    curvature_val = cross / (speed**(3))
    if clip:
        curvature_val = np.clip(curvature_val, -1000., 1000.)
    return curvature_val


def compute_lambdas(datas):
    """
    Input: datas :: C x T x N ndarray
    Return: eigenvalues of diff cov matrix
    """
    #compute differential covariance
    X = np.concatenate([x[:-1] for x in datas])
    X_dot = np.concatenate([np.diff(x, axis=0) for x in datas])
    Xeig = np.linalg.eigvals(X_dot.T @ X,)
    return Xeig

def compute_gyration(datas):
    Xeig = compute_lambdas(datas)
    xnom = np.abs(np.real(Xeig[0])) + np.abs(np.real(Xeig[1]))
    ynom = np.abs(np.imag(Xeig[0])) + np.abs(np.imag(Xeig[1]))
    x = xnom / (np.abs(Xeig).sum())
    y = ynom / (np.abs(Xeig).sum())
    return x, y


def func(t, mu, sigma, amp):
    return amp * np.exp(-((t - mu) / sigma)**2)


def fit_running_wave(datas, times):
    datas_cap = np.zeros_like(datas)
    r2_list_across_conds = []
    param_list_across_conds = []
    for c, cond in tqdm(enumerate(datas)):
        r2_list = []
        param_list = []
        t = times
        for i in range(cond.shape[1]):
            y = cond[:, i]
            popt, pcov = curve_fit(func, t, y, bounds=([-100, 10., 0.], [600, 5000, 1.5]))
            y_cap = func(t, *popt)
            datas_cap[c, :, i] = y_cap
            r2 = round(pearsonr(y, y_cap)[0]**2, 3)
            r2_list.append(r2)
            param_list.append(popt)
        r2_list_across_conds.append(r2_list)
        param_list_across_conds.append(param_list)

    return datas_cap, np.asarray(r2_list_across_conds), np.asarray(param_list_across_conds)


def eigenPCA(datas_list, npca=4, plot=True):
    if type(datas_list) == list:
        datas = np.stack(datas_list, axis=0)
    else:
        datas = datas_list.copy()
    variances = []
    for i in range(0, datas.shape[0]):
        cond = datas[i]
        pca = PCA(npca)
        Xp = pca.fit_transform(cond)
        variances.append(pca.explained_variance_ratio_.sum())
        if plot:
            plt.plot(Xp[:, 0], Xp[:, 1])
    return Xp

def compress_datas(datas_list, npca=4, plot=True):
    if type(datas_list) == list:
        datas = np.stack(datas_list, axis=0)
    else:
        datas = datas_list.copy()
    # datas = np.stack(datas, axis=0)
    num_conditions, num_time_bins, num_units = datas.shape
    pca = PCA(npca)
    datas = pca.fit_transform(datas)
    datas = datas.reshape(num_conditions, num_time_bins, npca)
    data_list = [x for x in datas]
    return data_list