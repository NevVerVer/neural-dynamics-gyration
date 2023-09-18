from utils.color_palettes import (
    GR1, RB1, YP1, YS1, BG1)
from utils.datasets import (
    GraspingSuresh,
    ReachingGallego,
    ReachingChurchland,
    ReachingKalidindi,
    BehaviouralMante,
)

DATA_PATH = 'datasets/'
data_pathes = {
    'church': 'church_processed/church_dataset.h5',
    'grasp': 'grasping_processed/Dataset4_Grasping_fg.h5',
    'lfp': 'spanish_processed/Han_CO_20171207_LeftS1Area2.h5',
    'khaild': 'kalidindi_processed/Patch_A5.h5',
    'pfc': 'pfc_processed/A_color_movement_context_36cond.h5',
}

pfc_params = {
    'name': 'Mante et at.',
    'class': BehaviouralMante,
    'size': '[36 x 751 x 762]',
    'year': 2013,
    'brain_area': 'PFC',
    'arr_size': 0.22,
    'c_size': 0.12,
    'arr_size_pca': 0.25,
    'c_size_pca': 0.15,
    'ts': 0,
    'te': 300, 
    'cmap': BG1(),
    'bins': 40,
    'hist_space': 150,
    'khalid': False,
    'pfc': True,
    'fs': 1,
                 }
church_params = {
    'name': 'Churchland et at.',
    'class': ReachingChurchland,
    'size': '[108 x 61 x 218]',
    'year': 2012,
    'brain_area': 'M1, PMd',
    'arr_size': 0.13,
    'c_size': 0.07,
    'arr_size_pca': 0.10,
    'c_size_pca': 0.07,
    'ts': 0,
    'te': 21, 
    'cmap': GR1(),
    'bins': 40,
    'hist_space': 750,
    'khalid': False,
    'pfc': False,
    'fs': 10,  # ms
                 }
grasp_params = {
    'name': 'Suresh et al.',  
    'class': GraspingSuresh,
    'size': '[35 x 70 x 37]',
    'year': 2020,
    'brain_area': 'M1',
    'arr_size': 0.08,
    'c_size': 0.031,
    'arr_size_pca': 0.1,
    'c_size_pca': 0.04,
    'ts': 0,
    'te': 60, 
    'cmap': RB1(),
    'bins': 55,
    'hist_space': 250,
    'khalid': False,
    'pfc': False,
    'fs': 10,  # ms
                 }
lfp_params = {
    'name': 'Gallego et al.', 
    'class': ReachingGallego,
    'size': '[8 x 68 x 72]',
    'year': 2020,
    'brain_area': 'S1',
    'arr_size': 0.08,
    'c_size': 0.06,
    'arr_size_pca': 0.1,
    'c_size_pca': 0.08,
    'ts': 0,
    'te': 60, 
    'cmap': YP1(),
    'bins': 35,
    'hist_space': 50,
    'khalid': False,
    'pfc': False,
    'fs': 30,
                 }
khalid_params = {
    'name': 'Kalidindi et al.',
    'class': ReachingKalidindi,
    'size': '[8 x 30 x 70]',
    'year': 2021,
    'brain_area': 'A5',
    'arr_size': 0.1,
    'c_size': 0.07,
    'arr_size_pca': 0.1,
    'c_size_pca': 0.08,
    'ts': 0, 
    'te': 21,
    'cmap': YS1(),
    'bins': 40,
    'hist_space': 30,
    'khalid': True,
    'pfc': False,
    'fs': 10,  # ms
                 }

dataset_params = {
    'pfc_params': pfc_params,
    'church_params': church_params,
    'grasp_params': grasp_params,
    'lfp_params': lfp_params,
    'khalid_params': khalid_params,
}