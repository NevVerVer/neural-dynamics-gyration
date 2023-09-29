![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![made with love](https://img.shields.io/badge/made%20with%20%E2%9D%A4%EF%B8%8F-8A2BE2)
# TODO: doi zinoda

# Unveiling Rotational Structure in Neural Data

A repository containing the code accompanying the research paper "Unveiling Rotational Structure in Neural Data" by Kuzmina E., Kriukov D., Lebedev M.

Published in ..., **TODO doi, etc** | [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.09.11.557230v1)

## Description

This repository contains code from a research paper focused on the analysis of neural activity data. Our work delves into the exploration and characterization of *rotational dynamic* prevalent in various neural datasets. We introduce a mathematical framework, informed by our research, designed to assess and quantify the "rotationess" of datasets. This framework leverages **Gyration Numbers**, a complex-valued metric derived from the eigenvalue decomposition of the differential covariance matrix of the data. The resulting **Gyration Plane** facilitates the comparison and simultaneous analysis of multiple datasets.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/NevVerVer/neural-dynamics-gyration.git
```

2. Navigate to the repository directory:
```bash
cd neural-dynamics-gyration
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```


## Datasets Downloading

1. Run script that downloads and unzip data: `bash prepare_data.sh` or `bash prepare_preproc_datasets.sh`. Or download archive with datasets manually from [here](https://drive.google.com/drive/folders/1AWO8XZpLBW1fkp5ylF6-w6J8gYcnnOkp?usp=sharing). You can also download datasets from the original source from here ... # TODO link to table.
`dataset.zip` contains raw data, as it is available at origin sourse. `preprocessed2h5.zip` contains already prerocessed datasets, saved to .h5 file as dictionary.
 
2. Add path to downloaded datasets to `datasets_config.py` file for convinience.

## Usage
### Data Preparation
Place your data in `./datasets/` directory or execture script `prepare_data.sh` to download data used in the study.
If you would like to work with datasets used in study, you can see how to pre-process and save datasets to `h5` file in notebook `datasets_analysis.ipynb`. If using other dataset, you can use parent class  `NeuralDataset` (from `utils/datasets.py`) and add specific methods to load data and pre-process.

### Content of Repository

- `datasets_analysis.ipynb` - contains tutorial on how to use special class created for working with datasets. It is easy modifiable and allows to use all visualization functions that were used in the study. Have code that was used to render **Fig. S2-S3-S4** and **Fig. 4** from our paper.

- `gyration_plane_tutorial.ipynb` - contains tutorial on how to use and interpret **Gyration Number** concept. How to plot datasets in **Gyration Plan** and compare them. Also, contains bonus explanation of **Curvature** concept.

- `traveling_wave_model.ipnyb` - Have code that was used to render **Fig. 3, 5** and **Fig. S5-S6-S7** from our paper.


### Gyration Plane

In short - **Gyration Plane** is an allocation of neural population dynamics datasets in the space spanned by real and imaginary parts of their first pair of complex-conjugated eigenvalues normalized by the spectrum power.

The normalized *imaginary part* measures the *structural rotation* component in the data and the normalized *real part* measures the *inflation/deflation* component of the data. 

This visualization approach provides a holistic view of the rotational dynamic problem allowing a comparison of different datasets with each other in the model-agnostic (no fitting as in jPCA) manner not requiring a dimensionality reduction procedure. 

![Gyration Plane Example](image/gif_for_README_80dpi.gif)


#### Shuffling Procedures
In `utils/shuffling.py` you can find implementation of *Shuffling Procedures* that are widely used for validation of "importance" of rotational dynamics in data.
We implemented three types of shufflings from original [Churchland et al.](https://doi.org/10.1038/nature11129), and CMPT shuffling from [Micaels et al.](https://doi.org/10.1371/journal.pcbi.1005175) Feel free to use them!


## Contributing
We welcome contributions to this repository. If you're found errors in code or experiments, please open an issue to discuss your ideas.

## License
This project is licensed under the XYZ License - see the LICENSE file for details.

## Acknowledgments/Citation
If you use the code or the findings from our paper, please cite:

Author A., Author B., Author C., (Year). Unveiling rotational structure in neural data. JournalName, volume(issue), pages.
**TODO add citation**
**TODO add bibtex**

Special thanks to @nosmokingsurfer for their valuable feedback and suggestions.
Thanks to @bantin for the [jPCA implementation](https://github.com/bantin/jPCA).

## Contact
For any questions or clarifications, please reach out to: *ekaterina.kuzmina@skoltech.ru*

## Datasets Used in Study
| Name & Authors                                                                                                                    | Year | Brain Areas                                                                                    | Behavior                                                                                              | Source                                                                                                                                                                                     | Original Paper                              |
|-----------------------------------------------------------------------------------------------------------------------------------|------|------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------|
| "Neural Population Dynamics During Reaching", Churchland et al.                                                                   | 2012 | primary motor cortex (M1), dorsal premotor cortex (PMd)                                        | straight hand reaches, curved hand reaches                                                            | ...                                                                                                                                                                                        | [link](https://doi.org/10.1038/nature11129) |
| "Local field potentials reflect cortical population dynamics in a region-specific and frequency-dependent manner", Gallego et al. | 2022 | primary motor cortex (M1), dorsal premotor cortex (PMd), primary somatosensory cortex (area 2) | delay centre-out reaching task using a manipulandum                                                   | [dryad](https://doi.org/10.5061/dryad.xd2547dkt)                                                                                                                                           | [link](https://doi.org/10.7554/eLife.73155) |
| "Neural Population Dynamics in Motor Cortex are Different for Reach and Grasp", Suresh et al.                                     | 2020 | primary motor cortex (M1), somatosensory cortex (SCx)                                          | isolated grasping task, center-out reaching task                                                      | [dryad](https://doi.org/10.5061/dryad.xsj3tx9cm)                                                                                                                                           | [link](https://doi.org/10.7554/eLife.58848) |
| "Rotational dynamics in motor cortex are consistent with a feedback controller", Kalidindi et al.                                 | 2021 | premotor and primary motor cortex (MC), primary somatosensory cortex (areas 1, 2, 3a, 5)       | delayed center-out reaching task, posture perturbation task                                           | [dryad](https://doi.org/10.5061/dryad.nk98sf7q7) [github](https://archive.softwareheritage.org/browse/revision/d61decd3cd750147ef098de1041326fd2be07ab2/?path=monkey_analysis/data_neural) | [link](https://doi.org/10.7554/eLife.67256) |
| "Context-dependent Computation by Recurrent Dynamics in Prefrontal Cortex", Mante et al.                                          | 2013 | prefrontal cortex (PFC): arcuate sulcus in and around the frontal eye field (FEF)              | context-dependent 2-alternative forced-choice visual discrimination task (perceptual decision-making) | [iniuzhch](https://www.ini.uzh.ch/en/research/groups/mante/data.html)                                                                                                                      | [link](https://doi.org/10.1038/nature12742) |