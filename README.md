# Unveiling Rotational Structure in Neural Data

A repository containing the code accompanying the research paper "Unveiling Rotational Structure in Neural Data" by Kuzmina E., Kriukov D., Lebedev M.

Published in ..., **TODO doi, etc**

## Description

**TODO write something short but explaining what, why and how of paper**
This repository hosts the code and relevant scripts used in the paper which explores rotational structures in neural datasets. By leveraging XYZ algorithm, we unveil patterns in the data that provide insights into ABC (some short specifics about the paper's content).

## Table of Contents??

## Badges ??

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
pip install -r requirements.txt

## Usage
### Data Preparation
Place your data in `./datasets/` directory or execture script `smth.py` to download data used in study.
If you use datasets used in study, you can see how to pre-process and save datasets to `h5` file in notebook `datasets_analysis.ipynb`. If using other dataset, you can use parent class  `NeuralDataset` (from `utils/datasets.py`) and add specific methods to load data and pre-process.

### Main Experiments
python main_experiment.py
Note: Additional configurations and parameters are provided in config.yml.
**TODO** Notebooks description
**TODO** Datasets used in this study (table with links)

This is `inline code`.

```python
def hello_world():
print("Hello, World!")
```

## Contributing
We welcome contributions to this repository. If you're interested in enhancing the code or adding new features, please submit a pull request. For major changes, please open an issue first to discuss your ideas.

## License
This project is licensed under the XYZ License - see the LICENSE file for details.

## Acknowledgments/Citation
If you use the code or the findings from our paper, please cite:

Author A., Author B., Author C., (Year). Unveiling rotational structure in neural data. JournalName, volume(issue), pages.

**TODO add citation**
**TODO add bibtex**

Special thanks to XYZ for their valuable feedback and suggestions.

## Contact
For any questions or clarifications, please reach out to:

---------------------
---------------------
---------------------

### НОУТБУКИ
1. Shuffling_Procedure - mostly contains shuffling procedures from original article. Also the version of 3rd shuffling which do not change variance of data. Also some exploratory analysis of how jPCA behaves if we change data a little bit. Make Rotation without jPCA. Checked how constant change PCA. 
2. pca_rotation - playground to create rotation plane with PCA, without jPCA
3. dev - pca_rotation and shuffle_data functions creation
4. CN1_fPCA - fPCA only
5. CN1 - tries to render nice runnig wave plot; тыкание в jPCA; генерация данных динамической модель

I propose to do beautiful README here, then just move it to repo to be published

