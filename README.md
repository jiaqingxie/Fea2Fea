![](https://img.shields.io/badge/language-python-orange.svg)
![](https://img.shields.io/badge/license-MIT-000000.svg)
![](https://img.shields.io/badge/github-v0.2.1-519dd9.svg)
# Feature Augmentation on Small Graphs
torch based program for graph feature to feature predictions
this project is supervised by Rex Ying
please do not use them for writing results currently since it has not been under reviewed.

## Installation
1. please ensure that torch has been installed successfully in your computer, check with:
```bash
$ python -c "import torch; print(torch.__version__)"
>>> 1.8.0
```
2. please install torch_geometric with correct torch version and cuda version.
In our experiment environment, torch version is 1.8.0 and cuda version is 11.1
for both GPU(GTX 2060Super) and torch. (It's important since different versions will cause imcompatible situations)
```bash
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-geometric
```
3. please install networkx. this package is used for calculating graph features and generating synthetic graphs.
```bash
pip install networkx
```

## pipeline model
![Fea2Fea-simple and Fea2Fea-multiple](https://github.com/JIAQING-XIE/Fea2Fea/blob/main/output.png)


## Dataset
### 1. node datasets (Planetoid)

### 2. graph datasets (TUDataset)

### 3. Synthetic datasets

## Calculating graph features
Graph features have been calculated already, where you can find it in \Result folder.

## single feature to feature prediction
For single feature to single feature prediction, go to the src folder and run f_f_Citation.py if you
want to run experiments on Planetoid datasets. If you want to perform experiments on TUDatasets, then
please run f_f_TU.py
```bash
   python f_f_Citation.py 
   python f_f_TU.py
```
## hyperparameter tunning

```python
    python hyper_param.py --param
```

hyperparameter list: 

| Name     | `EXPERIMENT_NAME` | Description  |
|----------|:-------------------:|--------------|
| Synthetic #1 | `syn1`  | Random BA graph with House attachments.  |
| Synthetic #2 | `syn2`  | Random BA graph with community features. | 
| Synthetic #3 | `syn3`  | Random BA graph with grid attachments.  |
| Synthetic #4 | `syn4`  | Random Tree with cycle attachments. |
| Synthetic #5 | `syn5`  | Random Tree with grid attachments. | 
| Enron        | `enron` | Enron email dataset [source](https://www.cs.cmu.edu/~enron/). |
| PPI          | `ppi_essential` | Protein-Protein interaction dataset. |
| | | |
| Reddit*      | `REDDIT-BINARY`  | Reddit-Binary Graphs ([source](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)). |
| Mutagenicity*      | `Mutagenicity`  | Predicting the mutagenicity of molecules ([source](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)). |
| Tox 21*      | `Tox21_AHR`  | Predicting a compound's toxicity ([source](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)). |


## choosing optimal R and graph embedding method.



## collect all possible concatenation


## Run feature concatenation tests

## visualization 

## jupyter notebook playground
