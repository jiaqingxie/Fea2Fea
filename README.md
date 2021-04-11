![](https://img.shields.io/badge/language-python-orange.svg)
![](https://img.shields.io/badge/license-MIT-000000.svg)
![](https://img.shields.io/badge/github-v0.1.0-519dd9.svg)
# Feature Augmentation on Small Graphs

## author: Jiaqing Xie, Rex Ying

### paper citation



## Installation

## Dependencies

## pipeline


## Dataset
### 1. node datasets (Planetoid)

### 2. graph datasets (TUDataset)

### 3.

## single feature to feature prediction

```python
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