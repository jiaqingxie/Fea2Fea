![](https://img.shields.io/badge/language-python-orange.svg)
![](https://img.shields.io/badge/license-MIT-000000.svg)
![](https://img.shields.io/badge/github-v1.0.0-519dd9.svg)
# Exploring Structural Feature Correlations via Graph Neural Networks (version 1.0.0)
function: give correlations between each two structural features, or between one and leftover features.

Brief: torch based program for graph feature to feature predictions.

This project is supervised by Zhitao (Rex) Ying from Stanford. Paper has been admitted by ECML PKDD Graph Embedding and Mining(GEM) workshop. Paper will be published in the proceedings of ECML PKDD workshop sooner or later.
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


## Dataset(Initial)
### 1. node classification datasets: 
[Cora](https://github.com/JIAQING-XIE/Fea2Fea/blob/main/data/Cora)
[CiteSeer](https://github.com/JIAQING-XIE/Fea2Fea/blob/main/data/Citeseer)
[PubMed](https://github.com/JIAQING-XIE/Fea2Fea/blob/main/data/PubMed)
### 2. graph classifcation datasets: 
[ENZYMES](https://github.com/JIAQING-XIE/Fea2Fea/blob/main/data/ENZYMES)
[PROTEINS](https://github.com/JIAQING-XIE/Fea2Fea/blob/main/data/PROTEINS)
[NCI1](https://github.com/JIAQING-XIE/Fea2Fea/blob/main/data/NCI1)

### 3. Synthetic datasets
Geometric synthetic datasets are generated by networkx.

<p float="left">
  <img src="https://github.com/JIAQING-XIE/Fea2Fea/blob/main/generate_syn_data/graph_200.png" width="200" />
  <img src="https://github.com/JIAQING-XIE/Fea2Fea/blob/main/generate_syn_data/graph_400.png" width="200" /> 
  <img src="https://github.com/JIAQING-XIE/Fea2Fea/blob/main/generate_syn_data/graph_800.png" width="200" />
</p>

# Procedures
## 1. Generating graph structural features
Graph features have been calculated already, where you can find it in \Result folder. {k}_property.txt, where k is the name of each dataset. Please make sure that `networkx` is installed.
If you want to generate them by youself, run:
```bash
python generate_property_planetoid.py ## for planetoid dataset
python generate_proprty_tuda.py ## for tudataset
```
under property_process folder. Functional file is graph_property.py, where you can add more structural features to the given list.

## 2. Single feature to feature prediction (Fea2Fea-single)
For single feature to single feature prediction, go to the src folder and run f_f_Citation.py if you
want to run experiments on Planetoid datasets. If you want to perform experiments on TUDatasets, then
please run f_f_TU.py
```bash
   python f_f_Citation.py 
   python f_f_TU.py
```
You need to correct the path at the moment before it's been corrected to the right relative path. Results are saved in Results/{dataset}.

## 3. Multiple features to single feature prediction (Fea2Fea-multiple)
What's different from Fea2Fea-single is that predicted objective should be set in advance in the parameter list which should be passed to the command line.
For example: run
```bash
   python aug_Citation.py --dataset=Cora --aim_feature = 2 ## prediction on Cora dataset and predict clustering coefficient
   python aug_TU.py --dataset=ENZYMES --aim_feature = 4 ## prediction on ENZYMES dataset and predict average path length (shortest)
```
The predicted objective will be filtered by function `all_possible_concatenation` firstly, generating the all possible concatenation group in the array `ans`. After that, it will go through graph neural network models to reach prediction accuracy for each element in the array and record average acc. The error bar-plot will be shown on the screen (for each dimension).

## 4. Real-world applications (important)
After analyzing the potential irredundant feature groups, we are going to concatenate them with initial graph features to make node or graph classifications. 
run:
```bash
   python citation_realworld.py --dataset=Cora --graphconv=GIN --o.concat_method=SimpleConcat
   ## prediction on Cora dataset, using GIN model in embedding layer and just simple concatenate each augmented structural feature.
```
Finally, you will reach acc for simple/bilinear/NTN + number of input features. This is a small trick where we only randomly choose from the array. You can choose the best one among all results. You can add your choice in the file, set the ans[0] to what you'd want the input feature idxes be.

## 5. Hyper-parameter tests
In this part, we introduce the hyper-parameter tests.
```bash
    python hyper_param.py [option list]
```

hyperparameter list: 

| Hyperparameter    | Type | Default Value| Description
|----------|:-------------------:|--------------|------------|
|input_feature | int| 0 | input feature index
|aim_feature|int| 1 | output feature index
|task | str |  `node`| node or graph dataset
|dataset | str | `Cora` | dataset name 
|hyperparameter | str | `binning` | binning or depth or threshold tunning
|min_bins | int | 2 | minimum number of bins
|max_bins | int | 6 | maximum number of bins
|min_depth| int | 2 | minimum depth of GNN architecture
|max_depth| int | 6 | maximum depth of GNN architecture
|hidden_dim | int | 2 | hidden dimension
|batchnorm | bool | 0 | if BatchNorm
|embedding | str | `GIN` | graph embedding method
|threshold | float | 0.8 | threshold for filtering irredundant features



## visualization 
### input feature distribution



### comparing concatenation methods

### embedding visualization

## jupyter notebook playground (for reference)

Most importantly, in the future, we are going to add more features and more graph neural network models to ensure the model's robustness.

## Citation
If you use to cite `Fea2Fea` in your research paper, please consider citing:
```
@misc{xie2021fea2fea,
      title={Fea2Fea: Exploring Structural Feature Correlations via Graph Neural Networks}, 
      author={Jiaqing Xie and Rex Ying},
      year={2021},
      eprint={2106.13061},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
Currently its not been published by Springer, but will be in a few months.