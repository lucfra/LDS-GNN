# LDS-GNN

This is the accompany python package for 
the ICML 2019 paper [_Learning Discrete Structures for Graph Neural Networks_](https://arxiv.org/abs/1903.11960)

It implements the method __LDS__ and its variant __KNN-LDS__ and reproduces experiments reported in the 
paper.

![alt text](https://github.com/lucfra/LDS-GNN/blob/master/illustration%20method.PNG 
"A cartoon of our approach for learning graph structures for graph neural networks")

### Requirements

The code is written Python 3.6 and TensorFlow version 1 (tested on versions 1.12 and 1.16). 
It requires scikit-learn >= 0.21.2 and the python 
packages 
- `FAR-HO`, available [here](https://github.com/lucfra/FAR-HO) (advised branch: final_ICML2019)
- `GCN`, available [here](https://github.com/tkipf/gcn) 


##### Datasets

UCI datasets should be loaded automatically, while graph-based datasets (Cora and Citeseer) should be downloaded and
included in the `lds/data` folder.
FMA dataset should also be downloaded, please email the authors if interested.

### Installation (optional)

```
python setup.py install
```

The scripts contained in `lds.py` should work also without installing the package. 

## Run

Navigate to `lds_gnn` folder.

The main script is in the file `lds.py`. The options are
```
-d: the evaluation dataset. Available datasets are iris, wine, breast_cancer, digits, 20newstrain, 
            20news10, cora, citeseer, fma. Default breast_cancer
-m: the method: lds or knnlds. Default knnlds
-s: the random seed. Default 1
-e: the percentage of missing edges (valid only for cora and citeseer dataset). Default 50
```

For experiments with incomplete graphs on Cora and Citeseer, run 
```
python lds.py -m lds -e {an integer between 0 and 100} -d {cora or citeseer} -s {if you want to specify random seed}
```

For experiments in semi-supervised learning (with no input graph), run
```
python lds.py -m knnlds -d {any available dataset} -s {if you want to specify random seed}
```

The code will run a small grid search to select some method's parameters such as the 
(outer) optimization learning rate and the number of truncation steps to compute the hypergradeient. It
will output the test accuracy of the best found model, according to the ''early stopping accuracy''. 
It will also create one log file per single experiment in the folder `lds/results`,
 which can be successively loaded (e.g. in a notebook) with the function `lds.load_results()` 
 for inspection and visualization .
 
 Please note that the package does not include implementations of baseline methods.
 
 ## Licence 
 
 Please take a look at LICENCE.txt
 
 ## Cite
 
 If you use this package, please cite 

 ```latex
@InProceedings{franceschi2019learning,
  title = 	 {Learning Discrete Structures for Graph Neural Networks},
  author = 	 {Luca Franceschi and Mathias Niepert and Massimiliano Pontil and Xiao He},
  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
  year = 	 {2019}
}
```
