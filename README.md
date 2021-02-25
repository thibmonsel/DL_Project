# DL_Project


Our Deep Learning Project uses the ogbg-molhiv dataset from the OGB platform found [here](https://ogb.stanford.edu/docs/graphprop/). It explores and test out graph neural network architectures.


## Environment setup
The main three package used are [OGB](https://github.com/snap-stanford/ogb), [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) and [Pytorch] (https://pytorch.org).

To create a workable virtual environment please run :
```
conda env create --name envname --file=environments.yml
```

## Dataset description 

The ogbg-molhiv dataset has the following characteritics
| #Graphs | #Nodes per graph | #Node features | #Edges per graph | #Edge features | #Positive label |
|---------|------------------|----------------|------------------|----------------|-----------------|
| 41,127  | 25.5             | 9              | 27.5             | 3              | 3.5%            |3              | 3.5%            |

## Repository structure

```
├── README.md
├── environment.yml
├── main.py
├── params.py
└── src
    ├── gnn.py
    ├── rgcn.py
    └── rgcn_gnn_inception.py
```