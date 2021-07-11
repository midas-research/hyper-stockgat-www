# Exploring the Scale-Free Nature of Stock Markets: Hyperbolic Graph Learning for Algorithmic Trading

This codebase contains the python scripts for HyperStock-GAT, the model for the WWW 2021 paper [link](https://dl.acm.org/doi/10.1145/3442381.3450095).

## Environment & Installation Steps
Python 3.6, Pytorch, Pytorch-Geometric and networkx.


```python
pip install -r requirements.txt
```

## Dataset and Preprocessing 

Download the dataset and follow preprocessing steps from [here](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking). 


## Run

Execute the following python command to train STHAN-SR: 
```python
python train.py -l 8 -a 2 --task nc --dataset pubmed --model HGCN --lr 0.0004 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.4 --weight-decay 0.0001 --manifold PoincareBall --log-freq 5 --cuda 0 --use-att 1
```

## Cite
Consider citing our work if you use our codebase

```c
@inproceedings{10.1145/3442381.3450095,
author = {Sawhney, Ramit and Agarwal, Shivam and Wadhwa, Arnav and Shah, Rajiv},
title = {Exploring the Scale-Free Nature of Stock Markets: Hyperbolic Graph Learning for Algorithmic Trading},
year = {2021},
isbn = {9781450383127},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3442381.3450095},
doi = {10.1145/3442381.3450095},
booktitle = {Proceedings of the Web Conference 2021},
pages = {11–22},
numpages = {12},
keywords = {graph neural network, hyperbolic learning, finance, stock market},
location = {Ljubljana, Slovenia},
series = {WWW '21}
}
}
```

