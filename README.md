# decision-tree

## Setup
1. conda
```bash
sh setup/conda-install.sh
. setup/conda-create.sh
. setup/conda-activate.sh
```

2. sklearn
```bash
pip install scikit-learn
```

3. graphviz
```bash
conda install python-graphviz
```

## iris
1. Run with sklearn
```bash
python iris/sklearn_iris.py
```

2. Run manually
- Download [iris data](https://archive.ics.uci.edu/ml/datasets/iris)
```bash
sh iris/load-data.sh
python iris/main.py
```