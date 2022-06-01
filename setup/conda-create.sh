#!/bin/sh

ROOT=$HOME
source $ROOT/miniconda3/bin/activate
conda create -n tree-py38-conda-env python=3.8

echo "Created conda environment: tree-py38-conda-env"