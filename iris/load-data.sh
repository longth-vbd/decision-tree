#!/bin/bash

ROOT=$PWD
DATA_DIR="data"
mkdir -p $DATA_DIR
cd $DATA_DIR

wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

cd $ROOT
