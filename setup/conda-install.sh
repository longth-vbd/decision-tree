#!/bin/bash

WORD_DIR=$PWD

# install conda in home
ROOT=$HOME

# install conda in the local dir
#ROOT=$PWD

# check exist
if [ -d "$ROOT/miniconda3" ]; then
  echo "miniconda3 exists: $HOME/miniconda3/"
else
  mkdir .tmp
  cd .tmp
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -p $ROOT/miniconda3
fi

cd $WORD_DIR

echo "Done!"