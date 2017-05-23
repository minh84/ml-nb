#!/bin/bash
set -e

if [ "$#" -ne 1 ]; then
    echo "You musht specify output hugo dir"
    exit 1
fi

INDIR=$(pwd)
OUTDIR=$1

deploy_one_nb() {
    cmd="python nb2hugo.py --input_nb $INDIR/$1 --hugo_dir $OUTDIR --outfile content/$2"
    echo "====================================="
    printf "Running cmd=%s\n" "$cmd"
    echo "-------------------------------------"
    eval $cmd
}

deploy_one_nb "machine_learning/index.ipynb" "blog/machine_learning.md"
deploy_one_nb "machine_learning/supervised/linear_regression_part01.ipynb" "blog/machine_learning/supervised/linear_regression_part01.md"
deploy_one_nb "machine_learning/supervised/linear_regression_part02.ipynb" "blog/machine_learning/supervised/linear_regression_part02.md"
deploy_one_nb "machine_learning/supervised/binary_classification_logistic.ipynb" "blog/machine_learning/supervised/binary_classification_logistic.md"

deploy_one_nb "machine_learning/unsupervised/rbm.ipynb" "blog/machine_learning/unsupervised/rbm.md"
