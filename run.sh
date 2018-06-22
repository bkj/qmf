#!/bin/bash

# run.sh

# ---------------------------------
# Run on MovieLens datasets

mkdir -p {_data,_results}/movielens

wget --header "Authorization:$TOKEN" https://hiveprogram.com/data/_v1/qmf/movielens/ml-1m-ratings-train.gz
wget --header "Authorization:$TOKEN" https://hiveprogram.com/data/_v1/qmf/movielens/ml-1m-ratings-test.gz

gunzip ml-*.gz && mv ml-* _data/movielens

./bin/wals \
    --train_dataset=_data/movielens/ml-1m-ratings-train \
    --test_dataset=_data/movielens/ml-1m-ratings-test \
    --user_factors=_results/movielens/ml-1m-user \
    --item_factors=_results/movielens/ml-1m-item \
    --nepochs=20 \
    --nfactors=100 \
    --nthreads=32 

# Evaluate quality of predictions (we're not interested in runtime of this)
# python eval.py \
#     --train-path _data/movielens/ml-1m-ratings-train \
#     --test-path _data/movielens/ml-1m-ratings-test \
#     --user-path _results/movielens/ml-1m-user \
#     --item-path _results/movielens/ml-1m-item

# Run on `ml-100k-ratings` or `ml-20m-ratings` for smaller or larger problem sizes

# ---------------------------------
# Run on Netflix datasets

mkdir -p {_data,_results}/netflix

wget --header "Authorization:$TOKEN" https://hiveprogram.com/data/_v1/qmf/netflix/netflix.tsv.gz
gunzip netflix.tsv.gz && mv netflix.tsv _data/netflix

./bin/wals \
    --train_dataset=_data/netflix/netflix.tsv \
    --user_factors=_results/movielens/ml-1m-user \
    --item_factors=_results/movielens/ml-1m-item \
    --nthreads=32 