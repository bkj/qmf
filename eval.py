#!/usr/bin/env python

"""
    eval.py
"""

from __future__ import print_function, division

import sys
import argparse
import numpy as np
import pandas as pd
from scipy import sparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', type=str)
    parser.add_argument('--test-path', type=str)
    parser.add_argument('--user-path', type=str)
    parser.add_argument('--item-path', type=str)
    
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--no-filter-train', action='store_true')
    parser.add_argument('--no-normalize', action='store_true')
    
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()
    
    print('eval.py: Load user embeddings', file=sys.stderr)
    user = pd.read_csv(args.user_path, sep=' ', header=None)
    user_lookup = dict(zip(user[0].values, range(user.shape[0])))
    user = user[user.columns[1:]].values
    if not args.no_normalize:
        user /= np.sqrt((user ** 2).sum(axis=-1, keepdims=True))
    
    print('eval.py: Load item embeddings', file=sys.stderr)
    item = pd.read_csv(args.item_path, sep=' ', header=None)
    item_lookup = dict(zip(item[0].values, range(item.shape[0])))
    item = item[item.columns[1:]].values
    if not args.no_normalize:
        item /= np.sqrt((item ** 2).sum(axis=-1, keepdims=True))
    
    print('eval.py: Load train set', file=sys.stderr)
    train = pd.read_csv(args.train_path, sep=' ', header=None)[[0,1]]
    train[0] = train[0].apply(lambda x: user_lookup[x])
    train[1] = train[1].apply(lambda x: item_lookup[x])
    train = train.values
    train = sparse.csr_matrix((np.ones(train.shape[0]), (train[:,0], train[:,1])), shape=(user.shape[0], item.shape[0]))
    
    print('eval.py: Load test set', file=sys.stderr)
    test = pd.read_csv(args.test_path, sep=' ', header=None)[[0,1]]
    test[0] = test[0].apply(lambda x: user_lookup.get(x, -1))
    test[1] = test[1].apply(lambda x: item_lookup.get(x, -1))
    test = test.values
    test = test[(test >= 0).all(axis=-1)]
    
    test = sparse.csr_matrix((np.ones(test.shape[0]), (test[:,0], test[:,1])), shape=(user.shape[0], item.shape[0]))
    test = test.toarray()
    
    print('eval.py: Compute similarity', file=sys.stderr)
    sim = user.dot(item.T)
    if not args.no_filter_train:
        sim[np.nonzero(train)] = -1
    topk = sim.argsort(axis=-1)[:,-args.k:]
    
    print('eval.py: Compute p_at_%d' % args.k, file=sys.stderr)
    p = []
    for i, act in enumerate(test):
        act = np.where(act)[0]
        p.append(len(set(topk[i]).intersection(act)) / len(topk[i]))
        
    p_at_k = np.mean(p)
    print('p_at_%d = %f' % (args.k, p_at_k))