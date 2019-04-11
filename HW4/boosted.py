#!/usr/bin/env python3.6
from sys import argv
from json import load
import numpy as np
import pandas as pd
import DecisionTree as dt
def combine_predict(predictions, alphas):
    hashMap = {}
    for i,label in enumerate(list(predictions)):
        if label in hashMap:
            hashMap[label] += alphas[i]
        else:
            hashMap[label] = alphas[i]
    return max(hashMap, key=hashMap.get)


def main(argv):

    #read parameters num of trees and max depth of decision tree
    n_trees = int(argv[1])
    max_d = int(argv[2])

    #get training and test data
    train = load(open(argv[3],'r'))
    meta = train['metadata']['features']
    train_data = np.array(train['data'])
    n_train = train_data.shape[0]   
    K = len(meta[-1][1])

    #get test data 
    test = load(open(argv[4],'r'))
    test_data = np.array(test['data'])
    n_test = test_data.shape[0]

    #initial weight
    w = np.full((n_train),1.0/n_train)
    predictions = []
    weights = [] 
    alphas = []

    #training decision tree by adaboost
    epsilon = 0
    for i in range(n_trees):
        tree = dt.DecisionTree()
        tree.fit(train_data[:,:-1], train_data[:,-1], meta, max_d,
                 instance_weights = w) 
        train_result = tree.predict(train_data[:,:-1], prob=False)
        test_result = tree.predict(test_data[:,:-1], prob=False)
        match = (train_result == train_data[:,-1]).astype(int)
        err = np.sum(w*(1-match))/np.sum(w)
        if (err >= 1-1.0/K):
            break
        weights.append(w)
        predictions.append(test_result)
        alpha = np.log((1-err)/err) + np.log(K-1)
        alphas.append(alpha)   
        w = w*np.exp(alpha*(1-match))
        w = w/np.sum(w)

    predictions = np.asarray(predictions).T
    alphas = np.asarray(alphas)
    weights = np.asarray(weights).T
 
    #calculate ensemble prediction and accuracy
    ens_prediction = np.apply_along_axis(combine_predict, 1, 
                                         predictions, alphas)
    test_Y = test_data[:,-1]
    return (meta[-1][1],ens_prediction,test_Y)

if __name__ == "__main__":
    np.random.seed(0)
    main(argv)

