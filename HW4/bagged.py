#!/usr/bin/env python3.6
from sys import argv
from json import load
import numpy as np
import pandas as pd
import DecisionTree as dt

def bagging_learning(sample_indices, meta, train_data, test_data, max_d, prob):
    resamples = train_data[sample_indices]
    tree = dt.DecisionTree()
    tree.fit(resamples[:,:-1], resamples[:,-1],meta, max_d)
    return tree.predict(test_data[:,:-1],prob=prob)

def main(argv):

    #read parameters num of trees and max depth of decision tree
    n_trees = int(argv[1])
    max_d = int(argv[2])

    #get training and test data
    train = load(open(argv[3],'r'))
    meta = train['metadata']['features']
    train_data = np.array(train['data'])
    n_train = train_data.shape[0]   

    #get test data 
    test = load(open(argv[4],'r'))
    test_data = np.array(test['data'])
    n_test = test_data.shape[0]

    #generate a n_train*n_trees table of indices of bootstrapped samples
    table = np.random.choice(n_train,(n_trees,n_train), True)
   
    #build and train an ensemble of decision trees
    predictions = np.apply_along_axis(bagging_learning, 1, table,
        meta, train_data, test_data, max_d, False).reshape((n_trees,-1))
    probs = np.sum(np.apply_along_axis(bagging_learning, 1, table,
        meta, train_data, test_data, max_d, True), axis=0)
    ens_prediction = np.apply_along_axis(
        lambda row:meta[-1][-1][np.argmax(row)], 1, probs) 
    test_Y = test_data[:,-1]
    return (meta[-1][1],ens_prediction,test_Y)

if __name__ == "__main__":
    np.random.seed(0)
    main(argv)

