#!/usr/bin/env python3.6
from sys import argv
from json import load
from math import floor
import numpy as np
import pandas as pd
from knn import knn_learning
from knn import cal_accuracy

def main():

    #Read parameters and load training and tesing data. 
    k = int(argv[1])
    training_data = {}
    testing_data = {}
    with open(argv[2],'r') as f:
        training_data = load(f)
    with open(argv[3],'r') as f:
        testing_data = load(f)
    features = training_data['metadata']['features']
    feature_names = [feature[0] for feature in features]
    training = pd.DataFrame.from_records(training_data['data'], 
                                         columns=feature_names)
    testing = pd.DataFrame.from_records(testing_data['data'], 
                                        columns=feature_names)
    testing_label = testing['label'].to_numpy()
    N = training.shape[0] 

    #Run classification with different subsets of training set and
    #calculate accuracy
    for i in range(10):
        n = floor((i+1)/10.0*N)
        prediction = knn_learning(features, training.head(n), testing, k, 
                                  learning_type = "classification")[:,-1]
        print ("%d,%s" %(n,str(cal_accuracy(prediction, testing_label))))

if __name__ == "__main__":
    main()
