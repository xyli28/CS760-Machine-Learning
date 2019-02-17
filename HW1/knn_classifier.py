#!/usr/bin/env python3.6
from sys import argv
from json import load
import numpy as np
import pandas as pd
from knn import knn_learning

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

    #Run classification and print results 
    results  = knn_learning(features, training, testing, k, 
                            learning_type = "classification")
    for row in results: 
        print (','.join(map(str,row)))

if __name__ == "__main__":
    main()
