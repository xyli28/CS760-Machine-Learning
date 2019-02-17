#!/usr/bin/env python3.6
from sys import argv
from json import load
import numpy as np
import pandas as pd
from knn import knn_learning
from knn import cal_accuracy

def main():

    #Read parameters and load training, validation and tesing data. 
    k = int(argv[1])
    training_data = {}
    val_data = {} 
    testing_data = {}
    with open(argv[2],'r') as f:
        training_data = load(f)
    with open(argv[3],'r') as f:
        val_data = load(f)
    with open(argv[4],'r') as f:
        testing_data = load(f)
    features = training_data['metadata']['features']
    feature_names = [feature[0] for feature in features]
    training = pd.DataFrame.from_records(training_data['data'], 
                                         columns=feature_names)
    val = pd.DataFrame.from_records(val_data['data'], 
                                    columns=feature_names)
    testing = pd.DataFrame.from_records(testing_data['data'], 
                                        columns=feature_names)
    val_label = val['label'].to_numpy()
    testing_label = testing['label'].to_numpy()

    #Run knn classification for validation set for each k
    accuracy = [0 for x in range(k)]
    for i in range(k):
        prediction = knn_learning(features, training, val, i+1, 
                                  learning_type = "classification")[:,-1]
        accuracy[i] = cal_accuracy(prediction, val_label)
        print ("%d,%s" %(i+1,str(accuracy[i])))

    #Pick optimal k and calculate accuracy for test set
    opt_k = np.asarray(accuracy).argmax()+1
    print (opt_k)
    prediction = knn_learning(features, pd.concat([training, val]), testing, 
                              opt_k, learning_type = "classification")[:,-1]
    print (cal_accuracy(prediction, testing_label))

if __name__ == "__main__":
    main()
