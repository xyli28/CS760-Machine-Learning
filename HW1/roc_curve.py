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
    labels = features[-1][1]
    testing_labels = (testing['label'] == labels[0]).astype(int)
    num = testing_labels.groupby(testing_labels).size()   
    num_pos = num[1]
    num_neg = num[0]

    #Run classification.
    confidence = knn_learning(features, training, testing, k, 
                            learning_type = "regression")
    roc_instances = pd.DataFrame({"confidence":confidence, "label":
                                 testing_labels}).sort_values(by = 
                                 ['confidence'], ascending=False, 
                                 kind='mergesort')

    #Calculate and print roc curve. 
    true_pos = 0
    false_pos = 0
    last_true_pos = 0
    last_confidence = 1.0
    for index, row in roc_instances.iterrows():     
        if row['label'] == 1:
            true_pos += 1
            last_confidence = row['confidence']
        elif (row['confidence'] != last_confidence) and (true_pos > last_true_pos):
            print ("%s,%s" %(false_pos/num_neg, true_pos/num_pos))
            last_true_pos = true_pos
            false_pos += 1
            last_confidence = row['confidence']
        else:
            false_pos += 1
            last_confidence = row['confidence']
    print ("%s,%s" %(false_pos/num_neg, true_pos/num_pos)) 

if __name__ == "__main__":
    main()
