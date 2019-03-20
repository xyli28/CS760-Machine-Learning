#!/usr/bin/env python3.6
from sys import argv
from json import load
import numpy as np
import pandas as pd
import time

def standardization(train_num, test_num):
    ave = train_num.mean(axis=0)
    stddev = train_num.std(axis=0, ddof = 0)
    stddev[stddev == 0.0] = 1.0
    train_num = (train_num-ave)/stddev
    test_num = (test_num-ave)/stddev
    return train_num, test_num

def oneHotEncoding(instance ,features, ctg_feature):
    value = []
    label = []
    for feature in ctg_feature:
        ctg = [(feature,c) for c in features[feature] ]
        label += ctg
        value += [int(instance[feature] == c)for c in features[feature]]
    return pd.Series(data = value, index = label)

def calActivation(instance, weight):
    return 1/(1+np.exp(-np.matmul(instance, weight))) 

def crossEntropy(activation, actual):
    return -np.sum(actual*np.log(activation)+(1-actual)*np.log(1-activation))

def calGradient(activation, actual, instance):
    return ((activation-actual)*instance).reshape(-1,1)

def main(argv):
    
    #read learning rate and num of epochs
    learning_rate = float(argv[1])
    num_epochs = int(argv[2]) 

    #Load training and testing data 
    train_data = {}
    test_data = {}
    with open(argv[3],'r') as f:
        train_data = load(f)
    with open(argv[4],'r') as f:
        test_data = load(f)
    features_data = train_data['metadata']['features']
    features = {}
    for feature in features_data:
        features[feature[0]] = feature[1]
    #label = features_data[-1] 
    train = pd.DataFrame.from_records(train_data['data'], 
                                      columns=features.keys())
    test = pd.DataFrame.from_records(test_data['data'], 
                                     columns=features.keys())
     
    #Classify numeric features and categorical features
    num_feature = []
    ctg_feature = []
    for feature, attr in features.items():
        if feature != 'class':
            if attr == 'numeric':
                num_feature.append(feature)
            else:
                ctg_feature.append(feature)
 
    #Standardize numeric features
    train_num = train[num_feature]
    test_num = test[num_feature]
    train_num, test_num = standardization(train_num, test_num)

    #One-hot encode the categorical features
    train_ctg = train[ctg_feature]
    test_ctg = test[ctg_feature]
    train_ctg = train_ctg.apply(oneHotEncoding, axis=1, 
                args = (features, ctg_feature)) 
    test_ctg = test_ctg.apply(oneHotEncoding, axis=1, 
                args = (features, ctg_feature))
    
    #Combine numeric features and catogorical features, add bias to features
    #Reorder the feature column
    feature_index = []
    for feature, attr in features.items():
        if feature != 'class':
            if attr == 'numeric':
                feature_index.append(feature)
            else:
                ctg_index = [(feature, x) for x in attr]
                feature_index += ctg_index
    train_feature = pd.concat([train_num, train_ctg],axis = 1)[feature_index].to_numpy()
    test_feature = pd.concat([test_num, test_ctg],axis = 1)[feature_index].to_numpy() 
    
    bias = np.ones(train_feature.shape[0]).reshape(-1,1)
    train_feature = np.concatenate([bias, train_feature], axis = 1)
    bias = np.ones(test_feature.shape[0]).reshape(-1,1)
    test_feature = np.concatenate([bias, test_feature], axis = 1)

    #Calculate the corresponding numerical values for training/test classification
    train_class = train['class'].apply(lambda x: 
                  int(x != features['class'][0])).to_numpy().reshape(-1,1)
    test_class = test['class'].apply(lambda x: 
                 int(x != features['class'][0])).to_numpy().reshape(-1,1)

    #Initialize the weight
    w = np.random.uniform(low=-0.01, high=0.01, size=(train_feature.shape[1],1))

    #Training 
    activation = np.zeros((train_feature.shape[0],1))
    for i in range(num_epochs):
       for j in range(train_feature.shape[0]): 
           activation[j] = calActivation(train_feature[j], w)
           grad = calGradient(activation[j], train_class[j], train_feature[j])
           w -= learning_rate*grad
       cross_entropy = crossEntropy(activation, train_class)
       prediction = np.round(activation)
       num_corr = np.sum(np.equal(prediction, train_class).astype(int))  
       num_incorr = train_class.shape[0] - num_corr       
       print ("%d %.12f %d %d" %(i+1, cross_entropy, num_corr, num_incorr)) 

    #Testing
    activation = calActivation(test_feature,w) 
    cross_entropy = crossEntropy(activation, test_class)
    prediction = np.round(activation)
    for i in range(test_feature.shape[0]):
        print ("%.12f %d %d" %(activation[i], prediction[i], test_class[i]))
    num_corr = np.sum(np.equal(prediction, test_class).astype(int))  
    num_incorr = test_class.shape[0] - num_corr       
    print ("%d %d" %(num_corr, num_incorr)) 
    
    #Calculate F1 score
    true_pos = np.sum(np.equal(prediction+test_class, 2).astype(int))
    act_pos = np.sum(test_class)
    pre_pos = np.sum(prediction)
    recall = true_pos*1.0/act_pos
    precision = true_pos*1.0/pre_pos
    F1 = 2.0*recall*precision/(recall+precision)
    print ("%.12f" %(F1))

if __name__ == "__main__":
    np.random.seed(0)
    main(argv)
