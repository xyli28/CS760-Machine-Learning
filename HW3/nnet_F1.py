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

def calActivationH(instance, w_i_h):
    intermediate = np.matmul(instance, w_i_h).reshape(-1,w_i_h.shape[1])
    return 1/(1+np.exp(-intermediate)) 

def calActivationO(intermediate, w_h_o):
    return 1/(1+np.exp(-np.matmul(intermediate, w_h_o))) 

def crossEntropy(activation, actual):
    return -np.sum(actual*np.log(activation)+(1-actual)*np.log(1-activation))

#def calGradient(activation, actual, instance):
#    return ((activation-actual)*instance).reshape(-1,1)

def calGradientO(activation, actual, intermediate):
    return ((activation-actual)*intermediate).reshape(-1,1)
    
def calGradientH(activation, actual, w_h_o,  intermediate, instance):
    sigma = (intermediate*(1-intermediate)*(activation-actual))*w_h_o.T
    return (np.matmul(instance.reshape(-1,1),sigma))
       

#def calGradientH():


def main(argv):
    
    #read learning rate and num of epochs
    learning_rate = float(argv[1])
    num_hiddens = int(argv[2])
    num_epochs = int(argv[3]) 

    #Load training and testing data 
    train_data = {}
    test_data = {}
    with open(argv[4],'r') as f:
        train_data = load(f)
    with open(argv[5],'r') as f:
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
    w_i_h = np.random.uniform(low=-0.01, high=0.01, size=(num_hiddens, train_feature.shape[1])).T
    w_h_o = np.random.uniform(low=-0.01, high=0.01, size=(1, num_hiddens+1)).T

    #Training 
    activation = np.zeros((train_feature.shape[0],1))
    for i in range(num_epochs):
       for j in range(train_feature.shape[0]): 
           intermediate = calActivationH(train_feature[j], w_i_h)
           bias = np.ones(intermediate.shape[0]).reshape(-1,1)
           intermediate = np.concatenate([bias, intermediate],axis = 1)
           activation[j] = calActivationO(intermediate, w_h_o)
           grad_o = calGradientO(activation[j], train_class[j], intermediate) 
           intermediate = np.delete(intermediate, 0, 1)
           grad_h = calGradientH(activation[j], train_class[j], w_h_o[1:], 
                                 intermediate, train_feature[j])
           w_h_o -= learning_rate*grad_o
           w_i_h -= learning_rate*grad_h
       #calculate F1 on train 
       intermediate = calActivationH(train_feature,w_i_h)
       bias = np.ones(intermediate.shape[0]).reshape(-1,1)
       activation_train = calActivationO(np.concatenate([bias, intermediate], axis = 1) , w_h_o) 
       cross_entropy = crossEntropy(activation_train, train_class)
       prediction = np.round(activation_train)
       num_corr = np.sum(np.equal(prediction, train_class).astype(int))  
       num_incorr = train_class.shape[0] - num_corr       
       true_pos = np.sum(np.equal(prediction+train_class, 2).astype(int))
       act_pos = np.sum(train_class)
       pre_pos = np.sum(prediction)
       recall = true_pos*1.0/act_pos
       precision = true_pos*1.0/pre_pos
       F1_train = 2.0*recall*precision/(recall+precision)

       #calculate F1 on test 
       intermediate = calActivationH(test_feature,w_i_h)
       bias = np.ones(intermediate.shape[0]).reshape(-1,1)
       activation_test = calActivationO(np.concatenate([bias, intermediate], axis = 1) , w_h_o) 
       cross_entropy = crossEntropy(activation_test, test_class)
       prediction = np.round(activation_test)
       num_corr = np.sum(np.equal(prediction, test_class).astype(int))  
       num_incorr = test_class.shape[0] - num_corr       
       true_pos = np.sum(np.equal(prediction+test_class, 2).astype(int))
       act_pos = np.sum(test_class)
       pre_pos = np.sum(prediction)
       recall = true_pos*1.0/act_pos
       precision = true_pos*1.0/pre_pos
       F1_test = 2.0*recall*precision/(recall+precision)
       print ("%d %.12f %.12f" %(i+1,F1_train,F1_test))

if __name__ == "__main__":
    np.random.seed(0)
    main(argv)
