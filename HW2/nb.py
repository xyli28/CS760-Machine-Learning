#!/usr/bin/env python3.6
from sys import argv
from json import load
import numpy as np
import pandas as pd

def cal_cpt(train, feature, features, N):
    """Calculate conditional probability P(X|Y)
    """
    n = len(features[feature])
    cpt = train.groupby([feature,'class']).size()
    for val in features[feature]:
        for label in features['class']:
            if (val,label) in cpt.index:
                cpt.loc[(val,label)] = (cpt.loc[(val,label)]+1)/\
                                       (N.loc[label]+n)
            elif (val,label) not in cpt.index and label in N.index:
                cpt.loc[(val,label)] = 1/(N.loc[label]+n)
            elif label in N.index:
                cpt.loc[(val,label)] = 1/n
            else:
                pass
    return cpt

def cal_pro(test, cpts, p_y):
    """Predict the class for test set
    """
    p_y_x = p_y.copy()
    for label in p_y.keys():
        for feature in cpts.keys():
            p_y_x.loc[label] *= cpts[feature].loc[(test.loc[feature],label)]   
    p_y_x = p_y_x/(p_y_x.sum()) 
    index_name = ['predicted','actual','probability']
    if p_y_x[0] >= 0.5:
        return pd.Series([p_y_x.keys()[0],test.loc['class'],p_y_x[0]], 
                         index = index_name)
    else:
        return pd.Series([p_y_x.keys()[1],test.loc['class'],p_y_x[1]], 
                         index = index_name)

def nb_learn(train, test, features):
    """Run naive baynes learning for training and test data
    """
    N = train.groupby('class').size()    #training set grouped bt class
    p_y = N.copy()                
    p_y = (p_y+1)/(N.sum()+N.size)       #probability of classes in training set 

    #Learning CPT using naive baynes structure(P(X|Y))
    cpts = {}    
    for feature in features.keys():
        if feature != 'class':
            cpts[feature] = cal_cpt(train, feature, features, N)
    
    #Run prediction for test data
    return test.apply(cal_pro, axis=1, args = (cpts, p_y))

def main():

    #Load training and testing data from files
    train_data = {}
    test_data = {}
    with open(argv[1],'r') as f:
        train_data = load(f)
    with open(argv[2],'r') as f:
        test_data = load(f)
    features_data = train_data['metadata']['features']
    features = {}
    for feature in features_data:
        features[feature[0]] = feature[1] 
    train = pd.DataFrame.from_records(train_data['data'], 
                                      columns=features.keys())
    test = pd.DataFrame.from_records(test_data['data'], 
                                     columns=features.keys())
    
    #Run naive bias learning
    results = nb_learn(train, test, features)

    #print out results
    for feature in features.keys():
        if feature != 'class':
            print (feature+" class")
    print ("")
    crt = 0
    for index,row in results.iterrows():
        if row.loc['predicted'] == row.loc['actual']:
            crt += 1
        print ("%s %s %.12f" %(row.loc['predicted'],row.loc['actual'],
               row.loc['probability'])) 
    print ("")
    print (crt)
    print ("") 
  
if __name__ == "__main__":
    main()
