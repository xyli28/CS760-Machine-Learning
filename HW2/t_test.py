#!/usr/bin/env python3.6
from sys import argv
from json import load
import numpy as np
import pandas as pd
from nb import nb_learn
from tan import tan_learn
from math import sqrt 

def cal_precision(results):
    N = results.shape[0]
    crt = 0
    for index,row in results.iterrows():
        if row.loc['predicted'] == row.loc['actual']:
            crt += 1
    return crt/N 


def main():
    data = {}
    with open(argv[1],'r') as f:
        data = load(f)
    features_data = data['metadata']['features']
    features = {}
    for feature in features_data:
        features[feature[0]] = feature[1] 
    data = pd.DataFrame.from_records(data['data'], 
                                      columns=features.keys())
    data = data.sample(frac=1).reset_index(drop=True)

    N = data.shape[0]
    n = round(N/10)
    nb = []
    tan = []
    for i in range(9):
        train = pd.concat([data[:n*i],data[n*(i+1):]], ignore_index=True)
        test = data[n*i:n*(i+1)]
        nb_results = nb_learn(train, test, features)
        tan_network, tan_results = tan_learn(train, test, features)
        nb.append(cal_precision(nb_results))
        tan.append(cal_precision(tan_results))
    nb_results = nb_learn(data[:n*9], data[n*9:],features)
    tan_network, tan_results = tan_learn(data[:n*9], data[n*9:],features)
    nb.append(cal_precision(nb_results))
    tan.append(cal_precision(tan_results))
    nb = np.asarray(nb)
    tan = np.asarray(tan)

    #print out average accuracy 
    nb_ave = nb.mean()
    tan_ave = tan.mean()
    print ("Aveage accuracy for Naive Baynes: %f" %(nb_ave))
    print ("Aveage accuracy for Tree Augmented Naive Baynes: %f" %(tan_ave))
    
    sigma = tan - nb
    sigma_mean = sigma.mean()
    sigma_stddev = sigma.std()
    t = sigma_mean/(sigma_stddev/sqrt(10-1))
    print ("t statistic %f" %(t))
    if t > 2.262:
        print ("Two systems have different accuracies")
    else:
        print ("Two systems have same accuracy")

if __name__ == "__main__":
    main()
