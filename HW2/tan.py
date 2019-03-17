#!/usr/bin/env python3.6
from sys import argv
from json import load
from math import log
import numpy as np
import pandas as pd

def cal_cpt_1(train, feature, features, N):
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


def cal_cpt_2(train, f_a, f_b, features, N):
    """Calculate conditional probability P(X1,X2|Y)
    """
    n = len(features[f_a])*len(features[f_b])
    cpt = train.groupby([f_a, f_b, 'class']).size()
    for v_a in features[f_a]:
        for v_b in features[f_b]:
            for label in features['class']:
                if (v_a,v_b,label) in cpt.index:
                    cpt.loc[(v_a,v_b,label)] = (cpt.loc[(v_a,v_b,label)]+1)/\
                    (N.loc[label]+n)
                elif (v_a,v_b,label) not in cpt.index and label in N.index:
                    cpt.loc[(v_a,v_b,label)] = 1/(N.loc[label]+n)
                elif label not in N.index:
                    cpt.loc[(v_a,v_b,label)] = 1/n
                else:
                    pass
    return cpt

def cal_cpt_3(train, f_a, f_b, features, N):
    """Calculate probability P(X1,X2,Y)
    """
    n = len(features[f_a])*len(features[f_b])*2
    n_train = train.shape[0]
    cpt = train.groupby([f_a, f_b, 'class']).size()
    for v_a in features[f_a]:
        for v_b in features[f_b]:
            for label in features['class']:
                if (v_a,v_b,label) in cpt.index:
                    cpt.loc[(v_a,v_b,label)] = (cpt.loc[(v_a,v_b,label)]+1)/\
                    (n_train+n)
                elif (v_a,v_b,label) not in cpt.index and label in N.index:
                    cpt.loc[(v_a,v_b,label)] = 1/(n_train+n)
                elif label not in N.index:
                    cpt.loc[(v_a,v_b,label)] = 1/n
                else:
                    pass
    return cpt

def cal_cpt_4(train, f_a, f_b, features):
    """Calculate conditional probability P(X1|X2,Y)
    """
    n = len(features[f_a])
    cpt = train.groupby([f_a, f_b, 'class']).size()
    den = train.groupby([f_b,'class']).size() 
    for v_a in features[f_a]:
        for v_b in features[f_b]:
            for label in features['class']:
                if (v_a,v_b,label) in cpt.index:
                    cpt.loc[(v_a,v_b,label)] = (cpt.loc[(v_a,v_b,label)]+1)/\
                    (den.loc[v_b,label]+n)
                elif (v_a,v_b,label) not in cpt.index and (v_b,label) in den.index:
                    cpt.loc[(v_a,v_b,label)] = 1/(den.loc[v_b,label]+n)
                elif (v_b,label) not in den.index:
                    cpt.loc[(v_a,v_b,label)] = 1/n
                else:
                    pass
    return cpt

def cal_info(cpts, f_a, f_b, features):
    """Calculate conditional mutual information
    """
    cpt_a = cpts[f_a]
    cpt_b = cpts[f_b]
    cpt_ab = cpts[(f_a, f_b)]
    cpt_abc = cpts[(f_a,f_b,'class')]
    info = 0
    for v_a in features[f_a]:
        for v_b in features[f_b]:
            for label in features['class']:
                info += cpt_abc[(v_a,v_b,label)]*log(cpt_ab[(v_a,v_b,label)]/
                        (cpt_a[v_a,label]*cpt_b[v_b,label]),2)
    return info
        
def mst(adj, n):
    """Calculate MST according to adjacency matrix
    """
    v = [0]
    e = []
    w = [0 for x in range(n)]
    ee = [[] for x in range(n)] 
    for i in range(1,n):
        w[i] = adj[0][i]
        ee[i] = [0,i]
    while len(v) < n:
        i = np.argmax(w)
        v.append(i)
        e.append(ee[i])
        w[i] = float("-inf")
        for j in range(n):
            if j not in v and adj[i][j] > w[j]:
                w[j] = adj[i][j]
                ee[j] = [i,j]
    return e
        
def cal_pro(test, p_y, cpts, tan_cpt, tan_network, first_feature):
    """Predict the class for test set
    """
    p_y_x = p_y.copy()
    for label in p_y.keys():
        p_y_x.loc[label] *= cpts[first_feature].\
                            loc[(test.loc[first_feature],label)]   
        for edge in tan_network:
            p_y_x.loc[label] *= tan_cpt[(edge[0],edge[1])].\
                                loc[(test.loc[edge[0]],test.loc[edge[1]],label)]   
    p_y_x = p_y_x/(p_y_x.sum()) 
    index_name = ['predicted','actual','probability']
    if p_y_x[0] >= 0.5:
        return pd.Series([p_y_x.keys()[0],test.loc['class'],p_y_x[0]], 
                         index = index_name)
    else:
        return pd.Series([p_y_x.keys()[1],test.loc['class'],p_y_x[1]], 
                         index = index_name)

def tan_learn(train, test, features):
    """Run tree augmented naive bayes learning for training and test data
    """
    n_f = len(features) - 1                 #number of features
    N = train.groupby('class').size()       #training set grouped by class
    p_y = N.copy()                          
    p_y = (p_y+1)/(N.sum()+N.size)          #probability of classes in training set 

    #Learning CPT for each feature(P(X|Y))
    cpts = {}    
    for feature in features.keys():
        if feature != 'class':
            cpts[feature] = cal_cpt_1(train, feature, features, N)

    #Learning CPT for each two features(P(X1,X2|Y)) and probability of P(X1,X2,Y)
    for i in range(n_f):
        f_a = list(features.keys())[i]
        for j in range(i+1, n_f):
            f_b = list(features.keys())[j]
            cpts[(f_a,f_b)] = cpts[(f_b,f_a)] =\
                cal_cpt_2(train, f_a, f_b, features, N)
            cpts[(f_a,f_b,'class')] = cpts[(f_b,f_a,'class')] =\
                cal_cpt_3(train, f_a, f_b, features, N)
   
    #Calculate mutual information 
    info = np.empty(shape=[n_f, n_f])
    for i in range(n_f):
        f_a = list(features.keys())[i]
        for j in range(i+1, n_f):
            f_b = list(features.keys())[j]
            info[i][j] = info[j][i] = cal_info(cpts, f_a, f_b, features)
    for i in range(n_f):
        info[i][i] = float("-inf")
          
    #Calculate Maximum Spanning Tree using Prim's Algorithm
    mst_edges = mst(info, n_f)
    mst_edges.sort(key = lambda x:x[1])
    tan_network = [[list(features.keys())[e[1]],
                   list(features.keys())[e[0]]] for e in mst_edges]

    #Calculate conditional probability (P(X1|X2,Y))according to tan structure
    tan_cpt = {}
    for e in tan_network:
        cpt = cal_cpt_4(train, e[0], e[1], features) 
        tan_cpt[(e[0],e[1])] = cpt

    #Run prediction for test data
    first_feature = list(features.keys())[0]
    return (tan_network, test.apply(cal_pro, axis=1, args = (p_y, 
           cpts, tan_cpt, tan_network, first_feature)))

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

    #Run TAN learning
    first_feature = list(features.keys())[0]
    tan_network, results = tan_learn(train, test, features)

    #print out results
    print (first_feature+" class") 
    for edge in tan_network:
        print ((" ").join(edge)+" class")
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
