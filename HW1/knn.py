import numpy as np
import pandas as pd

def cal_accuracy(a ,b):
    """Return the accuracy between prediction and actually data
    """
    return np.equal(b, a).astype(int).sum()/b.size

def num_distance(a, b):
    """Calculate numeric distance bwteen two sets of features
    """
    return np.absolute(a-b).sum(axis=1)

def ctg_distance(a, b):
    """Calculate categorical distance between two sets of features 
    """
    return np.not_equal(b, a).astype(int).sum(axis=1)

def find_knn(dist, training_label, k, labels):
    """Find k nearest neighbous of testing set,
    classify the testing set correspondingly.

    params: 
        dist: distances between training set and testing set
        trainning_label: labels for all training set
        k: number of nearest neighbours need to be considered
        labels: list of all labels

    returns:
        A list contains nearest neighbours' distribution on all labels 
        and the classification result for testing set
    """
    #Find k nearest neghours and group them by labels.
    knn = pd.DataFrame({"distance":dist, "label":training_label})\
          .nsmallest(k, 'distance', keep='first')\
          .groupby(['label']).size()
    #Calculate the distribution for all labels.
    s = pd.Series(0, index = labels)
    for (index, value) in s.iteritems():
        s.loc[index] = knn.loc[index] if index in knn else 0
    #Find the most populated label and classify the testing data into 
    #this category.
    s['label'] = s.idxmax() 
    return s.to_numpy()

def knn_confidence(dist, training_label, k, labels):
    """Find k nearest neighbous of testing set,
    and calculate the confidence value

    params: 
        dist: distances between training set and testing set
        trainning_label: labels for all training set
        k: number of nearest neighbours need to be considered
        labels: list of all labels

    returns:
        Confidence value
    """
    #Find k nearest neghours and group them by labels.
    knn = pd.DataFrame({"distance":dist, "label":training_label})\
          .nsmallest(k, 'distance', keep='first')\
    #calculate the confidence value
    epsilon = 1e-5  
    knn['distance'] = 1/(knn['distance']**2+epsilon) 
    knn['label'] = (knn['label'] == labels[0]).astype(int)
    return (knn['distance']*knn['label']).sum()/knn['distance'].sum() 

def knn_learning(features, training, testing, k, learning_type):  
    """Classfy the testing data using KNN

    params: 
        features: features
        training: training data
        testing: testing data
        k: number of nearest neighbours need to be considered
        learning_type: choose classification or regression for learning

    returns:
        Classification:
        An array contains nearest neighbours' distribution on all labels 
        and the classification result for testing set
        Regression:
        An array of confidence value for each test data
    """
    
    #Split features into numeric features and categorical features.
    num_features = []
    ctg_features = [] 
    for feature in features:
        if feature[0] == 'label':
            pass
        elif feature[1] == 'numeric':
            num_features.append(feature[0])
        else:
            ctg_features.append(feature[0])
    training_num = training[num_features].to_numpy()
    training_ctg = training[ctg_features].to_numpy()
    testing_num = testing[num_features].to_numpy()
    testing_ctg = testing[ctg_features].to_numpy()
    #Find all labels and labels for traing data.
    labels = features[-1][1]
    training_label = training['label'].to_numpy()

    #Standardize numeric features.
    ave = training_num.mean(axis=0)
    stddev = training_num.std(axis=0)
    stddev[stddev == 0.0] = 1.0
    training_num = (training_num-ave)/stddev
    testing_num = (testing_num-ave)/stddev

    #Calculate the distance between training set and testing set.
    dist_num = np.apply_along_axis(num_distance, 1 , 
                                           testing_num, training_num)
    dist_ctg = np.apply_along_axis(ctg_distance, 1 , 
                                           testing_ctg, training_ctg)
    dist_total = dist_num + dist_ctg

    if learning_type == "classification":
        #Find k nearest neighbours, classfy the testing set and print out results.
        return np.apply_along_axis(find_knn, 1, dist_total, 
                                   training_label, k, labels)
    elif learning_type == "regression":
        #Calculate the confidence value for all testing data
        return np.apply_along_axis(knn_confidence, 1, dist_total, 
                                   training_label, k, labels)
         
