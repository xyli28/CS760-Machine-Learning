#!/usr/bin/env python3.6
from sys import argv
import numpy as np
import bagged
import boosted

def main(argv):
    meta = []
    predicted = []
    acutal = []
    if argv[1] == "bag":
        meta, predicted, actual = bagged.main(argv[1:]) 
    else:
        meta, predicted, actual = boosted.main(argv[1:]) 
    hashMap = {}
    for i,label in enumerate(meta):
        hashMap[label] = i
    c_matrix = np.zeros((len(meta),len(meta)))    
    for i in range(predicted.shape[0]):
        c_matrix[hashMap[predicted[i]]][hashMap[actual[i]]] += 1    
    c_matrix = c_matrix.astype(int).astype(str)
    for i in range(len(meta)):
        print (','.join(c_matrix[i])) 
 
if __name__ == "__main__":
    np.random.seed(0)
    main(argv)

