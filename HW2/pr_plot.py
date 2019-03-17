from sys import argv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pr_conversion(row):
    """Data conversion
    """
    if row.loc['predicted'] == 'negative':
        return (1 - float(row.loc['confidence']))
    else: 
        return float(row.loc['confidence'])

def cal_recall_precision(confidence):
    """Calculate recall and precision
    """
    n_pos = confidence.groupby('class').size().loc['positive']
    recall = []
    precision = []
    true_pos = 0
    predict_pos = 0
    last_true_pos = 0
    last_confidence = 1.0
    for index,row in confidence.iterrows():
        if row.loc['class'] == 'positive':
            true_pos += 1
            last_confidence = row['confidence']
        elif (row['confidence'] != last_confidence) and (true_pos > last_true_pos):
            recall.append(true_pos/n_pos)
            precision.append(true_pos/predict_pos)
            last_true_pos = true_pos
            last_confidence = row['confidence']
        else:
            last_confidence = row['confidence'] 
        predict_pos += 1
    return [recall,precision]


def main():

    nb = []
    with open(argv[1],'r') as f:
        for line in f:
            if line == '\n':
                break 
        for line in f:
            if line == '\n':
                break 
            nb.append(line.split())
    nb = pd.DataFrame.from_records(nb,columns=['predicted','actual','confidence'])  
    pr = nb.apply(pr_conversion, axis=1)
    nb = pd.DataFrame.from_records({'confidence':pr,
                                    'class':nb['actual']})  
    nb = nb.sort_values(by=['confidence'],ascending=False)
    tan = []
    with open(argv[2],'r') as f:
        for line in f:
            if line == '\n':
                break 
        for line in f:
            if line == '\n':
                break 
            tan.append(line.split())
    tan = pd.DataFrame.from_records(tan,columns=['predicted','actual','confidence'])
    pr = tan.apply(pr_conversion, axis=1)
    tan = pd.DataFrame.from_records({'confidence':pr,
                                     'class':tan['actual']})  
    tan = tan.sort_values(by=['confidence'],ascending=False)

    #calculate recall and precision
    nb_pr = cal_recall_precision(nb) 
    tan_pr = cal_recall_precision(tan)

    plt.plot(nb_pr[0],nb_pr[1], 'ro-', linewidth = 1, label = "NB")
    plt.plot(tan_pr[0],tan_pr[1], 'bo-', linewidth = 1, label = "TAN")
    plt.legend()
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("Precision/Recall")
    plt.savefig("pr.png")
    plt.show()

if __name__ == "__main__":
    main()


