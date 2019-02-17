#!/usr/bin/env bash

#./knn_classifier 10 ./datasets/votes_train.json ./datasets/votes_test.json > output_knn_classifier_10_votes.txt 
#./knn_classifier 10 ./datasets/digits_train.json ./datasets/digits_test.json > output_knn_classifier_10_digits.txt 
#./hyperparam_tune 20 ./datasets/votes_train.json ./datasets/votes_val.json ./datasets/votes_test.json > output_hyperparam_tune_20_votes.txt
#./hyperparam_tune 20 ./datasets/digits_train.json ./datasets/digits_val.json ./datasets/digits_test.json > output_hyperparam_tune_20_digits.txt
#for k in {5,9,10,15,20};
#do
#    ./learning_curve $k ./datasets/votes_train.json ./datasets/votes_test.json > output_learning_curve_${k}_votes.txt 
#    ./learning_curve $k ./datasets/digits_train.json ./datasets/digits_test.json > output_learning_curve_${k}_digits.txt
#done 
for k in {2,5,10,15,20,30};
do
    ./roc_curve $k ./datasets/votes_train.json ./datasets/votes_test.json > output_roc_curve_${k}_votes.txt 
#    ./roc_curve $k ./datasets/digits_train.json ./datasets/digits_test.json > output_roc_curve_${k}_digits.txt
done 
