#bagged_trees 2 3 ./datasets/digits_train.json datasets/digits_test.json > bagged_trees_2_3_digits.txt 
#bagged_trees 2 3 ./datasets/heart_train.json datasets/heart_test.json > bagged_trees_2_3_heart.txt 
#bagged_trees 2 3 ./datasets/mushrooms_train.json datasets/mushrooms_test.json > bagged_trees_2_3_mushrooms.txt 
#bagged_trees 2 3 ./datasets/winequality_train.json datasets/winequality_test.json > bagged_trees_2_3_winequality.txt 
#bagged_trees 5 2 ./datasets/digits_train.json datasets/digits_test.json > bagged_trees_5_2_digits.txt 
#bagged_trees 5 2 ./datasets/heart_train.json datasets/heart_test.json > bagged_trees_5_2_heart.txt 
#bagged_trees 5 2 ./datasets/mushrooms_train.json datasets/mushrooms_test.json > bagged_trees_5_2_mushrooms.txt 
#bagged_trees 5 2 ./datasets/winequality_train.json datasets/winequality_test.json > bagged_trees_5_2_winequality.txt 
#boosted_trees 2 3 ./datasets/digits_train.json datasets/digits_test.json > boosted_trees_2_3_digits.txt 
#boosted_trees 2 3 ./datasets/heart_train.json datasets/heart_test.json > boosted_trees_2_3_heart.txt 
#boosted_trees 2 3 ./datasets/mushrooms_train.json datasets/mushrooms_test.json > boosted_trees_2_3_mushrooms.txt 
#boosted_trees 2 3 ./datasets/winequality_train.json datasets/winequality_test.json > boosted_trees_2_3_winequality.txt 
#boosted_trees 5 2 ./datasets/digits_train.json datasets/digits_test.json > boosted_trees_5_2_digits.txt 
#boosted_trees 5 2 ./datasets/heart_train.json datasets/heart_test.json > boosted_trees_5_2_heart.txt 
#boosted_trees 5 2 ./datasets/mushrooms_train.json datasets/mushrooms_test.json > boosted_trees_5_2_mushrooms.txt 
#boosted_trees 5 2 ./datasets/winequality_train.json datasets/winequality_test.json > boosted_trees_5_2_winequality.txt
confusion_matrix boost 5 2 ./datasets/digits_train.json datasets/digits_test.json > confusion_matrix_boost_5_2_digits.txt 
confusion_matrix bag 5 2 ./datasets/digits_train.json datasets/digits_test.json > confusion_matrix_bag_5_2_digits.txt 
python3 bagged_precision.py 10 5 ./datasets/digits_train.json datasets/digits_test.json > confusion_matrix_bag_5_2_digits.txt 
