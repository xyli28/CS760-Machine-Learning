./logistic 0.01 10 ./datasets/banknote_train.json ./datasets/banknote_test.json > logistic_0.01_10_banknote.txt
./logistic 0.05 20 ./datasets/heart_train.json ./datasets/heart_test.json > logistic_0.05_20_heart.txt
./logistic 0.01 10 ./datasets/magic_train.json ./datasets/magic_test.json > logistic_0.01_10_magic.txt 
./nnet 0.01 5 10 ./datasets/banknote_train.json ./datasets/banknote_test.json > nnet_0.01_5_10_banknote.txt
./nnet 0.05 7 20 ./datasets/heart_train.json ./datasets/heart_test.json > nnet_0.05_7_20_heart.txt
./nnet 0.01 10 5 ./datasets/magic_train.json ./datasets/magic_test.json > nnet_0.01_10_5_magic.txt 
python3 logistic_F1.py 0.01 30 ./datasets/magic_train.json ./datasets/magic_test.json > logistic_F1.txt 
python3 nnet_F1.py 0.01 10 30 ./datasets/magic_train.json ./datasets/magic_test.json > nnet_F1.txt 
