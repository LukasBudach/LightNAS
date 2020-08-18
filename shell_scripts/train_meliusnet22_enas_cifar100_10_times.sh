#!/bin/bash

cd ..
for VARIABLE in {1..10}
do
 echo TRAINING NUMBER: $VARIABLE
 python train_enas.py -c configs/default_config_melius22_cifar100.yml &> "Training_$VARIABLE.txt"

done