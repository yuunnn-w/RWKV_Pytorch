#! /bin/bash

for i in $(ls -F data | grep /);
do
    rm -- "data/${i}state.npy" "data/${i}tokens.pkl";
done
