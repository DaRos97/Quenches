#!/bin/bash

for tau in 100 200 400 1000
do
    for gamma in 0 0.001 0.005 0.01 0.05 0.1
    do 
        echo "Computing tau = $tau and gamma = $gamma"
        python main.py --tau $tau --gamma $gamma --steps 200
    done
done


