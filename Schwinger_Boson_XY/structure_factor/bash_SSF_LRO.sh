#!/bin/bash
ex2=( ["20"]=0 ["19"]=0.06 ["16"]=-0.3 ["15"]=0.3)
ex3=( ["20"]=0 ["19"]=0.02 ["16"]=0 ["15"]=0)

for ans in 20 16 15
do
    for DM in 000 209
    do
        python ssf_LRO.py -a $ans -S 50 --DM $DM --j2 ${ex2[$ans]} --j3 ${ex3[$ans]} --Nq 33
    done
done

