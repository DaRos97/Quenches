#!/bin/bash
ex2=( ["20"]=0 ["19"]=0.06 ["16"]=-0.16 ["15"]=0.2)
ex3=( ["20"]=0 ["19"]=0.02 ["16"]=0 ["15"]=0)

for ans in 20 19 #16 15
do
    for S in 50 36 #30 20
    do
        python Ssf_SL.py -a $ans -S $S --DM 000 --j2 ${ex2[$ans]} --j3 ${ex3[$ans]} --Nq 71
    done
done

