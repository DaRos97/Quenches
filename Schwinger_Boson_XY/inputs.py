import numpy as np
####
header = ['J','h','Energy','Gap','L','A','B']
MaxIter = 3000
prec_L = 1e-10       #precision required in L maximization
L_method = 'Brent'
L_bounds = (0,50)
L_b_2 = 0.05
cutoff_L = cutoff_O = 1e-7
#phase diagram
hi = 10
hf = 0
hpts = 100
H = []
for i in range(hpts):
    H.append(hi+(hf-hi)/(hpts-1)*i)
