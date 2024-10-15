import numpy as np
import functions as fs
from time import time
import sys,os
import random

machine = 'loc'

save_to_file = True

header = ['J','h','delta','Energy','Gap','L','A','argA','B','argB']
MaxIter = 3000
prec_L = 1e-10       #precision required in L maximization
L_method = 'Brent'
L_bounds = (0,50)       #bound of Lagrange multiplier
cutoff_L = 1e-4
pars_L = (prec_L,L_method,L_bounds)
cutoff_O = 1e-4
#phase diagram: staggered magnetic field h
hi = 10
hf = 0
hpts = 100
h_field_array = [hi+(hf-hi)/(hpts-1)*i for i in range(hpts)]
#
index_h = 0      #inp.H point in phase diagram
Spin = 0.5    #Spin value
K_points = 30      #number ok cuts in BZ
J_nn = 1
h_field = h_field_array[index_h]
delta = 0.1
pars = (J_nn,h_field,delta,Spin)
print("Using parameters: h=",str(h_field),", S=","{:.3f}".format(Spin),", points in BZ=",str(K_points))
#BZ points
Kx = K_points;     Ky = K_points
Kx_reference=13;   Ky_reference=13
#Filenames
#ReferenceDir = fs.get_res_final_dn(Kx_reference,Ky_reference,txt_S,machine)
filname = fs.get_res_final_fn(pars,Kx,Ky,machine)
#BZ points
kxg = np.linspace(0,np.pi/2,Kx)
kyg = np.linspace(-np.pi,np.pi,Ky)
k_grid = np.zeros((2,Kx,Ky))
for i in range(Kx):
    for j in range(Ky):
        k_grid[0,i,j] = kxg[i]
        k_grid[1,i,j] = kyg[j]
#### vectors of 1nn, 2nn and 3nn
a1 = (2,0)
a2 = (0,1)
########################
########################    Initiate routine
########################
Args_O = (k_grid,pars)
Args_L = (k_grid,pars,pars_L)
P_initial = [0.54,0,0.11,0]     #initial mean field parameters (|A|,arg(a),|B|,arg(B))
"""Can actually use values of previous h point in phase diagram"""
#Initiate variables
new_O = P_initial;      old_O_1 = new_O;      old_O_2 = new_O
new_L = (L_bounds[0]+L_bounds[1])/2;       old_L_1 = 0;    old_L_2 = 0
#
initial_time = time()
step = 0
continue_loop = True
while continue_loop:
    step += 1
    converged_L = 0
    converged_O = 0
    #Update old L variables
    old_L_2 = float(old_L_1)
    old_L_1 = float(new_L)
    #Compute L with newO
    new_L = fs.compute_L(new_O,Args_L)
    #Update old O variables
    old_O_2 = np.array(old_O_1)
    old_O_1 = np.array(new_O)
    #Compute O with new_L
    temp_O = fs.compute_O_all(new_O,new_L,Args_O)
    #Mix with previous result
    mix_factor = 0.5
    for i in range(len(P_initial)):
        new_O[i] = old_O_1[i]*mix_factor + temp_O[i]*(1-mix_factor)
    #Check if L steady solution
    if np.abs(old_L_2-new_L)/Spin > cutoff_L:
        converged_L = True
    #Check if O steady solution
    for i in range(len(P_initial)):
        if np.abs(old_O_1[i]-new_O[i])/Spin > cutoff_O or np.abs(old_O_2[i]-new_O[i])/Spin > cutoff_O:
            converged_O = 0
            break
        if i == len(P_initial)-1:
            converged_O = True
    if converged_O and converged_L:
        continue_loop = False
        new_L = fs.compute_L(new_O,Args_L)
    #
    if disp:
        print("Step ",step,": ",new_L,*new_O,end='\n')
    #Margin in number of steps
    if step > MaxIter:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Exceeded number of steps!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        break
######################################################################################################
######################################################################################################
print("\nNumber of iterations: ",step,'\n')
conv = True if (converged_O and converged_L) else False
if not conv:
    print("\n\nFound final parameters NOT converged: ",new_L,new_O,"\n")
    exit()
if new_L < inp.L_bounds[0] + 0.01 or new_L > inp.L_bounds[1] - 0.01:
    print("Suspicious L value: ",new_L," NOT saving")
    exit()
################################################### Save solution
E,gap = fs.total_energy(new_O,new_L,Args_O)
if E == 0:
    print("Something wrong with the energy=",E)
    exit()
data = [J_nn,h,E,gap,new_L]
for i in range(len(P_initial)):
    data.append(new_O[i])
DataDic = {}
header = inp.header
for ind in range(len(data)):
    DataDic[header[ind]] = data[ind]
if save_to_file:
    sf.SaveToCsv(DataDic,csvfile)

print(DataDic)
print("Total time: ",'{:5.2f}'.format((t()-Ti)/60),' minutes.')                           ################














































































