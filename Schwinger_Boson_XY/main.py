import numpy as np
import inputs as inp
import functions as fs
import system_functions as sf
from time import time as t
import sys
import getopt
import random
######################
###################### Set the initial parameters
######################
####### Outside inputs
Ti = t()
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "N:S:K:",["disp"])
    N = 0      #inp.J point in phase diagram
    txt_S = '50'
    K = 40      #number ok cuts in BZ
    numb_it = 3
    save_to_file = True
    disp = True
except:
    print("Error in input parameters",argv)
    exit()
for opt, arg in opts:
    if opt in ['-N']:
        N = int(arg)
    if opt in ['-S']:
        txt_S = arg
    if opt in ['-K']:
        K = int(arg)
    if opt == '--disp':
        disp = True 

J_nn = 1
h = inp.H[N]
J = (J_nn,h)
S_label = {'50':0.5,'36':(np.sqrt(3)-1)/2,'34':0.34,'30':0.3,'20':0.2}
S = S_label[txt_S]
#BZ points
Nx = K;     Ny = K
#Filenames
DirName = 'Data/S'+txt_S+'/'
DataDir = DirName + str(Nx) + '/'
ReferenceDir = DirName + str(13) + '/'
csvname = 'J_h=('+'{:5.4f}'.format(J_nn).replace('.','')+'_'+'{:5.4f}'.format(h).replace('.','')+').csv'
csvfile = DataDir + csvname
#BZ points
kxg = np.linspace(0,2*np.pi,Nx)
kyg = np.linspace(0,np.pi,Ny)
k_grid = np.zeros((2,Nx,Ny))
for i in range(Nx):
    for j in range(Ny):
        k_grid[0,i,j] = kxg[i]
        k_grid[1,i,j] = kyg[j]
#### vectors of 1nn, 2nn and 3nn
a1 = (1,0)
a2 = (0,2)
#### product of lattice vectors with K-matrix
KM = fs.compute_KM(k_grid,a1,a2)     #large unit cell
########################
########################    Initiate routine
########################
Args_O = (KM,K,S,J)
Args_L = (KM,K,S,J,inp.L_bounds)
Ti = t()
#
P_initial = [0.1,0.5]
new_O = P_initial;      old_O_1 = new_O;      old_O_2 = new_O
new_L = (inp.L_bounds[1]-inp.L_bounds[0])/2 + inp.L_bounds[0];       old_L_1 = 0;    old_L_2 = 0
#
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
    if np.abs(old_L_2-new_L)/S > inp.cutoff_L:
        converged_L = 1
    #Check if O steady solution
    conv = np.ones(len(P_initial))
    for i in range(len(P_initial)):
        if np.abs(old_O_1[i]-new_O[i])/S > inp.cutoff_O or np.abs(old_O_2[i]-new_O[i])/S > inp.cutoff_O:
            converged_O = 0
            break
        if i == len(P_initial)-1:
            converged_O = 1
    if converged_O and converged_L:
        continue_loop = False
        new_L = fs.compute_L(new_O,Args_L)
    if disp:
        print("Step ",step,": ",new_L,*new_O[:],end='\n')
    #Margin in number of steps
    if step > inp.MaxIter:
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














































































