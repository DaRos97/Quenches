import numpy as np
import system_functions as sf
from scipy import linalg as LA
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline as RBS
from pathlib import Path
import csv
import os
#Libraries needed only for debug and plotting -> not used in the cluster
#from colorama import Fore
#import matplotlib.pyplot as plt
#from matplotlib import cm

def total_energy(P,L,args):
    KM,K_,S,J = args
    J1,h = J
    m = 2
    J_ = np.zeros((2*m,2*m))
    for i in range(m):
        J_[i,i] = -1
        J_[i+m,i+m] = 1
    Res = -J1*4*(P[0]**2+P[1]**2+S**2)
    Res -= L*(2*S+1)            #part of the energy coming from the Lagrange multiplier
    #Compute now the (painful) part of the energy coming from the Hamiltonian matrix by the use of a Bogoliubov transformation
    args2 = (KM,K_,S,J)
    N = big_Nk(P,L,args2)                #compute Hermitian matrix from the ansatze coded in the ansatze.py script
    res = np.zeros((m,K_,K_))
    for i in range(K_):                 #cicle over all the points in the Brilluin Zone grid
        for j in range(K_):
            Nk = N[:,:,i,j]                 #extract the corresponding matrix
            try:
                Ch = LA.cholesky(Nk)        #not always the case since for some parameters of Lambda the eigenmodes are negative
            except LA.LinAlgError:          #matrix not pos def for that specific kx,ky
                print("Matrix is not pos def for some K at end of minimization")
                exit()
                return 0,0           #if that's the case even for a single k in the grid, return a defined value
            temp = np.dot(np.dot(Ch,J_),np.conjugate(Ch.T))    #we need the eigenvalues of M=KJK^+ (also Hermitian)
            a = LA.eigvalsh(temp)      #BOTTLE NECK -> compute the eigevalues
            res[:,i,j] = a[m:]
    gap = np.amin(res.ravel())           #the gap is the lowest value of the lowest gap (not in the fitting if not could be negative in principle)
    #Now fit the energy values found with a spline curve in order to have a better solution
    r2 = 0
    for i in range(m):
        func = RBS(np.linspace(0,1,K_),np.linspace(0,1,K_),res[i])
        r2 += func.integral(0,1,0,1)        #integrate the fitting curves to get the energy of each band
    r2 /= m
    #normalize
    #Summation over k-points
    #r3 = res.ravel().sum() / len(res.ravel())
    energy = Res + r2
    return energy, gap

#### Computes Energy from Parameters P, by maximizing it wrt the Lagrange multiplier L. Calls only totEl function
def compute_L(P,args):
    pars_L = args[2]
    pqs = compute_pqs(P,args)   #p1,p2,q1,q2
    res = minimize_scalar(lambda l: optimize_L(pqs,l,args),  #maximize energy wrt L with fixed P
            method = pars_L[1],          #can be 'bounded' or 'Brent'
            bracket = pars_L[2],
            options={'xtol':pars_L[0]}
            )
    return res.x        #Lambda

def compute_pqs(P,args):
    k_grid = args[0]
    J_nn,h_field,delta,Spin = args[1]
    #We take alpha_0=aplha_1 and alpha_x=alpha_y (same for beta)
    p1 = -J_nn*P[2]*np.exp(1j*P[3])*(1-delta)*2*(np.cos(k_grid[0,:,:])+np.cos(k_grid[1,:,:]))
    p2 = p1
    q1 = -J_nn*P[0]*np.exp(1j*P[1])*(1+delta)*2*(np.cos(k_grid[0,:,:])+np.cos(k_grid[1,:,:]))
    q2 = q1
    return (p1,p2,q1,q2)

#### Computes the Energy given the paramters P and the Lagrange multiplier L. 
#### This is the function that does the actual work.
def optimize_L(pqs,L,args):
    k_grid,pars,pars_L = args
    K_points = pqs[0].shape[0]
    L_bounds = pars_L[2]
    m = 2   #UC size
    J_ = np.zeros((4*m,4*m))
    VL = 100
    for i in range(4*m):
        J_[i,i] = (-1)**(i+1)
    if L < L_bounds[0]:
        Res = -VL-(L_bounds[0]-L)
        return -Res
    elif L > L_bounds[1]:
        Res = -VL-(L-L_bounds[1])
        return -Res
    Res = -L*(2*pars[-1]+1)
    #Compute now the (painful) part of the energy coming from the Hamiltonian matrix by the use of a Bogoliubov transformation
    matrix_N = big_Nk(pqs,L,pars)               #compute Hermitian matrix from the ansatze coded in the ansatze.py script
    res = np.zeros((2*m,K_points,K_points))
    for i in range(K_points):                 #cicle over all the points in the Brilluin Zone grid
        for j in range(K_points):
            Nk = matrix_N[:,:,i,j]                 #extract the corresponding matrix
            try:
                Ch = LA.cholesky(Nk)        #not always the case since for some parameters of Lambda the eigenmodes are negative
            except LA.LinAlgError:          #matrix not pos def for that specific kx,ky
                r4 = -VL+(L-L_bounds[0])
                result = -(Res+r4)
                print('e:\t',L,result)
                return result           #if that's the case even for a single k in the grid, return a defined value
            temp = np.dot(np.dot(Ch,J_),np.conjugate(Ch.T))    #we need the eigenvalues of M=KJK^+ (also Hermitian)
            res[:,i,j] = LA.eigvalsh(temp)[2*m:]      #BOTTLE NECK -> compute the eigevalues
    gap = np.amin(res[0].ravel())           #the gap is the lowest value of the lowest map (not in the fitting if not could be negative in principle)
    #Now fit the energy values found with a spline curve in order to have a better solution
    r2 = 0
    for i in range(m):
        func = RBS(np.linspace(0,1,K_points),np.linspace(0,1,K_points),res[i])
        r2 += func.integral(0,1,0,1)        #integrate the fitting curves to get the energy of each band
    r2 /= m                             #normalize
    #Summation over k-pts
    #r2 = res.ravel().sum() / len(res.ravel())
    result = -(Res+r2)
#    print('g:\t',L,result)
    return result

def compute_O_all(old_O,L,args):
    new_O = np.zeros(len(old_O))
    KM,K_,S,J = args
    m = 2
    J_ = np.zeros((2*m,2*m))
    for i in range(m):
        J_[i,i] = -1
        J_[i+m,i+m] = 1
    #Compute first the transformation matrix M at each needed K
    args_M = (KM,K_,S,J)
    N = big_Nk(old_O,L,args_M)
    M = np.zeros(N.shape,dtype=complex)
    for i in range(K_):
        for j in range(K_):
            N_k = N[:,:,i,j]
            Ch = LA.cholesky(N_k) #upper triangular-> N_k=Ch^{dag}*Ch
            w,U = LA.eigh(np.dot(np.dot(Ch,J_),np.conjugate(Ch.T)))
            w = np.diag(np.sqrt(np.einsum('ij,j->i',J_,w)))
            M[:,:,i,j] = np.dot(np.dot(LA.inv(Ch),U),w)
    #for each parameter need to know what it is
    dic_O = [compute_A,compute_B]
    for p in range(2):
        rrr = np.zeros((K_,K_),dtype=complex)
        for i in range(K_):
            for j in range(K_):
                U,X,V,Y = split(M[:,:,i,j],m,m)
                U_,V_,X_,Y_ = split(np.conjugate(M[:,:,i,j].T),m,m)
                rrr[i,j] = dic_O[p](U,X,V,Y,U_,X_,V_,Y_,0,1)
        interI = RBS(np.linspace(0,1,K_),np.linspace(0,1,K_),np.imag(rrr))
        res2I = interI.integral(0,1,0,1)
        interR = RBS(np.linspace(0,1,K_),np.linspace(0,1,K_),np.real(rrr))
        res2R = interR.integral(0,1,0,1)
        res = (res2R+1j*res2I)/2
        new_O[p] = np.absolute(res)                   #renormalization of amplitudes 
    return new_O
#
def compute_A(U,X,V,Y,U_,X_,V_,Y_,li_,lj_):
    return (np.einsum('ln,nm->lm',U,V_)[li_,lj_]
            - np.einsum('nl,mn->lm',Y_,X)[li_,lj_])
########
def compute_B(U,X,V,Y,U_,X_,V_,Y_,li_,lj_):
    return (np.einsum('nl,mn->lm',X_,X)[li_,lj_]
            + np.einsum('ln,nm->lm',V,V_)[li_,lj_])

def split(array, nrows, ncols):
    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))


#### Ansatze encoded in the matrix
def big_Nk(pqs,L,pars):
    J_nn,h_field,delta,Spin = pars
    K_points = pqs[0].shape[0]
    m = 2
    p1,p2,q1,q2 = pqs
    ################
    Naa = np.zeros((2*m,2*m,K_points,K_points), dtype=complex)
    Nab = np.zeros((2*m,2*m,K_points,K_points), dtype=complex)
    Naa[0,2] = p1
    Naa[0,3] = q1
    Naa[1,2] = q2
    Naa[1,3] = p2
    Naa += np.conjugate(np.transpose(Naa,axes=(1,0,2,3)))
    #
    Nab[0,0] = -1j*h_field
    Nab[1,1] = 1j*h_field
    Nab[2,2] = 1j*h_field
    Nab[3,3] = -1j*h_field
    #
    final_N = np.zeros((4*m,4*m,K_points,K_points), dtype=complex)
    final_N[:2*m,:2*m] = Naa
    final_N[2*m:,2*m:] = Naa
    final_N[2*m:,:2*m] = Nab
    final_N[:2*m,2*m:] = Nab
    #################################### L
    for i in range(4*m):
       final_N[i,i] += L
    return final_N

def compute_KM(k_grid,a1,a2):
    """
    Compute the product exp(i k . a) of momenta and lattice directions
    """
    ka1 = np.exp(1j*np.tensordot(a1,k_grid,axes=1));   ka1_ = np.conjugate(ka1);
    ka2 = np.exp(1j*np.tensordot(a2,k_grid,axes=1));   ka2_ = np.conjugate(ka2);
    KM_ = (ka1,ka1_,ka2,ka2_)
    return KM_

##################################################################
################################################################## 
##################################################################
##################################################################
"""
Filenames and I/O functions
"""
def get_res_final_fn(pars,Kx,Ky,machine):
    J_nn,h_field,delta,Spin = pars
    return get_res_final_dn(Kx,Ky,Spin,machine) + 'J_h=('+str(J_nn)+'_'+'{:5.4f}'.format(h_field).replace('.','')+').csv'

def get_res_final_dn(Kx,Ky,Spin,machine):
    return get_res_S_dn(Spin,machine)+'Kx_'+str(Kx)+'Ky_'+str(Ky)+'/'

def get_res_S_dn(Spin,machine):
    return get_res_dn(machine)+"S_"+"{:.4f}".format(Spin)+'/'

def get_res_dn(machine):
    return get_home_dn(machine)+'results/'

def get_home_dn(machine):
    if machine == 'loc':
        return '/home/dario/Desktop/git/MoireBands/last_lap/1_tight_binding/'
    elif machine == 'hpc':
        return '/home/users/r/rossid/1_tight_binding/'
    elif machine == 'maf':
        return '/users/rossid/1_tight_binding/'


def SaveToCsv(Data,csvfile):
    header = Data.keys()
    with open(csvfile,'a') as f:
        writer = csv.DictWriter(f, fieldnames = header)
        writer.writeheader()
        writer.writerow(Data)













