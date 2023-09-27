import numpy as np
from scipy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib import cm

cutoff_solution = 1e-3
CO_phase = 1e-3
#
def import_data(filename):
    P = []
    done = False
    with open(filename, 'r') as f:
        lines = f.readlines()
    N = (len(lines)-1)//2 + 1
    for i in range(N):
        data = lines[i*2+1].split(',')
        head = lines[i*2].split(',')
        ans2 = data[0]
        head[-1] = head[-1][:-1]
        print('Gap value found: ',data[head.index('Gap')])
        for p in range(head.index('L'),len(data)):
            P.append(float(data[p]))
        return P

def M(K,P,args):
    m = 2
    J = np.zeros((2*m,2*m))
    for i in range(m):
        J[i,i] = -1
        J[i+m,i+m] = 1
    N = Nk(K,P,args)
    Ch = LA.cholesky(N) #upper triangular
    w,U = LA.eigh(np.dot(np.dot(Ch,J),np.conjugate(Ch.T)))
    w = np.diag(np.sqrt(np.einsum('ij,j->i',J,w)))
    Mk = np.dot(np.dot(LA.inv(Ch),U),w)
    return split(Mk,m,m) 
####
def split(array, nrows, ncols):
    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))
#
def Nk(K,P_,args):
    S,J_all = args
    J1,h = J_all
    m = 2
    a1 = (1,0)
    a2 = (0,2)
    ka1 = np.exp(1j*np.dot(a1,K));   ka1_ = np.conjugate(ka1);
    ka2 = np.exp(1j*np.dot(a2,K));   ka2_ = np.conjugate(ka2);
    L = P_[0]
    A,B = P_[1:]
    ################
    N = np.zeros((2*m,2*m), dtype=complex)
    ##################################### B
    N[0,0] = J1*B*(ka1+ka1_) + h/2
    N[1,1] = J1*B*(ka1+ka1_) - h/2
    N[2,2] = J1*B*(ka1+ka1_) - h/2
    N[3,3] = J1*B*(ka1+ka1_) + h/2
    #
    N[0,1] = J1*B*(1+ka2_)
    N[0,2] = J1*A*(ka1+ka1_)
    N[0,3] = J1*A*(1+ka2_)
    N[1,2] = J1*A*(1+ka2)
    N[1,3] = J1*A*(ka1+ka1_)
    N[2,3] = J1*B*(1+ka2_)
    for i in range(2*m):
        for j in range(i+1,2*m):
            N[j,i] += np.conjugate(N[i,j])
    #################################### L
    for i in range(2*m):
        N[i,i] += L
    return N
