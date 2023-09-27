import numpy as np
import functions as fs
import matplotlib.pyplot as plt
from matplotlib import cm
import getopt
import sys
import os
from tqdm import tqdm

#input arguments
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "N:S:K:", ['disp','Nq='])
    S = '50'
    K = 30          #cuts in BZ of minimization
    Nq = 30         #number of points to evaluate the SF in
    N = 0           #index of h value --> not needed
    disp = True
except:
    print("Error")
for opt, arg in opts:
    if opt in ['-S']:
        S = arg
    if opt in ['-K']:
        K = int(arg)
    if opt in ['-N']:
        N = int(arg)
    if opt == '--disp':
        disp = True
    if opt == '--Nq':
        Nq = int(arg)

#Parameters
J_nn = 1
hi = 10
hf = 0
hpts = 100
H = []
for i in range(hpts):
    H.append(hi+(hf-hi)/(hpts-1)*i)
h = H[N]
#
Nx = Nq     #SSF points to compute in BZ (Q)
Ny = Nx
dirname = '../Data/S'+S+'/'+str(K)+'/'
filename = dirname + 'J_h=('+str(J_nn)+'_'+'{:5.4f}'.format(h).replace('.','')+').csv'
savenameSFzz = "data_SF/SFzz_J1_"+str(J_nn)+'_h_'+'{:5.4f}'.format(h).replace('.','')+'_S_'+S+'_Nq_'+str(Nq)+'.npy'
savenameSFxy = "data_SF/SFxy_J1_"+str(J_nn)+'_h_'+'{:5.4f}'.format(h).replace('.','')+'_S_'+S+'_Nq_'+str(Nq)+'.npy'
command_plot = 'python plot_SF.py -S '+str(S)+' -K '+str(K)+' -N '+str(N) + ' --Nq '+str(Nq)

if not os.path.isfile(filename):
    print(J_nn,h,K," values are not valid or the point was not computed")
if os.path.isfile(savenameSFxy):
    os.system(command_plot)
    printed = True
else:
    printed = False
#
if printed:
    exit()
#
print("Using arguments: h = ",h,", S = ",S,", K = ",K)
#import data from file
params = fs.import_data(filename)
#Arguments
J = (J_nn,h)
S_dic = {'50':0.5,'36':(np.sqrt(3)-1)/2,'34':0.34,'30':0.3,'20':0.2}
S_val = S_dic[S]
#
args = (S_val,J)
#######################################################################
Kx = 30     #points for summation over BZ
Ky = 30
######
D = np.array([  [0,0],
                [0,1]])

Kxg = np.linspace(0,2*np.pi,Kx)
Kyg = np.linspace(0,np.pi,Ky)

##
Qxg = np.linspace(0,2*np.pi,Nx)
Qyg = np.linspace(0,np.pi,Ny)

#Result store
SFzz = np.zeros((Nx,Ny))
SFxy = np.zeros((Nx,Ny))
#
m = 2
#Compute Xi(Q) for Q in BZ
for xx in tqdm(range(Nx*Ny)):
    ii = xx//Nx
    ij = xx%Ny
    Q = np.array([Qxg[ii],Qyg[ij]])
    #
    delta = np.zeros((m,m),dtype=complex)
    for u in range(m):
        for g in range(m):
            delta[u,g] = np.exp(1j*np.dot(Q,D[:,g]-D[:,u]))
    #
    resxy = 0
    #summation over BZ
    for x in range(Kx*Ky):
        i = x//Kx
        j = x%Ky
        #
        K__ = np.array([Kxg[i],Kyg[j]])
        U1,X1,V1,Y1 = fs.M(K__,params,args)
        U2,X2,V2,Y2 = fs.M(-K__,params,args)
        U3,X3,V3,Y3 = fs.M(Q+K__,params,args)
        U4,X4,V4,Y4 = fs.M(-K__-Q,params,args)
        ##############################################
        temp1 = np.einsum('ua,ga->ug',np.conjugate(X1),X1) * np.einsum('ua,ga->ug',np.conjugate(Y4),Y4)
        temp2 = np.einsum('ua,ga->ug',np.conjugate(X1),Y1) * np.einsum('ua,ga->ug',np.conjugate(Y4),X4)
        temp3 = np.einsum('ua,ga->ug',V2,np.conjugate(V2)) * np.einsum('ua,ga->ug',U3,np.conjugate(U3))
        temp4 = np.einsum('ua,ga->ug',V2,np.conjugate(U2)) * np.einsum('ua,ga->ug',U3,np.conjugate(V3))
        temp = (temp1 + temp2 + temp3 + temp4) * delta
        resxy += temp.ravel().sum()
    #
    SFxy[ii,ij] = np.real(resxy)/(Kx*Ky)
#
#np.save(savenameSFzz,SFzz)
np.save(savenameSFxy,SFxy)
print("Finished computing, now plotting ...")

os.system(command_plot)




















