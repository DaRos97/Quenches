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
    opts, args = getopt.getopt(argv, "S:K:a:", ['DM=','j2=','j3=','Nq='])
    S = '50'
    DM = '000'
    ans = '15'
    K = '13'
    J2 = J3 = 0
    Nq = 17
except:
    print("Error")
for opt, arg in opts:
    if opt in ['-S']:
        S = arg
    if opt in ['-K']:
        K = arg
    if opt in ['-a']:
        ans = arg
    if opt == '--DM':
        DM = arg
    if opt == '--j2':
        J2 = float(arg)
    if opt == '--j3':
        J3 = float(arg)
    if opt == '--Nq':
        Nq = int(arg)

dirname = '../../Data/self_consistency/S'+S+'/phi'+DM+'/'+K+'/'
#dirname = '../../Data/self_consistency/test/'+K+'/'
filename = dirname+'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
savenameSFzz = "data_SF/LRO_SFzz_"+ans+'_'+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+S+DM+str(Nq)+'.npy'
savenameSFxy = "data_SF/LRO_SFxy_"+ans+'_'+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+S+DM+str(Nq)+'.npy'
command_plot = 'python plot_SF.py -S '+S+' -K '+K+' --DM '+DM+' -a '+ans+' --j2 '+str(J2)+' --j3 '+str(J3)+' --Nq '+str(Nq)

if not os.path.isfile(filename):
    print(J2,J3,ans," values are not valid or the point was not computed")
if os.path.isfile(savenameSFzz):
    os.system(command_plot)
    printed = True
else:
    printed = False
    print("Computing it ....")
#
if printed:
    exit()

########################################
########################################
print("Using arguments: ans-> ",ans," Dm angle = ",DM," spin S = ",S," J2 = ",J2," J3 = ",J3)
#import data from file
data = fs.import_data(ans,filename)
if J2:
    list_p = {'15':[(1,1),], '16':[(0,0),], '20':[(1,1),], '17':[(1,1),], '19':[(1,1),], '18':[(0,0),]}
    if J3:
        list_p = {'15':[(1,1),], '16':[(0,0),], '20':[(1,1,0,0),], '17':[(1,1),], '19':[(1,1,0,0),], '18':[(0,0),]}
elif J3:
    list_p = {'15':[(2,2),], '16':[(2,2),], '20':[(0,0),], '17':[(2,2),], '19':[(0,0),], '18':[(2,2),]}
else:
    list_p = {'15':[(2,2),], '16':[(2,2),], '20':[(2,2),], '17':[(2,2),], '19':[(2,2),], '18':[(2,2),]}
PpP = list_p[ans][0]
#for temp_p in range(int(ans[3])):
#    PpP.append(int(ans[4+temp_p]))
#compute the Ks of the minimum band
Nx = 13     #points for looking at minima in BZ
Ny = 13
#Arguments
J = (1,J2,J3)
S_dic = {'50':0.5,'36':(np.sqrt(3)-1)/2,'34':0.34,'30':0.3,'20':0.2}
S_val = S_dic[S]
DM_dic = {'000':0,'005':0.05,'104':np.pi/3,'209':np.pi*2/3}
DM_val = DM_dic[DM]
DM1 = DM_val;   DM2 = 0;    DM3 = 2*DM_val
t1 = np.exp(-1j*DM1);    t1_ = np.conjugate(t1)
t2 = np.exp(-1j*DM2);    t2_ = np.conjugate(t2)
t3 = np.exp(-1j*DM3);    t3_ = np.conjugate(t3)
Tau = (t1,t1_,t2,t2_,t3,t3_)
args = (Tau,S_val,J,ans,PpP)
K_,is_LRO = fs.find_minima(data,args,Nx,Ny)
if not is_LRO:
    print("Not LRO, there is a gap now")
    exit()
#Compute the M in those K and extract the relative columns
V,degenerate = fs.get_V(K_,data,args)
#construct the spin matrix for each sublattice and compute the coordinates
#of the spin at each lattice site
UC = 16          #even
p1 = 0 if ans in fs.ans_p0 else 1
m = fs.Mm[p1]
factor = 2 if m == 3 else 1
S_l = np.zeros((3,m,UC,UC)) #3 spin components, 6 sites in UC, ij coordinates of UC
#Pauli matrices
sigma = np.zeros((3,2,2),dtype = complex)
sigma[0] = np.array([[0,1],[1,0]])
sigma[1] = np.array([[0,-1j],[1j,0]])
sigma[2] = np.array([[1,0],[0,-1]])
a1 = np.array([1,0])
a2 = np.array([-1/factor,np.sqrt(3)/factor])
if degenerate:                   #FIX degeneracy with multi gap-closing points!!!!!!!!!!!!!!!!!
    if len(K_) > 1:
        print("Not supported multi-gap closing points with degeneracy")
        exit()
    K_.append(K_[0])
k1 = K_[0]
k2 = K_[-1]
v1 = V[0]/np.linalg.norm(V[0])
v2 = V[-1]/np.linalg.norm(V[-1])
#constants of modulo 1 (?) whic give the orientation of the condesate
c1 = (1)/np.sqrt(2)
c1_ = np.conjugate(c1)
c2 = (1)/np.sqrt(2)
c2_ = np.conjugate(c2)
c = [1j,1,1,1]
f = np.sqrt(3)/4
d = np.array([  [1/2,-1/4,1/4,0  ,-3/4,-1/4],
                [0  ,f   ,f  ,2*f,3*f ,3*f ]])
r = np.zeros((2,m,UC,UC))
for i in range(UC):
    for j in range(UC):
        R = i*a1 + j*a2
        for s in range(m):
            r_ = R# + d[:,s]
            r[:,s,i,j] = R + d[:,s]
            cond = np.zeros(2,dtype=complex)
            for xx in range(len(K_)):
                cond[0] += c[xx]*V[xx][s]/np.linalg.norm(V[xx])*np.exp(1j*np.dot(K_[xx],r_))
                cond[1] += np.conjugate(c[xx])*np.conjugate(V[xx][s+m])/np.linalg.norm(V[xx])*np.exp(-1j*np.dot(K_[xx],r_))
            #cond[0] = c1*v1[s]*np.exp(1j*np.dot(k1,r_)) + c2*v2[s]*np.exp(1j*np.dot(k2,r_))
            #cond[1] = c2_*np.conjugate(v2[m+s])*np.exp(-1j*np.dot(k2,r_)) + c1_*np.conjugate(v1[m+s])*np.exp(-1j*np.dot(k1,r_))
            for x in range(3):
                S_l[x,s,i,j] = np.real(1/2*np.dot(np.conjugate(cond.T),np.einsum('ij,j->i',sigma[x],cond)))
            S_l[:,s,i,j] /= np.linalg.norm(S_l[:,s,i,j])
#Plotting spin values
#plt.figure()
#plt.subplot(2,2,1)
#plt.title("S_x")
#plt.scatter(r[0].ravel(),r[1].ravel(),c = S[0].ravel(), cmap = cm.plasma)
#plt.colorbar()
#plt.subplot(2,2,2)
#plt.title("S_y")
#plt.scatter(r[0].ravel(),r[1].ravel(),c = S[1].ravel(), cmap = cm.plasma)
#plt.colorbar()
#plt.subplot(2,2,3)
#plt.title("S_z")
#plt.scatter(r[0].ravel(),r[1].ravel(),c = S[2].ravel(), cmap = cm.plasma)
#plt.colorbar()
#plt.show()
if 0:
    for i in range(1,4):
        for j in range(1,4):
            print(i,j,':')
            for m in range(3):
                print(S_l[:,m,i,j])
    print("Chirality single triangles: ")
    for i in range(1,2):
        for j in range(1,2):
            ch_u1 = np.dot(S_l[:,1,i,j],np.cross(S_l[:,2,i,j],S_l[:,3,i,j]))      #up
            ch_u2 = np.dot(S_l[:,4,i,j],np.cross(S_l[:,5,i,j],S_l[:,0,i,j+1]))    #up
            ch_d1 = np.dot(S_l[:,0,i,j],np.cross(S_l[:,1,i+1,j],S_l[:,2,i,j]))      #down
            ch_d2 = np.dot(S_l[:,3,i,j],np.cross(S_l[:,4,i+1,j],S_l[:,5,i,j]))      #down
            Ch_r = np.dot(S_l[:,2,i-1,j],np.cross(S_l[:,3,i,j],S_l[:,4,i,j]))      #right
            Ch_l = np.dot(S_l[:,1,i,j],np.cross(S_l[:,5,i,j],S_l[:,3,i-1,j]))      #left
            CH_a = np.dot(S_l[:,1,i+1,j],np.cross(S_l[:,3,i,j],S_l[:,2,i,j]))      #2nn
            CH_b = np.dot(S_l[:,2,i,j],np.cross(S_l[:,4,i+1,j],S_l[:,3,i,j]))      #2nn
            print("Small triangles up ",i,",",j,": ",ch_u1,"\t",ch_u2)
            print("Small triangles down ",i,",",j,": ",ch_d1,"\t",ch_d2)
            print("Big triangles right and left ",i,",",j,": ",Ch_r,"\t",Ch_l)
            print("Big (2nn) triangles a and b ",i,",",j,": ",CH_a,"\t",CH_b)
#exit()
savenameS = "data_SpinOrientations/S_"+ans+'_'+str(J2).replace('.','')+'_'+str(J3).replace('.','')+'.npy'
np.save(savenameS,S_l)
print("Spins computed, now compute the spin structure factor")
#
#Now compute the SSF
Kx = Nq     #point to compute the SSF in the EBZ
Ky = Kx
kxg = np.linspace(-8*np.pi/3,8*np.pi/3,Kx)
kyg = np.linspace(-4*np.pi/np.sqrt(3),4*np.pi/np.sqrt(3),Ky)
K = np.zeros((2,Kx,Ky))
SFzz = np.zeros((Kx,Ky))
SFxy = np.zeros((Kx,Ky))
for nn in tqdm(range(Kx**2)):
    i = nn//Kx
    j = nn%Kx
    K[:,i,j] = np.array([kxg[i],kyg[j]])
    if not fs.EBZ(K[:,i,j]):
        SFxy[i,j], SFzz[i,j] = ('nan','nan')
        continue
    SFxy[i,j], SFzz[i,j] = fs.SpinStructureFactor(K[:,i,j],S_l,UC,m)

np.save(savenameSFzz,SFzz)
np.save(savenameSFxy,SFxy)
print("Finished")
##################################
print("\n\nPlotting...")

os.system(command_plot)

































