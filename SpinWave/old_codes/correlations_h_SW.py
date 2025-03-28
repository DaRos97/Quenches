import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
from matplotlib import cm
import sys
from tqdm import tqdm
import scipy
from pathlib import Path

"""
Here we compute the correlations SzSz for all h.
"""
L = 6 if len(sys.argv)<2 else int(sys.argv[1])
Ns = L*L
nkx = L//2
nky = L
a1 = np.array([2,0])
a2 = np.array([-1,1])
b1 = np.array([np.pi,np.pi])
b2 = np.array([0,2*np.pi])
#BZ of 2 site unit cell
gridk = np.zeros((nkx,nky,2))
Gamma = np.zeros((nkx,nky))
ind_mk = np.zeros((nkx,nky,2),dtype=int)
for i1 in range(nkx):
    for i2 in range(nky):
        d = 2*np.pi/L
        gx = -np.pi + d*(1+i1+i2//2)
        gy = d*((1+i2)//2-i1)
        gridk[i1,i2] = np.array([gx,gy])
        Gamma[i1,i2] = np.cos(gridk[i1,i2,0]) + np.cos(gridk[i1,i2,1])
        #
for ikx in range(nkx):
    for iky in range(nky):
        try:    #get indexes of -k
            new_ikx,new_iky = np.argwhere(np.all(abs(gridk+gridk[ikx,iky])<1e-7,axis=2))[0]
        except: #-k is related to k by b1 or b2
            new_ikx,new_iky = (ikx,iky)
        ind_mk[i1,i2] = np.array([new_ikx,new_iky])
#
J_nn = 1        #J_nn = 1 -> AFM, J_nn = -1 -> FM
tit = "FM" if J_nn < 0 else "AFM"
delta = 1       #parameter for ZZ
sign = np.sign(J_nn)
S = 0.5
dx = 6
n_T = 3
n_H = 26
H_list = np.linspace(0,2.5,n_H)
indh = 1
list_t = np.linspace(0,20,n_T)
J_ = np.identity(4)
J_[0,0] = -1
J_[1,1] = -1
#
def fun_E0(J_nn,S,th,h):
    return 2*S*J_nn*(-sign*np.sin(th)**2*(S+1)-2*delta*np.cos(th)**2) - h*np.cos(th)*(2*S+1/2)
def fun_th(h,J_nn,S):
    return 0 if h>2*abs(J_nn) else np.arccos(h/4/abs(J_nn)/S/(1+delta))
def get_index_minus_k(ikx,iky,gridk):
    input()

fn = "results/correlations_h_J"+"{:.1f}".format(J_nn)+"_d"+"{:.1f}".format(delta)+"_t"+"{:.3f}".format(list_t[0])+"-"+"{:.3f}".format(list_t[-1])+"_"+str(len(list_t))+"_L"+str(L)+".npy"
save = False
plot_temp = True
if not Path(fn).is_file() or plot_temp:
    w_h = np.zeros((nkx,nky,2))
    h = H_list[indh]
    ###
    th = 0 if (delta==1 and sign==1) else fun_th(h,J_nn,S)
    print("h: ","{:.2f}".format(h),", J: ",J_nn,", delta: ",delta,", theta: ",th/np.pi*180)
    #
    p1 = 2*J_nn*S*(sign*np.sin(th)**2+delta*np.cos(th)**2)*np.ones((nkx,nky)) + h*np.cos(th)/2
    p2 = Gamma*J_nn*S/2*(-np.cos(th)**2-sign*delta*np.sin(th)**2+1)
    p3 = Gamma*J_nn*S/2*(-np.cos(th)**2-sign*delta*np.sin(th)**2-1)
    z = np.zeros((nkx,nky))
    Nk = np.array([
        [p1,p2,z,p3],
        [p2,p1,p3,z],
        [z,p3,p1,p2],
        [p3,z,p2,p1]
    ])
    Mk = np.zeros((4,4,nkx,nky))
    for ikx in range(nkx):
        for iky in range(nky):
            try:
                Ch = scipy.linalg.cholesky(Nk[:,:,ikx,iky],lower=False)
            except:
                print("One non-Bogoliubov point")
                w_h[ikx,iky] = np.zeros(w_h.shape[-1])*np.nan
                Mk[:,:,ikx,iky] = np.zeros((Mk.shape[0],Mk.shape[1]))*np.nan
                continue
            w0,U = np.linalg.eigh(Ch@J_@Ch.T.conj())
            w_h[ikx,iky] = w0[2:]
            temp = np.diag(np.sqrt(J_@w0))
            Mk[:,:,ikx,iky] = scipy.linalg.inv(Ch)@U@temp
#                Mk[:,:,ikx,iky] = Mk[:,:,ikx,iky].T.conj()
    #Correlations
    corr = np.zeros((dx,len(list_t)),dtype=complex)
    #0th part
    d = np.zeros((2,nkx,nky))       #A or B, kx,ky
    for ikx in range(nkx):
        for iky in range(nky):
            d[0,ikx,iky] = Mk[2,0,ikx,iky]*Mk[0,2,ikx,iky]+Mk[2,1,ikx,iky]+Mk[0,3,ikx,iky]
            d[1,ikx,iky] = Mk[3,0,ikx,iky]*Mk[1,2,ikx,iky]+Mk[3,1,ikx,iky]+Mk[1,3,ikx,iky]
    c0 = np.zeros(2)        #j in A or B
    c0[0] = (S*np.cos(th)-np.cos(th)/Ns*2*np.sum(d[0]))*(S*np.cos(th)-np.cos(th)/Ns*2*np.sum(d[0]))
    c0[1] = (S*np.cos(th)-np.cos(th)/Ns*2*np.sum(d[0]))*(-S*np.cos(th)+np.cos(th)/Ns*2*np.sum(d[1]))
    for indt in range(len(list_t)):
        print("t = ",list_t[indt])
        t = list_t[indt]
        for idx in range(dx):
            c1 = 0
            sign1 = 1 if idx%2 == 0 else sign
            c2 = 0
            sign2 = 1 if idx%2 == 0 else -1
            for ikx in range(nkx):
                for iky in range(nky):
                    if np.linalg.norm(gridk[ikx,iky])<1e-4: #skip k=0
                        continue
                    m1 = Mk[0+idx%2,2,ikx,iky] + Mk[2+idx%2,2,ikx,iky]
                    m2 = Mk[0+idx%2,3,ikx,iky] + Mk[2+idx%2,3,ikx,iky]
                    mikx,miky = ind_mk[ikx,iky]
                    m3 = Mk[0,0,mikx,miky] + Mk[2,0,mikx,miky]
                    m4 = Mk[0,1,mikx,miky] + Mk[2,1,mikx,miky]
                    c1 += np.exp(1j*np.dot(gridk[ikx,iky],np.array([-idx,0])))*(m1*m3*np.exp(-1j*w_h[mikx,miky,0]*t)+m2*m4*np.exp(-1j*w_h[mikx,miky,1]*t))
                    for ikx2 in range(nkx):
                        for iky2 in range(nky):
                            if np.linalg.norm(gridk[ikx2,iky2])<1e-4:   #skip k=0
                                continue
                            mikx2,miky2 = ind_mk[ikx2,iky2]
                            c2 += np.exp(1j*np.dot(gridk[ikx,iky]-gridk[ikx2,iky2],np.array([idx,0])))*(
                                  np.exp(-1j*t*(w_h[ikx,iky,0]+w_h[mikx2,miky2,0]))*Mk[2+idx%2,2,mikx,miky]*Mk[0+idx%2,2,ikx2,iky2]*(Mk[2,0,ikx,iky]*Mk[0,0,mikx2,miky2]+Mk[2,0,mikx2,miky2]*Mk[0,0,ikx,iky])
                                + np.exp(-1j*t*(w_h[ikx,iky,1]+w_h[mikx2,miky2,1]))*Mk[2+idx%2,3,mikx,miky]*Mk[0+idx%2,3,ikx2,iky2]*(Mk[2,1,ikx,iky]*Mk[0,1,mikx2,miky2]+Mk[2,1,mikx2,miky2]*Mk[0,1,ikx,iky])
                                + np.exp(-1j*t*(w_h[ikx,iky,0]+w_h[mikx2,miky2,1]))*Mk[2+idx%2,2,mikx,miky]*Mk[0+idx%2,3,ikx2,iky2]*(Mk[2,0,ikx,iky]*Mk[0,1,mikx2,miky2]+Mk[2,1,mikx2,miky2]*Mk[0,0,ikx,iky])
                                + np.exp(-1j*t*(w_h[ikx,iky,1]+w_h[mikx2,miky2,0]))*Mk[2+idx%2,3,mikx,miky]*Mk[0+idx%2,2,ikx2,iky2]*(Mk[2,1,ikx,iky]*Mk[0,0,mikx2,miky2]+Mk[2,0,mikx2,miky2]*Mk[0,1,ikx,iky])
                            )
            corr[idx,indt] = c0[idx%2] + sign1*np.sin(th)**2*S/Ns*c1 + sign2*np.cos(th)**2/Ns**2*4*c2
    if save:
        np.save(fn,corr)
else:
    corr = np.load(fn)

print("Correlator at same time and position: ",corr[0,0])

#Plot
X,Y = np.meshgrid(list_t,np.arange(dx))
X = X.flatten()
Y = Y.flatten()
#
fig = plt.figure(figsize=(10,6))
#
ax = fig.add_subplot(121)
colors = np.real(corr).flatten()
sc = ax.scatter(X,Y,c=colors,marker='s',s=100,cmap=cm.plasma)
cbar = fig.colorbar(sc)
ax.set_title("Real part",size=20)
ax.set_xlabel("time",size=20)
ax.set_ylabel("x-distance",size=20)
ax.tick_params(axis='both', which='major', labelsize=20)
cbar.ax.tick_params(labelsize=20)
#ax.tick_params(axis='both', which='major', labelsize=10)
#
ax = fig.add_subplot(122)
colors = np.imag(corr).flatten()
sc = ax.scatter(X,Y,c=colors,marker='s',s=100,cmap=cm.plasma)
cbar = fig.colorbar(sc)
ax.set_title("Imaginary part",size=20)
ax.set_xlabel("time",size=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_yticks([])
cbar.ax.tick_params(labelsize=20)
#
plt.suptitle(tit+r", $\Delta=$"+str(delta),size=30)

fig.tight_layout()
plt.show()





















