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
Here we compute the GS energy of XX+YY+delta*ZZ + staggered Z using Holstein-Primakof.
We have a 2-site UC always (except for FM and h=0).
We take J_nn=+1 or -1 to have AFM and FM case and vary the staggered field 'h' from high values to 0 (phase transition at h=2|J| from gapped (h>2|J|) to gapless (0<h<2|J|)).
We perform the HP transformation around the quantization axis defined by the classical orientation of the magnetization.
We compute the Hamiltonian and do the Bogoliubov triansformation (BT) numerically at each k of the BZ.
We do not consider the points where the Bogoliubov transformation is not possible, which should be only k=0.
"""
L = 100
Ns = L*L
nkx = L//2
nky = L
b1 = np.array([np.pi,np.pi])
b2 = np.array([0,2*np.pi])
gridk = np.zeros((nkx,nky,2))
Gamma = np.zeros((nkx,nky))
for i1 in range(nkx):
    for i2 in range(nky):
        d = 2*np.pi/L
        gx = -np.pi + d*(i1+i2//2+1)
        gy = i2//2*d - d*i1
        gridk[i1,i2] = np.array([gx,gy])
        Gamma[i1,i2] = np.cos(gridk[i1,i2,0]) + np.cos(gridk[i1,i2,1])
#
J_nn = -1        #J_nn = 1 -> AFM, J_nn = -1 -> FM
delta = 1       #parameter for ZZ
sign = np.sign(J_nn)
tit = "FM" if J_nn < 0 else "AFM"
S = 0.5
n_H = 26
H_list = np.linspace(0,2.5,n_H)
def fun_E0(J_nn,S,th,h):
#    return -2*abs(J_nn)*S*np.sin(th)**2*(S+1)-h*np.cos(th)*(S+1/2)
    return 2*S*J_nn*(-sign*np.sin(th)**2*(S+1)-2*delta*np.cos(th)**2) - h*np.cos(th)*(2*S+1/2)
def fun_th(h,J_nn,S):
    return 0 if h>2*abs(J_nn) else np.arccos(h/4/abs(J_nn)/S/(1+delta))

fn = "results/result_J"+"{:.3f}".format(J_nn)+"_d"+"{:.3f}".format(delta)+"_h"+"{:.3f}".format(H_list[0])+"-"+"{:.3f}".format(H_list[-1])+"_"+str(n_H)+"_nkx"+str(nkx)+"_nky"+str(nky)+".npy"
save = False
plot_temp = True
if not Path(fn).is_file() or plot_temp:
    w_h = np.zeros((n_H,nkx,nky,4))
    for ind_h in range(n_H):
        h = H_list[ind_h]
        ###
        th = 0 if (delta==1 and sign==1) else fun_th(h,J_nn,S)
        print("h: ","{:.2f}".format(h),", J: ",J_nn,", delta: ",delta,", theta: ",th/np.pi*180)
        #
        J_ = np.identity(4)
        J_[0,0] = J_[1,1] = -1
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
        for ikx in range(nkx):
            for iky in range(nky):
                try:
                    Ch = scipy.linalg.cholesky(Nk[:,:,ikx,iky])
#                    a = fakearg
                except:
                    print("One non-Bogoliubov point")
                    if 0:
                        w_h[ind_h,ikx,iky] = np.linalg.eigvalsh(Nk[:,:,ikx,iky])[:]
                    else:
                        for iii in range(w_h.shape[-1]):
                            w_h[ind_h,ikx,iky,iii] = np.nan
#                        w_h[ind_h,ikx,iky,1] = np.nan
                    continue
                w_h[ind_h,ikx,iky] = np.linalg.eigvalsh(Ch@J_@Ch.T.conj())      #last 2 eigvals are to be summed
        if plot_temp:   #plot w
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(projection='3d')
#            ax = fig.add_subplot(221,projection='3d')
            ax.plot_surface(gridk[:,:,0],gridk[:,:,1],w_h[ind_h,:,:,0],cmap=cm.plasma)
#            ax.set_title("theta="+"{:.2f}".format(th*180/np.pi)+'°',size=30)
#            ax = fig.add_subplot(222,projection='3d')
            ax.plot_surface(gridk[:,:,0],gridk[:,:,1],w_h[ind_h,:,:,1],cmap=cm.plasma)
#            ax = fig.add_subplot(223,projection='3d')
            ax.plot_surface(gridk[:,:,0],gridk[:,:,1],w_h[ind_h,:,:,2],cmap=cm.plasma)
#            ax = fig.add_subplot(224,projection='3d')
            ax.plot_surface(gridk[:,:,0],gridk[:,:,1],w_h[ind_h,:,:,3],cmap=cm.plasma)
            #
            ax.set_xlabel('Kx')
            ax.set_ylabel('Ky')
            ax.set_aspect('equalxy')
            ax.set_title(tit+r" ,$\Delta=$"+str(delta),size=30)
            fig.tight_layout()
            plt.show()
            exit()
    if save:
        np.save(fn,w_h)
else:
    w_h = np.load(fn)

#
fig = plt.figure()#figsize=(20,20))
ax = fig.add_subplot()
tot_E = np.zeros(n_H)
gap = np.zeros(n_H)
for ind_h in range(n_H):
    h = H_list[ind_h]
    th = fun_th(h,J_nn,S)
    wh = w_h[ind_h,:,:,2:]
    tot_E[ind_h] = fun_E0(J_nn,S,th,h)+np.sum(wh[~np.isnan(wh)])/Ns/2
    val = w_h[ind_h,:,:,1]      #valence band
    con = w_h[ind_h,:,:,2]      #conduction band
    gap[ind_h] = np.min(con[~np.isnan(con)])-np.max(val[~np.isnan(val)])

l1 = ax.plot(H_list,tot_E,'k',marker='*',label='energy')
ax.set_ylabel("Energy")

ax_r = ax.twinx()
l2 = ax_r.plot(H_list,gap,'g',marker='^',label='gap')
ax_r.set_ylabel("gap")

ax.xaxis.set_inverted(True)

ax.set_title(tit+r", $S=$"+"{:.1f}".format(S)+r", $\Delta=$"+str(delta),size=20)
ax.set_xlabel(r'$h$')

labels = [l.get_label() for l in l1+l2]
ax.legend(l1+l2,labels,loc='center right')
plt.show()


















































