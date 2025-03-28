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
We take J_1=+1 or -1 to have AFM and FM nn case and vary the staggered field 'h' from high values to 0.
There is a phase transition at h=2|J_1| from gapped (h>2|J_1|) to gapless (0<h<2|J_1|).
We perform the HP transformation around the quantization axis defined by the classical orientation of the magnetization.
We compute the Hamiltonian and do the Bogoliubov triansformation numerically at each k of the BZ.
We do not consider the points where the Bogoliubov transformation is not possible, which should be only k=0 (gap closing point).
"""
L = 102
Ns = L*L    #number of sites
nkx = L//2      #number of unit cells (momentum points) in x and y direction
nky = L
b1 = np.array([np.pi,np.pi])
b2 = np.array([0,2*np.pi])
gridk = np.zeros((nkx,nky,2))  #BZ of 2-site UC
Gamma = np.zeros((nkx,nky))     #
for i1 in range(nkx):
    for i2 in range(nky):
        d = 2*np.pi/L
        gx = -np.pi + d*(i1+i2//2+1)
        gy = i2//2*d - d*i1
        gridk[i1,i2] = np.array([gx,gy])
        Gamma[i1,i2] = np.cos(gridk[i1,i2,0]) + np.cos(gridk[i1,i2,1])
#
J_1 = 1        #J_1 = 1 -> AFM, J_1 = -1 -> FM
delta = 0       #parameter for ZZ
#
sign = np.sign(J_1)
tit = "FM" if J_1 < 0 else "AFM"
S = 0.5     #spin value
n_H = 3#26    #number of h-field points
#H_list = np.linspace(0,2.5,n_H)
H_list = np.linspace(1.5,2.5,n_H)
def fun_E0(J_1,S,th,h):
    """Energy coming from 0 order terms in HP expansion"""
    return 2*S*J_1*(-sign*np.sin(th)**2*(S+1)-2*delta*np.cos(th)**2) - h*np.cos(th)*(2*S+1/2)
def fun_th(h,J_1,S):
    """Angle theta of quantization axis given by classical configuration"""
    return 0 if h>2*abs(J_1) else np.arccos(h/4/abs(J_1)/S/(1+delta))

energy_fn = "results/energy_J1_"+"{:.3f}".format(J_1)+"_delta_"+"{:.3f}".format(delta)+"_h_"+"{:.3f}".format(H_list[0])+"-"+"{:.3f}".format(H_list[-1])+"_"+str(n_H)+"_nkx"+str(nkx)+"_nky"+str(nky)+".npy"
bogTrsf_fn = "results/bogTrsf_J1_"+"{:.3f}".format(J_1)+"_delta_"+"{:.3f}".format(delta)+"_h_"+"{:.3f}".format(H_list[0])+"-"+"{:.3f}".format(H_list[-1])+"_"+str(n_H)+"_nkx"+str(nkx)+"_nky"+str(nky)+".npy"
save = True
plot_temp = True#False
#Matrix for Bogoliubov
J_ = np.identity(4)
J_[0,0] = J_[1,1] = -1
if not Path(energy_fn).is_file() or plot_temp:
    w_h = np.zeros((n_H,nkx,nky))
    Mk_h = np.zeros((n_H,2,2,nkx,nky))    #Transformation matrix for Bogoliubov trsf
    for ind_h in range(n_H):
        h = H_list[ind_h]
        ###
        th = 0 if (delta==1 and sign==1) else fun_th(h,J_1,S)
        print("h: ","{:.2f}".format(h),", J: ",J_1,", delta: ",delta,", theta: ",th/np.pi*180)
        #
        p1 = 2*J_1*S*(sign*np.sin(th)**2+delta*np.cos(th)**2)*np.ones((nkx,nky)) + h*np.cos(th)/2
        p2 = -sign*Gamma*J_1*S/2*(-np.cos(th)**2-sign*delta*np.sin(th)**2+1)
        p3 = Gamma*J_1*S/2*(-np.cos(th)**2-sign*delta*np.sin(th)**2-1)
        #
        w_h[ind_h] = np.sqrt(abs((p1+p2+p3)*(p1+p2-p3)))
        for ikx in range(nkx):
            for iky in range(nky):
                if abs(w_h[ind_h,ikx,iky])<1e-7:  #skip gap closing points
                    #print("Gapless at k=",gridk[ikx,iky])
                    continue
                tanh = -p3[ikx,iky]/(p1[ikx,iky]+p2[ikx,iky])
                if abs(tanh)<1:
                    rk = 1/2*np.arctanh(tanh)
                    Mk_h[ind_h,:,:,ikx,iky] = np.array([[np.cosh(rk),np.sinh(rk)],[np.sinh(rk),np.cosh(rk)]])
        if plot_temp:   #plot w
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(projection='3d')
            ax.plot_surface(gridk[:,:,0],gridk[:,:,1],w_h[ind_h,:,:],cmap=cm.plasma)
            #
            ax.set_xlabel('Kx')
            ax.set_ylabel('Ky')
            ax.set_aspect('equalxy')
            ax.set_title(tit+r" ,$\Delta=$"+str(delta)+', h='+"{:.3f}".format(h),size=30)
            fig.tight_layout()
            plt.show()
    if save:
        np.save(energy_fn,w_h)
        np.save(bogTrsf_fn,Mk_h)
else:
    w_h = np.load(energy_fn)
    Mk_h = np.load(bogTrsf_fn)
#
tot_E = np.zeros(n_H)
gap = np.zeros(n_H)
th_h = np.zeros(n_H)
for ind_h in range(n_H):
    h = H_list[ind_h]
    th_h[ind_h] = fun_th(h,J_1,S)
    tot_E[ind_h] = fun_E0(J_1,S,th_h[ind_h],h)+np.sum(w_h[ind_h])/Ns/2
    gap[ind_h] = np.min(w_h[ind_h])

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot()

#Q-angle
l2 = ax.plot(H_list,th_h,label=r"$\theta$ of quantization axis for GS",color='b',marker='*',ls='')
ax.set_yticks([np.pi/12*i for i in range(7)],["$0°=z-AFM$",r"$15°$",r"$30°$",r"$45°$",r"$60°$",r"$75°$",r"$90°=\hat{x}$"])
ax.yaxis.set_tick_params(labelsize=20,color='b')
ax.tick_params(axis='y',colors='b')

ax_r = ax.twinx()
#GS energy
l1 = ax_r.plot(H_list,tot_E,label="GS energy",color='r')
ax_r.yaxis.set_tick_params(labelsize=20)
ax_r.tick_params(axis='y',colors='r')

#Gap
ax_r = ax.twinx()
l3 = ax_r.plot(H_list,gap,color='g',label="Gap")
ax_r.tick_params(axis='y',colors='g')

#legend
labels = [l.get_label() for l in l1+l2+l3]
ax.legend(l1+l2+l3,labels,loc='center right')
#ax.xaxis.set_inverted(True)

ax.set_title(tit+r", $S=$"+"{:.1f}".format(S)+r", $\Delta=$"+str(delta),size=20)
ax.xaxis.set_tick_params(labelsize=20)
ax.set_xlabel(r"staggered field $h$",size=20)

plt.show()


















































