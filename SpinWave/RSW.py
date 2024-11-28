import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
from matplotlib import cm
from matplotlib.lines import Line2D
import sys
from tqdm import tqdm
import scipy
from pathlib import Path
import itertools

L = 4 if len(sys.argv)<2 else int(sys.argv[1])
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
for i1 in range(nkx):
    for i2 in range(nky):
        gridk[i1,i2] = i1*2/L*b1 + i2/L*b2
        Gamma[i1,i2] = np.cos(gridk[i1,i2,0]) + np.cos(gridk[i1,i2,1])
if 0:   #Plot real space and BZ points
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(121)
    ax.axis('off')
    ax.set_aspect('equal')
    d = -0.3
    ax.arrow(d,d,0,0.8,color='k',head_width=0.05)
    ax.arrow(d,d,0.8,0,color='k',head_width=0.05)
    ax.text(d+0.8,d+0.1,r'$x$',size=20)
    ax.text(d+0.1,d+0.8,r'$y$',size=20)
    for i in range(L):
        ax.plot([i,i],[0,L-1],color='k',lw=0.2,zorder=0)
        ax.plot([0,L-1],[i,i],color='k',lw=0.2,zorder=0)
    for i1 in range(nkx):
        for i2 in range(nky):
            vA = i1*a1 + i2*a2
            vB = i1*a1 + i2*a2 + np.array([1,0])
            if vA[0]<0:
                vA += L//2*a1
            if vB[0]<0:
                vB += L//2*a1
            ax.scatter(vA[0],vA[1],color='k',marker='o',s=70)
            ax.scatter(vB[0],vB[1],color='r',marker='o',s=70)
    ax.set_title("Real space",size=20)
    #
    ax = fig.add_subplot(122)
    ax.axis('off')
    ax.set_aspect('equal')
    f = 1.2
    v = 0.15
    ax.arrow(-np.pi*f,0,2*np.pi*f,0,color='k',head_width=0.1)
    ax.arrow(0,-np.pi*f,0,2*np.pi*f,color='k',head_width=0.1)
    ax.text(np.pi*f,v,r'$k_x$',size=20)
    ax.text(v,np.pi*f,r'$k_y$',size=20)
    ax.plot([-np.pi,0],[0,np.pi],color='orange',lw=1)
    ax.plot([0,np.pi],[np.pi,0],color='orange',lw=1)
    ax.plot([np.pi,0],[0,-np.pi],color='orange',lw=1)
    ax.plot([0,-np.pi],[-np.pi,0],color='orange',lw=1)
    for i1 in range(nkx):
        for i2 in range(nky):
            gx,gy = gridk[i1,i2]
            if gy>np.pi-gx:
                gx,gy = gridk[i1,i2]-b2
            if gy>np.pi-gx:
                gx,gy = (gx-b2[0],gy-b2[1])
            if gy<= -np.pi+gx:
                gx,gy = (gx-b1[0]+b2[0],gy-b1[1]+b2[1])
            ax.scatter(gx,gy,color='k',marker='o')

    ax.set_title("Brillouin zone",size=20)
#    plt.suptitle(r'$L=4$')
    fig.tight_layout()
    plt.show()
    exit()
#
J_nn = -1        #J_nn = 1 -> AFM, J_nn = -1 -> FM
delta = 0       #parameter for ZZ
sign = 1 if J_nn>0 else -1
tit = "FM" if J_nn < 0 else "AFM"
S = 0.5
n_H = 26
H_list = np.linspace(2.5,0,n_H)
def fun_E0(J_nn,S,th,h):
    """The part of GS energy that does not depend on the Bogoliubov dispersion."""
    return 2*J_nn*S*Ns*(-sign*np.sin(th)**2*(S+1)-2*delta*np.cos(th)**2) - h*Ns*np.cos(th)*(2*S+1/2) + J_nn*(1-delta)/2/(Ns-1)
def fun_th(h,J_nn,S):
    """Theta of classical state. For Delta different from 0 does not work, unless J is FM."""
    return 0 if h>2*abs(J_nn) else np.arccos(h/4/abs(J_nn)/S/(1+delta))

#No need to save, takes a second to compute
for ind_h in range(n_H-1,-1,-1):
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
    wh = np.zeros((nkx,nky,2))
    for ikx in range(nkx):
        for iky in range(nky):
            try:
                Ch = scipy.linalg.cholesky(Nk[:,:,ikx,iky])
            except:
                print("One non-Bogoliubov point")
                if 0:
                    wh[ikx,iky] = np.linalg.eigvalsh(Nk[:,:,ikx,iky])[:]
                else:
                    for iii in range(wh.shape[-1]):
                        wh[ikx,iky,iii] = np.nan
                continue
            wh[ikx,iky] = np.linalg.eigvalsh(Ch@J_@Ch.T.conj())[2:]      #last 2 eigvals are to be summed

    #Compute ll possible energies
    n_Mk = int(Ns*S+1)  if Ns<=16 else 8#number of possible Mk's
    E_gs = fun_E0(J_nn,S,th,h) + np.sum(wh[~np.isnan(wh)])/2
    #No spin waves
    E0 = np.zeros(n_Mk)
    for Mk in range(n_Mk):
        E0[Mk] = E_gs - 2*J_nn*(1-delta)/(Ns-1)*Mk**2
    #1 spin wave
    w1 = np.unique(wh.reshape(2*nkx*nky)[2:])
    E1=np.zeros((w1.shape[0],n_Mk))
    for ind in range(w1.shape[0]):
        for Mk in range(n_Mk):
            E1[ind,Mk] = E_gs - 2*J_nn*(1-delta)/(Ns-1)*Mk**2 + w1[ind]
    #2 spin waves
    w2 = []
    for i1 in range(w1.shape[0]):
        for i2 in range(i1,w1.shape[0]):
            w2.append(w1[i1]+w1[i2])
    E2 = np.zeros((len(w2),n_Mk))
    for ind in range(len(w2)):
        for Mk in range(n_Mk):
            E2[ind,Mk] = E_gs - 2*J_nn*(1-delta)/(Ns-1)*Mk**2 + w2[ind]
    #3 spin waves
    w3 = []
    for i1 in range(w1.shape[0]):
        for i2 in range(len(w2)):
            w3.append(w1[i1]+w2[i2])
    E3 = np.zeros((len(w3),n_Mk))
    for ind in range(len(w3)):
        for Mk in range(n_Mk):
            E3[ind,Mk] = E_gs - 2*J_nn*(1-delta)/(Ns-1)*Mk**2 + w3[ind]
    #Per-site energy
    E0 /= Ns
    E1 /= Ns
    E2 /= Ns
    E3 /= Ns
    #Plot
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot()
    list_Mk = np.array([Mk**2 for Mk in range(n_Mk)])

    ax.scatter(list_Mk,E0,color='b',marker='s',zorder=4)
    for Mk in range(n_Mk):
        ax.scatter(np.ones(E1.shape[0])*list_Mk[Mk],E1[:,Mk],color='orange',marker='o',zorder=3)
        ax.scatter(np.ones(E2.shape[0])*list_Mk[Mk],E2[:,Mk],color='r',marker='D',zorder=2)
        ax.scatter(np.ones(E3.shape[0])*list_Mk[Mk],E3[:,Mk],color='maroon',marker='p',zorder=1)

    ax.set_xlabel(r"$\langle (J^z)^2\rangle$",size=30)
    ax.set_ylabel("Energy",size=30)
    ax.set_title(tit+", h="+"{:.1f}".format(h)+", L="+str(L),size=30)

    #legend
    list_leg = [
        Line2D([0],[0],color='b',marker='s',lw=0,markersize=8),
        Line2D([0],[0],color='orange',marker='o',lw=0,markersize=8),
        Line2D([0],[0],color='r',marker='D',lw=0,markersize=8),
        Line2D([0],[0],color='maroon',marker='p',lw=0,markersize=8),
    ]
    list_labels = ['RSW - rotor','+ 1-SW','+ 2-SW','+ 3-SW']
    ax.legend(list_leg,list_labels,fontsize=16,loc='lower right')
    #
    fig_fn = 'results/figures/L'+str(L)+'/h='+"{:.1f}".format(h).replace('.',',')+'_3SW.png'
    fig.savefig(fig_fn)
    plt.close()





