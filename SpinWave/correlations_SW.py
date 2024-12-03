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
Here we compute the correlations.
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
for i1 in range(nkx):
    for i2 in range(nky):
        d = 2*np.pi/L
        gx = -np.pi + d*(1+i1+i2//2)
        gy = d*((1+i2)//2-i1)
        gridk[i1,i2] = np.array([gx,gy])
        Gamma[i1,i2] = np.cos(gridk[i1,i2,0]) + np.cos(gridk[i1,i2,1])
if 0:
    fig = plt.figure()
    ax = fig.add_subplot()
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
    ax.scatter(gridk[:,:,0],gridk[:,:,1])
    plt.show()
    exit()
#
J_nn = 1        #J_nn = 1 -> AFM, J_nn = -1 -> FM
tit = "FM" if J_nn < 0 else "AFM"
delta = 0       #parameter for ZZ
sign = np.sign(J_nn)
S = 0.5
n_H = 26
dx = 6
H_list = np.linspace(0,2.5,n_H)
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

fn = "results/correlations_J"+"{:.3f}".format(J_nn)+"_d"+"{:.3f}".format(delta)+"_h"+"{:.3f}".format(H_list[0])+"-"+"{:.3f}".format(H_list[-1])+"_"+str(n_H)+"_nkx"+str(nkx)+"_nky"+str(nky)+".npy"
save = False
plot_temp = True
if not Path(fn).is_file() or plot_temp:
    w_h = np.zeros((n_H,nkx,nky,2))
    for ind_h in range(n_H):
        h = H_list[ind_h]
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
                    w_h[ind_h,ikx,iky] = np.zeros(w_h.shape[-1])*np.nan
                    Mk[:,:,ikx,iky] = np.zeros((Mk.shape[0],Mk.shape[1]))*np.nan
                    continue
                w0,U = np.linalg.eigh(Ch@J_@Ch.T.conj())
                w_h[ind_h,ikx,iky] = w0[2:]
                temp = np.diag(np.sqrt(J_@w0))
                Mk[:,:,ikx,iky] = scipy.linalg.inv(Ch)@U@temp
#                Mk[:,:,ikx,iky] = Mk[:,:,ikx,iky].T.conj()
        if 0:#plot_temp:   #plot w
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(projection='3d')
#            ax = fig.add_subplot(221,projection='3d')
            ax.plot_surface(gridk[:,:,0],gridk[:,:,1],w_h[ind_h,:,:,0],cmap=cm.plasma)
#            ax.set_title("theta="+"{:.2f}".format(th*180/np.pi)+'°',size=30)
#            ax = fig.add_subplot(222,projection='3d')
            ax.plot_surface(gridk[:,:,0],gridk[:,:,1],w_h[ind_h,:,:,1],cmap=cm.plasma)
#            ax = fig.add_subplot(223,projection='3d')
#            ax.plot_surface(gridk[:,:,0],gridk[:,:,1],w_h[ind_h,:,:,2],cmap=cm.plasma)
#            ax = fig.add_subplot(224,projection='3d')
#            ax.plot_surface(gridk[:,:,0],gridk[:,:,1],w_h[ind_h,:,:,3],cmap=cm.plasma)
            #
            ax.set_xlabel('Kx')
            ax.set_ylabel('Ky')
            ax.set_aspect('equalxy')
            ax.set_title(r" ,$\Delta=$"+str(delta),size=30)
            fig.tight_layout()
            plt.show()
            exit()
        if 0:#plot Mk
            for i2 in range(4):
                fig = plt.figure(figsize=(20,20))
                ax = fig.add_subplot(projection='3d')
                for i1 in range(1):
                    ax.plot_surface(gridk[:,:,0],gridk[:,:,1],Mk[i1,i2],cmap=cm.plasma)
                plt.show()
            exit()
        #Correlations
        list_t = np.linspace(0,20,101)
        corr = np.zeros((dx,len(list_t)),dtype=complex)
        for indt in range(len(list_t)):
            print("t = ",list_t[indt])
            t = list_t[indt]
            for idx in range(dx):
#                print("x = ",idx)
                for ikx in range(nkx):
                    for iky in range(nky):
                        if np.linalg.norm(gridk[ikx,iky])<1e-4:
#                            print("K=0 point at coordinates ",ikx,iky)
                            continue
                        m1 = Mk[0+idx%2,2,ikx,iky] + Mk[2+idx%2,2,ikx,iky]
                        m2 = Mk[0+idx%2,3,ikx,iky] + Mk[2+idx%2,3,ikx,iky]
                        if 0:
                            try:    #get indexes of -k
                                new_ikx,new_iky = np.argwhere(np.all(abs(gridk+gridk[ikx,iky])<1e-7,axis=2))[0]
                            except: #-k is related to k by b1 or b2
                                new_ikx,new_iky = (ikx,iky)
                        else:
                            new_ikx,new_iky = (ikx,iky)
                        m3 = Mk[0,0,new_ikx,new_iky] + Mk[2,0,new_ikx,new_iky]
                        m4 = Mk[0,1,new_ikx,new_iky] + Mk[2,1,new_ikx,new_iky]
                        corr[idx,indt] += np.exp(1j*np.dot(gridk[ikx,iky],np.array([-idx,0])))*(m1*m3*np.exp(-1j*w_h[ind_h,new_ikx,new_iky,0]*t)+m2*m4*np.exp(-1j*w_h[ind_h,new_ikx,new_iky,1]*t))
                if idx == 0 and indt == 0:
                    print("Correlator at same time and position: ",corr[idx,indt]/Ns/2)
                    #exit()
        #
        corr /= Ns*2

        #
        X,Y = np.meshgrid(list_t,np.arange(dx))
        X = X.flatten()
        Y = Y.flatten()
        #
        fig = plt.figure(figsize=(20,20))
        #
        ax = fig.add_subplot(121)
        colors = np.real(corr).flatten()
        sc = ax.scatter(X,Y,c=colors,marker='s',s=100,cmap=cm.plasma)
        fig.colorbar(sc)
        ax.set_title("Real part",size=30)
        ax.set_xlabel("time",size=30)
        ax.set_ylabel("x-distance",size=30)
        #
        ax = fig.add_subplot(122)
        colors = np.imag(corr).flatten()
        sc = ax.scatter(X,Y,c=colors,marker='s',s=100,cmap=cm.plasma)
        fig.colorbar(sc)
        ax.set_title("Imaginary part",size=30)
        ax.set_xlabel("time",size=30)
        ax.set_ylabel("x-distance",size=30)
        #
        plt.suptitle(tit+r", $\Delta=$"+str(delta),size=30)
        plt.show()

        exit()




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














