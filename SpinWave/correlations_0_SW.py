import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
from matplotlib import cm
from matplotlib.colors import Normalize
import sys
from tqdm import tqdm
import scipy
from pathlib import Path
from time import time
import functions as fs

"""
Here we compute the correlations SzSz for h=0.
We compute the dynamical structure factor.
"""
L = 10 if len(sys.argv)<2 else int(sys.argv[1])
Ns = L*L
UC = 1
nkx = L//UC
nky = L
#Vectors
a1 = np.array([1,0]) if UC==1 else np.array([2,0])
a2 = np.array([0,1]) if UC==1 else np.array([-1,1])
b1 = np.array([2*np.pi,0]) if UC==1 else np.array([np.pi,np.pi])
b2 = np.array([0,2*np.pi])
#BZ
gridk = fs.BZgrid(nkx,nky,UC)
ind_path = np.zeros((3*L//2,2),dtype=int)     #indices of path G-X-M-G
lk = ind_path.shape[0]
for i in range(L//2):
    ind_path[i,0] = L//2-1 + i
    ind_path[i,1] = L//2-1
for i in range(L//2,L):
    ind_path[i,0] = L-1
    ind_path[i,1] = i-1
for i in range(L,3*L//2):
    ind_path[i,0] = L-1-(i-L)
    ind_path[i,1] = L-1-(i-L)
if 0:
    fig = plt.figure()
    ax = fig.add_subplot()
    for ik in range(lk):
        x,y = ind_path[ik]
        ax.scatter(gridk[x,y,0],gridk[x,y,1])
    plt.show()
    exit()
Gamma = np.cos(gridk[:,:,0]) + np.cos(gridk[:,:,1])
#Parameters
J_nn = 1 if len(sys.argv)<3 else int(sys.argv[2])        #J_nn = 1 -> AFM, J_nn = -1 -> FM
tit = "FM" if J_nn < 0 else "AFM"
delta = 1 if len(sys.argv)<4 else int(sys.argv[3])       #parameter for ZZ
sign = np.sign(J_nn)
S = 0.5
dx = 1#5
n_T = 1#31
list_t = np.linspace(0,10,n_T)
J_ = np.identity(2*UC)
for i in range(UC):
    J_[i,i] = -1
#
def fun_E0(J_nn,S,th,h):
    return 2*S*J_nn*(-sign*np.sin(th)**2*(S+1)-2*delta*np.cos(th)**2) - h*np.cos(th)*(2*S+1/2)
def fun_th(h,J_nn,S):
    return 0 if h>2*abs(J_nn) else np.arccos(h/4/abs(J_nn)/S/(1+delta))
#
fn = "results/dssf_J"+"{:.1f}".format(J_nn)+"_d"+"{:.1f}".format(delta)+"_t"+"{:.3f}".format(list_t[0])+"-"+"{:.3f}".format(list_t[-1])+"_"+str(len(list_t))+"_L"+str(L)+"_UC"+str(UC)+".npy"
save = False
plot_temp = True
if not Path(fn).is_file() or plot_temp:
    n_H = 5
    list_H = [0,1.5,1.9,2,3]
    for ih in range(n_H):
        h = list_H[ih]
        ###
        th = 0 if delta==1 else fun_th(h,J_nn,S)
        print("h: ","{:.2f}".format(h),", J: ",J_nn,", delta: ",delta,", theta: ",th/np.pi*180)
        #
        p1 = 2*J_nn*S*(sign*np.sin(th)**2+delta*np.cos(th)**2)*np.ones((nkx,nky)) + h*np.cos(th)/2
        p2 = Gamma*J_nn*S/2*(-np.cos(th)**2-sign*delta*np.sin(th)**2+1)
        p3 = Gamma*J_nn*S/2*(-np.cos(th)**2-sign*delta*np.sin(th)**2-1)
        #
        w_h = np.sqrt(abs((p1+p2+p3)*(p1+p2-p3)))
        Mk = np.zeros((2,2,nkx,nky))
        for ikx in range(nkx):
            for iky in range(nky):
                if abs(w_h[ikx,iky])<1e-7:  #skip gap closing points
    #                    Mk[:,:,ikx,iky] = np.zeros((2,2))*np.nan
                    continue
                if abs(p3[ikx,iky]/(p1[ikx,iky]+p2[ikx,iky]))<1:
                    rk = 1/2*np.arctanh(-p3[ikx,iky]/(p1[ikx,iky]+p2[ikx,iky]))
                    Mk[:,:,ikx,iky] = np.array([[np.cosh(rk),np.sinh(rk)],[np.sinh(rk),np.cosh(rk)]])
                else:
                    continue
        #Correlations
        uk = Mk[0,0]
        vk = Mk[0,1]
        #
        dssf = np.zeros((lk,nkx*nky,2))
        dssfx = np.zeros((lk,nkx*nky,2))
        for iQ in range(lk):
            iQx,iQy = ind_path[iQ]
            ie = 0
            dssf[iQ,ie,0] = np.sin(th)**2*S/2*(uk[iQx,iQy]+vk[iQx,iQy])**2        #dssf
            dssf[iQ,ie,1] = w_h[iQx,iQy]        #energy
            dssfx[iQ,ie,0] = np.cos(th)**2*S/2*(uk[iQx,iQy]+vk[iQx,iQy])**2        #dssf
            dssfx[iQ,ie,1] = w_h[iQx,iQy]        #energy
            ie += 1
            for ikx in range(nkx):
                for iky in range(nky):
                    temp_en = w_h[ikx,iky] + w_h[(ikx+iQx)%nkx,(iky+iQy)%nky]
                    temp_ssf = 1/Ns*np.cos(th)**2*uk[ikx,iky]*vk[(ikx+iQx)%nkx,(iky+iQy)%nky]*(vk[ikx,iky]*uk[(ikx+iQx)%nkx,(iky+iQy)%nky]+uk[ikx,iky]*vk[(ikx+iQx)%nkx,(iky+iQy)%nky])
                    temp_ssfx = 1/Ns*np.sin(th)**2*uk[ikx,iky]*vk[(ikx+iQx)%nkx,(iky+iQy)%nky]*(vk[ikx,iky]*uk[(ikx+iQx)%nkx,(iky+iQy)%nky]+uk[ikx,iky]*vk[(ikx+iQx)%nkx,(iky+iQy)%nky])
                    if (abs(temp_en-dssf[iQ,:,1])<1e-8).any():
                        ind_e = np.argwhere(abs(temp_en-dssf[iQ,:,1])<1e-8)
                        dssf[iQ,ind_e,0] += temp_ssf
                        dssfx[iQ,ind_e,0] += temp_ssf
                    else:
                        dssf[iQ,ie,0] += temp_ssf
                        dssf[iQ,ie,1] = temp_en
                        dssfx[iQ,ie,0] += temp_ssfx
                        dssfx[iQ,ie,1] = temp_en
                        ie += 1
        dssf = dssfx
        emin = np.min(dssf[:,:,1])
        emax = np.max(dssf[:,:,1])
        nbin = 200
        im = np.zeros((nbin,lk))
        for iQ in range(lk):
            for ie in range(nkx*nky):
                inde = int(nbin-(dssf[iQ,ie,1]-emin)/(emax-emin)*nbin-1)
                im[inde,iQ] += dssf[iQ,ie,0]
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot()
        sc = ax.imshow(im,cmap=cm.Blues,aspect=0.7)
        #
        cbar = plt.colorbar(sc)
        cbar.ax.tick_params(labelsize=20)
        ax.axvline(L//2,color='k',zorder=1,lw=0.5)
        ax.axvline(L,color='k',zorder=1,lw=0.5)
        ax.set_xticks([0,L//2,L,3*L//2],[r"$\Gamma$",r"X",r"$M$",r"$\Gamma$"],size=20)
        ax.axhline(nbin+emin/(emax-emin)*nbin-1,color='k',zorder=1,lw=0.5,ls='--')
        ax.set_yticks([nbin+emin/(emax-emin)*nbin-1,],["0",],size=20)
        ax.set_ylabel(r"$\omega$",size=20)
        ax.set_title("h="+"{:.2f}".format(h),size=30)
        fig.tight_layout()
        plt.show()
        continue
#        exit()
        #Plot
        maxn = np.max(dssf[:,:,0].flatten()[~np.isnan(dssf[:,:,0].flatten())])
        norm = Normalize(vmin=0,vmax=maxn)
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot()
        for iQ in range(lk):
            if iQ in [L//2,L]:
                ax.axvline(iQ,color='k',zorder=0)
            sc = ax.scatter(np.ones(dssf.shape[1])*iQ,dssf[iQ,:,1],c=dssf[iQ,:,0],zorder=1,norm=norm,cmap=cm.gray_r)
        ax.set_title("h="+"{:.2f}".format(h),size=30)
        ax.set_xticks([0,L//2,L,3*L//2],[r"$\Gamma$",r"X",r"$M$",r"$\Gamma$"],size=20)
        ax.set_ylabel(r"$\omega$",size=20)
        plt.colorbar(sc)
        plt.show()
#        exit()
        continue


        if 0:   #old stuff
            uv = uk[:,:,None,None]*vk[None,None,:,:]    #first 2 indices for k, last two for kp
            vu = vk[:,:,None,None]*uk[None,None,:,:]
            ssf = np.cos(th)**2*((S-vk**2)**2+uk**2*vk**2) + np.sin(th)**2*4*uk*vk
            if pl_ssf:
                fig = plt.figure(figsize=(10,6))
                ax = fig.add_subplot()
                sc = ax.scatter(gridk[:,:,0],gridk[:,:,1],c=ssf,marker='s')#,markersize=20)
                fig.colorbar(sc)
                plt.show()
                continue
            #
            for indt in range(len(list_t)):
                t = list_t[indt]
                c2t = np.exp(-1j*t*w_h)[:,:,None,None]*np.exp(-1j*t*w_h)[None,None,:,:]
                for idx in range(dx):
                    sign02 = (-1)**(idx%2)
                    c0 = sign02*(S*np.cos(th)-np.cos(th)/Ns*np.sum(vk**2))**2
                    #
                    sign1 = sign**(2-idx%2) #1 if idx%2 == 0 else sign
                    c1 = sign1*np.sin(th)**2*S/2/Ns*np.sum(np.exp(-1j*gridk[:,:,0]*idx)*np.exp(-1j*t*w_h)*(uk+vk)**2)
                    #
                    c2d = np.exp(-1j*gridk[:,:,0]*idx)[:,:,None,None] * np.exp(-1j*gridk[:,:,0]*idx)[None,None,:,:]
                    c2 = np.sum(c2t*c2d*uv*(uv+vu))*sign02*np.cos(th)**2/Ns**2
                    #
                    corr[ih,idx,indt] = c0+c1+c2
    if save:
        np.save(fn,corr)
else:
    corr = np.load(fn)

exit()



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
plt.suptitle(tit+r", $\Delta=$"+str(delta)+", L="+str(L)+", h="+"{:.2f}".format(h),size=30)

fig.tight_layout()
plt.show()























