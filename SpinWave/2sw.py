import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
from tqdm import tqdm
import scipy
from pathlib import Path

if 0:   #theta independent, t=0
    h = 1
    J_nn = 0.4
    print("h: ",h,", J: ",J_nn)
    nkx = 500
    nky = 2*nkx
    Ns = nkx*nky
    list_kx = np.linspace(0,np.pi,nkx,endpoint=False)
    list_ky = np.linspace(-np.pi,np.pi,nky,endpoint=False)
    Kx,Ky = np.meshgrid(list_kx,list_ky)
    g = np.cos(Kx)+np.cos(Ky)

    E0 = -2*h
    #
    u = 1/2*np.arctanh(J_nn*g/h)
    omega = (h-J_nn**2*g*g/2/h)*(np.cosh(u)**2+np.sinh(u)**2)
    #Now some points k don't have a well defined Bogoliubov transformation
    E1 = np.sum(omega)/Ns
    if 0:   #plot dispersion
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(Kx,Ky,omega.T)
        plt.show()
        exit()
    print(E0,E1)
    print("Total: ",E0+E1)
elif 1:   #Theta dependent, 2-site UC, numerical Bogoliubov (only possibility)
    """Here we compute the GS energy of XY+staggered Z using Holstein-Primakof.
    We have a 2-tie UC.
    We take J_nn=1 and vary the staggered field 'h' from high values to 0 (phase transition at h=2J from gapped (h>2J) to gapless (h<2J)).
    For each (J_nn,h) we compute the GS energy for n_th values of angle of quantization axis from th=0 (AFM in z) to th=pi/2 (FM in x).
    For each th we compute the Hamiltonian and do the Bogoliubov transformation (BT) numerically at each k of the BZ.
    For some values of (J_nn,h,th) the BT is not possible for all k: for example at (1,h,pi/2) the Hamiltonian at k=0 does not allow the BT. This
    is because the excitation spectrum is gapless. There are also situations where the spectrum ha NEGATIVE eigenvalues. This cannot be since we
    are working with bosons, and a negative eigenvalue would mean that the bosons would all condense there and break the n<<1 constraint of
    Holstein-Primakof. We then discard all solutions which have negative eigenvalues.
    """
    nkx = 100
    nky = 2*nkx
    Ns = nkx*nky
    list_kx = np.linspace(0,np.pi,nkx,endpoint=False)
    list_ky = np.linspace(-np.pi,np.pi,nky,endpoint=False)
    Kx,Ky = np.meshgrid(list_ky,list_kx)
    g = np.cos(Kx)+np.cos(Ky)
    #
    J_nn = 1
    H_list = np.linspace(2.5,0,26)
    ind_h = 0 if len(sys.argv)<2 else int(sys.argv[1])
    h = H_list[ind_h]
    print("h: ",h,", J: ",J_nn)
    #
    n_th = 201
    list_th = np.linspace(np.pi/2,0,n_th)
    fn0 = "results/energy0_J"+"{:.3f}".format(J_nn)+"_h"+"{:.3f}".format(h)+"_nkx"+str(nkx)+"_nky"+str(nky)+"_th"+str(n_th)+".npy"
    fn1 = "results/dispersion_J"+"{:.3f}".format(J_nn)+"_h"+"{:.3f}".format(h)+"_nkx"+str(nkx)+"_nky"+str(nky)+"_th"+str(n_th)+".npy"
    fn2 = "results/minimaNk_J"+"{:.3f}".format(J_nn)+"_h"+"{:.3f}".format(h)+"_nkx"+str(nkx)+"_nky"+str(nky)+"_th"+str(n_th)+".npy"
    save = True
    if not Path(fn0).is_file():# or not save:
        E0_th = np.zeros(n_th)
        J = np.identity(4)
        J[0,0] = J[1,1] = -1
        #
        w_th = np.zeros((n_th,nkx,nky,2))
        min_th = np.zeros((n_th,nkx,nky))
        for i in tqdm(range(n_th)):
            t = J_nn*np.sin(list_th[i])**2
            r = h*np.cos(list_th[i])
            E0_th[i] = -3/2*t-r
            #
            d = (t+r/2)*np.ones((nkx,nky))
            b = (J_nn*2-t)*g/4
            c = -t*g/4
            z = np.zeros((nkx,nky))
            Nk = np.array([
                [d,c,z,b],
                [c,d,b,z],
                [z,b,d,c],
                [b,z,c,d]
            ])
            w = np.zeros((nkx,nky,2))
            min_local_eigval = np.zeros((nkx,nky))
            for ikx in range(nkx):
                for iky in range(nky):
                    min_local_eigval[ikx,iky] = np.linalg.eigvalsh(Nk[:,:,ikx,iky])[0]
                    try:
                        Ch = scipy.linalg.cholesky(Nk[:,:,ikx,iky])
                    except:
                        w[ikx,iky] = np.nan
                        continue
                    w[ikx,iky] = np.linalg.eigvalsh(Ch@J@Ch.T.conj())[2:]
            if 0:   #plot w
                print(E0_th[-1],np.sum(w[~np.isnan(w)])/Ns/2,np.min(w[~np.isnan(w)]))
                fig = plt.figure(figsize=(20,20))
                ax = fig.add_subplot(projection='3d')
                ax.plot_surface(Kx,Ky,w[:,:,0],cmap=cm.plasma)
#                ax.plot_surface(Kx,Ky,w[:,:,1],cmap=cm.plasma_r)
                ax.set_title("theta="+"{:.2f}".format(list_th[i]*180/np.pi)+'°',size=30)
                plt.show()
                #exit()
            w_th[i] = w
            min_th[i] = min_local_eigval
        if save:
            np.save(fn0,E0_th)
            np.save(fn1,w_th)
            np.save(fn2,min_th)
    else:
        E0_th = np.load(fn0)
        w_th = np.load(fn1)
        min_th = np.load(fn2)

if 1:       #Plotting energy vs theta
    E1_th = np.zeros(n_th)
    minNk_th = np.zeros(n_th)
    gap_th = np.zeros(n_th)
    Ns = w_th[0].shape[0]*w_th[0].shape[1]
    for i in range(n_th):
        if np.min(min_th[i]) < -1e-7:
            not_good = True
        else:
            not_good = False
        try:
            E1_th[i] = np.sum(w_th[i][~np.isnan(w_th[i])])/Ns/2 #if nnn==0 else np.nan
            if not not_good:
                gap_th[i] = np.min(w_th[i][~np.isnan(w_th[i])])
            else:
                gap_th[i] = np.nan
        except:     #w_th all nan
            print("aaaaaaa",i)
            E1_th[i] = 0
            gap_th[i] = np.nan
#        minNk_th[i] = np.min(min_th[i])
    #
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot()
    E_tot = E0_th+E1_th
    E_min = np.min(E_tot[~np.isnan(E_tot)])
    E_max = np.max(E_tot[~np.isnan(E_tot)])
    #
    edgecolor = []
    color1 = []
    color2 = []
    i_best = -1
    for i in range(n_th):
        if not np.isnan(gap_th[i]):
            temp1 = gap_th[i]
            temp2 = 'none'
        else:
            temp1 = np.nan
            temp2 = 'k'
        color1.append(temp1)
        color2.append(temp2)
    try:
        i_best = color2.index('k')-1
    except:
        i_best = n_th-1
    sc = ax.scatter(list_th,E_tot,c=color1,label='total',cmap=cm.plasma_r,marker='o',s=100)
    ax.scatter(list_th,E_tot,c=color2,marker='o',s=100,alpha=0.1,label='unphysical states')
    for i in range(1,3):
        th = -np.pi/2+np.pi/2*i
        ax.plot([th,th],[E_min,E_max],lw=0.5,c='b')
    ax.plot([list_th[i_best],list_th[i_best]],[E_min,E_max],lw=0.5,c='r')
    ax.set_xlabel(r"$\theta$"+" angle of quantization axis",size=30)
    ax.set_ylabel("GS energy per site",size=30)
    ax.set_xticks([0,list_th[i_best],np.pi/2],["$0°=z-AFM$","{:.2f}".format(list_th[i_best]*180/np.pi)+'°',r"$90°=\hat{x}$"])
    ax.xaxis.set_tick_params(labelsize=20)
    ax.set_title(r"$h=$"+"{:.3f}".format(h)+r", $J=$"+"{:.3f}".format(J_nn),size=30)
#    plt.legend(fontsize=20)
    cb = plt.colorbar(sc)
    cb.set_label(label='Gap',size=30)
    if 0:
        plt.savefig("figures/"+fn0[8:-4]+'.png')
    if 1:
        print(E0_th)
        print(E1_th)
        print(E_tot)
        plt.show()




































