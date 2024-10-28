import numpy as np
import matplotlib.pyplot as plt
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
    nkx = 100
    nky = 2*nkx
    Ns = nkx*nky
    list_kx = np.linspace(0,np.pi,nkx,endpoint=False)
    list_ky = np.linspace(-np.pi,np.pi,nky,endpoint=False)
    Kx,Ky = np.meshgrid(list_ky,list_kx)
    g = np.cos(Kx)+np.cos(Ky)
    #
    J_nn = 1
    H_list = np.linspace(2,0,11)
    ind_h = 0 if len(sys.argv)<2 else int(sys.argv[1])
    h = H_list[ind_h]
    print("h: ",h,", J: ",J_nn)
    #
    n_th = 101
    list_th = np.linspace(-np.pi,np.pi,n_th)
    fn0 = "results/energy0_J"+"{:.3f}".format(J_nn)+"_h"+"{:.3f}".format(h)+"_nkx"+str(nkx)+"_nky"+str(nky)+"_th"+str(n_th)+".npy"
    fn1 = "results/energy1_J"+"{:.3f}".format(J_nn)+"_h"+"{:.3f}".format(h)+"_nkx"+str(nkx)+"_nky"+str(nky)+"_th"+str(n_th)+".npy"
    save = False
    if not Path(fn1).is_file():# or not save:
        E0_th = np.zeros(n_th)
        E1_th = np.zeros(n_th)
        J = np.identity(4)
        J[0,0] = J[1,1] = -1
        J[2,2] = J[3,3] = 1
        #
        for i in tqdm(range(n_th)):
            t = np.sin(list_th[i])**2
            r = h*np.cos(list_th[i])
            E0_th[i] = -3/2*t*J_nn-r
            #
            d = (t*J_nn+r/2)*np.ones((nkx,nky))
            b = J_nn*(2-t)*g/4
            c = -J_nn*t*g/4
            z = np.zeros((nkx,nky))
            Nk = np.array([
                [d,c,z,b],
                [c,d,b,z],
                [z,b,d,c],
                [b,z,c,d]
            ])
            w = np.zeros((nkx,nky,2))
            nnn = 0
            for ikx in range(nkx):
                for iky in range(nky):
                    try:
                        Ch = scipy.linalg.cholesky(Nk[:,:,ikx,iky])
                    except:
                        nnn = 1
                        w[ikx,iky] = np.nan
                        continue
                    w[ikx,iky] = np.linalg.eigvalsh(Ch@J@Ch.T.conj())[2:]
            E1_th[i] = np.sum(w[~np.isnan(w)])/Ns/2 #if nnn==0 else np.nan
        if save:
            np.save(fn0,E0_th)
            np.save(fn1,E1_th)
    else:
        E1_th = np.load(fn1)
        E0_th = np.load(fn0)

if 1:
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot()
    E_tot = E0_th+E1_th
    try:
        E_min = np.min(E_tot[~np.isnan(E_tot)])
        E_max = np.max(E_tot[~np.isnan(E_tot)])
        ax.plot(list_th,E_tot,c='k',label='total')
    except:
        print("All E1s are nan")
        E_min = np.min(E0_th)
        E_max = np.max(E0_th)
        ax.plot(list_th,E0_th,c='k',label='E0')
    for i in range(3):
        th = -np.pi/2+np.pi/2*i
        ax.plot([th,th],[E_min,E_max],lw=0.5,c='b')
    ax.set_xlabel(r"$\theta$",size=20)
    ax.set_ylabel(r"$E/N_s$",size=20)
    ax.set_title("h="+"{:.3f}".format(h)+", J="+"{:.3f}".format(J_nn),size=30)
    plt.legend(fontsize=20)
    if 0:
        plt.savefig("figures/"+fn0[9:-4]+'.png')
    else:
        plt.show()




































