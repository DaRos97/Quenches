import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

"""
Here we compute the lowest energy configuration for the classical spin model varying J2 and h.
This is given by minimizing a function of 5 angles: orientation of two spins in the unit cell (3 angles),
rotation angle of translation in the 2 directions. This is equivalent to a RMO construction
considering only translations and the U(1) symmetry of the Hamiltonian.
"""

if 0:   #energy of simply twisting two spins
    J1 = 1

    def e1(j2,h,th,ph):
        return -J1/4*np.sin(th)**2*(1+np.cos(ph)) + j2/2*np.sin(th)**2*np.cos(ph) - h/2*np.cos(th)

    l_j2 = np.linspace(0,1,10)
    l_h = np.linspace(0,3,10)

    l_th = np.linspace(0,np.pi/2,100)
    l_ph = np.linspace(0,np.pi,150)
    X,Y = np.meshgrid(l_th,l_ph)

    en = np.zeros((len(l_j2),len(l_h),len(l_th),len(l_ph)))

    fig = plt.figure(figsize=(15,15))
    for ij2 in range(len(l_j2)):
        for ih in range(len(l_h)):
            j2 = l_j2[ij2]
            h = l_h[ih]
            en[ij2,ih] = e1(j2,h,l_th[:,None],l_ph[None,:])
            ax = fig.add_subplot(len(l_j2),len(l_h),ij2*len(l_h)+ih+1,projection='3d')
            ax.plot_surface(X,Y,en[ij2,ih].T)
            ax.set_title("j2: "+"{:.1f}".format(j2)+", h:"+"{:.1f}".format(h))
    plt.show()

if 1:   #energy of RMO
    def e2(angles,*args):
        j1,j2,h = args
        thA,phB,ph2 = angles
        thB = np.pi-thA
        ph1 = 0
        return (j1/2*np.sin(thA)*np.sin(thB)*(np.cos(phB)+np.cos(phB+ph2)+np.cos(phB-ph1)+np.cos(phB-ph1-ph2))
                + j2/2*(np.cos(ph1+ph2)+np.cos(ph2))*(np.sin(thA)**2+np.sin(thB)**2)
                + h*(np.cos(thB)-np.cos(thA))
               )
    from scipy.optimize import minimize
    from scipy.optimize import differential_evolution as D_E
    from time import time
    from tqdm import tqdm
    J1 = 1
    l_j2 = np.linspace(0,1,30)
    l_h = np.linspace(0.01,3,20)
    if not Path('temp.npy').is_file():
        en = np.zeros((len(l_j2),len(l_h),4))   #energy and 3 angles
        for ij2 in tqdm(range(len(l_j2))):
            for ih in range(len(l_h)):
    #            ti = time()
                args = (J1,l_j2[ij2],l_h[ih])
                res = D_E(     e2,
    #                           x0 = (0.5,np.pi-0.2,0.3,0.8,0.9),
                               bounds = [(0,np.pi),(-np.pi,np.pi),(-np.pi,np.pi)],
                               args=args,
                               #method='Nelder-Mead',
                               strategy='rand1exp',
                               tol=1e-8,
    #                           options={'disp':False}
                              )
                en[ij2,ih,0] = res.fun
                en[ij2,ih,1:] = res.x
    #            print("time: ",time()-ti)
        np.save('temp.npy',en)
    else:
        en = np.load('temp.npy')
    #process
    for ij2 in tqdm(range(len(l_j2))):
        for ih in range(len(l_h)):
            if abs(en[ij2,ih,1])<1e-4:
                en[ij2,ih,3] = np.nan
                en[ij2,ih,2] = np.nan
            if abs(abs(en[ij2,ih,3])-np.pi)<1e-4:
                en[ij2,ih,3] = abs(en[ij2,ih,3])

    fig = plt.figure(figsize=(15,10))
    X,Y = np.meshgrid(l_j2,l_h)
    titles = ['energy','thA','thB','phB','ph2']
    if 1:
        for i in range(4):
            ax = fig.add_subplot(2,3,i+1,projection='3d')
            ax.plot_surface(X,Y,en[:,:,i].T,cmap='plasma_r')
            ax.set_title(titles[i])
    if 0:
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X,Y,abs(en[:,:,2].T+en[:,:,3].T)%(2*np.pi),cmap='plasma_r')
        plt.show()
        exit()
    #PD
    plt.show()
    exit()
    ax = fig.add_subplot()
    for ij2 in range(len(l_j2)):
        for ih in range(len(l_h)):
            e,th,phb,ph2 = en[ij2,ih]
            if abs(th)<1e-3:
                color='k'
            elif abs(abs(phb)-np.pi)<1e-3 and abs(ph2)<1e-3:  #neel
                color='r'
            elif abs(abs(phb)-np.pi)<1e-3:#
                color='y'
            elif abs(phb)<1e-3:#
                color='orange'
            else:   #unknown
                color = 'b'
                print(phb)
            ax.scatter(l_j2[ij2],l_h[ih],color=color,marker='o')
    plt.show()

    exit()
