import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import scipy

ind_h = 0 if len(sys.argv)<2 else int(sys.argv[1])
list_h = np.linspace(0,100,101)
h = list_h[ind_h]
J_nn = 1

print("h=",h,", J=",J_nn)

if 1:   #Theta dependent, 2-site UC, numerical Bogoliubov (only possibility)
    nkx = 70
    nky = 2*nkx
    Ns = nkx*nky
    list_kx = np.linspace(0,np.pi,nkx,endpoint=False)
    list_ky = np.linspace(-np.pi,np.pi,nky,endpoint=False)
    Kx,Ky = np.meshgrid(list_ky,list_kx)
    g = np.cos(Kx)+np.cos(Ky)
    #
    n_th = 101
    list_th = np.linspace(0,np.pi/2,n_th)
    E0_th = np.zeros(n_th)
    E1_th = np.zeros(n_th)
    J = np.identity(4)
    J[0,0] = J[1,1] = -1
    J[2,2] = J[3,3] = 1
    #
    for i in tqdm(range(n_th)):
        t = np.sin(list_th[i])**2
        r = h*np.cos(list_th[i])
        E0_th[i] = -3/2*t*J_nn
        #
        d1 = (4*t*J_nn+r)*np.ones((nkx,nky))
        d2 = (4*t*J_nn-r)*np.ones((nkx,nky))
        a = J_nn*(t-2)*g
        b = -J_nn*t*g
        z = np.zeros((nkx,nky))
        Nk = np.array([
            [d1,a,z,b],
            [a,d2,b,z],
            [z,b,d1,a],
            [b,z,a,d2]
        ])
        w = np.zeros((nkx,nky,2))
        for ikx in range(nkx):
            for iky in range(nky):
                try:
                    Ch = scipy.linalg.cholesky(Nk[:,:,ikx,iky])
                    nn = 0
                except:
                    w[ikx,iky] = np.nan
                    continue
                w[ikx,iky] = np.linalg.eigvalsh(Ch@J@Ch.T.conj())[2:]/4
        if 0:   #plot dispersion
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot_surface(Kx,Ky,w[:,:,0])
            ax.plot_surface(Kx,Ky,w[:,:,1])
            plt.show()
        E1_th[i] = np.sum(w[~np.isnan(w)])/Ns/2
elif 0: #Theta dependent, 1-site UC, numerical Bogoliubov
    nkx = nky = 100
    Ns = nkx*nky
    list_kx = np.linspace(-np.pi,np.pi,nkx,endpoint=False)
    list_ky = np.linspace(-np.pi,np.pi,nky,endpoint=False)
    Kx,Ky = np.meshgrid(list_kx,list_ky)
    g = np.cos(Kx)+np.cos(Ky)
    #
    n_th = 21
    list_th = np.linspace(0,np.pi/2,n_th)
    E0_th = np.zeros(n_th)
    E1_th = np.zeros(n_th)
    J = np.array([[-1,0],[0,1]])
    for i in tqdm(range(n_th)):
        t = np.sin(list_th[i])**2
        E0_th[i] = -3/2*t
        a = 4*t-(2-t)*g
        b = -t*g
        Nk = np.array([
            [a,b],
            [b,a]
        ])
        w = np.zeros((nkx,nky))
        for ikx in range(nkx):
            for iky in range(nky):
                try:
                    Ch = scipy.linalg.cholesky(Nk[:,:,ikx,iky])
                    nn = 0
                except:
                    try:
                        Ch = scipy.linalg.cholesky(-Nk[:,:,ikx,iky])
                        nn = 1
                    except:
                        w[ikx,iky] = np.nan
                        continue
                w[ikx,iky] = (-1)**nn*np.linalg.eigvalsh(Ch@J@Ch.T.conj())[1]/4
        E1_th[i] = np.sum(w[~np.isnan(w)])/Ns
        if 0:   #plot dispersion
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot_surface(Kx,Ky,w[:,:,0].T)
            plt.show()
            exit()
elif 0: #Theta dependent, 1-site UC
    nkx = nky = 2000
    Ns = nkx*nky
    list_kx = np.linspace(-np.pi,np.pi,nkx,endpoint=False)
    list_ky = np.linspace(-np.pi,np.pi,nky,endpoint=False)
    Kx,Ky = np.meshgrid(list_kx,list_ky)
    g = np.cos(Kx)+np.cos(Ky)

    n_th = 51
    list_th = np.linspace(0,np.pi/2,n_th)
    E0_th = np.zeros(n_th)
    E1_th = np.zeros(n_th)

    for i in tqdm(range(n_th)):
        t = np.sin(list_th[i])**2
        E0_th[i] = -3/2*t
        #
        u = 1/2*np.arctanh(-t*g/(4*t-(2-t)*g))
        omega = (2*t-g)*(np.cosh(u)-np.sinh(u))**2
        #Now some points k don't have a well defined Bogoliubov transformation
        E1_th[i] = omega[~np.isnan(omega)].sum()/2/Ns
        if 0:   #plot dispersion
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot_surface(Kx,Ky,omega.T)
            plt.show()
            exit()
elif 0: #Theta independent, 1-site UC, Q-axis along y, numerical bogoliubov
    nkx = 500
    nky = 500
    Ns = nkx*nky
    list_kx = np.linspace(-np.pi,np.pi,nkx)
    list_ky = np.linspace(-np.pi,np.pi,nky)

    E0 = -3/2
    Kx,Ky = np.meshgrid(list_kx,list_ky)
    g = np.cos(Kx)+np.cos(Ky)
    Nk = np.array([[4-g,-g],[-g,4-g]])
    J = np.array([[-1,0],[0,1]])
    E1 = 0
    for ikx in range(nkx):
        for iky in range(nky):
            Ch = scipy.linalg.cholesky(Nk[:,:,ikx,iky])
            E1 += np.linalg.eigvalsh(Ch@J@Ch.T.conj())[1]/4/Ns
    print("E0:",E0)
    print("E1:",E1)
    print("Tot:",E0+E1)
    exit()
else: #Theta independent, 1-site UC, Q-axis along y
    nkx = 10000
    nky = 10000
    Ns = nkx*nky
    list_kx = np.linspace(-np.pi,np.pi,nkx)
    list_ky = np.linspace(-np.pi,np.pi,nky)

    E0 = -3/2
    Kx,Ky = np.meshgrid(list_kx,list_ky)
    g = np.cos(Kx)+np.cos(Ky)
    E1 = np.sqrt(1-g/2).sum()/Ns
    print("E0:",E0)
    print("E1:",E1)
    print("Tot:",E0+E1)
    exit()

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot()
#ax.plot(list_th,E0_th,c='r')
#ax.plot(list_th,E1_th,c='b',label='E1')
ax.plot(list_th,E0_th+E1_th,c='k',label='total')
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$E/N_s$")
print("E1:",E1_th)
print("Tot:",E0_th+E1_th)
plt.legend()
plt.show()
