import numpy as np
import matplotlib.pyplot as plt
import functions as fs

#Parameters Hamiltonian
S = 0.5     #spin value
J1 = 1
J2 = 0
D1 = 0
D2 = 0
J = (J1,J2)
D = (D1,D2)
#
Lx = 100     #Linear size of lattice
Ly = 100
Ns = Lx*Ly    #number of sites
gridk = fs.BZgrid(Lx,Ly)#BZ grid
Gamma1 = np.cos(gridk[:,:,0])+np.cos(gridk[:,:,1])  #cos(kx) + cos(ky)
Gamma2 = np.cos(gridk[:,:,0]+gridk[:,:,1])+np.cos(gridk[:,:,0]-gridk[:,:,1])  #cos(kx+ky) + cos(kx-ky)
Gamma = (Gamma1,Gamma2)
#
n_H = 26    #number of h-field points
H_list = np.linspace(2.5,0,n_H)
#H_list = np.linspace(1.5,2.5,n_H)

#Actual code
e_gs = np.zeros(n_H)
thetas = np.zeros(n_H)
ps = np.zeros(n_H)
for i,h in enumerate(H_list):
    theta,phi = fs.get_angles(S,J,D,h)
    ps[i] = fs.get_p_xy(theta,phi,J,D)[1]
    thetas[i] = theta
    parameters = (S,Gamma,h,theta,phi,J,D)
    e_gs[i] = fs.get_E_GS(*parameters)
    if 1:
        epsilon = fs.get_epsilon(*parameters)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        X,Y = np.meshgrid(gridk[:,:,0],gridk[:,:,1])
        ax.plot_surface(gridk[:,:,0],gridk[:,:,1],epsilon,cmap='plasma')
        ax.set_aspect('equalxy')
        ax.set_title("H="+"{:.4f}".format(h))
        plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(H_list,e_gs,'r*-',label='e_gs')
ax.legend(loc='lower left')
ax_r = ax.twinx()
ax_r.plot(H_list,thetas,'b*-',label='theta')
#ax_r.plot(H_list,ps,'g*-',label='p')
ax_r.legend(loc='upper right')
plt.show()















