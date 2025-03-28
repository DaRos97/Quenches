import numpy as np
import matplotlib.pyplot as plt
import functions as fs

stop_ratio_list = np.linspace(0.1,1,10)     #ratios of ramp where we stop and measure
#Parameters Hamiltonian
S = 0.5     #spin value
J1_fin = 40     #MHz
J2 = 0
D1 = 0
D2 = 0
h_in = 30       #MHz
#
Lx = 7#30     #Linear size of lattice
Ly = 6#30
Ns = Lx*Ly    #number of sites
gridk = fs.BZgrid(Lx,Ly)#BZ grid
Gamma1 = np.cos(gridk[:,:,0])+np.cos(gridk[:,:,1])  #cos(kx) + cos(ky)
Gamma2 = np.cos(gridk[:,:,0]+gridk[:,:,1])+np.cos(gridk[:,:,0]-gridk[:,:,1])  #cos(kx+ky) + cos(kx-ky)
Gamma = (Gamma1,Gamma2)
#
Ntimes = 401        #number of time steps in the measurement
measurement_time = 0.8
t_list = np.linspace(0,measurement_time,Ntimes)      #800 ns
corr = np.zeros((len(stop_ratio_list),Lx,Ly,Ntimes),dtype=complex)
#e^{i*k*(ri-rj)}
x_val = np.arange(Lx)
y_val = np.arange(Ly)
X,Y = np.meshgrid(x_val,y_val)
grid_real_space = np.zeros((Lx,Ly,2))
grid_real_space[:,:,0] = X.T
grid_real_space[:,:,1] = Y.T
#e^{-i*eps*t}
#Actual code
for i_sr,stop_ratio in enumerate(stop_ratio_list):
    J1 = J1_fin*stop_ratio
    h = h_in*(1-stop_ratio)
    print("Step ",i_sr," with J=",J1," and h=",h)
    J = (J1,J2)
    D = (D1,D2)
    theta,phi = fs.get_angles(S,J,D,h)
    px = np.sin(theta)*np.cos(phi)
    py = np.sin(theta)*np.sin(phi)
    pz = np.cos(theta)
    parameters = (S,Gamma,h,theta,phi,J,D)
    epsilon = fs.get_epsilon(*parameters)       #dispersion
    exp_e = np.zeros((Lx,Ly,Ntimes),dtype=complex)
    for it in range(Ntimes):    #e^{-i*t*eps}   ->  [Lx,Ly,Nt]
        exp_e[:,:,it] = np.exp(-1j*2*np.pi*t_list[it]*epsilon)
    exp_e = np.reshape(exp_e, shape=(Lx*Ly,Ntimes))
    exp_e[np.isnan(exp_e)] = 1          #right??
    rk = fs.get_rk(*parameters)
    phik = fs.get_phik(*parameters).reshape(Lx*Ly)
    phik[np.isnan(phik)] = 1
    cosh_rk = np.cosh(rk).reshape(Lx*Ly)
    cosh_rk[np.isnan(cosh_rk)] = 0
    cosh_rk[np.absolute(cosh_rk)>1e3] = 0
    sinh_rk = np.sinh(rk).reshape(Lx*Ly)
    sinh_rk[np.isnan(sinh_rk)] = 0
    sinh_rk[np.absolute(sinh_rk)>1e3] = 0
    for ii in range(Lx*Ly):     #assuming phi=0 for now -> py=0
        ix = ii%Lx
        iy = ii//Lx
        exp_k = (np.exp(-1j*np.dot(gridk,grid_real_space[ix,iy]))).reshape(Lx*Ly)
        corr1 = S/2/Ns*np.einsum('i,i,ij->j',exp_k,px**2*np.absolute(cosh_rk+phik.conj()*sinh_rk)**2,exp_e,optimize=True)
        corr2 = np.sum( (-1)**(2*(ix+iy))*pz**2*(S-1/Ns*sinh_rk**2) )
        corr3 = (-1)**(2*(ix+iy))*pz**2/Ns**2*(np.einsum('ik,jk,i,j,i,j->k',exp_e,exp_e,exp_k,exp_k,
                                                    sinh_rk**2,cosh_rk**2,
                                                    optimize=True) +
                                           np.einsum('ik,jk,i,j,i,j->k',exp_e,exp_e,exp_k,exp_k,
                                                    phik.conj()*cosh_rk*sinh_rk,phik*cosh_rk*sinh_rk,
                                                    optimize=True) )
        corr[i_sr,ix,iy] = 2*1j*np.imag(corr1 + corr2 + corr3)

fs.plot_corr(corr)
























