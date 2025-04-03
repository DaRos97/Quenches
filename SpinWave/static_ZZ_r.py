import numpy as np
import functions as fs
import matplotlib.pyplot as plt
import scipy
import sys
from time import time

disp = 0
#Parameters of the ramp
S = 0.5     #spin value
full_time_ramp = 0.5 if len(sys.argv)<3 else float(sys.argv[2])/1000    #ramp time in ms
time_steps = 500        #of ramp
time_step = full_time_ramp/time_steps  #time step of ramp
#
use_experimental_parameters = 0#False
txt_exp = 'expPars' if use_experimental_parameters else 'uniform'
if use_experimental_parameters:
    initial_parameters_fn = 'exp_input/20250324_6x7_2D_StaggeredFrequency_0MHz_5.89_.p'
    final_parameters_fn = 'exp_input/20250324_6x7_2D_IntFrequency_10MHz_5.89_.p'
    Lx,Ly,g1_in,g2_in,d1_in,h_in = fs.extract_experimental_parameters(initial_parameters_fn)
    Lx,Ly,g1_fin,g2_fin,d1_fin,h_fin = fs.extract_experimental_parameters(final_parameters_fn)
    Ns = Lx*Ly
else:
    Lx = 6
    Ly = 7
    g_val = 40
    h_val = 30
    #
    Ns = Lx*Ly
    g1_in = np.zeros((Ns,Ns))
    g1_fin = np.zeros((Ns,Ns))
    for ix in range(Lx):
        for iy in range(Ly):
            ind = iy+ix*Ly
            ind_plus_y = ind+1
            if ind_plus_y//Ly==ind//Ly:
                g1_fin[ind,ind_plus_y] = g1_fin[ind_plus_y,ind] = g_val
            ind_plus_x = ind+Ly
            if ind_plus_x<Lx*Ly:
                g1_fin[ind,ind_plus_x] = g1_fin[ind_plus_x,ind] = g_val
    g2_in = np.zeros((Ns,Ns))
    g2_fin = np.zeros((Ns,Ns))
    d1_in = np.zeros((Ns,Ns))
    d1_fin = np.zeros((Ns,Ns))
    h_in = np.zeros((Ns,Ns))
    for ix in range(Lx):
        for iy in range(Ly):
            h_in[iy+ix*Ly,iy+ix*Ly] = (-1)**(ix+iy)*    h_val
    h_fin = np.zeros((Ns,Ns))
site_j = (2,3) if (Lx==6 and Ly==7) else (Lx//2,Ly//2)      #site wrt with compute the correlator
indj = site_j[1] + site_j[0]*Ly
site0 = 0 if h_in[0,0]<h_in[1,1] else 1     #decide sublattice A and B of reference lattice site -> see fs.get_ts
#
if disp:    print("hamiltonian parameters..")
t_i = time()
g1_t_i,g2_t_i,d1_t_i,h_t_i = fs.get_Hamiltonian_parameters(time_steps,g1_in,g2_in,d1_in,h_in,g1_fin,g2_fin,d1_fin,h_fin)   #parameters of Hamiltonian which depend on time
if disp:    print("..computed in ","{:.2f}".format(time()-t_i)," sec")
#
full_time_measure = 0.8     #measure time in ms
Nt = 401        #time steps after ramp for the measurement
measure_time_list = np.linspace(0,full_time_measure,Nt)
stop_ratio_list = np.linspace(0.1,1,10)     #ratios of ramp where we stop and measure
#Diagonal matrix
Gamma = np.zeros((2*Ns,2*Ns))
for i in range(Ns):
    Gamma[i,i] = 1
    Gamma[i+Ns,i+Ns] = -1
#Actual code
thetas = np.zeros(10)
ens = np.zeros(10)
corr = np.zeros((len(stop_ratio_list),Lx,Ly,Nt),dtype=complex)
for i_sr,stop_ratio in enumerate(stop_ratio_list):
    print("Stop ratio ",stop_ratio)
    indt = int(time_steps*stop_ratio)
    if indt==time_steps:
        indt -= 1
    #
    J_i = (g1_t_i[indt,:,:],np.zeros((Ns,Ns)))
    D_i = (d1_t_i[indt,:,:],np.zeros((Ns,Ns)))
    h_i = h_t_i[indt,:,:]
    theta,phi = fs.get_angles(S,J_i,D_i,h_i)
    thetas[i_sr] = theta
    #
    parameters = (S,Lx,Ly,h_i,theta,phi,J_i,D_i)
    hamiltonian = fs.get_Hamiltonian_rs(*parameters)
    if np.max(abs(hamiltonian-hamiltonian.T.conj()))>1e-7:
        print("Non-Hermitian Hamiltonian")
        exit()
    if 0:
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(projection='3d')
        X,Y = np.meshgrid(np.arange(2*Ns),np.arange(2*Ns))
        ax.plot_surface(X,Y,hamiltonian.T,cmap='plasma')
        plt.show()
    if disp:    print("Diagonalization..")
    t_i = time()
    eps,U = scipy.linalg.eigh(Gamma@hamiltonian)
    if disp:    print("..computed in ","{:.2f}".format(time()-t_i)," sec")

    if disp:
        if np.max(abs(U-U.T.conj()))>1e-7:
            print("Non-Hermitian U")
        else:
            print("Hermitian U")

#    ens[i_sr] = eps[:Ns].sum()/Ns + 2*S*(S+1)*fs.get_p_zz(theta,phi,J_i,D_i)[0][0,1]-abs(h_i[0,0])*np.cos(theta)*(S+1/2)
#    ens[i_sr] /= np.sum(J_i[0])/2/Ns
    #
    exp_e = np.exp(-1j*2*np.pi*measure_time_list[:,None]*eps[None,Ns:])
    big_matrix = np.einsum('tk,ik,jk->ijt',exp_e,U[:,:Ns],U[:,Ns:],optimize=True)
    t_ab = fs.get_ts(theta,phi,site0)
    ts_j = t_ab[(site_j[0]+site_j[1])%2]
    G,H = big_matrix[Ns:,:Ns], big_matrix[:Ns,Ns:]
    B,A = big_matrix[:Ns,:Ns], big_matrix[Ns:,Ns:]

    if disp:    print("Computing correlator")
    for ind in range(Ns):
        ind_x,ind_y = (ind//Ly,ind%Ly)
        ts_i = t_ab[(ind_x+ind_y)%2]
        corr[i_sr,ind_x,ind_y] = fs.get_correlator(ts_i,ts_j,S,
                                                   G[ind,indj],
                                                   H[ind,indj],
                                                   A[ind,indj],
                                                   B[ind,indj],
                                                  )
        if 0:
            fig = plt.figure(figsize=(20,20))
            X,Y = np.meshgrid(np.arange(2*Ns),np.arange(2*Ns))
            for i in range(3):
                ax = fig.add_subplot(2,3,i+1,projection='3d')
                ax.plot_surface(X,Y,np.real(big_matrix[:,:,i*(Nt//2-1)]),cmap='plasma')
                ax = fig.add_subplot(2,3,i+4,projection='3d')
                ax.plot_surface(X,Y,np.imag(big_matrix[:,:,i*(Nt//2-1)]),cmap='plasma')
            plt.show()
    if 0:
        import matplotlib
        cmap = matplotlib.colormaps['plasma']
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot()
        for ix in range(Lx):
            for iy in range(Ly):
                if (ix,iy)==site_j:
                    ax.plot(measure_time_list,np.imag(corr[i_sr,ix,iy]),label=str(ix)+'_'+str(iy),color='k')
                    continue
                ax.plot(measure_time_list,np.imag(corr[i_sr,ix,iy]),label=str(ix)+'_'+str(iy),color=cmap((iy+ix*Ly)/Ns))
        ax.legend()
        plt.show()

if 0:
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(stop_ratio_list,thetas,'b^',label='theta')
    ax.legend()
    ax_r = ax.twinx()
    ax_r.plot(stop_ratio_list,ens,'r*',label='energy')
    ax_r.legend()
    plt.show()


fs.plot_corr(corr)













