import numpy as np
import functions as fs
import matplotlib.pyplot as plt
import scipy


use_experimental_parameters = 1#False
txt_exp = 'expPars' if use_experimental_parameters else 'uniform'
if use_experimental_parameters:
    initial_parameters_fn = 'exp_input/20250324_6x7_2D_StaggeredFrequency_0MHz_5.89_.p'
    final_parameters_fn = 'exp_input/20250324_6x7_2D_IntFrequency_10MHz_5.89_.p'
    g_in,h_in = fs.extract_experimental_parameters(initial_parameters_fn)
    g_fin,h_fin = fs.extract_experimental_parameters(final_parameters_fn)
    N = len(g_in)
else:
    N = 42          #chain sites
    g_in = np.zeros(N)
    g_fin = np.ones(N)*10 #MHz
    h_in = np.ones(N)*15 #MHz
    for i in range(N):
        h_in[i] *= (-1)**i
    h_fin = np.zeros(N)
g_t_i,h_t_i = fs.get_Hamiltonian_parameters(time_steps,g_in,g_fin,h_in,h_fin)   #parameters of Hamiltonian which depend on time

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
site_j = (2,3) if (Lx==6 and Ly==7) else (Lx//2,Ly//2)      #site wrt with compute the correlator
indj = site_j[1] + site_j[0]*Ly
#
Gamma = np.zeros(2*Ns)
for i in range(Ns):
    Gamma[i,i] = 1
    Gamma[i+Ns,i+Ns] = -1

Ntimes = 401        #number of time steps in the measurement
measurement_time = 0.8
t_list = np.linspace(0,measurement_time,Ntimes)      #800 ns
corr = np.zeros((len(stop_ratio_list),Lx*Ly,Ntimes),dtype=complex)
for i_sr,stop_ratio in enumerate(stop_ratio_list):
    J1 = J1_fin*stop_ratio
    h = h_in*(1-stop_ratio)
    print("Step ",i_sr," with J=",J1," and h=",h)
    J = (J1,J2)
    D = (D1,D2)
    theta,phi = fs.get_angles(S,J,D,h)
    parameters = (S,Lx,Ly,h,theta,phi,J,D)
    hamiltonian = fs.get_Hamiltonian_rs(*parameters)
    eps,U = scipy.linalg.eigh(Gamma@hamiltonian)

    big_matrix = np.einsum('tk,ik,jk->ijt',exp_e,U.conj()[:,Ns:],U.conj()[:,:Ns],optimize=True)
    G_i = big_matrix[:Ns,Ns+indj,:]
    H_i = big_matrix[Ns:,indj,:]
    A_i = big_matrix[:Ns,indj,:]
    B_i = big_matrix[Ns:,Ns+indj,:]
    params = (S,G_i,H_i,A_i,B_i)
    t_ab = fs.get_ts(theta,phi)
    ts_j = t_ab[indj%2]
    for indi in range(Ns):
        ts_i = t_ab[indi%2]
        corr[i_sr,indi] = fs.get_correlator(ts_i,ts_j,S,
                                            big_matrix[indi,indj+Ns,:],     #G
                                            big_matrix[indi+Ns,indj,:],     #H
                                            big_matrix[indi,indj,:],        #A
                                            big_matrix[indi+Ns,indj+Ns,:]   #B
                                           )


fs.plot_corr(corr)
