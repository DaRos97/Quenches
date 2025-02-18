import numpy as np
from scipy.linalg import expm
import scipy
import os
from tqdm import tqdm

def get_ramp_evolution(*args):
    """Compute the time evolved wave function along the ramp and at each time step evaluate populations and fidelity."""
    N,g_t,h_t,time_step = args
    time_steps = len(g_t)
    time_evolved_psi = np.zeros((time_steps,N,N),dtype=complex)    #wavefunction at all time steps of the ramp
    fidelity = np.zeros(time_steps)
    populations = np.zeros((time_steps,N))
    print("Time evolution")
    for it in tqdm(range(time_steps)):
        if it==0:
            time_evolved_psi[it] = np.linalg.eigh(compute_H(g_t[it],h_t[it],N))[1]      #exact GS at t=0
        else:
            exp_H = expm(-1j*2*np.pi*compute_H(g_t[it],h_t[it],N)*time_step)
            for m in range(N):  #evolve each mode independently
                time_evolved_psi[it,:,m] = exp_H @ time_evolved_psi[it-1,:,m]
        #Fidelity wrt real GS
        E_GS,psi_GS = scipy.linalg.eigh(compute_H(g_t[it],h_t[it],N))
        fidelity[it] = np.absolute(np.linalg.det(time_evolved_psi[it,:,:N//2].T.conj()@psi_GS[:,:N//2]))**2
        #Occupation of populations
        populations[it] = compute_populations(time_evolved_psi[it],psi_GS)
    return time_evolved_psi, fidelity,populations

def compute_zz_correlator(*args):
    """Compute the ZZ correlator and its Fourier transform."""
    N,omega_list,stop_ratio_list,measure_time_list,g_t,h_t,time_evolved_psi,use_time_evolved = args
    Nt = len(measure_time_list)
    Nomega = len(omega_list)
    time_steps = len(g_t)
    #
    zz_correlator = np.zeros((len(stop_ratio_list),N,Nomega),dtype=complex)
    for i_sr in tqdm(range(len(stop_ratio_list))):
        stop_ratio = stop_ratio_list[i_sr]
        i_t = int(time_steps*stop_ratio)-1
        E_GS,beta = scipy.linalg.eigh( compute_H(g_t[i_t],h_t[i_t],N) )
#        _,beta = np.linalg.eigh(compute_H(g[-1],h[-1],N))
        if use_time_evolved:
            alpha = np.copy(time_evolved_psi[i_t][:,:N//2])
        else:
            alpha = np.copy(beta[:,:N//2])
        #Time evolution and correlators
        U_t = [np.exp(-1j*2*np.pi*E_GS*time) for time in measure_time_list]
        G_ij = get_Gij(alpha,beta,U_t)
        H_ij = get_Hij(alpha,beta,U_t)
        #Correlator in space-time
        ZZ = np.zeros((N,Nt),dtype=complex)
        for i in range(N):
            ZZ[i] = np.array([2*1j*np.imag(G_ij[i_t,i,0]*H_ij[i_t,i,0]) for i_t in range(Nt)])/N*2
        #2D FFT
        zz_correlator[i_sr] = np.fft.fftshift(np.fft.fft2(ZZ,[N,Nomega]))
    return zz_correlator

def compute_H(g_nn,h_field,N_sites):
    """NxN fermion Hamiltonian with PBC"""
    ham = np.zeros((N_sites,N_sites))
    for i in range(N_sites-1):
        ham[i,i+1] = g_nn
        ham[i+1,i] = g_nn
        ham[i,i] = h_field*(-1)**(i+1)
    ham[-1,-1] = h_field*(-1)**(N_sites)
    #PBC
    ham[-1,0] = -g_nn
    ham[0,-1] = -g_nn
    return ham

def compute_populations(psi_t,psi_GS):
    """Compute the populations of the single modes"""
    N = psi_GS.shape[0]
    alpha = psi_t[:,:N//2]
    beta = psi_GS
    result = np.real(np.diagonal(beta.T.conj()@alpha@alpha.T.conj()@beta))
    return result

def get_Gij(alpha,beta,U):
    """Compute correlator <c_i(t)^\dag c_j(0)>"""
    return np.array([(alpha@alpha.T.conj()@beta@np.diag(U[i_t])@beta.T.conj()).T for i_t in range(len(U))])

def get_Hij(alpha,beta,U):
    """Compute correlator <c_i(t) c_j(0)^\dag>"""
    N = beta.shape[0]
    return np.array([N/2*beta@np.diag(U[i_t].conj())@beta.T.conj() - beta@np.diag(U[i_t].conj())@beta.T.conj()@alpha@alpha.T.conj() for i_t in range(len(U))])

def get_Hamiltonian_parameters(time_steps,g_,h_):
    """Compute g(t) and h(t) for each time of the ramp."""
    g = np.zeros(time_steps)
    h = np.zeros(time_steps)
    for i in range(time_steps):
        g[i] = 0 + g_/(time_steps-1)*i
        h[i] = h_ - h_/(time_steps-1)*i
    return g,h

def get_data_filename(spec_name,pars,extension):
    """Define the filename for data. Pars is a list of tuple, each tuple has the argument and the number of decimals precision."""
    if extension=='.npy':
        dirname = os.getcwd()+'/data/'
    elif extension=='.png':
        dirname = os.getcwd()+'/figures/'
    fn = dirname+spec_name
    for i in pars:
        if type(i[0])==str:
            fn += '_'+i[0]
            continue
        fn += '_'+f"{i[0]:.{i[1]}f}"
    fn += extension
    return fn
