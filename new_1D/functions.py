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
            time_evolved_psi[it] = scipy.linalg.eigh(compute_H(g_t[it],h_t[it],N,1))[1]      #exact GS at t=0
        else:
            exp_H = expm(-1j*2*np.pi*compute_H(g_t[it],h_t[it],N,1)*time_step)
            #exp_H = expm(-1j*compute_H(g_t[it],h_t[it],N,1)*time_step)
            for m in range(N):  #evolve each mode independently
                time_evolved_psi[it,:,m] = exp_H @ time_evolved_psi[it-1,:,m]
        #Fidelity wrt real GS
        E_GS,psi_GS = scipy.linalg.eigh(compute_H(g_t[it],h_t[it],N,1))
        fidelity[it] = np.absolute(scipy.linalg.det(time_evolved_psi[it,:,:N//2].T.conj()@psi_GS[:,:N//2]))**2
        #Occupation of populations
        populations[it] = compute_populations(time_evolved_psi[it],psi_GS)
    return time_evolved_psi, fidelity,populations

def compute_H(g_nn,h_field,N_sites,P=1):
    """NxN fermion Hamiltonian with PBC"""
    ham = np.zeros((N_sites,N_sites))
    h_list = np.ones(N_sites)*h_field
    for i in range(N_sites-1):
        ham[i,i+1] = g_nn
        ham[i+1,i] = g_nn
        ham[i,i] = h_list[i]*(-1)**(i+1)
    ham[-1,-1] = h_list[-1]*(-1)**(N_sites)
    #PBC
    ham[-1,0] = P*g_nn    #*P fermion number
    ham[0,-1] = P*g_nn
    return ham

def compute_correlator(correlator_type,*args):
    """Compute the correlator and its Fourier transform."""
    N,omega_list,stop_ratio_list,measure_time_list,g_t,h_t,time_evolved_psi,use_time_evolved = args
    Nt = len(measure_time_list)
    Nomega = len(omega_list)
    time_steps = len(g_t)
    #
    corr_kw = np.zeros((len(stop_ratio_list),N,Nomega),dtype=complex)
    for i_sr in tqdm(range(len(stop_ratio_list))):
        stop_ratio = stop_ratio_list[i_sr]
        i_t = int(time_steps*stop_ratio)-1
        E_GS,beta = scipy.linalg.eigh( compute_H(g_t[i_t],h_t[i_t],N,1) )
        if use_time_evolved:
            alpha = np.copy(time_evolved_psi[i_t][:,:N//2])
        else:
            alpha = np.copy(beta[:,:N//2])
        #Time evolution and correlators
        U_t = [np.exp(-1j*2*np.pi*E_GS*time) for time in measure_time_list]
        G_ij = get_Gij(alpha,beta,U_t)/np.sqrt(N/2)
        H_ij = get_Hij(alpha,beta,U_t)/np.sqrt(N/2)
        #Correlator in space-time: here we decide which correlator we are computing
        corr_xt = np.zeros((N,Nt),dtype=complex)
        func_corr = dic_correlators[correlator_type]
        for i in range(N):
            corr_xt[i] = np.array([func_corr(G_ij[i_t],H_ij[i_t],i) for i_t in range(Nt)])
        #2D FFT
        corr_kw[i_sr] = np.fft.fftshift(np.fft.fft2(corr_xt,[N,Nomega]))
    return corr_kw

def correlator_zz(G,H,i):
    ip1 = (i+1)%G.shape[0]  #i+1 is modulo N -> N+1=1
    return 2*1j*np.imag(G[i,0]*H[i,0])
def correlator_ee(G,H,i):
    ip1 = (i+1)%G.shape[0]
    return 2*1j*np.imag(G[i,1]*H[ip1,0]+G[i,0]*H[ip1,1]+G[ip1,1]*H[i,0]+G[ip1,0]*H[i,1])
def correlator_jj(G,H,i):
    ip1 = (i+1)%G.shape[0]
    return -2*1j*np.imag(G[i,1]*H[ip1,0]-G[i,0]*H[ip1,1]-G[ip1,1]*H[i,0]+G[ip1,0]*H[i,1])
def correlator_ez(G,H,i):
    ip1 = (i+1)%G.shape[0]
    return 2*1j*np.imag(G[i,0]*H[ip1,0]+G[ip1,0]*H[i,0])
def correlator_ze(G,H,i):
    ip1 = (i+1)%G.shape[0]
    return 2*1j*np.imag(G[i,1]*H[i,0]+G[i,0]*H[i,1])
def correlator_jz(G,H,i):
    ip1 = (i+1)%G.shape[0]
    return -2*np.imag(G[i,0]*H[ip1,0]-G[ip1,0]*H[i,0])
def correlator_zj(G,H,i):
    ip1 = (i+1)%G.shape[0]
    return -2*np.imag(G[i,1]*H[i,0]-G[i,0]*H[i,1])
def correlator_ej(G,H,i):
    ip1 = (i+1)%G.shape[0]
    return -2*np.imag(G[i,1]*H[ip1,0]-G[i,0]*H[ip1,1]+G[ip1,1]*H[i,0]-G[ip1,0]*H[i,1])
def correlator_je(G,H,i):
    ip1 = (i+1)%G.shape[0]
    return -2*np.imag(G[i,1]*H[ip1,0]+G[i,0]*H[ip1,1]-G[ip1,1]*H[i,0]-G[ip1,0]*H[i,1])

dic_correlators = {
    'zz':correlator_zz,
    'ee':correlator_ee,
    'jj':correlator_jj,
    'ez':correlator_ez,
    'ze':correlator_ze,
    'jz':correlator_jz,
    'zj':correlator_zj,
    'ej':correlator_ej,
    'je':correlator_je,
}

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
    g_t = np.linspace(0,10,time_steps)
    h_t = np.linspace(15,0,time_steps)
    return g_t,h_t

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
