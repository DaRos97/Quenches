import numpy as np
from scipy.linalg import expm
import scipy
import os
from tqdm import tqdm
import pickle

def get_ramp_evolution(*args):
    """Compute the time evolved wave function along the ramp and at each time step evaluate populations and fidelity."""
    N,g_t_i,h_t_i,time_step,filling = args
    modes = int(N*filling)
    time_steps = g_t_i.shape[0]
    time_evolved_psi = np.zeros((time_steps,N,N),dtype=complex)    #wavefunction at all time steps of the ramp
    fidelity = np.zeros(time_steps)
    populations = np.zeros((time_steps,N))
    energies = np.zeros(time_steps)
    print("Time evolution")
    for it in tqdm(range(time_steps)):
        if it==0:
            g_i = np.zeros(N)
            h_i = np.zeros(N)
            indices_p = {3/6:[0,2,4], 2/6:[0,2,4,5], 1/6:[0,2,3,4,5], 4/6: [0,2], 5/6: [0,]}
            indices_m = {3/6:[1,3,5], 2/6:[1,3], 1/6:[1,], 4/6: [1,3,4,5], 5/6: [1,2,3,4,5]}
            for i in range(N//6):
                for ip in indices_p[filling]:
                    h_i[(N//6-1)*i+ip] = -15
                for im in indices_m[filling]:
                    h_i[(N//6-1)*i+im] = 15
            E_GS,psi_GS = scipy.linalg.eigh(compute_H(g_i,h_i,N,1))      #exact GS at t=0
            time_evolved_psi[it] = psi_GS
            #
            fidelity[it] = np.absolute(scipy.linalg.det(time_evolved_psi[it,:,:modes].T.conj()@psi_GS[:,:modes]))**2
            populations[it] = compute_populations(time_evolved_psi[it],psi_GS)
            energies[it] = np.sum(populations[it]*E_GS)/N - np.sum(h_i)/N/2   #remove offset energy of magnetif field
        else:
            exp_H = expm(-1j*2*np.pi*compute_H(g_t_i[it],h_t_i[it],N,1)*time_step)
            #exp_H = expm(-1j*compute_H(g_t[it],h_t[it],N,1)*time_step)
            for m in range(N):  #evolve each mode independently
                time_evolved_psi[it,:,m] = exp_H @ time_evolved_psi[it-1,:,m]
            #Fidelity wrt real GS
            E_GS,psi_GS = scipy.linalg.eigh(compute_H(g_t_i[it],h_t_i[it],N,1))
            fidelity[it] = np.absolute(scipy.linalg.det(time_evolved_psi[it,:,:modes].T.conj()@psi_GS[:,:modes]))**2
            #Occupation of populations
            populations[it] = compute_populations(time_evolved_psi[it],psi_GS)
            energies[it] = np.sum(populations[it]*E_GS)/N - np.sum(h_t_i[it])/N/2   #remove offset energy of magnetif field
    return time_evolved_psi, fidelity, populations, energies

def compute_H(g_nn,h_field,N_sites,P=1):
    """NxN fermion Hamiltonian with PBC"""
    ham = np.zeros((N_sites,N_sites))
    h_list = h_field
    for i in range(N_sites-1):
        ham[i,i+1] = g_nn[i]
        ham[i+1,i] = g_nn[i]
        ham[i,i] = h_list[i]#*(-1)**(i+1)
    ham[-1,-1] = h_list[-1]#*(-1)**(N_sites)
    #PBC
    ham[-1,0] = P*g_nn[i]    #*P fermion number
    ham[0,-1] = P*g_nn[i]
    return ham

def compute_correlator(correlator_type,*args):
    """Compute the correlator and its Fourier transform."""
    N,omega_list,stop_ratio_list,measure_time_list,g_t_i,h_t_i,time_evolved_psi,use_time_evolved,filling = args
    modes = int(N*filling)
    Nt = len(measure_time_list)
    Nomega = len(omega_list)
    time_steps = g_t_i.shape[0]
    #
    corr_kw = np.zeros((len(stop_ratio_list),N,Nomega),dtype=complex)
    corr_spacetime = np.zeros((len(stop_ratio_list),N,Nt),dtype=complex)
    for i_sr in tqdm(range(len(stop_ratio_list))):
        stop_ratio = stop_ratio_list[i_sr]
        i_t = int(time_steps*stop_ratio)-1
        E_GS,beta = scipy.linalg.eigh( compute_H(g_t_i[i_t],h_t_i[i_t],N,1) )
        if use_time_evolved:
            alpha = np.copy(time_evolved_psi[i_t][:,:modes])
        else:
            alpha = np.copy(beta[:,:modes])
        #Time evolution and correlators
        U_t = [np.exp(-1j*2*np.pi*E_GS*time) for time in measure_time_list]
        G_ij = get_Gij(alpha,beta,U_t)/np.sqrt(modes)
        H_ij = get_Hij(alpha,beta,U_t)/np.sqrt(modes)
        #Correlator in space-time: here we decide which correlator we are computing
        corr_xt = np.zeros((N,Nt),dtype=complex)
        func_corr = dic_correlators[correlator_type]
        for i in range(N):
            corr_xt[i] = np.array([func_corr(G_ij[i_t],H_ij[i_t],i) for i_t in range(Nt)])
        corr_spacetime[i_sr] = corr_xt
        #2D FFT
        corr_kw[i_sr] = np.fft.fftshift(np.fft.fft2(corr_xt,[N,Nomega]))
    return corr_kw, corr_spacetime

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
    """Compute correlator <c_i(t)^dag c_j(0)>"""
    return np.array([(alpha@alpha.T.conj()@beta@np.diag(U[i_t])@beta.T.conj()).T for i_t in range(len(U))])

def get_Hij(alpha,beta,U):
    """Compute correlator <c_i(t) c_j(0)^dag>"""
    N = beta.shape[0]
    return np.array([N/2*beta@np.diag(U[i_t].conj())@beta.T.conj() - beta@np.diag(U[i_t].conj())@beta.T.conj()@alpha@alpha.T.conj() for i_t in range(len(U))])

def get_Hamiltonian_parameters(time_steps,g_in,g_fin,h_in,h_fin):
    """Compute g(t) and h(t) for each time and site of the ramp."""
    g_t_i = np.zeros((time_steps,len(g_in)))   #time and space
    h_t_i = np.zeros((time_steps,len(g_in)))   #time and space
    for it in range(time_steps):
        for i in range(len(g_in)):
            g_t_i[it,i] = np.linspace(g_in[i],g_fin[i],time_steps)[it]
            h_t_i[it,i] = np.linspace(h_in[i],h_fin[i],time_steps)[it]
    return g_t_i,h_t_i

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

def extract_experimental_parameters(fn):
    """Import experimental Hamiltonian parameters"""
    with open(fn,'rb') as f:
        data = pickle.load(f)
    pairs = list(data['xx'].keys())
    g = np.zeros(len(pairs))
    for i in range(len(pairs)):
        g[i] = data['xx'][pairs[i]]
    sites = list(data['z'].keys())
    h = np.zeros(len(sites))
    for i in range(len(sites)):
        h[i] = data['z'][sites[i]]
    return g*1000,h*1000        #put it in MHz
