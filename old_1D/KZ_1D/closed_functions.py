import numpy as np
import scipy
from scipy.linalg import expm
import inputs

#
def time_evolve(args):
    #Time evolve wavefunction for each quench time
    h_t,J_t,steps,tau,gamma,result_dir,save_data,cluster = args
    #
    N = len(h_t)
    times = np.linspace(0,tau,steps)
    dt = times[1]-times[0]
    #
    filename = result_dir+inputs.names['t-ev']+'wf_'+inputs.pars_name(tau,dt,gamma)+'.npy'
    try:
        psi = np.load(filename)
    except:
        print("Closed time-evolution of tau = ",tau," ...")
        psi = np.zeros((steps,N,N),dtype=complex)     #steps->time, N -> number of sites,  N -> number of modes (evoleve all of them even if just the first N/2 will be occupied)
        H_0 = inputs.H_t(N,J_t,h_t,0)
        E_0, psi_0 = scipy.linalg.eigh(H_0)
        psi[0] = psi_0
        for s in range(1,steps):
            H_temp = -1j*2*np.pi*inputs.H_t(N,J_t,h_t,s)*dt
            exp_H = expm(H_temp)
            for m in range(N):
                psi[s,:,m] = np.matmul(exp_H,psi[s-1,:,m])
        #
        if save_data:
            np.save(filename,psi)
    return psi

def compute_fidelity(args):
    #Compute fidelity at all times for each quench time
    h_t,J_t,steps,tau,gamma,result_dir,save_data,cluster = args
    #
    N = len(h_t)
    times = np.linspace(0,tau,steps)
    dt = times[1]-times[0]
    #
    filename = result_dir+inputs.names['fid']+'wf_'+inputs.pars_name(tau,dt,gamma)+'.npy'
    try:
        fid = np.load(filename)
    except:
        #Find time evolved states
        filename_psi = result_dir+inputs.names['t-ev']+'wf_'+inputs.pars_name(tau,dt,gamma)+'.npy'
        try:
            psi = np.load(filename_psi)
        except:
            args2 = (h_t,J_t,tau,gamma,result_dir,True,cluster)
            psi = time_evolve(args2)
        #Compute fidelity
        print("Closed fidelity of tau = ",tau," ...")
        fid = np.zeros(steps)
        for s in range(steps):
            E_, evec = np.linalg.eigh(inputs.H_t(N,J_t,h_t,s))
            phi_gs = evec[:,:N//2]             #GS modes
            overlap_matrix = np.matmul(np.conjugate(psi[s,:,:N//2]).T,phi_gs)
            det = np.linalg.det(overlap_matrix)
            fid[s] = np.linalg.norm(det)**2
        if save_data:
            np.save(filename,fid)
    return fid

def compute_populations(args):
    #Compute mode population for low energy modes (N//2+-rg) at the critical point
    h_t,J_t,steps,tau,gamma,result_dir,save_data,cluster = args
    #
    N = len(h_t)
    times = np.linspace(0,tau,steps)
    dt = times[1]-times[0]
    #
    filename = result_dir+inputs.names['pop']+'wf_'+inputs.pars_name(tau,dt,gamma)+'.npy'
    try:
        pop = np.load(filename)
    except:
        #Find time evolved states
        filename_psi = result_dir+inputs.names['t-ev']+'wf_'+inputs.pars_name(tau,dt,gamma)+'.npy'
        try:
            psi = np.load(filename_psi)
        except:
            args2 = (h_t,J_t,tau,gamma,result_dir,True,cluster)
            psi = time_evolve(args2)
        print("Closed populations of tau = ",tau,"...")
        #
        ind_T = -1
        pop = np.zeros(N)
        H_F = inputs.H_t(N,J_t,h_t,ind_T)
        E_F, psi_0 = scipy.linalg.eigh(H_F) #energy and GS of system at time ind_T
        
        G = np.matmul(np.conjugate(psi[ind_T]).T,psi_0)
        G2 = G*np.conjugate(G)
        for k in range(N):
            pop[k] = np.real(G2[:N//2,k].sum())
        if save_data:
            np.save(filename,pop)
    return pop

def compute_CF_zz(args):
    #Compute mode population for low energy modes (N//2+-rg) at the critical point
    h_t,J_t,steps,tau,gamma,result_dir,save_data,cluster = args
    #
    N = len(h_t)
    times = np.linspace(0,tau,steps)
    dt = times[1]-times[0]
    #
    filename = result_dir+inputs.names['CF_zz']+'wf_'+inputs.pars_name(tau,dt,gamma)+'.npy'
    ind_t = -1       #time index in the quench
    try:
        CFzz = np.load(filename)
    except:
        #Find time evolved states
        filename_psi = result_dir+inputs.names['t-ev']+'wf_'+inputs.pars_name(tau,dt,gamma)+'.npy'
        try:
            psi = np.load(filename_psi)
        except:
            args2 = (h_t,J_t,tau,gamma,result_dir,True,cluster)
            psi = time_evolve(args2)
        print("Computing corr. fun. 'zz' of tau = ",tau," ...")
        CFzz = np.zeros(N)
        psi_t = psi[ind_t]
        out_ = 0
        #Test
        for r in range(1,N):
            Gr = 0
            Nr = 0
            for i in range(out_,N-out_):
                if r+i >= N-out_:
                    continue
                Nr += 1
                temp_i = 0          #REDUCE
                for s in range(N//2):       #just the GS
                    temp_i += psi_t[i,s]*np.conjugate(psi_t[i+r,s])
                Gr += temp_i*np.conjugate(temp_i)           #REDUCE
            if Nr:
                Gr /= Nr
                CFzz[r] = np.real(Gr) if abs(np.real(Gr)) > 1e-15 else np.nan
        if save:
            np.save(filename,CF)
    return CF

def compute_CF_xx(args):
    pass
def compute_nex(args):
    pass



