import numpy as np
import scipy
from scipy.linalg import expm, sqrtm
from tqdm import tqdm

names = {   't-ev':'time_evolved_DM_',
            'fid':'fidelity_DM_',
            'pop':'populatins_DM_',
        }
def pars_name(tau,gamma,dt):
    return "{:.1f}".format(tau)+'_'+"{:.5f}".format(gamma).replace('.',',')+"_"+"{:.5f}".format(dt).replace('.',',')

def H_t(N_,h_,J_,t_):
    H_ = np.zeros((N_,N_))
    for i in range(N_-1):
        H_[i,i+1] = H_[i+1,i] = J_[i][t_]/2
        H_[i,i] = h_[i][t_]*2
    H_[-1,-1] = h_[-1][t_]*2
    H_[-1,0] = H_[0,-1] = -J_[-1][t_]/2   #- for PBC
    return H_

def Lindblad_i(N_,site):
    L = np.zeros((N_,N_))
    L[site,site] = 1
    return L

def L_A(A):     #Left action
    return np.kron(A,np.identity(A.shape[0]))
def R_A(A):     #Right action
    return np.kron(np.identity(A.shape[0]),A.T)

def time_evolve(args):
    h_t,J_t,times,tau,gamma,result_dir,save_data = args
    #
    dt = times[1]-times[0]
    name_pars = pars_name(tau,gamma,dt) 
    N = len(h_t)
    steps = len(times)
    #
    filename = result_dir+names['t-ev']+name_pars+'.npy'
    try:
        eta = np.load(filename)
    except:
        print("Open time-evolution of tau = ",tau,", gamma = ",gamma," ...")
        rho = np.zeros((steps,N**2),dtype=complex)
        eta = np.zeros((steps,N,N),dtype=complex)
        #initial state
        E_0, psi_0 = scipy.linalg.eigh(H_t(N,h_t,J_t,0))
        for m in range(N//2):
            rho[0] += np.reshape(np.outer(psi_0[:,m],psi_0[:,m]),(N**2))
        eta[0] = np.reshape(rho[0],(N,N))
        #Lindblad part of master equation -> t-independent
        S_L = np.zeros((N**2,N**2),dtype=complex)
        for i in range(N):
            L_i = Lindblad_i(N,i)
            S_L += gamma*(np.matmul(L_A(L_i),R_A(L_i))-1/2*(L_A(L_i) + R_A(L_i)))
        #time evolution
        for s in tqdm(range(1,steps)):
            ham = H_t(N,h_t,J_t,s)
            S_H = -1j*2*np.pi*(L_A(ham)-R_A(ham))
            exp_S = expm((S_H+S_L)*dt)      #dt is IMPORTANT !!!
            rho[s] = np.matmul(exp_S,rho[s-1])
            eta[s] = np.reshape(rho[s],(N,N))
        #
        if save_data:
            np.save(filename,eta)
    return eta

def compute_fidelity(args):
    #Compute fidelity at all times for each quench time
    h_t,J_t,times,tau,gamma,result_dir,save_data = args
    #
    dt = times[1]-times[0]
    name_pars = pars_name(tau,gamma,dt) 
    N = len(h_t)
    steps = len(times)
    #Time evolution
    filename = result_dir+names['fid']+name_pars+'.npy'
    try:
        fid = np.load(filename)
    except:
        #Find time evolved states
        filename_rho = result_dir+names['t-ev']+name_pars+'.npy'
        try:
            rho = np.load(filename_rho)
        except:
            args2 = (h_t,J_t,times,tau,gamma,result_dir,1)
            rho = time_evolve(args2)
        #Compute fidelity
        print("Open fidelity of tau = ",tau,", gamma = ",gamma," ...")
        fid = np.zeros(steps)
        for s in range(steps):
            rho_g = np.zeros((N,N),dtype=complex)     #steps->time, N**2 -> number of components of DM, N -> number of modes
            E_, psi_gs = np.linalg.eigh(H_t(N,h_t,J_t,s))   #GS at time-step s
            for m in range(N//2):
                rho_g += np.outer(psi_gs[:,m],psi_gs[:,m])
            D_g, M_g = np.linalg.eigh(rho_g)
            rho_tilde = np.matmul(np.matmul(np.conjugate(M_g).T,rho[s]),M_g)
            nksp = np.real(np.diagonal(rho_tilde)[:N//2])
            nksm = np.real(np.diagonal(rho_tilde)[N//2:])
            fid[s] = np.prod(nksm)
        if save_data:
            np.save(filename,fid)
    return fid

def compute_populations(args):
    #Compute mode population for low energy modes (N//2+-rg) at the critical point
    h_t,J_t,times,tau,gamma,result_dir,save_data = args
    #
    dt = times[1]-times[0]
    name_pars = pars_name(tau,gamma,dt) 
    N = len(h_t)
    steps = len(times)
    #Time evolution
    filename = result_dir+names['pop']+name_pars+'.npy'
    try:
        pop = np.load(filename)
    except:
        #Find time evolved states
        filename_rho = result_dir+names['t-ev']+name_pars+'.npy'
        try:
            rho = np.load(filename_rho)
        except:
            args2 = (h_t,J_t,times,tau,gamma,home_dirname,1)
            rho = time_evolve(args2)
        print("Open populations of tau = ",tau,", gamma = ",gamma," ...")
        #
        ind_T = -1
        rho_g = np.zeros((N,N))     #steps->time, N**2 -> number of components of DM, N -> number of modes
        E_, psi_gs = np.linalg.eigh(H_t(N,h_t,J_t,ind_T))   #GS at time-step s
        for m in range(N//2):
            rho_g += np.outer(psi_gs[:,m],psi_gs[:,m])
        D_g, M_g = np.linalg.eigh(rho_g)
        M_g = psi_gs
        pop = np.real(np.diagonal(np.matmul(np.matmul(M_g.T,rho[ind_T]),M_g)))
        
        #sm = np.real(np.diagonal(np.matmul(np.matmul(np.conjugate(M_g).T,rho[ind_T]),M_g)))[:N//2]
        #bg = np.real(np.diagonal(np.matmul(np.matmul(np.conjugate(M_g).T,rho[ind_T]),M_g)))[N//2:]
        #
        #res = list(np.flip(np.sort(bg)))+list(np.flip(np.sort(sm)))
        #print(np.sum(res))

        if save_data:
            np.save(filename,pop)
    return pop

def compute_zphases(dm,h_t,J_t,type_):
    import matplotlib.pyplot as plt
    N = dm[0][0].shape[0]
    ind_T = -1
    tau = 0
    if type_=='random':
        n_p = 100
        list_ph = np.linspace(0,2*np.pi,n_p,endpoint=True)
        for i in range(1,N-1):
            h1 = h_t[i][ind_T]
            h2 = h_t[i+1][ind_T]
            J = J_t[i][ind_T]
            E = J*2*np.real(dm[tau][ind_T,i,i+1]*np.exp(1j*list_ph)) + J*2*np.real(dm[tau][ind_T,i-1,i]*np.exp(1j*list_ph)) #+ h1*np.real(dm[tau][ind_T,i,i] + h2*dm[tau][ind_T,i+1,i+1])
            plt.plot(list_ph,E,label=str(i))
#            plt.text(-0.05-(i%5)/10,E[0],str(i))
#        plt.xlim(-0.5,2*np.pi)
#        plt.legend()
        plt.show()



























