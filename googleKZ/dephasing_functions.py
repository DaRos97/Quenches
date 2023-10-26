import numpy as np
import scipy
from scipy.linalg import expm, sqrtm
from tqdm import tqdm

dirname_open = 'open_Data/'

cols = ['r','g','y','b','k','m','orange','forestgreen']

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
    h_t,J_t,gamma,times_dic,list_Tau,home_dirname,save_data = args
    result_dirname = home_dirname+dirname_open
    #
    eta_ = []
    for n,Tau in enumerate(list_Tau):
        dt = times_dic[Tau][1]-times_dic[Tau][0]
        name_pars = str(Tau)+'_'+"{:.4f}".format(dt).replace('.',',')+"_"+"{:.5f}".format(gamma).replace('.',',')
        N = len(h_t)
        steps = len(times_dic[Tau])
        #
        filename = result_dirname+'time_evolved_DM_'+name_pars+'.npy'
        try:
            eta = np.load(filename)
            eta_.append(eta)
        except:
            print("Time evolution of gamma = ",gamma," ...")
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
            eta_.append(eta)
            #
            if save_data:
                np.save(filename,eta)
    return eta_

def compute_fidelity(args):
    #Compute fidelity at all times for each quench time
    h_t,J_t,gamma,times_dic,list_Tau,home_dirname,save_data = args
    result_dirname = home_dirname+dirname_open
    #
    fid_ = []
    for n,Tau in enumerate(list_Tau):
        dt = times_dic[Tau][1]-times_dic[Tau][0]
        name_pars = str(Tau)+'_'+"{:.4f}".format(dt).replace('.',',')+"_"+"{:.5f}".format(gamma).replace('.',',')
        N = len(h_t)
        steps = len(times_dic[Tau])
        #Time evolution
        filename = result_dirname+'fidelity_'+name_pars+'.npy'
        try:
            fid_.append(np.load(filename))
        except:
            #Find time evolved states
            filename_rho = result_dirname+'time_evolved_DM_'+name_pars+'.npy'
            try:
                rho = np.load(filename_rho)
            except:
                args2 = (h_t,J_t,gamma,times_dic,[Tau,],home_dirname,1)#save_data)
                rho = time_evolve(args2)[0]
            #Compute fidelity
            print("Computing fidelity of gamma = ",gamma," and Tau = ",str(Tau)," ...")
            fid_.append(np.zeros(steps))
            for s in range(steps):
                rho_g = np.zeros((N,N),dtype=complex)     #steps->time, N**2 -> number of components of DM, N -> number of modes
                E_, psi_gs = np.linalg.eigh(H_t(N,h_t,J_t,s))   #GS at time-step s
                for m in range(N//2):
                    rho_g += np.outer(psi_gs[:,m],psi_gs[:,m])
                D_g, M_g = np.linalg.eigh(rho_g)
                rho_tilde = np.matmul(np.matmul(np.conjugate(M_g).T,rho[s]),M_g)
                nksp = np.real(np.diagonal(rho_tilde)[:N//2])
                nksm = np.real(np.diagonal(rho_tilde)[N//2:])
                fid_[-1][s] = np.prod(nksm)
            if save_data:
                np.save(filename,fid_[-1])
    return fid_

def compute_populations(args):
    #Compute mode population for low energy modes (N//2+-rg) at the critical point
    h_t,J_t,gamma,times_dic,list_Tau,home_dirname,save_data = args
    result_dirname = home_dirname+dirname_open
    #
    N = len(h_t)
    #
    n_q = []
    for n,Tau in enumerate(list_Tau):
        dt = times_dic[Tau][1]-times_dic[Tau][0]
        name_pars = str(Tau)+'_'+"{:.4f}".format(dt).replace('.',',')+"_"+"{:.5f}".format(gamma).replace('.',',')
        steps = len(times_dic[Tau])
        filename = result_dirname+'populations_'+name_pars+'.npy'
        try:
            n_q.append(np.load(filename))
        except:
            print("Computing n_q of Tau = ",Tau," ...")
            #Find time evolved states
            filename_rho = result_dirname+'time_evolved_DM_'+name_pars+'.npy'
            try:
                rho = np.load(filename_rho)
            except:
                args2 = (h_t,J_t,gamma,times_dic,[Tau,],home_dirname,1)#save_data)
                rho = time_evolve(args2)[0]
            #
            ind_T = -1
            if 0: #Use diagonalization of rho
                H_F = H_t(N,h_t,J_t,ind_T)
                E_F, psi_0 = scipy.linalg.eigh(H_F) #energy and GS of system at time ind_T
                rho_g = np.zeros((N,N),dtype=complex)     #GS(t) DM
                for m in range(N//2): #GS excited states
                    rho_g += np.outer(psi_0[:,m],psi_0[:,m])
                #Diagonalize
                D_t, M_t = np.linalg.eigh(rho[ind_T])
                D_g, M_g = np.linalg.eigh(rho_g)
                #
                res = np.zeros(N)
                G = np.matmul(np.conjugate(M_t).T,M_g)
                G2 = G*np.conjugate(G)
                for k in range(N):
                    res[k] = np.real(G2[:N//2,k].sum())
            else:   #Use Trace
                #H_fic = np.zeros((N,N),dtype=complex)
                #for i in range(N):
                #    H_fic[i,i] = (-1)**i
                H_fic = H_t(N,h_t,np.zeros((N,N)),0)
                D_g, M_g = np.linalg.eigh(H_fic)
                res = np.real(np.diagonal(np.matmul(np.matmul(np.conjugate(M_g).T,rho[ind_T]),M_g)))
                
                #res = np.zeros(N)
                #for k in range(N):
                #    n_k = np.zeros((N,N),dtype=complex)
                #    for i in range(N):
                #        for j in range(N):
                #            n_k[i,j] = np.exp(-1j*np.pi*(k)/N*(i-j))
                #    n_k_tilde = np.matmul(np.matmul(np.conjugate(M_t).T,n_k),M_t)
                #    print(k,np.sum(np.diagonal(n_k_tilde))/N)
                #exit()
                    #exp_val = np.trace(np.matmul(rho[ind_T],n_k))/N
                    #res[k] = np.absolute(exp_val)
            #
            n_q.append(res)
            #
            if save_data:
                np.save(filename,res)
    return n_q






























