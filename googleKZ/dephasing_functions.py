import numpy as np
import scipy
from scipy.linalg import expm
from tqdm import tqdm

dirname_open = 'open_Data/'

cols = ['r','g','y','b','k','m','orange','forestgreen']

def H_t(N_,J_,h_,t_):
    H_ = np.zeros((N_,N_))
    for i in range(N_-1):
        H_[i,i+1] = H_[i+1,i] = J_[i][t_]/2
        H_[i,i] = h_[i][t_]*2
    H_[-1,-1] = h_[-1][t_]*2
    H_[-1,0] = H_[0,-1] = -J_[-1][t_]/2   #- for PBC
    return H_

def L_j(N_,site):
    L = np.zeros((N_,N_))
    L[site,site] = 1
    return L

def S_t(N_,J_,h_,t_,gamma):
    ham = H_t(N_,J_,h_,t_)
    S = -1j*2*np.pi*(L_A(ham)-R_A(ham))
    for i in range(N_):
        L_i = L_j(N_,i)
        S += gamma*(np.matmul(L_A(L_i),R_A(L_i))-1/2*(L_A(np.matmul(L_i,L_i)) + R_A(np.matmul(L_i,L_i))))
    return S

def L_A(A):     #Left action
    return np.kron(A,np.identity(A.shape[0]))
def R_A(A):     #Right action
    return np.kron(np.identity(A.shape[0]),A.T)

def time_evolve(args):
    h_t,J_t,list_gamma,times_dic,Tau,home_dirname,save_data = args
    result_dirname = home_dirname+dirname_open
    #
    rho_ = []
    for n,gamma in enumerate(list_gamma):
        dt = times_dic[Tau][1]-times_dic[Tau][0]
        name_pars = str(Tau)+'_'+"{:.4f}".format(dt).replace('.',',')+"_"+"{:.5f}".format(gamma).replace('.',',')
        N = len(h_t)
        steps = len(times_dic[Tau])
        #
        filename = result_dirname+'time_evolved_wf_'+name_pars+'.npy'
        try:
            rho = np.load(filename)
            rho_.append(rho)
        except:
            print("Time evolution of gamma = ",gamma," ...")
            rho = np.zeros((steps,N**2,N),dtype=complex)     #steps->time, N**2 -> number of components of DM, N -> number of modes
#            psi = np.zeros((steps,N,N),dtype=complex)     #steps->time, N -> number of sites,  N -> number of modes (evoleve all of them even if just the first N/2 will be occupied)
            E_0, psi_0 = scipy.linalg.eigh(H_t(N,J_t,h_t,0))
            rho_0 = np.zeros((N**2,N),dtype=complex)     #steps->time, N**2 -> number of components of DM, N -> number of modes
            for m in range(N):
                rho_0[:,m] = np.reshape(np.outer(psi_0[:,m],psi_0[:,m]),(N**2))
            rho[0] = rho_0
#            psi[0] = psi_0
            S_L = np.zeros((N**2,N**2),dtype=complex)
            for i in range(N):
                L_i = L_j(N,i)
                S_L += gamma*(np.matmul(L_A(L_i),R_A(L_i))-1/2*(L_A(np.matmul(L_i,L_i)) + R_A(np.matmul(L_i,L_i))))
            for s in tqdm(range(1,steps)):
                ham = H_t(N,J_t,h_t,s)
                S_H = -1j*2*np.pi*(L_A(ham)-R_A(ham))
                exp_S = expm(S_H+S_L)
                for m in range(N):
                    rho[s,:,m] = np.matmul(exp_S,rho[s-1,:,m])
            rho_.append(rho)
            #
            if save_data:
                np.save(filename,rho)
    return rho_

def compute_fidelity(args):
    #Compute fidelity at all times for each quench time
    h_t,J_t,list_gamma,times_dic,Tau,home_dirname,save_data = args
    result_dirname = home_dirname+dirname_open
    #
    fid_ = []
    for n,gamma in enumerate(list_gamma):
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
            filename_rho = result_dirname+'time_evolved_wf_'+name_pars+'.npy'
            try:
                rho = np.load(filename_rho)
            except:
                args2 = (h_t,J_t,[gamma,],times_dic,Tau,home_dirname,True)
                rho = time_evolve(args2)[0]
            #Compute fidelity
            print("Computing fidelity of gamma = ",gamma," ...")
            fid_.append(np.zeros(steps))
            for s in range(steps):
                rho_g = np.zeros((N,N),dtype=complex)     #steps->time, N**2 -> number of components of DM, N -> number of modes
                rho_s = np.zeros((N,N),dtype=complex)     #steps->time, N**2 -> number of components of DM, N -> number of modes
                E_, psi_gs = np.linalg.eigh(H_t(N,J_t,h_t,s))   #GS at time-step s
                for m in range(N//2):
                    rho_g += np.outer(psi_gs[:,m],psi_gs[:,m])
                    rho_s += np.reshape(rho[s,:,m],(N,N))
                D_s, M_s = np.linalg.eigh(rho_s)
                D_g, M_g = np.linalg.eigh(rho_g)
                overlap_matrix = np.matmul(np.conjugate(M_s[:,:N//2]).T,M_g[:,:N//2])
                det = np.linalg.det(overlap_matrix)
                fid_[-1][s] = np.absolute(det)**2
                #print(fid_[-1][s])
            if save_data:
                np.save(filename,fid_[-1])
    return fid_































