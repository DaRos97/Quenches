import numpy as np
import scipy
from scipy.linalg import expm

def H_t(N_,J_,h_,t_):
    H_ = np.zeros((N_,N_))
    for i in range(N_-1):
        H_[i,i+1] = H_[i+1,i] = J_[i][t_]/2
        H_[i,i] = h_[i][t_]*2
    H_[-1,-1] = h_[-1][t_]*2
    H_[-1,0] = H_[0,-1] = -J_[-1][t_]/2   #- for PBC
    return H_
#
def time_evolve(args):
    h_t,J_t,times_dic,list_Tau,result_dirname,save_data = args
    #
    psi_ = []
    for n,Tau in enumerate(list_Tau):
        dt = times_dic[Tau][1]-times_dic[Tau][0]
        name_pars = str(Tau)+'_'+"{:.4f}".format(dt).replace('.',',')
        N = len(h_t)
        steps = len(times_dic[Tau])
        #Time evolution
        filename = result_dirname+'time_evolved_wf_'+name_pars+'.npy'
        try:
            psi = np.load(filename)
            psi_.append(psi)
        except:
            print("Time evolution of Tau = ",Tau," ...")
            psi = np.zeros((steps,N,N),dtype=complex)     #steps->time, N -> number of sites,  N -> number of modes
            H_0 = H_t(N,J_t,h_t,0)
            E_0, psi_0 = scipy.linalg.eigh(H_0)
            psi[0] = psi_0[:,:N]
            for s in range(1,steps):
                H_temp = -1j*2*np.pi*H_t(N,J_t,h_t,s)*dt
                exp_H = expm(H_temp)
                for m in range(N):
                    psi[s,:,m] = np.matmul(exp_H,psi[s-1,:,m])
            psi_.append(psi)
            #
            if save_data:
                np.save(filename,psi)
    return psi_

def compute_fidelity(args):
    h_t,J_t,times_dic,list_Tau,result_dirname,save_data = args
    #
    fid_ = []
    for n,Tau in enumerate(list_Tau):
        dt = times_dic[Tau][1]-times_dic[Tau][0]
        name_pars = str(Tau)+'_'+"{:.4f}".format(dt).replace('.',',')
        N = len(h_t)
        steps = len(times_dic[Tau])
        #Time evolution
        filename = result_dirname+'fidelity_'+name_pars+'.npy'
        try:
            fid_.append(np.load(filename))
        except:
            #Find time evolved states
            filename_psi = result_dirname+'time_evolved_wf_'+name_pars+'.npy'
            try:
                psi = np.load(filename_psi)
            except:
                psi = time_evolve(args)[n]
            #Compute fidelity
            print("Computing fidelity of Tau = ",Tau," ...")
            fid_.append(np.zeros(steps))
            for s in range(steps):
                E_, evec = np.linalg.eigh(H_t(N,J_t,h_t,s))
                phi_gs = evec[:,:N//2]             #GS modes
                overlap_matrix = np.matmul(np.conjugate(psi[s,:,:N//2]).T,phi_gs)
                det = np.linalg.det(overlap_matrix)
                fid_[n][s] = np.linalg.norm(det)**2
            if save_data:
                np.save(filename,fid_[n])
    return fid_


def compute_nex(ind_T,args):
    h_t,J_t,times_dic,list_Tau,result_dirname,save_data = args
    #
    N = len(h_t)
    nex = []
    for n,Tau in enumerate(list_Tau):
        dt = times_dic[Tau][1]-times_dic[Tau][0]
        name_pars = str(Tau)+'_'+"{:.4f}".format(dt).replace('.',',')
        steps = len(times_dic[Tau])
        filename = result_dirname+'nex_'+name_pars+'.npy'
        try:
            nex.append(np.load(filename))
        except:
            print("Computing nex of Tau = ",Tau," ...")
            #Find time evolved states
            filename_psi = result_dirname+'time_evolved_wf_'+name_pars+'.npy'
            try:
                psi = np.load(filename_psi)
            except:
                psi = time_evolve(args)[n]
            #
            ind_t = len(times)//2 if ind_T == 2 else ind_T
            res = 0
            #
            H_F = H_t(N,J_t,h_t,ind_t)
            E_F, psi_0 = scipy.linalg.eigh(H_F) #energy and GS of system
            for k in range(N//2,N):
                temp_k = 0
                for l in range(N//2):
                    Gamma_kl = np.matmul(np.conjugate(psi[ind_t,:,l]).T,psi_0[:,k]).sum()
                    temp_k += np.linalg.norm(Gamma_kl)**2
                res += temp_k
            nex.append(res)

            if save_data:
                np.save(filename,res)
    return nex

def compute_pop_ev(args,rg):
    h_t,J_t,times_dic,list_Tau,result_dirname,save_data = args
    #
    N = len(h_t)
    #
    n_q = []
    for n,Tau in enumerate(list_Tau):
        dt = times_dic[Tau][1]-times_dic[Tau][0]
        name_pars = str(Tau)+'_'+"{:.4f}".format(dt).replace('.',',')
        steps = len(times_dic[Tau])
        filename = result_dirname+'n_q_'+name_pars+'.npy'
        try:
            n_q.append(np.load(filename))
        except:
            print("Computing n_q of Tau = ",Tau," ...")
            #Find time evolved states
            filename_psi = result_dirname+'time_evolved_wf_'+name_pars+'.npy'
            try:
                psi = np.load(filename_psi)
            except:
                psi = time_evolve(args)[n]
            #
            res = np.zeros((N,steps))
            for s in range(steps):
                H_F = H_t(N,J_t,h_t,s)
                E_F, psi_0 = scipy.linalg.eigh(H_F) #energy and GS of system
                for k in range(N//2-rg,N//2+rg):
                    temp_k = 0
                    for l in range(N//2):
                        Gamma_kl = np.matmul(np.conjugate(psi[s,:,l]).T,psi_0[:,k]).sum()
                        temp_k += np.linalg.norm(Gamma_kl)**2
                    res[k,s] = temp_k
            n_q.append(res)
            #
            if save_data:
                np.save(filename,res)
    return n_q


def pow_law(x,a,b):
    return a*np.abs(x)**(b)




