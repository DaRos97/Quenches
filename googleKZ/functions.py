import numpy as np
import scipy
from scipy.linalg import expm
from tqdm import tqdm

dirname_closed = 'closed_Data/'

cols = ['r','g','y','b','k','m','orange','forestgreen']

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
    #Time evolve wavefunction for each quench time
    h_t,J_t,times_dic,list_Tau,home_dirname,save_data = args
    result_dirname = home_dirname+dirname_closed
    #
    psi_ = []
    for n,Tau in enumerate(list_Tau):
        dt = times_dic[Tau][1]-times_dic[Tau][0]
        name_pars = str(Tau)+'_'+"{:.4f}".format(dt).replace('.',',')
        N = len(h_t)
        steps = len(times_dic[Tau])
        #
        filename = result_dirname+'time_evolved_wf_'+name_pars+'.npy'
        try:
            psi = np.load(filename)
            psi_.append(psi)
        except:
            print("Time evolution of Tau = ",Tau," ...")
            psi = np.zeros((steps,N,N),dtype=complex)     #steps->time, N -> number of sites,  N -> number of modes (evoleve all of them even if just the first N/2 will be occupied)
            H_0 = H_t(N,J_t,h_t,0)
            E_0, psi_0 = scipy.linalg.eigh(H_0)
            psi[0] = psi_0
            for s in tqdm(range(1,steps)):
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
    #Compute fidelity at all times for each quench time
    h_t,J_t,times_dic,list_Tau,home_dirname,save_data = args
    result_dirname = home_dirname+dirname_closed
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
                args2 = (h_t,J_t,times_dic,[Tau,],home_dirname,save_data)
                psi = time_evolve(args2)[0]
            #Compute fidelity
            print("Computing fidelity of Tau = ",Tau," ...")
            fid_.append(np.zeros(steps))
            for s in range(steps):
                E_, evec = np.linalg.eigh(H_t(N,J_t,h_t,s))
                phi_gs = evec[:,:N//2]             #GS modes
                overlap_matrix = np.matmul(np.conjugate(psi[s,:,:N//2]).T,phi_gs)
                det = np.linalg.det(overlap_matrix)
                fid_[-1][s] = np.linalg.norm(det)**2
            if save_data:
                np.save(filename,fid_[-1])
    return fid_

def compute_nex(ind_T,args):
    #Compute density of excitations at time ind_T for each quench time
    h_t,J_t,times_dic,list_Tau,home_dirname,save_data = args
    result_dirname = home_dirname+dirname_closed
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
                args2 = (h_t,J_t,times_dic,[Tau,],home_dirname,save_data)
                psi = time_evolve(args)[0]
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

def compute_populations(args):
    #Compute mode population for low energy modes (N//2+-rg) at the critical point
    h_t,J_t,times_dic,list_Tau,home_dirname,save_data = args
    result_dirname = home_dirname+dirname_closed
    #
    N = len(h_t)
    #
    n_q = []
    for n,Tau in enumerate(list_Tau):
        dt = times_dic[Tau][1]-times_dic[Tau][0]
        name_pars = str(Tau)+'_'+"{:.4f}".format(dt).replace('.',',')
        steps = len(times_dic[Tau])
        filename = result_dirname+'populations_'+name_pars+'.npy'
        try:
            n_q.append(np.load(filename))
        except:
            print("Computing n_q of Tau = ",Tau," ...")
            #Find time evolved states
            filename_psi = result_dirname+'time_evolved_wf_'+name_pars+'.npy'
            try:
                psi = np.load(filename_psi)
            except:
                args2 = (h_t,J_t,times_dic,[Tau,],home_dirname,save_data)
                psi = time_evolve(args2)[0]
            #
            res = np.zeros(N)
            ind_T = -1
            H_F = H_t(N,J_t,h_t,ind_T)
            E_F, psi_0 = scipy.linalg.eigh(H_F) #energy and GS of system at time ind_T
            G = np.matmul(np.conjugate(psi[ind_T]).T,psi_0)
            G2 = G*np.conjugate(G)
            for k in range(N):
                res[k] = np.real(G2[:N//2,k].sum())
            n_q.append(res)
            #
            if save_data:
                np.save(filename,res)
    return n_q

def compute_energy_state(args):
    h_t,J_t,times_dic,list_Tau,home_dirname,save_data = args
    result_dirname = home_dirname+dirname_closed
    #
    N = len(h_t)
    #
    en_CP = np.zeros(len(list_Tau))
    for n,Tau in enumerate(list_Tau):
        dt = times_dic[Tau][1]-times_dic[Tau][0]
        name_pars = str(Tau)+'_'+"{:.4f}".format(dt).replace('.',',')
        steps = len(times_dic[Tau])
        filename = result_dirname+'energy_state_'+name_pars+'.npy'
        try:
            en_CP.append(np.load(filename))
        except:
            print("Computing energy of state at CP of Tau = ",Tau," ...")
            #Find time evolved states
            filename_psi = result_dirname+'time_evolved_wf_'+name_pars+'.npy'
            try:
                psi = np.load(filename_psi)
            except:
                args2 = (h_t,J_t,times_dic,[Tau,],home_dirname,save_data)
                psi = time_evolve(args2)[0]
            #
            ind_T = -1      #Critical point
            #
            Ham = H_t(N,J_t,h_t,ind_T)
            eigs_GS, psi_GS = scipy.linalg.eigh(Ham)
            En_GS = sum(eigs_GS[:N//2])     #Energy of GS
            En_psi = 0 
            for k in range(N):
                temp_k = 0
                for l in range(N//2):
                    Gamma_kl = np.matmul(np.conjugate(psi[ind_T,:,l]).T,psi_GS[:,k]).sum()
                    temp_k += np.linalg.norm(Gamma_kl)**2
                En_psi += temp_k*eigs_GS[k]
            en_CP[n] = abs((En_psi-En_GS)/En_GS)
            if save_data:
                np.save(filename,en_CP[n])
    return en_CP

def pow_law(x,a,b):
    return a*np.abs(x)**(b)




