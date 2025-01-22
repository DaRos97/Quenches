import numpy as np
from scipy.linalg import expm
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
from pfapack import pfaffian as pf
plt.rcParams.update({"text.usetex": True,})
from contextlib import redirect_stdout
from itertools import combinations


def H_t(N_,J_,h_,BC_):
    H_ = np.zeros((N_,N_))
    for i in range(N_-1):
        H_[i,i+1] = H_[i+1,i] = J_       
        H_[i,i] = h_*(-1)**(i+1)
    H_[-1,-1] = h_*(-1)**(N_)
    H_[-1,0] = H_[0,-1] = -J_
    return H_

#Real and linear quenches
def h_t_real(t_,Tau_):
    return np.arctan(t_/Tau_)
def h_t_linear(t_,Tau_):
    return -t_/Tau_
h_t_dic = {'real':h_t_real, 'linear':h_t_linear}
def J_t_real(t_,Tau_):
    sigma = Tau_*np.tan(1/2)/np.sqrt(np.log(2))
    return np.exp(-(t_/sigma)**2)
def J_t_linear(t_,Tau_):
    jj = 1
    try:
        return np.ones(len(t_))*jj
    except:
        return jj
J_t_dic = {'real':J_t_real, 'linear':J_t_linear}
def time_span_real(Tau,steps):
    return np.linspace(-np.tan(1)*Tau,np.tan(1)*Tau,steps,endpoint=True)
def time_span_linear(Tau,steps):
    return np.linspace(-Tau,Tau,steps,endpoint=True)
time_span_dic = {'real':time_span_real, 'linear':time_span_linear}
datadir_dic = {'real':'Real_Quench/', 'linear':'Linear_Quench'}
#
def compute_gap(args):
    N,dt,list_Tau,BC,type_of_quench = args
    datadir = datadir_dic[type_of_quench]
    h_t = h_t_dic[type_of_quench]
    J_t = J_t_dic[type_of_quench]
    time_span = time_span_dic[type_of_quench]
    #
    gap_ = []
    Tau = list_Tau[-1]
    total_ev_time_per_Tau = time_span(1,2)[-1]-time_span(1,2)[0]
    steps = int(total_ev_time_per_Tau * Tau / dt)
    times = time_span(Tau,steps)
    J = J_t(times,Tau)
    h = h_t(times,Tau)
    gap = np.zeros(steps)
    fig = plt.figure()
    for s in range(steps):
        H_0 = H_t(N,J[s],h[s],BC)
        E_0 = scipy.linalg.eigvalsh(H_0)
        gap[s] = E_0[N//2]-E_0[N//2-1]
    plt.plot(times,gap)
    plt.title("Gap at 0 ="+str(gap[steps//2]))
    plt.show()
    return gap
def compute_GSE(args):
    N,dt,list_Tau,BC,type_of_quench = args
    datadir = datadir_dic[type_of_quench]
    h_t = h_t_dic[type_of_quench]
    J_t = J_t_dic[type_of_quench]
    time_span = time_span_dic[type_of_quench]
    Tau = list_Tau[-1]
    total_ev_time_per_Tau = time_span(1,2)[-1]-time_span(1,2)[0]
    steps = int(total_ev_time_per_Tau * Tau / dt)
    times = time_span(Tau,steps)
    J = J_t(times,Tau)
    h = h_t(times,Tau)
    GSE = np.zeros(steps)
    for s in range(steps):
        H_0 = H_t(N,J[s],h[s],BC)
        E_0 = scipy.linalg.eigvalsh(H_0)
        GSE[s] = np.sum(E_0[:N//2])
    fig = plt.figure()
    plt.plot(h,GSE)
    plt.show()
    if False:
        savefile = "en_GS.txt"
        with open(savefile, 'w') as f:
            with redirect_stdout(f):
                print("J\th\tGSE")
                for s in range(steps):
                    print(str(J[s]),'\t',"{:.3f}".format(h[s]),'\t',"{:.4f}".format(GSE[s]))
    return GSE

def time_evolve(args):
    N,dt,list_Tau,BC,type_of_quench = args
    datadir = datadir_dic[type_of_quench]
    h_t = h_t_dic[type_of_quench]
    J_t = J_t_dic[type_of_quench]
    time_span = time_span_dic[type_of_quench]
    #
    psi_ = []
    times_ = []
    total_ev_time_per_Tau = time_span(1,2)[-1]-time_span(1,2)[0]
    for n,Tau in enumerate(list_Tau):
        name_pars = str(N)+'_'+str(Tau)+'-'+str(dt)+'_'+BC
        steps = int(total_ev_time_per_Tau * Tau / dt)
        steps = steps+1 if steps%2==0 else steps
        times = time_span(Tau,steps)
        times_.append(times)
        #Time evolution
        filename = datadir+'time_evolved_wf_'+name_pars+'.npy'
        try:
            psi = np.load(filename)
            psi_.append(psi)
        except:
            print("Time evolution of Tau = ",Tau," ...")
            J = J_t(times,Tau)
            h = h_t(times,Tau)
            psi = np.zeros((steps,N,N),dtype=complex)#N//2 (last)     #steps->time, N -> number of sites,  N -> number of modes
            H_0 = H_t(N,J[0],h[0],BC)
            E_0, psi_0 = scipy.linalg.eigh(H_0)
            psi[0] = psi_0[:,:N]            #N//2
            for s in tqdm(range(1,steps)):
                H_temp = -1j*H_t(N,J[s],h[s],BC)*dt
                exp_H = expm(H_temp)
                for m in range(N):  #N//2
                    psi[s,:,m] = np.matmul(exp_H,psi[s-1,:,m])
            np.save(filename,psi)
            psi_.append(psi)
    return times_, psi_

def compute_fidelity(args):
    N,dt,list_Tau,BC,type_of_quench = args
    datadir = datadir_dic[type_of_quench]
    h_t = h_t_dic[type_of_quench]
    J_t = J_t_dic[type_of_quench]
    time_span = time_span_dic[type_of_quench]
    #
    fid_ = []
    total_ev_time_per_Tau = time_span(1,2)[-1]-time_span(1,2)[0]
    for n,Tau in enumerate(list_Tau):
        name_pars = str(N)+'_'+str(Tau)+'-'+str(dt)+'_'+BC
        filename = datadir+'fidelity_'+name_pars+'.npy'
        #Find if already computed
        try:
            fid_.append(np.load(filename))
        except:
            #Find time evolved states
            filename_psi = datadir+'time_evolved_wf_'+name_pars+'.npy'
            try:
                psi = np.load(filename_psi)
            except:
                times_, psi_ = time_evolve(args)
                psi = psi_[n]
            #Compute fidelity
            print("Computing fidelity of Tau = ",Tau," ...")
            steps = int(total_ev_time_per_Tau * Tau / dt)
            steps = steps+1 if steps%2==0 else steps
            times = time_span(Tau,steps)
            J = J_t(times,Tau)
            h = h_t(times,Tau)
            #
            fid_.append(np.zeros(steps))
            for s in range(steps):
                E_, evec = np.linalg.eigh(H_t(N,J[s],h[s],BC))
                phi_gs = evec[:,:N//2]             #GS modes
                overlap_matrix = np.matmul(np.conjugate(psi[s,:,:N//2]).T,phi_gs)
                det = np.linalg.det(overlap_matrix)
                fid_[n][s] = np.linalg.norm(det)**2
            np.save(filename,fid_[n])
    if 0:
        savefile = datadir+"Fid_"+name_pars+".txt"
        with open(savefile, 'w') as f:
            with redirect_stdout(f):
                print("J\th\tFidelity")
                for n in range(len(list_Tau)):
                    Tau = list_Tau[n]
                    print("Tau = ",str(Tau))
                    total_ev_time = 2*np.tan(1)*Tau if type_of_quench == "real" else 2*Tau
                    steps = int(total_ev_time/dt)
                    times = time_span(Tau,steps)
                    J = J_t(times,Tau)
                    h = h_t(times,Tau)
                    for s in range(steps):
                        print(str(J[s]),'\t',"{:.3f}".format(h[s]),'\t',"{:.4f}".format(fid_[n][s]))
    return fid_

def compute_nex(ind_T,args):
    N,dt,list_Tau,BC,type_of_quench = args
    datadir = datadir_dic[type_of_quench]
    h_t = h_t_dic[type_of_quench]
    J_t = J_t_dic[type_of_quench]
    time_span = time_span_dic[type_of_quench]
    #
    nex = []
#    n_q = []
    total_ev_time_per_Tau = time_span(1,2)[-1]-time_span(1,2)[0]
    for n,Tau in enumerate(list_Tau):
        name_pars = str(N)+'_'+str(Tau)+'-'+str(dt)+'_'+BC
        filename = datadir+'nex_'+name_pars+'.npy'
        try:
            nex.append(np.load(filename))
        except:
            print("Computing nex of Tau = ",Tau," ...")
            #Find time evolved states
            filename_psi = datadir+'time_evolved_wf_'+name_pars+'.npy'
            try:
                psi = np.load(filename_psi)
                steps = int(total_ev_time_per_Tau * Tau / dt)
                steps = steps+1 if steps%2==0 else steps
                times = time_span(Tau,steps)
            except:
                times_, psi_ = time_evolve(args)
                times = times_[n]
                psi = psi_[n]
            #
            ind_t = len(times)//2 if ind_T == 2 else ind_T
            res = 0
            #
            J = J_t(times[ind_t],Tau)
            h = h_t(times[ind_t],Tau)
            H_F = H_t(N,J,h,BC)
            E_F, psi_0 = scipy.linalg.eigh(H_F) #energy and GS of system
#            nnn = np.zeros(N//2)
            for k in range(N//2,N):
                temp_k = 0
                for l in range(N//2):
                    Gamma_kl = np.matmul(np.conjugate(psi[ind_t,:,l]).T,psi_0[:,k]).sum()
                    temp_k += np.linalg.norm(Gamma_kl)**2
#                nnn[k-N//2] = temp_k
                res += temp_k
            np.save(filename,res)
            nex.append(res)
#            n_q.append(nnn)
    return nex#,n_q

def compute_en_CP(args):
    N,dt,list_Tau,BC,type_of_quench = args
    datadir = datadir_dic[type_of_quench]
    h_t = h_t_dic[type_of_quench]
    J_t = J_t_dic[type_of_quench]
    time_span = time_span_dic[type_of_quench]
    #
    en_CP = np.zeros((len(list_Tau),2))
    total_ev_time_per_Tau = time_span(1,2)[-1]-time_span(1,2)[0]
    for n,Tau in enumerate(list_Tau):
        name_pars = str(N)+'_'+str(Tau)+'-'+str(dt)+'_'+BC
        filename = datadir+'en_CP_'+name_pars+'.npy'
        try:
            en_CP.append(np.load(filename))
        except:
            print("Computing en_CP of Tau = ",Tau," ...")
            steps = int(total_ev_time_per_Tau * Tau / dt)
            steps = steps+1 if steps%2==0 else steps
            #Find time evolved states
            filename_psi = datadir+'time_evolved_wf_'+name_pars+'.npy'
            try:
                psi = np.load(filename_psi)
                times = time_span(Tau,steps)
            except:
                times_, psi_ = time_evolve(args)
                times = times_[n]
                psi = psi_[n]
            #
            t_ = [steps//2,-1]     #time of critical point
            en_Tau = np.array([0,0])
            for i,t_CP in enumerate(t_):
                J = J_t(times,Tau)[t_CP]
                h = h_t(times,Tau)[t_CP]
                Ham = H_t(N,J,h,BC)
                eigs_GS, psi_GS = scipy.linalg.eigh(Ham)
                En_GS = sum(eigs_GS[:N//2])     #Energy of GS
                En_psi = 0 
                for k in range(N):
                    temp_k = 0
                    for l in range(N//2):
                        Gamma_kl = np.matmul(np.conjugate(psi[t_CP,:,l]).T,psi_GS[:,k]).sum()
                        temp_k += np.linalg.norm(Gamma_kl)**2
                    En_psi += temp_k*eigs_GS[k]
                en_CP[n,i] = abs((En_psi-En_GS)/En_GS)
    return en_CP

def compute_pop_ev(args,rg):
    N,dt,list_Tau,BC,type_of_quench = args
    datadir = datadir_dic[type_of_quench]
    h_t = h_t_dic[type_of_quench]
    J_t = J_t_dic[type_of_quench]
    time_span = time_span_dic[type_of_quench]
    #
    n_q = []
    total_ev_time_per_Tau = time_span(1,2)[-1]-time_span(1,2)[0]
    for n,Tau in enumerate(list_Tau):
        name_pars = str(N)+'_'+str(Tau)+'-'+str(dt)+'_'+BC
        filename = datadir+'n_q_'+name_pars+'.npy'
        try:
            n_q.append(np.load(filename))
        except:
            print("Computing n_q of Tau = ",Tau," ...")
            steps = int(total_ev_time_per_Tau * Tau / dt)
            steps = steps+1 if steps%2==0 else steps
            #Find time evolved states
            filename_psi = datadir+'time_evolved_wf_'+name_pars+'.npy'
            try:
                psi = np.load(filename_psi)
                times = time_span(Tau,steps)
            except:
                times_, psi_ = time_evolve(args)
                times = times_[n]
                psi = psi_[n]
            #
            J = J_t(times,Tau)
            h = h_t(times,Tau)
            res = np.zeros((N,steps))
            for s in range(steps):
                H_F = H_t(N,J[s],h[s],BC)
                E_F, psi_0 = scipy.linalg.eigh(H_F) #energy and GS of system
                for k in range(N//2-rg,N//2+rg):
                    temp_k = 0
                    for l in range(N//2):
                        Gamma_kl = np.matmul(np.conjugate(psi[s,:,l]).T,psi_0[:,k]).sum()
                        temp_k += np.linalg.norm(Gamma_kl)**2
                    res[k,s] = temp_k
            np.save(filename,res)
            n_q.append(res)
    return n_q

def compute_pop_T(args):
    N,dt,list_Tau,BC,type_of_quench = args
    datadir = datadir_dic[type_of_quench]
    h_t = h_t_dic[type_of_quench]
    J_t = J_t_dic[type_of_quench]
    time_span = time_span_dic[type_of_quench]
    #
    n_q = []
    total_ev_time_per_Tau = time_span(1,2)[-1]-time_span(1,2)[0]
    for n,Tau in enumerate(list_Tau):
        name_pars = str(N)+'_'+str(Tau)+'-'+str(dt)+'_'+BC
        filename = datadir+'n_qt_'+name_pars+'.npy'
        try:
            n_q.append(np.load(filename))
        except:
            print("Computing n_qt of Tau = ",Tau," ...")
            steps = int(total_ev_time_per_Tau * Tau / dt)
            steps = steps+1 if steps%2==0 else steps
            #Find time evolved states
            filename_psi = datadir+'time_evolved_wf_'+name_pars+'.npy'
            try:
                psi = np.load(filename_psi)
                times = time_span(Tau,steps)
            except:
                times_, psi_ = time_evolve(args)
                times = times_[n]
                psi = psi_[n]
            #
            J = J_t(times,Tau)
            h = h_t(times,Tau)
            list_t = [steps//2,steps-1]
            res = np.zeros((N,len(list_t)))
            for i,s in enumerate(list_t):
                H_F = H_t(N,J[s],h[s],BC)
                E_F, psi_0 = scipy.linalg.eigh(H_F) #energy and GS of system
                for k in range(N):
                    temp_k = 0
                    for l in range(N//2):
                        Gamma_kl = np.matmul(np.conjugate(psi[s,:,l]).T,psi_0[:,k]).sum()
                        temp_k += np.linalg.norm(Gamma_kl)**2
                    res[k,i] = temp_k
            np.save(filename,res)
            n_q.append(res)
    return n_q

def compute_Enex(args):
    N,dt,list_Tau,BC,type_of_quench = args
    datadir = datadir_dic[type_of_quench]
    h_t = h_t_dic[type_of_quench]
    J_t = J_t_dic[type_of_quench]
    time_span = time_span_dic[type_of_quench]
    #
    Enex = []
    total_ev_time_per_Tau = time_span(1,2)[-1]-time_span(1,2)[0]
    for n,Tau in enumerate(list_Tau):
        name_pars = str(N)+'_'+str(Tau)+'-'+str(dt)+'_'+BC
        filename = datadir+'Enex_'+name_pars+'.npy'
        try:
            Enex.append(np.load(filename))
        except:
            print("Computing Enex of Tau = ",Tau," ...")
            #Find time evolved states
            filename_psi = datadir+'time_evolved_wf_'+name_pars+'.npy'
            steps = int(total_ev_time_per_Tau * Tau / dt)
            steps = steps+1 if steps%2==0 else steps
            try:
                psi = np.load(filename_psi)
            except:
                times_, psi_ = time_evolve(args)
                psi = psi_[n]
            times = time_span(Tau,steps)
            J = J_t(times,Tau)
            h = h_t(times,Tau)
            Enex.append(np.zeros(steps))
            for s in range(steps):
                Ham = H_t(N,J[s],h[s],BC)
                E_GS, psi_GS = scipy.linalg.eigh(Ham)
                En_GS = sum(E_GS[:N//2])
                En_psi = 0 
                for k in range(N):
                    temp_k = 0
                    for l in range(N//2):
                        Gamma_kl = 0
                        for i in range(N):
                            Gamma_kl += psi_GS[i,k]*np.conjugate(psi[s,i,l])
                        temp_k += np.linalg.norm(Gamma_kl)**2
                    En_psi += temp_k*E_GS[k]
                Enex[-1][s] = En_psi-En_GS
            np.save(filename,Enex[-1])
    if 0:
        savefile = datadir+"Enex_"+name_pars+".txt"
        with open(savefile, 'w') as f:
            with redirect_stdout(f):
                print("J\th\tEnergy of exited state")
                for n in range(len(list_Tau)):
                    Tau = list_Tau[n]
                    print("Tau = ",str(Tau))
                    total_ev_time = 2*np.tan(1)*Tau if type_of_quench == "real" else 2*Tau
                    steps = int(total_ev_time/dt)
                    times = time_span(Tau,steps)
                    J = J_t(times,Tau)
                    h = h_t(times,Tau)
                    for s in range(steps):
                        print(str(J[s]),'\t',"{:.3f}".format(h[s]),'\t',"{:.4f}".format(Enex[n][s]))
    return Enex

def compute_CF_zz(ind_T,args,wf,in_,out_,step_site):
    N,dt,list_Tau,BC,type_of_quench = args
    datadir = datadir_dic[type_of_quench]
    h_t = h_t_dic[type_of_quench]
    J_t = J_t_dic[type_of_quench]
    time_span = time_span_dic[type_of_quench]
    #
    CF = []
    total_ev_time_per_Tau = time_span(1,2)[-1]-time_span(1,2)[0]
    if wf=="t-ev":
        print("Using time-evolved wf at time ",ind_T)
    else:
        print("Using istantaneous GS at time ",ind_T)
        list_Tau = [0,]
    for n,Tau in enumerate(list_Tau):
        name_pars = str(N)+'_'+str(Tau)+'-'+str(dt)+'_'+BC
        t_text = r"$t_{CP}$" if ind_T==2 else r"$t_f$"
        name_spec = t_text+'_'+wf+'_'+str(in_)+'_'+str(out_)+'_'+str(step_site)+'_'
        filename = datadir+'CFzz_'+name_spec+name_pars+'.npy'
        try:
            CF.append(np.load(filename))
        except:
            print("Computing Corr. fun. of Tau = ",Tau," ...")
            steps = int(total_ev_time_per_Tau * Tau / dt)
            steps = steps+1 if steps%2==0 else steps
            #Find time evolved states
            filename_psi = datadir+'time_evolved_wf_'+name_pars+'.npy'
            try:
                psi = np.load(filename_psi)
                times = time_span(Tau,steps)
            except:
                times_, psi_ = time_evolve(args)
                psi = psi_[n]
                times = times_[n]
            #
            ind_t = int(steps//2) if ind_T == 2 else int(ind_T)
            J = J_t(times,Tau)
            h = h_t(times,Tau)
            if wf=="t-ev":
                psi_t = psi[ind_t]
            else:
                E,evec = np.linalg.eigh(H_t(N,J[ind_t],h[ind_t],BC))
                psi_t = evec
            #Test
            CF.append(np.zeros(N))
            for r in range(in_,N):
                if (r-in_)%step_site==1:
                    CF[-1][r] = np.nan
                    continue
                Gr = 0
                Nr = 0
                for i in range(out_,N-out_):
                    if r+i >= N-out_:
                        continue
                    Nr += 1
                    temp_i = 0
                    for s in range(N//2):
                        temp_i += psi_t[i,s]*np.conjugate(psi_t[i+r,s])
                    Gr += temp_i*np.conjugate(temp_i)
                if Nr:
                    Gr /= Nr
                    CF[-1][r] = np.real(Gr) if abs(np.real(Gr)) > 1e-15 else np.nan
            np.save(filename,CF[-1])
    return CF

def compute_CF_pm(ind_T,args,wf,in_,out_,step_site,end_distance):
    N,dt,list_Tau,BC,type_of_quench = args
    datadir = datadir_dic[type_of_quench]
    h_t = h_t_dic[type_of_quench]
    J_t = J_t_dic[type_of_quench]
    time_span = time_span_dic[type_of_quench]
    #
    CF = []
    total_ev_time_per_Tau = time_span(1,2)[-1]-time_span(1,2)[0]
    if wf=="t-ev":
        print("Using time-evolved wf at time ",ind_T)
    else:
        print("Using istantaneous GS at time ",ind_T)
        list_Tau = [0,]
    for n,Tau in enumerate(list_Tau):
        name_pars = str(N)+'_'+str(Tau)+'-'+str(dt)+'_'+BC
        t_text = r"$t_{CP}$" if ind_T==2 else r"$t_f$"
        name_spec = t_text+'_'+wf+'_'+str(in_)+'_'+str(out_)+'_'+str(step_site)+'_'+str(end_distance)+'_'
        filename = datadir+'CFpm_'+name_spec+name_pars+'.npy'
        try:
            CF.append(np.load(filename))
        except:
            print("Computing Corr. fun. of Tau = ",Tau," ...")
            steps = int(total_ev_time_per_Tau * Tau / dt)
            steps = steps+1 if steps%2==0 else steps
            #Find time evolved states
            filename_psi = datadir+'time_evolved_wf_'+name_pars+'.npy'
            try:
                psi = np.load(filename_psi)
                times = time_span(Tau,steps)
            except:
                times_, psi_ = time_evolve(args)
                psi = psi_[n]
                times = times_[n]
            #
            ind_t = int(steps//2) if ind_T == 2 else int(ind_T)
            J = J_t(times,Tau)
            h = h_t(times,Tau)
            if wf=="t-ev":
                psi_t = psi[ind_t]
            else:
                E,evec = np.linalg.eigh(H_t(N,J[ind_t],h[ind_t],BC))
                psi_t = evec
            #Test
            CF.append(np.zeros(N))
            #
            def G_ij(wf,i_,j_):      #   <c^dag_ic_j>
                res = 0
                for s_ in range(wf.shape[0]//2):
                    res += wf[i_,s_]*np.conjugate(wf[j_,s_])
                return res
            #
            def matrix_ij(i_,r_,s_):
                s_x = np.array([0] + list(s_)) +i_
                s_y = np.array(list(s_) + [r_]) + i_
                xx,yy = np.meshgrid(s_x,s_y)
                return big_G[xx,yy]#.T
            #
            big_G = np.zeros((N,N),dtype=complex)
            for i in range(N):
                for j in range(N):
                    big_G[i,j] = G_ij(psi_t,i,j)
            #
            for distance in range(in_,N):
                if distance > end_distance or (distance-in_)%step_site==1:
                    CF[-1][distance] = np.nan
                    continue
                print(distance)
#                rho_xx = 0
#                rho_yy = 0
#                rho_zz = 0
                rho_pm = 0
                Nr = 0
                for site in range(out_,N-out_):
                    #print(site,distance)
                    if distance+site >= N-out_:
                        continue
                    Nr += 1
#                    mat_xx = big_G[site:site+distance,site+1:site+distance+1]
#                    mat_yy = big_G[i+1:i+r+1,i:i+r]
#                    rho_xx += np.linalg.det(mat_xx)
#                    rho_yy += np.linalg.det(mat_yy)
#                    rho_zz += big_G[i+r,i]*big_G[i,i+r]
                    rho_pm += big_G[site,site+distance]
                    #print(big_G[site,site+distance])
                    for iteration in range(1,distance):
                        ss = list(combinations(np.arange(1,distance),iteration))
                        #print(ss)
                        for step in ss:
                            rho_pm += 2**iteration*np.linalg.det(matrix_ij(site,distance,step))

                if Nr:
#                    rho_xx /= Nr
#                    rho_yy /= Nr
#                    rho_zz /= Nr
                    rho_pm /= Nr
                    res = rho_pm
                    #print("r = ",distance,res)
                    CF[-1][distance] = np.real(res) if abs(np.real(res)) > 1e-15 else np.nan
                #input()
            np.save(filename,CF[-1])
    return CF

def pow_law(x,a,b):
    return a*np.abs(x)**(b)

def exp_dec(x,a,b):
    return a*np.exp(-x/b) 

def exp_2dec(x,a,b):
    return a*np.exp(-x**2/b**2)/b**2
##################################################
##################################################
##################################################
##################################################

def compute_S(ind_T,args,list_L):
    N,dt,list_Tau,BC,type_of_quench = args
    datadir = 'Real_Quench/' if type_of_quench == "real" else 'Linear_Quench/'
    h_t = h_t_real if type_of_quench == "real" else h_t_linear
    J_t = J_t_real if type_of_quench == "real" else J_t_linear
    time_span = time_span_real if type_of_quench == "real" else time_span_linear
    S = []
    for n,Tau in enumerate(list_Tau):
        name_pars = str(N)+'_'+str(Tau)+'-'+str(dt)+'_'+BC
        filename = datadir+'S_'+name_pars+'.npy'
        try:
            S.append(np.load(filename))
        except:
            print("Computing ent. entropy of Tau = ",Tau," ...")
            total_ev_time = 2*np.tan(1)*Tau if type_of_quench == "real" else 2*Tau
            steps = int(total_ev_time/dt)
            #Find time evolved states
            filename_psi = datadir+'time_evolved_wf_'+name_pars+'.npy'
            try:
                psi = np.load(filename_psi)
                times = time_span(Tau,steps)
            except:
                times_, psi_ = time_evolve(args)
                psi = psi_[n]
                times = times_[n]
            #
            ind_t = steps//2 if ind_T == 2 else ind_T
            psi_t = psi[ind_t]
            S.append(np.zeros((N)))
            alpha = np.zeros((N,N),dtype=complex)
            d = np.eye(N)
            for i in range(N):
                for j in range(i,N):
                    alpha[i,j] = -corr_fun(j,i,psi_t)  + d[i,j]
                    alpha[j,i] = np.conjugate(alpha[i,j])
            for L in list_L:
                Pi = np.zeros((2*L,2*L),dtype=complex)
                alpha_L = alpha[:L,:L]
                Pi[:L,:L] = alpha_L
                Pi[L:,L:] = 1-alpha_L
                Pi_log_Pi = np.matmul(Pi,scipy.linalg.logm(Pi)/np.log(2.0))
                res = -np.trace(Pi_log_Pi)
                S[-1][L] = np.linalg.norm(res)/(2*L)
#            np.save(filename,S[-1])
    return S
















