import numpy as np
import scipy
import functions as fs
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from pathlib import Path

""" Parameetr choice from terminal """
if len(sys.argv)!=5:
    print("Usage: py static_ZZ_r.py arg1 arg2 arg3 arg4")
    print("\targ1: correlator type (zz,ee, ecc)")
    print("\targ2,arg3: Lx and Ly (int)")
    print("\targ4: fourier type (fft->plane waves(scipy.fft.fft), dst->scipy.fft.dst (type 1), dct->scipy.fft.dct (type 2), dat->amazing functions)")
    exit()
else:
    correlator_type = sys.argv[1]
    Lx = int(sys.argv[2])
    Ly = int(sys.argv[3])
    Ns = Lx*Ly
    fourier_type = sys.argv[4]
    if fourier_type not in ['fft','dst','dct','dat']:
        print("Unacceted value of Fourier type: ", sys.argv[4])

""" Other options of the calculation """
use_experimental_parameters = False
""" Diagonalization """
save_wf = 1
plot_wf = 0 #True
""" Correlator """
save_correlator = 1
save_bonds = 0
plot_correlator = 1#True
save_fig = 0#False
""" Magnon terms to include in the plot """
exclude_zero_mode = 0 #True
include_list = [2,4,6,8,]
full_exclude_list = [2,4,6,8]
exclude_list = []
for e in full_exclude_list:
    if e not in include_list:
        exclude_list.append(e)
if include_list ==  [2,4,6,8]:
    txt_magnon = 'allMagnon'
else:
    txt_magnon = ''
    for i,term in enumerate(include_list):
        txt_magnon += str(int(term/2))
        if i!=len(include_list)-1:
            txt_magnon += ','
    txt_magnon += 'Magnon'
""" Parameters of the Hamiltonian and of the ramp """
S = 0.5     #spin value
full_time_ramp = 0.5  #ramp time in ms
time_steps = 500        #of ramp
time_step = full_time_ramp/time_steps  #time step of ramp
stop_ratio_list = np.linspace(0.1,1,10)     #ratios of full_time_ramp where we stop and measure
txt_exp = 'expPars' if use_experimental_parameters else 'uniform'
g_val = 20      #factor of 2 from experiment due to s^xs^x -> s^+s^-
h_val = 15
g1_t_i,g2_t_i,d1_t_i,h_t_i = fs.get_parameters(use_experimental_parameters,Lx,Ly,time_steps,g_val,h_val)   #parameters of Hamiltonian which depend on time
""" Correlator parameters """
site_j = (2,2) if (Lx==6 and Ly==7) else (Lx//2,Ly//2-1)      #site wrt which compute the correlator
ind_j = site_j[1] + site_j[0]*Ly
site0 = 0 if h_t_i[0,0,0]<0 else 1     #decide sublattice A and B of reference lattice site
full_time_measure = 0.8     #measure time in ms
Ntimes = 401        #time steps after ramp for the measurement
measure_time_list = np.linspace(0,full_time_measure,Ntimes)
N_omega = 2000      #default from Jeronimo

""" Print out parameters"""
print("2D XY model with staggered field")
print("Initial magnetic field: ",h_val," MHz")
print("Final nn coupling     : ",g_val/2," MHz")
print("System size: %i x %i"%(Lx,Ly))
print("Computing correlator "+correlator_type+" acting on site %i,%i -> index %d"%(site_j[0],site_j[1],ind_j))
if exclude_zero_mode:
    print("Removing zero energy mode")
else:
    print("Keeping zero energy mode")
print(txt_magnon+' terms')

""" Bogoliubov transformation"""
U_ = np.zeros((len(stop_ratio_list),Ns,Ns),dtype=complex)
V_ = np.zeros((len(stop_ratio_list),Ns,Ns),dtype=complex)
evals = np.zeros((len(stop_ratio_list),Ns))
txt_pars = 'expPars' if use_experimental_parameters else 'uniform'
args_fn = (Lx,Ly,txt_pars)
transformation_fn = 'Data/rs_bogoliubov_' + fs.get_fn(*args_fn) + '.npz'
if not Path(transformation_fn).is_file():
    for i_sr in tqdm(range(len(stop_ratio_list)),desc="Computing Bogoliubov transformation"):
        stop_ratio = stop_ratio_list[i_sr]
        indt = int(time_steps*stop_ratio)
        if indt==time_steps:    indt -= 1
        # Hamiltonian parameters
        J_i = (g1_t_i[indt,:,:],np.zeros((Ns,Ns)))  #site-dependent hopping
        D_i = (d1_t_i[indt,:,:],np.zeros((Ns,Ns)))
        h_i = h_t_i[indt,:,:]
        theta,phi = fs.get_angles(S,J_i,D_i,h_i)
        ts = fs.get_ts(theta,phi)       #All t-parameters for A and B sublattice
        parameters = (S,Lx,Ly,h_i,ts,theta,phi,J_i,D_i)
        #
        hamiltonian = fs.get_Hamiltonian_rs(*parameters)
        if np.max(np.absolute(hamiltonian-hamiltonian.conj()))>1e-5:
            print("Hamiltonian is not real! Procedure might be wrong")
        # Para-diagonalization, see notes (appendix) for details
        A = hamiltonian[:Ns,:Ns]
        B = hamiltonian[:Ns,Ns:]
        try:
            K = scipy.linalg.cholesky(A-B)
            k = scipy.linalg.cholesky(hamiltonian)
        except:
#            print("Negative or null eigenvalue in Hamiltonian: ",scipy.linalg.eigvalsh(A-B)[0])
            K = scipy.linalg.cholesky(A-B+np.identity(Ns)*1e-5)
        lam2,chi_ = scipy.linalg.eigh(K@(A+B)@K.T.conj())
        evals[i_sr] = np.sqrt(lam2)         #dispersion -> positive
        #
        chi = chi_ / evals[i_sr]**(1/2)     #normalized eigenvectors: divide each column of chi_ by the corresponding eigenvalue -> of course for the gapless mode there is a problem here
        phi_ = K.T.conj()@chi
        psi_ = (A+B)@phi_/evals[i_sr]       # Problem also here
        U_[i_sr] = 1/2*(phi_+psi_)
        V_[i_sr] = 1/2*(phi_-psi_)
    # Save
    if save_wf:
        np.savez(transformation_fn,amazingU=U_,amazingV=V_,evals=evals)
else:
    print("Loading Bogoliubov transformation from file")
    U_ = np.load(transformation_fn)['amazingU']
    V_ = np.load(transformation_fn)['amazingV']
    evals = np.load(transformation_fn)['evals']

if exclude_zero_mode:
    for i_sr in range(2,len(stop_ratio_list)):
        U_[i_sr,:,0] *= 0
        V_[i_sr,:,0] *= 0

""" Plot wavefunction """
if plot_wf or 0: # Plot wavefunction
    X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly))
    if 0:   #Plot phi
        i_sr = 9
        A_ik = np.real(U_[i_sr] - V_[i_sr])
        B_ik = np.real(U_[i_sr] + V_[i_sr])
        Bj_k = B_ik[ind_j]
        phi_ik = A_ik / Bj_k[None,:]
        for i in range(Ns):
            phi_ik[i,:] *= 2/np.pi*(-1)**(i//Ly+i%Ly+1)
        fig = plt.figure(figsize=(20,15))
        for k in range(15):
            ax = fig.add_subplot(3,5,k+1,projection='3d')
            ax.plot_surface(X,Y,phi_ik[:,k].reshape(Lx,Ly).T,cmap='plasma')
        plt.show()

        fig = plt.figure(figsize=(20,15))
        xline = np.arange(Lx)
        yline = np.arange(Ly)
        for kx in range(5):
            for ky in range(3):
                ax = fig.add_subplot(3,5,kx*3+ky+1,projection='3d')
                c = np.outer(np.cos(np.pi*kx*(2*xline+1)/2/Lx), np.cos(np.pi*ky*(2*yline+1)/2/Ly) )
                ax.plot_surface(X,Y,c.T,cmap='plasma')
        plt.show()
        exit()
    if 0:   #Plot U and phi
        i_sr = 9
        A_ik = np.real(U_[i_sr] - V_[i_sr])
        B_ik = np.real(U_[i_sr] + V_[i_sr])
        Bj_k = B_ik[ind_j]
        phi_ik = A_ik / Bj_k[None,:]
        for i in range(Ns):
            phi_ik[i,:] *= 2/np.pi*(-1)**(i//Ly+i%Ly+1)
        for k in range(Ns-1,-1,-1):
            fig = plt.figure(figsize=(10,6))
            ax = fig.add_subplot(121,projection='3d')
            ax.plot_surface(X,Y,phi_ik[:,k].reshape(Lx,Ly).T,cmap='plasma')
            ax.set_title(r"Amazing function, $\varphi_{ik}$",size=20)
            ax = fig.add_subplot(122,projection='3d')
            ax.plot_surface(X,Y,np.real(U_[i_sr,:,k]).reshape(Lx,Ly).T,cmap='plasma')
            ax.set_title("U function",size=20)
            plt.suptitle("K = %d"%k,size=20)
            fig.tight_layout()
            plt.show()
    if 0:       #Plot some phi modes
        fig = plt.figure(figsize=(20,15))
        i_sr = 9
        for k in range(15):
            ik  = Ns-1-k
            val = np.real(U_[i_sr,:,ik]).reshape(Lx,Ly)
            ax = fig.add_subplot(3,5,k+1,projection='3d')
            ax.plot_surface(X,Y,val.T,cmap='plasma')
            kx,ky = fs.get_momentum_Bogoliubov(val)
            ax.set_title("Kx,Ky = %d,%d"%(kx,ky),size=15)
        fig.tight_layout()
        plt.show()
        exit()
    if 1:       #DCT on phi_ik
        fig = plt.figure(figsize=(20,15))
        for i_sr in range(9,10):
            A_ik = np.real(U_[i_sr] - V_[i_sr])
            B_ik = np.real(U_[i_sr] + V_[i_sr])
            Bj_k = B_ik[ind_j]
            phi_ik = A_ik / Bj_k[None,:]
            for i in range(Ns):
                phi_ik[i,:] *= 2/np.pi*(-1)**(i//Ly+i%Ly+1)
            ax = fig.add_subplot()#2,5,i_sr+1)
            for k in range(Ns):
                f_in = phi_ik[:,k].reshape(Lx,Ly)
                kx,ky = fs.get_momentum_Bogoliubov2(f_in)
                ax.scatter(kx,ky,color='r')
                #ax.text(kx,ky+0.1*np.random.rand(),str(k))
            ax.set_aspect('equal')
            ax.set_ylim(Ly-1,0)
        fig.tight_layout()
        plt.show()
    if 0:
        i_sr = 9
        sins1 = np.zeros((Lx,Ly,Lx,Ly))
        xline = np.arange(Lx)
        yline = np.arange(Ly)
        for kx in range(Lx):
            for ky in range(Ly):
                sins1[kx,ky] = np.outer( np.sin(np.pi*(kx+1)*(xline+1)/(Lx+1)), np.sin(np.pi*(ky+1)*(yline+1)/(Ly+1)) )
        sins2 = -sins1
        for k in range(Ns-1,-1,-1):
            f_in = np.real(U_[i_sr,:,k]).reshape(Lx,Ly)
            f_in /= np.max(abs(f_in))
            min1 = np.min(np.sum(np.abs(f_in[None,None,:,:] - sins1),axis=(2,3)))
            min2 = np.min(np.sum(np.abs(f_in[None,None,:,:] - sins2),axis=(2,3)))
            ind = np.argmin(np.sum(np.abs(f_in[None,None,:,:] - sins1),axis=(2,3))) if min1<min2 else np.argmin(np.sum(np.abs(f_in[None,None,:,:] - sins2),axis=(2,3)))
            kx,ky = ind//Ly, ind%Ly
            fig = plt.figure(figsize=(20,20))
            ax = fig.add_subplot(121,projection='3d')
            ax.plot_surface(X,Y,sins1[kx,ky].T,cmap='plasma')
            ax = fig.add_subplot(122,projection='3d')
            ax.plot_surface(X,Y,f_in.T,cmap='plasma')
            ax.set_title("ind:%d, kx,ky:%d,%d"%(k,kx,ky),size=20)
            plt.show()
        exit()

    if 0:       #plot bogoliuybov momenta
        fig = plt.figure(figsize=(20,7))
        for i_sr in range(9,10):
            ax = fig.add_subplot()#2,5,i_sr+1)
            ks = []
            for i in range(Ns):
                kx,ky = fs.get_momentum_Bogoliubov(np.real(U_[i_sr,:,i]).reshape(Lx,Ly))
                ks.append([kx,ky])
#                ax.scatter(ks[-1][0],ks[-1][1],marker='o',color='r')
                #ax.text(ks[-1][0],ks[-1][1]-0.2,str(i))
            ks = np.array(ks)
            ax.scatter(ks[:,0],ks[:,1],marker='o',color='r',zorder=-1)
            ax.set_aspect('equal')
            ax.set_ylim(Ly-1,0)
        plt.show()
        exit()

""" Explicit ZZ transformation """
if 0:
    #Definition of A and B
    i_sr = 9
    A_ik = np.real(U_[i_sr] - V_[i_sr])
    B_ik = np.real(U_[i_sr] + V_[i_sr])

    if 1: #Check orthogonality of U vs V
        for m in range(Ns):
            for n in range(Ns):
                p1 = U_[i_sr,:,m] @ U_[i_sr,:,n].T - V_[i_sr,:,m] @ V_[i_sr,:,n].T
                p2 = U_[i_sr,:,m] @ V_[i_sr,:,n].T - V_[i_sr,:,m] @ U_[i_sr,:,n].T
                if (abs(p1) > 1e-10 and m!=n) or abs(p2)>1e-10:
                    print("m,n : %d,%d"%(m,n))
                    print(p1)
                    print(p2)
                    exit()
        for i in range(Ns):
            for j in range(Ns):
                p1 = U_[i_sr,i,:] @ U_[i_sr,j,:].T - V_[i_sr,i,:] @ V_[i_sr,j,:].T
                p2 = U_[i_sr,i,:] @ V_[i_sr,j,:].T - V_[i_sr,i,:] @ U_[i_sr,j,:].T
                if (abs(p1) > 1e-10 and m!=n) or abs(p2)>1e-10:
                    print("i,j : %d,%d"%(m,n))
                    print(p1)
                    print(p2)
                    exit()
        print("U and V functions are orthogonal")

    """ We are chosing the j site carfully so that B[j,:] does not have any zeros, which are difficult to deal with """
    while(True):
        if min(abs(B_ik[ind_j])) < 1e-10:
            ind_j -= 1
        else:
            print("Using j index %d"%ind_j)
            break
    Bj_k = B_ik[ind_j]
    phi_ik = A_ik / Bj_k[None,:]

    # Correlator
    corr2 = np.zeros((10,Ns,Ntimes),dtype=complex)
    for ind in range(Ns):
        for it in range(Ntimes):
            corr2[i_sr,ind,it] = 2*1j * np.sum(np.sin(2*np.pi*measure_time_list[it]*evals[i_sr]) * B_ik[ind,:] * Bj_k)

    if 1:       #check orthogonality of transformation function phi_ik
        for k in range(Ns):
            for m in range(Ns):
                t =  Bj_k[m] * (phi_ik[:,k] @ B_ik[:,m])
                if abs(t)>1e-7 and m!=k:
                    print("k,m: %d,%d with t=%f"%(k,m,t))
                    exit()
        print("Transformation function is orthogonal")

    # Transform in x -> k
    c_kt = np.zeros((10,Ns,Ntimes),dtype=complex)
    for k in range(Ns):
        c_kt[i_sr,k] = np.sum(phi_ik[:,k,None]*corr2[i_sr,:,:],axis=0)

    if 0:       #plot c_kt calculated vs analytical. Works with AND without 0 mode
        for k in range(Ns):
            fig = plt.figure(figsize=(20,20))
            ax = fig.add_subplot()
#            ax.plot(measure_time_list,np.real(c_kt[i_sr,k]),color='k',label='real')
            ax.plot(measure_time_list,np.imag(c_kt[i_sr,k]),color='r',label='imag')
            ax.plot(measure_time_list,2*np.sin(2*np.pi*evals[i_sr,k]*measure_time_list),color='b',label='analytical')
            ax.set_title("k: %d"%k, size=20)
            ax.legend()
            plt.tight_layout()
            plt.show()
        exit()
    if 1:       #check c_kt calculated is the same as the analytical one
        for k in range(Ns):
            p = np.sum( abs( np.imag(c_kt[i_sr,k]) - 2*np.sin(2*np.pi*measure_time_list*evals[i_sr,k]) ) )
            if p>1e-4:
                print("k: %d, p: %f"%(k,p))
                exit()
        print("Transform working as analytical result")

    from scipy.fft import fft, fftshift
    corr_kw = np.zeros((10,Lx,Ly,N_omega),dtype=complex)
    corr2_kw = np.zeros((10,Lx,Ly,Ntimes),dtype=complex)

    # Transform in t -> w
    for k in range(Lx*Ly):
        kx,ky = fs.get_momentum_Bogoliubov(np.real(U_[i_sr,:,Ns-1-k].reshape(Lx,Ly)))
        corr_kw[i_sr,kx,ky] = fftshift(fft(c_kt[i_sr,k],n=N_omega)) / Ntimes
        corr2_kw[i_sr,kx,ky] = fftshift(fft(2*1j*np.sin(2*np.pi*measure_time_list*evals[i_sr,k])))/Ntimes

    if 0:       #Plot corr_kw for each k
        for k in range(Lx*Ly):
            fig = plt.figure(figsize=(25,15))
            ax = fig.add_subplot(131)
            ax.plot(np.linspace(-250,250,N_omega),np.real(corr_kw[i_sr,kx,ky]),color='r',label='real')
            ax.plot(np.linspace(-250,250,Ntimes),np.real(corr2_kw[i_sr,kx,ky]),color='r',label='2')
            ax.legend()
            ax = fig.add_subplot(132)
            ax.plot(np.linspace(-250,250,N_omega),np.imag(corr_kw[i_sr,kx,ky]),color='b',label='imag')
            ax.plot(np.linspace(-250,250,Ntimes),np.imag(corr2_kw[i_sr,kx,ky]),color='r',label='2')
            ax.legend()
            ax = fig.add_subplot(133)
            ax.plot(np.linspace(-250,250,N_omega),np.absolute(corr_kw[i_sr,kx,ky]),color='b',label='abs')
            ax.plot(np.linspace(-250,250,Ntimes),np.absolute(corr2_kw[i_sr,kx,ky]),color='r',label='2')
            ax.legend()
            print("k,kx,ky: %d,%d,%d -> "%(k,kx,ky),corr_kw[i_sr,kx,ky])
            plt.show()

    print("Plotting 2")
    fs.plot(
        corr_kw,
        n_bins=100,
        fourier_type=fourier_type,
        title='',
#        figname='Figures/analytic_zz.png',
        showfig=True,
    )
    exit()

""" Computation of correlator"""
correlator = np.zeros((len(stop_ratio_list),Lx,Ly,Ntimes),dtype=complex)
txt_0energy = 'without0energy' if exclude_zero_mode else 'with0energy'
args_fn = (correlator_type,Lx,Ly,txt_pars,txt_0energy,txt_magnon)
correlator_fn = 'Data/rs_corr_' + fs.get_fn(*args_fn) + '.npy'
if save_bonds:
    corr_bonds_h = np.zeros((len(stop_ratio_list),Lx-1,Ly,Ntimes),dtype=complex)
    corr_bonds_v = np.zeros((len(stop_ratio_list),Lx,Ly-1,Ntimes),dtype=complex)
    corr_bonds_fn = 'Data/rs_corr_bonds_' + fs.get_fn(*args_fn) + '.npz'
if not Path(correlator_fn).is_file():
    for i_sr in tqdm(range(len(stop_ratio_list)),desc="Computing correlator"):
        stop_ratio = stop_ratio_list[i_sr]
        indt = int(time_steps*stop_ratio)
        if indt==time_steps:    indt -= 1
        # Hamiltonian parameters
        J_i = (g1_t_i[indt,:,:],np.zeros((Ns,Ns)))  #site-dependent hopping
        D_i = (d1_t_i[indt,:,:],np.zeros((Ns,Ns)))
        h_i = h_t_i[indt,:,:]
        theta,phi = fs.get_angles(S,J_i,D_i,h_i)
        ts = fs.get_ts(theta,phi)       #All t-parameters for A and B sublattice
        #
        U = np.zeros((2*Ns,2*Ns),dtype=complex)
        U[:Ns,:Ns] = U_[i_sr]
        U[:Ns,Ns:] = V_[i_sr]
        U[Ns:,:Ns] = V_[i_sr]
        U[Ns:,Ns:] = U_[i_sr]
        #Correlator -> can make this faster, we actually only need U_ and V_
        exp_e = np.exp(-1j*2*np.pi*measure_time_list[:,None]*evals[i_sr,None,:])
        A = np.einsum('tk,ik,jk->ijt',exp_e,U[Ns:,:Ns],U[Ns:,Ns:],optimize=True)
        B = np.einsum('tk,ik,jk->ijt',exp_e,U[:Ns,:Ns],U[:Ns,Ns:],optimize=True)
        G = np.einsum('tk,ik,jk->ijt',exp_e,U[Ns:,:Ns],U[:Ns,Ns:],optimize=True)
        H = np.einsum('tk,ik,jk->ijt',exp_e,U[:Ns,:Ns],U[Ns:,Ns:],optimize=True)
        #
        for ind_i in range(Ns):
            correlator[i_sr,ind_i//Ly,ind_i%Ly] = fs.get_correlator[correlator_type](
                S,Lx,Ly,
                ts,
                site0,
                ind_i,ind_j,
                A,B,G,H,
                exclude_list
            )
        if save_bonds:
            for ihx in range(Lx-1):
                for ihy in range(Ly):
                    corr_bonds_h[i_sr,ihx,ihy] = fs.correlator_jj_bond(
                        S,Lx,Ly,
                        ts,
                        site0,
                        ihx*Ly+ihy,ind_j,
                        A,B,G,H,
                        'h',        #orientation of bond
                        exclude_list
                    )
            for ivx in range(Lx):
                for ivy in range(Ly-1):
                    corr_bonds_v[i_sr,ivx,ivy] = fs.correlator_jj_bond(
                        S,Lx,Ly,
                        ts,
                        site0,
                        ivx*Ly+ivy,ind_j,
                        A,B,G,H,
                        'v',        #orientation of bond
                        exclude_list
                    )
    if save_correlator:
        np.save(correlator_fn,correlator)
    if save_bonds:
        np.savez(corr_bonds_fn,horizontal_bonds=corr_bonds_h,vertical_bonds=corr_bonds_v)
else:
    print("Loading real-space correlator from file")
    correlator = np.load(correlator_fn)

###############################################################################
# FOURIER TRANSFORM AND FIGURE
###############################################################################
if fourier_type=='fft':
    correlator_kw = fs.fourier_fft(correlator,N_omega)
elif fourier_type=='dst':
    type_dst = 1
    correlator_kw = fs.fourier_dst(correlator,N_omega,type_dst)
elif fourier_type=='dct':
    type_dst = 2
    correlator_kw = fs.fourier_dct(correlator,N_omega,type_dst)

elif fourier_type=='dat':
    correlator_kw = fs.fourier_dat(correlator,U_,V_,ind_j,N_omega)

title = 'Commutator '+correlator_type+', fourier: '+fourier_type+', '+txt_magnon
args_fn += (fourier_type,)
figname = 'Figures/' + fs.get_fn(*args_fn) + '.png'


print("Plotting")
fs.plot(
    correlator_kw,
    n_bins=50,
    fourier_type=fourier_type,
    title=title,
    figname=figname,
    showfig=True,
)














