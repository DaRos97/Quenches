import numpy as np
import scipy
import functions as fs
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from pathlib import Path

if len(sys.argv)!=5:
    print("Usage: py static_ZZ_r.py arg1 arg2 arg3 arg4")
    print("\targ1: correlator type (zz,ee, ecc)")
    print("\targ2,arg3: Lx and Ly (int)")
    print("\targ4: fourier type (fft->plane waves(scipy.fft.fft), dst->scipy.fft.dst, dst2->sin, dat->amazing functions)")
    exit()
else:
    correlator_type = sys.argv[1]
    Lx = int(sys.argv[2])
    Ly = int(sys.argv[3])
    Ns = Lx*Ly
    fourier_type = sys.argv[4]
    if fourier_type not in ['fft','dst','dst2','dat']:
        print("Unacceted value of Fourier type: ", sys.argv[4])

"""Other options of the calculation"""
use_experimental_parameters = False
"""Diagonalization"""
save_wf = 1
plot_wf = 0#True
"""Correlator"""
save_correlator = 1
plot_correlator = 1#True
save_fig = 1#False
"""Magnon terms to include in the plot"""
exclude_zero_mode = 1#True
include_list = [2,4,6,8]
full_exclude_list = [2,4,6,8]
exclude_list = []
for e in full_exclude_list:
    if e not in include_list:
        exclude_list.append(e)
"""Parameters of the Hamiltonian and of the ramp"""
S = 0.5     #spin value
full_time_ramp = 0.5  #ramp time in ms
time_steps = 500        #of ramp
time_step = full_time_ramp/time_steps  #time step of ramp
stop_ratio_list = np.linspace(0.1,1,10)     #ratios of full_time_ramp where we stop and measure
txt_exp = 'expPars' if use_experimental_parameters else 'uniform'
g_val = 20      #factor of 2 from experiment due to s^xs^x -> s^+s^-
h_val = 15
g1_t_i,g2_t_i,d1_t_i,h_t_i = fs.get_parameters(use_experimental_parameters,Lx,Ly,time_steps,g_val,h_val)   #parameters of Hamiltonian which depend on time
"""Correlator parameters"""
site_j = (2,3) if (Lx==6 and Ly==7) else (Lx//2,Ly//2)      #site wrt which compute the correlator
ind_j = site_j[1] + site_j[0]*Ly
site0 = 0 if h_t_i[0,0,0]<0 else 1     #decide sublattice A and B of reference lattice site
full_time_measure = 0.8     #measure time in ms
Ntimes = 401        #time steps after ramp for the measurement
measure_time_list = np.linspace(0,full_time_measure,Ntimes)
N_omega = 2000      #default from Jeronimo

"""Print out parameters"""
print("2D XY model with staggered field")
print("Initial magnetic field: ",h_val," MHz")
print("Final nn coupling     : ",g_val/2," MHz")
print("System size: %i x %i"%(Lx,Ly))
print("Computing correlator "+correlator_type+" acting on site %i,%i"%(site_j[0],site_j[1]))
if exclude_zero_mode:
    print("Removing zero energy mode")
else:
    print("Keeping zero energy mode")

"""Bogoliubov transformation"""
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
        if theta!=0 and exclude_zero_mode:   # When theta!=0 -> gapless spectrum -> do not consider the 0 energy eigenmode and eigenstate
            lam2[0] = 1
        evals[i_sr] = np.sqrt(lam2)         #dispersion -> positive
        #
        chi = chi_ / evals[i_sr]**(1/2)     #normalized eigenvectors: divide each column of chi_ by the corresponding eigenvalue -> of course for the gapless mode there is a problem here
        phi_ = K.T.conj()@chi
        psi_ = (A+B)@phi_/evals[i_sr]       # Problem also here
        U_[i_sr] = 1/2*(phi_+psi_)
        V_[i_sr] = 1/2*(phi_-psi_)
        if theta!=0 and exclude_zero_mode:        # Remove the 0-energy eigenvector
            U_[i_sr,:,0] *= 0
            V_[i_sr,:,0] *= 0
    # Save
    if save_wf:
        np.savez(transformation_fn,amazingU=U_,amazingV=V_,evals=evals)
else:
    print("Loading Bogoliubov transformation from file")
    U_ = np.load(transformation_fn)['amazingU']
    V_ = np.load(transformation_fn)['amazingV']
    evals = np.load(transformation_fn)['evals']

if exclude_zero_mode:
    for i_sr in range(len(stop_ratio_list)):
        U_[i_sr,:,0] *= 0
        V_[i_sr,:,0] *= 0

if plot_wf: # Plot wavefunction
    X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly))
    if 0:       #First n_modes of U and V on rows
        fig = plt.figure(figsize=(25,10))
        n_modes = 7
        i_sr = 0
        for i_n in range(n_modes):
            #k = Lx*Ly-1-ik
            n = Ns-1-i_n
            U_x = U_[i_sr,:,n].reshape(Lx,Ly)     #-1 -> last stop ratio
            V_x = V_[i_sr,:,n].reshape(Lx,Ly)
            #U
            ax = fig.add_subplot(2,n_modes,i_n+1,projection='3d')
            ax.plot_surface(X,Y,np.real(U_x).T,cmap='plasma')
            ax.set_title("U, mode k=%d"%(i_n))
            #V
            ax = fig.add_subplot(2,n_modes,i_n+n_modes+1,projection='3d')
            ax.plot_surface(X,Y,np.real(V_x).T,cmap='plasma')
            ax.set_title("V, mode k=%d"%(i_n)+',e:'+"{:.3f}".format(evals[i_sr,n]))
        plt.subplots_adjust(left=0.02,right=0.974,top=0.983,bottom=0.058,wspace=0.1,hspace=0.02)
        plt.show()
    if 0:       #Only U modes on a grid
        fig = plt.figure(figsize=(20,20))
        n_modes = 10
        i_sr = 2
        for i in range(n_modes**2):
            n = Ns-1-i
            U_x = np.real(U_[i_sr,:,n]).reshape(Lx,Ly)     #-1 -> last stop ratio
            nx,ny = fs.get_momentum_Bogoliubov(U_x)
            ax = fig.add_subplot(n_modes,n_modes,i+1,projection='3d')
            ax.plot_surface(X,Y,U_x.T,cmap='plasma')
            ax.set_title("ind:%d, mode:(%d,%d)"%(i,nx,ny)+",e="+"{:.3f}".format(evals[i_sr,n]))
            if i==0:
                ax.set_xlabel('x')
                ax.set_ylabel('y')
        plt.show()
    if 1:       #plot bogoliuybov momenta
        fig = plt.figure(figsize=(20,7))
        for i_sr in range(10):
            ax = fig.add_subplot(2,5,i_sr+1)
            ks = []
            for i in range(Ns):
                disp = True if i_sr == 9 else False
                kx,ky = fs.get_momentum_Bogoliubov(np.real(U_[i_sr,:,i]).reshape(Lx,Ly),disp)
                ks.append([kx,ky])
            ks = np.array(ks)
            ax.scatter(ks[:,0],ks[:,1],marker='o',color='r')
            ax.set_aspect('equal')
        plt.show()
    if 0:       #primnt absolute value of momentum
        kx = np.pi * np.arange(1, Lx + 1) / (Lx + 1)
        ky = np.pi * np.arange(1, Ly + 1) / (Ly + 1)
        ks = np.zeros(Ns)
        for i in range(Ns):
            ks[i] = np.sqrt(kx[i%Lx]**2+ky[i//Lx]**2)
        inds = np.argsort(ks)
        for i in range(Ns):
            print(i,' -> (',inds[i]%Lx,inds[i]//Lx,'):',ks[inds[i]])
    exit()

"""Computation of correlator"""
correlator = np.zeros((len(stop_ratio_list),Lx,Ly,Ntimes),dtype=complex)
args_fn = (correlator_type,Lx,Ly,txt_pars)
correlator_fn = 'Data/rs_corr_' + fs.get_fn(*args_fn) + '.npy'
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
    if save_correlator:
        np.save(correlator_fn,correlator)
else:
    print("Loading real-space correlator from file")
    correlator = np.load(correlator_fn)



###############################################################################
# FOURIER TRANSFORM AND FIGURE
###############################################################################

if 0:   #Plot time trace
    i_sr = 9
    x = 2
    iy = 6
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(measure_time_list,np.imag(correlator[i_sr,ix,iy,:]))
    plt.show()
    exit()

print("Fourier transforming")
if fourier_type=='fft':
    correlator_kw = fs.fourier_fft(correlator,N_omega)
elif fourier_type=='dst':
    type_dst = 1
    correlator_kw = fs.fourier_dst(correlator,N_omega,type_dst)
elif fourier_type=='dst2':
    correlator_kw = fs.fourier_dst2(correlator,N_omega)
elif fourier_type=='dat':
    correlator_kw = fs.fourier_dat(correlator,U_,V_,N_omega)

title = 'Commutator '+correlator_type+', fourier: '+fourier_type
args_fn += (fourier_type,)
figname = 'Figures/' + fs.get_fn(*args_fn) + '.png'


print("Plotting")
fs.plot(
    correlator_kw,
    n_bins=30,
    fourier_type=fourier_type,
    title=title,
    figname=figname,
    showfig=True,
)















