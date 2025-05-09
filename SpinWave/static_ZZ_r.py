import numpy as np
import functions as fs
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import scipy
import sys
from time import time


if len(sys.argv)!=5:
    print("Usage: py static_ZZ_r.py corr_type(zz) Lx Ly new_fourier")

save_corr = 0
plot_correlator = 1#True
plot_wf = 0#True
new_fourier = int(sys.argv[4])#True
superimpose_dispersion = False
save_fig = 1#False
corr_type = 'zz'if len(sys.argv)<2 else sys.argv[1]
include_list = [8]

#
full_exclude_list = [2,4,6,8]
exclude_list = []
for e in full_exclude_list:
    if e not in include_list:
        exclude_list.append(e)
#Parameters of the ramp
S = 0.5     #spin value
full_time_ramp = 0.5  #ramp time in ms
time_steps = 500        #of ramp
time_step = full_time_ramp/time_steps  #time step of ramp
#Hamiltonian parameters
use_experimental_parameters = 0#False
txt_exp = 'expPars' if use_experimental_parameters else 'uniform'
if use_experimental_parameters:     #Not yet implemented
    print("Using experimental parameters")
    initial_parameters_fn = 'exp_input/20250324_6x7_2D_StaggeredFrequency_0MHz_5.89_.p'
    final_parameters_fn = 'exp_input/20250324_6x7_2D_IntFrequency_10MHz_5.89_.p'
    Lx,Ly,g1_in,g2_in,d1_in,h_in = fs.extract_experimental_parameters(initial_parameters_fn)
    Lx,Ly,g1_fin,g2_fin,d1_fin,h_fin = fs.extract_experimental_parameters(final_parameters_fn)
    Ns = Lx*Ly
    g1_in *= -4
    g1_fin *= -4
    h_in -= np.identity(Ns)*np.sum(h_in)/Ns
    h_fin -= np.identity(Ns)*np.sum(h_fin)/Ns
    h_in *= 2
    h_fin *= 2
    print(np.diagonal(h_in))
    print(np.diagonal(h_fin))
    exit()
else:
    Lx = 6
    Ly = 7
    if len(sys.argv)>=4:
        Lx = int(sys.argv[2])
        Ly = int(sys.argv[3])
    g_val = 40
    h_val = 30
    #
    Ns = Lx*Ly
    g1_in = np.zeros((Ns,Ns))
    g1_fin = np.zeros((Ns,Ns))
    for ix in range(Lx):
        for iy in range(Ly):
            ind = iy+ix*Ly
            ind_plus_y = ind+1
            if ind_plus_y//Ly==ind//Ly:
                g1_fin[ind,ind_plus_y] = g1_fin[ind_plus_y,ind] = g_val
            ind_plus_x = ind+Ly
            if ind_plus_x<Lx*Ly:
                g1_fin[ind,ind_plus_x] = g1_fin[ind_plus_x,ind] = g_val
    g2_in = np.zeros((Ns,Ns))
    g2_fin = np.zeros((Ns,Ns))
    d1_in = np.zeros((Ns,Ns))
    d1_fin = np.zeros((Ns,Ns))
    h_in = np.zeros((Ns,Ns))
    for ix in range(Lx):
        for iy in range(Ly):
            h_in[iy+ix*Ly,iy+ix*Ly] = (-1)**(ix+iy)*    h_val
    h_fin = np.zeros((Ns,Ns))
site_j = (2,3) if (Lx==6 and Ly==7) else (Lx//2,Ly//2)      #site wrt with compute the correlator
ind_j = site_j[1] + site_j[0]*Ly
site0 = 0 if h_in[0,0]<0 else 1     #decide sublattice A and B of reference lattice site
#
print("System size: %i x %i"%(Lx,Ly))
g1_t_i,g2_t_i,d1_t_i,h_t_i = fs.get_Hamiltonian_parameters(time_steps,g1_in,g2_in,d1_in,h_in,g1_fin,g2_fin,d1_fin,h_fin)   #parameters of Hamiltonian which depend on time
#
full_time_measure = 0.8     #measure time in ms
Ntimes = 401        #time steps after ramp for the measurement
measure_time_list = np.linspace(0,full_time_measure,Ntimes)
N_omega = 2000      #default from Jeronimo
omega_list = np.linspace(-250,250,N_omega)
stop_ratio_list = np.linspace(0.1,1,10)     #ratios of full_time_ramp where we stop and measure
#Actual code
corr = np.zeros((len(stop_ratio_list),Lx,Ly,Ntimes),dtype=complex)
corr_fn = 'data/rs_corr_zz_6x7_uniform.npy'
for i_sr,stop_ratio in enumerate(stop_ratio_list):
    if plot_wf and i_sr!=9:
        continue
    print("Stop ratio ",stop_ratio)
    indt = int(time_steps*stop_ratio)
    if indt==time_steps:    indt -= 1
    #
    J_i = (g1_t_i[indt,:,:],np.zeros((Ns,Ns)))  #site-dependent hopping
    D_i = (d1_t_i[indt,:,:],np.zeros((Ns,Ns)))
    h_i = h_t_i[indt,:,:]
    theta,phi = fs.get_angles(S,J_i,D_i,h_i)
    ts = fs.get_ts(theta,phi) #All t-parameters for A and B sublattice
    parameters = (S,Lx,Ly,h_i,ts,theta,phi,J_i,D_i)
    hamiltonian = fs.get_Hamiltonian_rs(*parameters)
    if np.max(np.absolute(hamiltonian-hamiltonian.conj()))>1e-5:
        print("Hamiltonian is not real! Procedure might be wrong")
    #Para-Diagonalization
    A = hamiltonian[:Ns,:Ns]
    B = hamiltonian[:Ns,Ns:]
    try:
        K = scipy.linalg.cholesky(A-B)
    except:
        print("Negative or null eigenvalue in Hamiltonian: ",scipy.linalg.eigvalsh(hamiltonian)[0])
        K = scipy.linalg.cholesky(A-B+np.identity(Ns)*1e-7)  #Upper triangular s.t. ham = K^dag @ K
    lam2,chi_ = scipy.linalg.eigh(K@(A+B)@K.T.conj())
    eps = np.sqrt(lam2)         #dispersion -> positive
    chi = chi_ / eps**(1/2)     #normalized eigenvectors
    phi_ = K.T.conj()@chi
    psi_ = (A+B)@phi_/eps
    U_ = 1/2*(phi_+psi_)
    V_ = 1/2*(phi_-psi_)
    U = np.zeros((2*Ns,2*Ns),dtype=complex)
    U[:Ns,:Ns] = U_
    U[:Ns,Ns:] = V_#.conj()
    U[Ns:,:Ns] = V_
    U[Ns:,Ns:] = U_#.conj()
    #Correlator
    exp_e = np.exp(-1j*2*np.pi*measure_time_list[:,None]*eps[None,:])
    A = np.einsum('tk,ik,jk->ijt',exp_e,U[Ns:,:Ns],U[Ns:,Ns:],optimize=True)
    B = np.einsum('tk,ik,jk->ijt',exp_e,U[:Ns,:Ns],U[:Ns,Ns:],optimize=True)
    G = np.einsum('tk,ik,jk->ijt',exp_e,U[Ns:,:Ns],U[:Ns,Ns:],optimize=True)
    H = np.einsum('tk,ik,jk->ijt',exp_e,U[:Ns,:Ns],U[Ns:,Ns:],optimize=True)
    #
    for ind_i in range(Ns):
        corr[i_sr,ind_i//Ly,ind_i%Ly] = fs.get_correlator[corr_type](
            S,Lx,Ly,
            ts,site0,
            ind_i,ind_j,
            A,B,G,H,
            exclude_list
        )
    #Save
    if save_corr:
        np.save(corr_fn,corr)
    if plot_wf and i_sr == 9:   #plot wavefunction
        J = U
#        J = np.linalg.inv(U)
        fig = plt.figure(figsize=(25,10))
        for ik in range(10):
            k = Lx*Ly-1-ik
            U_ = J[:Ns,k].reshape(Lx,Ly)
            V_ = J[Ns:,k].reshape(Lx,Ly)
#            U_[0,0] = np.nan
#            V_[0,0] = np.nan
            ax = fig.add_subplot(2,5,ik+1,projection='3d')
            X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly))
            ax.plot_surface(X,Y,np.real(U_).T,cmap='plasma')
#            ax = fig.add_subplot(122,projection='3d')
#            ax.plot_surface(X,Y,np.real(V_).T,cmap='plasma')
            ax.set_title("Mode k=%d"%(ik))
        fig.tight_layout()
        plt.show()

def fourier(xs_ts, Lx, Ly, kx_list, ky_list):
    sx,sy = (3,4) if (Lx==6 and Ly==7) else (Lx//2,Ly//2)
    #
    x = np.linspace(1, Lx, Lx)
    y = np.linspace(1, Ly, Ly)
    # Compute the 2D sine arrays
    sin_kx_x_mes = np.sin(np.outer(kx_list, x))
    sin_ky_y_mes = np.sin(np.outer(ky_list, y))

    sin_kx_x_act = np.sin(np.outer(kx_list, [sx]*Lx))
    sin_ky_y_act = np.sin(np.outer(ky_list, [sy]*Ly))

    sin_kx_x = sin_kx_x_mes * sin_kx_x_act
    sin_ky_y = sin_ky_y_mes * sin_ky_y_act

    # Now combine them with einsum
    trans = np.einsum('ki,qj->kqij', sin_kx_x, sin_ky_y)
    ks_ts =  np.einsum('ijt,kqij->kqt', xs_ts, trans)
    return ks_ts

if plot_correlator and new_fourier:
    n_sr = len(stop_ratio_list)
    #momenta for plotting
    kx_list = np.arange(1,Lx+1,1)  * np.pi / (Lx+1)
    ky_list = np.arange(1,Ly+1,1)  * np.pi / (Ly+1)
    ks = []     #mod k
    ks_m = []   #mod_k +- 0.01
    ddd = 0.05
    for kx in kx_list:
        for ky in ky_list:
            ks.append(np.sqrt(kx**2+ky**2))
            ks_m.append([np.sqrt(kx**2+ky**2)-ddd, np.sqrt(kx**2+ky**2)+ddd])
    ks = np.array(ks)
    k_inds = np.argsort(ks)
    ks_m = np.array(ks_m)[k_inds]   #ordered
    vals, idx = np.unique(ks[k_inds], return_index=True)    #take only unique |k| points
    idx = np.append(idx, len(ks))
    #Uniform backgound
    omega_mesh = np.linspace(-250,250,N_omega)
    bla_x = np.linspace(0.,ks_m[-1][1],2)       #specific of Lx=7,Ly=6
    bla_y = np.linspace(-250,250,2)
    X0, Y0 = np.meshgrid(bla_x, bla_y)
    #
    fig = plt.figure(figsize=(20.8,8))
    for p in range(n_sr):
        ax = fig.add_subplot(2,5,p+1)
        ks_ts = fourier(corr[p],Lx,Ly,kx_list,ky_list)
        #Fourier transform t->Omega and flatten kx,ky
        ks_ws_flat = np.zeros((Lx*Ly,N_omega), dtype=complex)
        for kx in range(Lx):
            for ky in range(Ly):
                ind = ky + Ly*kx
                ks_ws_flat[ind,:] = np.fft.fftshift(np.fft.fft(ks_ts[kx,ky,:], n=N_omega))
        #Take absolute value and order like the absolute values of momenta
        ks_ws_flat = np.abs(ks_ws_flat[k_inds,:])
        #Sum values of Fourier transform in the same |k| interval
        ks_ws_plot = []
        ks_m_plot  = []
        for i in range(len(vals)):
            val_c  = np.sum(ks_ws_flat[idx[i]:idx[i+1],:], axis=0)/(idx[i+1]-idx[i])
            ks_ws_plot.append(val_c)
#            ks_m_plot.append(ks_m[idx[i]])
#            continue
        for i in range(len(vals)):
            if i>0 and i!=len(vals)-1:
                ks_m_plot.append([(vals[i]+vals[i-1])*0.5, (vals[i+1]+vals[i])/2])
            elif i==0:
                ks_m_plot.append([0, (vals[i+1]+vals[i])/2])
            else:
                ks_m_plot.append([(vals[i-1]+vals[i])/2,vals[i]+0.1])
        vma = np.amax(ks_ws_plot)
        #Plot 0 background
        ax.pcolormesh(X0, Y0, np.zeros((1,1)), cmap='magma', vmin=0, vmax=vma)
        #Plot single columns
        sm = ScalarMappable(cmap='magma',norm=Normalize(vmin=0,vmax=vma))
        sm.set_array([])
        for i in range(len(vals)):
            X, Y = np.meshgrid(np.array(ks_m_plot[i]), omega_mesh)
            ax.pcolormesh(X, Y, ks_ws_plot[i].reshape((1,N_omega)).T, cmap='magma', vmin=0, vmax=vma)
        plt.colorbar(sm,ax=ax,label='FFT (a.u.)')
        ax.set_ylim(-70,70)
        if p > 4:
            ax.set_xlabel('$|k|$')
        if p%5 == 0:
            ax.set_ylabel('$\\omega$')
        ax.set_title('Stop ratio :'+"{:.3f}".format(stop_ratio_list[p]))

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.2, hspace=0.3, left=0.04, right=0.97)

    if save_fig:
        figname = 'Figures/'+corr_type+'_'+str(Lx)+'x'+str(Ly)+'_rs'
        if superimpose_dispersion:
            figname += '_disp'
        figname += '.png'
        plt.savefig(figname)
    plt.show()

    exit()

if plot_correlator:
    n_sr = len(stop_ratio_list)
    #momenta for plotting
    kx_list = np.fft.fftshift(np.fft.fftfreq(Lx,d=1))
    ky_list = np.fft.fftshift(np.fft.fftfreq(Ly,d=1))
    ks = []     #mod k
    ks_m = []   #mod_k +- 0.01
    for kx in kx_list:
        for ky in ky_list:
            ks.append(np.sqrt(kx**2+ky**2))
            ks_m.append([np.sqrt(kx**2+ky**2)-0.01, np.sqrt(kx**2+ky**2)+0.01])
    ks = np.array(ks)
    k_inds = np.argsort(ks)
    ks_m = np.array(ks_m)[k_inds]   #ordered
    vals, idx = np.unique(ks[k_inds], return_index=True)    #take only unique |k| points
    idx = np.append(idx, len(ks))
    #Uniform backgound
    omega_mesh = np.linspace(-250,250,N_omega+1)
    bla_x = np.linspace(0.,ks_m[-1][1],2)       #specific of Lx=7,Ly=6
    bla_y = np.linspace(-250,250,2)
    X0, Y0 = np.meshgrid(bla_x, bla_y)
    #
    fig = plt.figure(figsize=(20.8,8))
    for p in range(n_sr):
        ax = fig.add_subplot(2,5,p+1)        #default 10 stop ratios
        a = corr[p] # insert matrix for stop ratio of shape (x,y,Ntimes), here (6,7,401)
        #Fourier transform x,y->kx,ky with fft2 for each time t
        ks_ts = np.zeros((Lx,Ly,Ntimes), dtype=complex)
        for t in range(Ntimes):
            ks_ts[:,:,t] =  np.fft.fftshift(np.fft.fft2(a[:,:,t]))      #/Lx/Ly
        #Fourier transform t->Omega and flatten kx,ky
        ks_ws_flat = np.zeros((Lx*Ly,N_omega), dtype=complex)
        for kx in range(Lx):
            for ky in range(Ly):
                ind = ky + Ly*kx
                ks_ws_flat[ind,:] = np.fft.fftshift(np.fft.fft(ks_ts[kx,ky,:], n=N_omega))      /np.sqrt(401)
        #Take absolute value and order like the absolute values of momenta
        ks_ws_flat = np.abs(ks_ws_flat[k_inds,:])
        #Sum values of Fourier transform in the same |k| interval
        ks_ws_plot = []
        ks_m_plot  = []
        for i in range(len(vals)):
            val_c  = np.sum(ks_ws_flat[idx[i]:idx[i+1],:], axis=0)/(idx[i+1]-idx[i])
            ks_ws_plot.append(val_c)
            ks_m_plot.append(ks_m[idx[i]])
        vma = np.amax(ks_ws_plot)
        #Plot 0 background
        ax.pcolormesh(X0, Y0, np.zeros((1,1)), cmap='magma', vmin=0, vmax=vma)
        #Plot single columns
        sm = ScalarMappable(cmap='magma',norm=Normalize(vmin=0,vmax=vma))
        sm.set_array([])
        for i in range(len(vals)):
            X, Y = np.meshgrid(np.array(ks_m_plot[i]), omega_mesh)
            ax.pcolormesh(X, Y, ks_ws_plot[i].reshape((1,N_omega)).T, cmap='magma', vmin=0, vmax=vma)
        plt.colorbar(sm,ax=ax,label='FFT (a.u.)')
        ax.set_ylim(-70,70)
        if p > 4:
            ax.set_xlabel('$|k|$')
        if p%5 == 0:
            ax.set_ylabel('$\\omega$')
        ax.set_title('Stop ratio :'+"{:.3f}".format(stop_ratio_list[p]))

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.2, hspace=0.3, left=0.04, right=0.97)

    if save_fig:
        figname = 'Figures/'+corr_type+'_'+str(Lx)+'x'+str(Ly)+'_rs'
        if superimpose_dispersion:
            figname += '_disp'
        list_dic = {'zz':[2,4],'ze':[2,4,6],'ez':[2,4,6],'ee':[2,4,6,8]}
        if len(exclude_list)!=0:
            mag = False
            figname += '_'
            include_list = list_dic[corr_type]
            for ii in include_list:
                if ii not in exclude_list:
                    figname += str(ii)+'-'
                    mag = True
            if mag:
                figname += 'magnon'
        ax = fig.add_subplot()
        ax.axis('off')
        ax.text(0.4,0.5,figname[8:],size=20)
        figname += '.png'
        plt.savefig(figname)
    plt.show()













