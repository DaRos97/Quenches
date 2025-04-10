import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import functions as fs

#Options
plot_correlator = True
superimpose_energy = 1#True
#
stop_ratio_list = np.linspace(0.1,1,10)     #ratios of ramp where we stop and measure
#Parameters Hamiltonian
S = 0.5     #spin value
J1_fin = 40     #MHz
J2 = 0
D1 = 0
D2 = 0
h_in = 30       #MHz
#
Lx = 15     #Linear size of lattice
Ly = 15
Ns = Lx*Ly    #number of sites
gridk = fs.BZgrid(Lx,Ly)#BZ grid
Gamma1 = np.cos(gridk[:,:,0])+np.cos(gridk[:,:,1])  #cos(kx) + cos(ky)
Gamma2 = np.cos(gridk[:,:,0]+gridk[:,:,1])+np.cos(gridk[:,:,0]-gridk[:,:,1])  #cos(kx+ky) + cos(kx-ky)
Gamma = (Gamma1,Gamma2)
#
Ntimes = 401        #number of time steps in the measurement
measurement_time = 0.8
t_list = np.linspace(0,measurement_time,Ntimes)      #800 ns
N_omega = 2000      #default from Jeronimo
omega_list = np.linspace(-250,250,N_omega)
#
corr = np.zeros((len(stop_ratio_list),Lx,Ly,Ntimes),dtype=complex)
dispersion = []
#e^{i*k*(ri-rj)}
x_val = np.arange(Lx)
y_val = np.arange(Ly)
X,Y = np.meshgrid(x_val,y_val)
grid_real_space = np.zeros((Lx,Ly,2))
grid_real_space[:,:,0] = X.T
grid_real_space[:,:,1] = Y.T
#Actual code
for i_sr,stop_ratio in enumerate(stop_ratio_list):
    J1 = J1_fin*stop_ratio
    h = h_in*(1-stop_ratio)
    print("Step ",i_sr," with J=",J1," and h=",h)
    J = (J1,J2)
    D = (D1,D2)
    theta,phi = fs.get_angles(S,J,D,h)
    print("Theta: ",theta)
    px = -np.sin(theta)
    py = 0
    pz = np.cos(theta)
    parameters = (S,Gamma,h,theta,phi,J,D)
    epsilon = fs.get_epsilon(*parameters)       #dispersion
    dispersion.append(epsilon)
    exp_e = np.zeros((Lx,Ly,Ntimes),dtype=complex)
    for it in range(Ntimes):    #e^{-i*t*eps}   ->  [Lx,Ly,Ntimes]
        exp_e[:,:,it] = np.exp(-1j*2*np.pi*t_list[it]*epsilon)
    exp_e = np.reshape(exp_e, shape=(Lx*Ly,Ntimes))
    exp_e[np.isnan(exp_e)] = 1          #right??
    rk = fs.get_rk(*parameters)
    phik = fs.get_phik(*parameters).reshape(Lx*Ly)
    phik[np.isnan(phik)] = 1
    cosh_rk = np.cosh(rk).reshape(Lx*Ly)
    cosh_rk[np.isnan(cosh_rk)] = 0
    cosh_rk[np.absolute(cosh_rk)>1e3] = 0
    sinh_rk = np.sinh(rk).reshape(Lx*Ly)
    sinh_rk[np.isnan(sinh_rk)] = 0
    sinh_rk[np.absolute(sinh_rk)>1e3] = 0
    for ii in range(Lx*Ly):     #assuming phi=0 for now -> py=0
        ix = ii%Lx
        iy = ii//Lx
        exp_k = (np.exp(-1j*np.dot(gridk,grid_real_space[ix,iy]))).reshape(Lx*Ly)
        corr1 = S/2/Ns*np.einsum('i,i,ij->j',exp_k,px**2*np.absolute(cosh_rk+phik.conj()*sinh_rk)**2,exp_e,optimize=True)*(-1)**(ix+iy)
        corr2 = np.sum( (-1)**(ix+iy)*pz**2*(S-1/Ns*sinh_rk**2) )
        corr3 = (-1)**(ix+iy)*pz**2/Ns**2*(np.einsum('ik,jk,i,j,i,j->k',exp_e,exp_e,exp_k,exp_k,
                                                    sinh_rk**2,cosh_rk**2,
                                                    optimize=True) +
                                           np.einsum('ik,jk,i,j,i,j->k',exp_e,exp_e,exp_k,exp_k,
                                                    phik.conj()*cosh_rk*sinh_rk,phik*cosh_rk*sinh_rk,
                                                    optimize=True) )
        corr[i_sr,ix,iy] = 2*1j*np.imag(corr1 + corr2 + corr3)

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
    omega_mesh = np.linspace(-250.250125063,250.250125063,2001)
    bla_x = np.linspace(0.,ks_m[-1][1],2)       #specific of Lx=7,Ly=6
    bla_y = np.linspace(-250.250125063,250.250125063,2)
    X0, Y0 = np.meshgrid(bla_x, bla_y)
    #
    fig = plt.figure(figsize=(20.8,8))
    for p in range(n_sr):
        ax = fig.add_subplot(2,5,p+1)        #default 10 stop ratios
        a = corr[p] # insert matrix for stop ratio of shape (x,y,Ntimes), here (6,7,401)
        #Fourier transform x,y->kx,ky with fft2 for each time t
        ks_ts = np.zeros((Lx,Ly,Ntimes), dtype=complex)
        for t in range(Ntimes):
            ks_ts[:,:,t] =  np.fft.fftshift(np.fft.fft2(a[:,:,t]))
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
            ks_m_plot.append(ks_m[idx[i]])
        vma = np.amax(ks_ws_plot)
        #Plot 0 background
        ax.pcolormesh(X0, Y0, np.zeros((1,1)), cmap='magma', vmin=0, vmax=vma)

#        ks_ws_plot.reverse()    ###################################################

        #Plot single columns
        sm = ScalarMappable(cmap='magma',norm=Normalize(vmin=0,vmax=vma))
        sm.set_array([])
        for i in range(len(vals)):
            X, Y = np.meshgrid(np.array(ks_m_plot[i]), omega_mesh)
            ax.pcolormesh(X, Y, ks_ws_plot[i].reshape((1,N_omega)).T, cmap='magma', vmin=0, vmax=vma)
            if superimpose_energy and 0:
                """Scatter a red * at the value of the dispersion for that momentum -> since e plot as a function of |k|, here I just take the first instance of ks with that specific |k|. Others hopefully have the same dispersion."""
                ind = list(ks).index(vals[i])
                ikx,iky = (ind//Ly,ind%Ly)
                ax.scatter(vals[i],dispersion[p][ikx,iky],color='r',marker='*',zorder=3)
        if superimpose_energy:  #Plot all dispersion points
            for ix,kx in enumerate(kx_list):
                for iy,ky in enumerate(ky_list):
                    ax.scatter(np.sqrt(kx**2+ky**2),dispersion[p][ix,iy],color='r',marker='*',zorder=3)
        plt.colorbar(sm,ax=ax,label='FFT (a.u.)')
        ax.set_ylim(-70,70)
        if p > 4:
            ax.set_xlabel('$|k|$')
        if p%5 == 0:
            ax.set_ylabel('$\\omega$')
        ax.set_title('Stop ratio :$'+"{:.1f}".format(0.1*(p+1))+'$')

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.2, hspace=0.3, left=0.04, right=0.97)

    plt.show()
























