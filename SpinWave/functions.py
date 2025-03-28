import numpy as np
import matplotlib.pyplot as plt
import pickle

#Lattice directions of square lattice
a1 = np.array([1,0])
a2 = np.array([0,1])
b1 = np.array([2*np.pi,0])
b2 = np.array([0,2*np.pi])

def get_N_11(*pars):
    """Compute N_11 as in notes."""
    S,Gamma,h,theta,phi,J,D = pars
    p_xx = get_p_xx(theta,phi,J,D)
    p_yy = get_p_yy(theta,phi,J,D)
    p_zz = get_p_zz(theta,phi,J,D)
    result = h/2*np.cos(theta)
    for i in range(2):
        result += S*(Gamma[i]*(p_xx[i]+p_yy[i])/2-2*p_zz[i])
    return result
def get_N_12(*pars):
    """Compute N_12 as in notes."""
    S,Gamma,h,theta,phi,J,D = pars
    p_xx = get_p_xx(theta,phi,J,D)
    p_yy = get_p_yy(theta,phi,J,D)
    p_xy = get_p_xy(theta,phi,J,D)
    result = 0
    for i in range(2):
        result += S/2*Gamma[i]*(p_xx[i]-p_yy[i]-2*1j*p_xy[i])
    return result
def get_p_xx(theta,phi,J,D,order='canted_Neel'):
    if order=='canted_Neel':
        return (-J[0]*(np.cos(theta)**2*np.cos(phi)**2-np.sin(phi)**2+D[0]*np.sin(theta)**2*np.cos(phi)**2),
                J[1]*(np.cos(theta)**2*np.cos(phi)**2+np.sin(phi)**2+D[1]*np.sin(theta)**2*np.cos(phi)**2))
def get_p_yy(theta,phi,J,D,order='canted_Neel'):
    if order=='canted_Neel':
        return (-J[0]*(np.cos(theta)**2*np.sin(phi)**2-np.cos(phi)**2+D[0]*np.sin(theta)**2*np.sin(phi)**2),
                J[1]*(np.cos(theta)**2*np.sin(phi)**2+np.cos(phi)**2+D[1]*np.sin(theta)**2*np.sin(phi)**2))
def get_p_zz(theta,phi,J,D,order='canted_Neel'):
    if order=='canted_Neel':
        return (-J[0]*(np.sin(theta)**2+D[0]*np.cos(theta)**2),
                J[1]*(np.sin(theta)**2+D[1]*np.cos(theta)**2))
def get_p_xy(theta,phi,J,D,order='canted_Neel'):
    if order=='canted_Neel':
        return (J[0]/2*np.sin(2*phi)*(np.cos(theta)**2+1+D[0]*np.sin(theta)**2),
                -J[1]/2*np.sin(2*phi)*(np.cos(theta)**2-1+D[1]*np.sin(theta)**2))

def get_E_0(*pars):
    """Compute E_0 as in notes."""
    S,Gamma,h,theta,phi,J,D = pars
    p_zz = get_p_zz(theta,phi,J,D)
    result = -h*(S+1/2)*np.cos(theta)
    for i in range(2):
        result += 2*S*(S+1)*p_zz[i]
    return result

def get_epsilon(*pars):
    """Compute dispersion epsilon as in notes."""
    N_11 = get_N_11(*pars)
    N_12 = get_N_12(*pars)
    result = np.sqrt(N_11**2-np.absolute(N_12)**2,where=(N_11**2>=np.absolute(N_12)**2))
    result[N_11**2<np.absolute(N_12)**2] = 0
    return result

def get_E_GS(*pars):
    """Compute ground state energy as in notes."""
    E_0 = get_E_0(*pars)
    epsilon = get_epsilon(*pars)
    Ns = epsilon.shape[0]*epsilon.shape[1]
    return E_0 + np.sum(epsilon[~np.isnan(epsilon)])/Ns

def get_angles(S,J,D,h):
    """Compute angles theta and phi of quantization axis depending on Hamiltonian parameters."""
    if J[1]<J[0]/2 and h<4*S*(J[0]*(1-D[0])-J[1]*(1-D[1])):
        theta = np.arccos(h/(4*S*(J[0]*(1-D[0])-J[1]*(1-D[1]))))
    else:
        theta = 0
    phi = 0
    return (theta,phi)

def get_rk(*pars):
    """Compute rk as in notes."""
    N_11 = get_N_11(*pars)
    N_12 = get_N_12(*pars)
    frac = np.divide(np.absolute(N_12),N_11,where=(N_11!=0))
    result = -1/2*np.arctanh(frac,where=(frac<1))
    result[frac>=1] = np.nan
    return result

def get_phik(*pars):
    """Compute e^i*phik as in notes."""
    N_12 = get_N_12(*pars)
    result = np.exp(1j*np.angle(N_12))
    return result

def BZgrid(Lx,Ly):
    """Compute BZ coordinates of points."""
    dx = 2*np.pi/Lx
    dy = 2*np.pi/Ly
    gridk = np.zeros((Lx,Ly,2))
    for i1 in range(Lx):
        for i2 in range(Ly):
            gridk[i1,i2,0] = dx*(1+i1) - np.pi
            gridk[i1,i2,1] = dy*(1+i2) - np.pi
    return gridk

def get_Hamiltonian_rs(*parameters):
    """
    Compute the real space Hamiltonian -> (2Ns x 2Ns).
    Conventions for the real space wavefunction and other things are in the notes.
    SECOND NEAREST-NEIGHBOR NOT IMPLEMENTED.
    """
    S,Lx,Ly,h,theta,phi,J,D = parameters
    if J[1]!=0:
        print("Second nn not implemented")
        exit()
    Ns = Lx*Ly
    ham = np.zeros((2*N_s,2*Ns),dtype=complex)
    p_zz = get_p_zz(theta,phi,J,D)
    p_xx = get_p_xx(theta,phi,J,D)
    p_yy = get_p_yy(theta,phi,J,D)
    p_xy = get_p_xy(theta,phi,J,D)
    #diagonal
    diag = h/2*np.cos(theta)-2*S*(p_zz[0]+p_zz[1])
    for i in range(2*Ns):
        ham[i,i] = diag
    #off_diag 1 - nn
    off_diag_1_nn = S/2*(p_xx[0]+p_yy[0])
    #off_diag 2 - nn
    off_diag_2_nn = S/2*(p_xx[0]-p_yy[0]+2*1j*p_xy[0])
    for ix in range(Lx):
        for iy in range(Ly):
            ind = iy+ix*Ly
            ind_plus_y = ind+1
            if ind_plus_y//Ly==ind//Ly:
                #off_diag 1 - nn
                ham[ind,ind_plus_y] = off_diag_1_nn
                ham[ind_plus_y,ind] = off_diag_1_nn
                ham[ind+Ns,ind_plus_y+Ns] = off_diag_1_nn
                ham[ind_plus_y+Ns,ind+Ns] = off_diag_1_nn
                #off_diag 2 - nn
                ham[ind,ind_plus_y+Ns] = off_diag_2_nn
                ham[ind_plus_y,ind+Ns] = off_diag_2_nn
                ham[ind+Ns,ind_plus_y] = off_diag_2_nn
                ham[ind_plus_y+Ns,ind] = off_diag_2_nn
            #
            ind_plus_x = ind+Ly
            if ind_plus_x<Lx*Ly:
                #off_diag 1 - nn
                ham[ind,ind_plus_x] = off_diag_1_nn
                ham[ind_plus_x,ind] = off_diag_1_nn
                ham[ind+Ns,ind_plus_x+Ns] = off_diag_1_nn
                ham[ind_plus_x+Ns,ind+Ns] = off_diag_1_nn
                #off_diag 2 - nn
                ham[ind,ind_plus_x+Ns] = off_diag_2_nn
                ham[ind_plus_x,ind+Ns] = off_diag_2_nn
                ham[ind+Ns,ind_plus_x] = off_diag_2_nn
                ham[ind_plus_x+Ns,ind] = off_diag_2_nn
    return ham
def get_correlator(ts_i,ts_j,S,G,H,A,B):
    """Compute real space zz correlator."""
    t_zz_i,t_zx_i,t_zy_i = ts_i
    t_zz_j,t_zx_j,t_zy_j = ts_j
    return 2*1j*(
        np.imag(H)*(t_zz_i*t_zz_j*n.real(G)+S/2*(t_zx_i*t_zx_j + t_zy_i*t_zy_j + t_zx_i*t_zy_j - t_zy_i*t_zx_j))    +
        np.imag(G)*(t_zz_i*t_zz_j*n.real(H)+S/2*(t_zx_i*t_zx_j + t_zy_i*t_zy_j - t_zx_i*t_zy_j + t_zy_i*t_zx_j))    +
        np.imag(A)*(t_zz_i*t_zz_j*n.real(B)+S/2*(t_zx_i*t_zx_j - t_zy_i*t_zy_j + t_zx_i*t_zy_j + t_zy_i*t_zx_j))    +
        np.imag(B)*(t_zz_i*t_zz_j*n.real(A)+S/2*(t_zx_i*t_zx_j - t_zy_i*t_zy_j - t_zx_i*t_zy_j - t_zy_i*t_zx_j))    +
                )


def plot_gridBZ(ax,UC):
    """Plot BZ axes and borders."""
    ax.axis('off')
    ax.set_aspect('equal')
    f = 1.2
    v = 0.15
    ax.arrow(-np.pi*f,0,2*np.pi*f,0,color='k',head_width=0.1)
    ax.arrow(0,-np.pi*f,0,2*np.pi*f,color='k',head_width=0.1)
    ax.text(np.pi*f,v,r'$k_x$',size=20)
    ax.text(v,np.pi*f,r'$k_y$',size=20)
    if UC == 2:
        ax.plot([-np.pi,0],[0,np.pi],color='orange',lw=1)
        ax.plot([0,np.pi],[np.pi,0],color='orange',lw=1)
        ax.plot([np.pi,0],[0,-np.pi],color='orange',lw=1)
        ax.plot([0,-np.pi],[-np.pi,0],color='orange',lw=1)
    elif UC ==1:
        ax.plot([-np.pi,-np.pi],[-np.pi,np.pi],color='g',lw=1)
        ax.plot([-np.pi,np.pi],[np.pi,np.pi],color='g',lw=1)
        ax.plot([np.pi,np.pi],[-np.pi,np.pi],color='g',lw=1)
        ax.plot([-np.pi,np.pi],[-np.pi,-np.pi],color='g',lw=1)

def plot_BZ(ax,gridk,UC):
    """Plot BZ."""
    plot_gridBZ(ax,UC)
    ax.scatter(gridk[:,:,0],gridk[:,:,1])
    ax.set_title("Brillouin zone",size=20)

def plot_real_space(ax,gridk,a1,a2,UC):
    """Plot lattice points in real space."""
    nkx, nky = gridk.shape[:2]
    L = nky
    ax.axis('off')
    ax.set_aspect('equal')
    d = -0.5    #axes
    ax.arrow(d,d,0,0.8,color='k',head_width=0.05)
    ax.arrow(d,d,0.8,0,color='k',head_width=0.05)
    ax.text(d+0.8,d+0.1,r'$x$',size=20)
    ax.text(d+0.1,d+0.8,r'$y$',size=20)
    for i in range(L):  #grid
        ax.plot([i,i],[0,L-1],color='k',lw=0.2,zorder=0)
        ax.plot([0,L-1],[i,i],color='k',lw=0.2,zorder=0)
    for i1 in range(nkx):   #points
        for i2 in range(nky):
            vA = i1*a1 + i2*a2
            vB = i1*a1 + i2*a2 + np.array([1,0])
            if vA[0]<0:
                vA += L//2*a1
            if vB[0]<0:
                vB += L//2*a1
            ax.scatter(vA[0],vA[1],color='k',marker='o',s=70)
            if UC == 2:
                ax.scatter(vB[0],vB[1],color='r',marker='o',s=70)
    ax.set_title("Real space",size=20)

def plot_opposite_BZ(k,mk,UC):
    """Plot points in BZ with their negative counterpoint."""
    fig = plt.figure()
    ax = fig.add_subplot()
    plot_gridBZ(ax,UC)
    ax.scatter(k[0],k[1],c='r')
    ax.scatter(mk[0],mk[1],c='b')
    plt.show()

def get_pars_fn(names,pars):
    J1,J2,D1,D2,Lx,Ly,S,H_i,H_f,n_H = pars
    dirname,filename,extension = names
    fn = dirname+filename+'_'
    for i in pars:
        if type(i) is float or type(i) is np.float64:
            fn += "{:.4f}".format(i).replace('.',',')
        elif type(i) is int:
            fn += str(i)
        elif type(i) is str:
            fn += i
        if not i==pars[-1]:
            fn += '_'
    fn += extension
    return fn

def plot_corr(corr):
    """Code from Johannes.
    Takes the i,j,t matrix of correlators and computes the Fourier transform and then plots it.
    corr has dimension [H_list,Lx,Ly,Nt].
    """
    n_sr,Lx,Ly,Nt = corr.shape
    N_omega = 2000
    omega_list = np.linspace(-250,250,N_omega)
    #
    for p in range(n_sr):
        plt.subplot(2,5,p+1)
        a = corr[p] # insert matrix for stop rate ra of shape (x,y,ts), here (7,6,401)
        #Bunch of ks
        kx_list = np.fft.fftshift(np.fft.fftfreq(Lx,d=1))
        ky_list = np.fft.fftshift(np.fft.fftfreq(Ly,d=1))
        ks = []
        ks_m = []
        for kx in kx_list:
            for ky in ky_list:
                ks.append(np.sqrt(kx**2+ky**2))
                ks_m.append([np.sqrt(kx**2+ky**2)-0.01, np.sqrt(kx**2+ky**2)+0.01])
        ks = np.array(ks)
        k_inds = np.argsort(ks)
        ks_m = np.array(ks_m)[k_inds]
        vals, idx = np.unique(ks[k_inds], return_index=True)
        idx = np.append(idx, len(ks))
        #
        ks_ts = np.zeros((Lx,Ly,Nt), dtype=complex)
        for t in range(Nt):
            ks_ts[:,:,t] =  np.fft.fftshift(np.fft.fft2(a[:,:,t]))
        ks_ws_flat = np.zeros((Lx*Ly,N_omega), dtype=complex)
        for kx in range(Lx):
            for ky in range(Ly):
                i = ky + Ly*kx
                ks_ws_flat[i,:] = np.fft.fftshift(np.fft.fft(ks_ts[kx,ky,:], n=N_omega))
        ks_ws_flat = np.abs(ks_ws_flat[k_inds,:])
        #
        ks_ws_plot = []
        ks_m_plot  = []
        for i in range(len(vals)):
            val_c  = np.sum(ks_ws_flat[idx[i]:idx[i+1],:], axis=0)/(idx[i+1]-idx[i])
            ks_ws_plot.append(val_c)
            ks_m_plot.append(ks_m[idx[i]])
        vma = np.amax(ks_ws_plot)
        #For plotting
        omega_mesh = np.linspace(-250.250125063,250.250125063,2001)
        bla_x = np.linspace(0.,0.7,2)
        bla_y = np.linspace(-250.250125063,250.250125063,2)
        X, Y = np.meshgrid(bla_x, bla_y)
        plt.pcolormesh(X, Y, np.zeros((1,1)), cmap='magma', vmin=0, vmax=vma)
        ks_ws_plot.reverse()
        for i in range(len(vals)):
            X, Y = np.meshgrid(np.array(ks_m_plot[i]), omega_mesh)
            plt.pcolormesh(X, Y, ks_ws_plot[i].reshape((1,N_omega)).T, cmap='magma', vmin=0, vmax=vma)
        plt.colorbar(label='FFT (a.u.)')
        plt.ylim(-60,60)
        if p > 4:
            plt.xlabel('$|k|$')
        if p%5 == 0:
            plt.ylabel('$\\omega$')
        #plt.gca().invert_xaxis()
        plt.title('Stop ratio :$'+"{:.1f}".format(0.1*(p+1))+'$')
    plt.tight_layout()

    plt.subplots_adjust(wspace=0.2, hspace=0.3, left=0.04, right=0.97)

    fi = plt.gcf()
    fi.set_size_inches(18.,7.)
    # fi.savefig('corr_same.png',bbox_inches='tight',dpi=300)

    plt.show()

def extract_experimental_parameters(fn):
    """Import experimental Hamiltonian parameters"""
    with open(fn,'rb') as f:
        data = pickle.load(f)
    print(data['xx'].keys())
    exit()
    pairs = list(data['xx'].keys())
    g = np.zeros(len(pairs))
    for i in range(len(pairs)):
        g[i] = data['xx'][pairs[i]]
    sites = list(data['z'].keys())
    h = np.zeros(len(sites))
    for i in range(len(sites)):
        h[i] = data['z'][sites[i]]
    return g*1000,h*1000        #put it in MHz

























