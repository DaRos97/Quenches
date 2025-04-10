import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
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
    """J and D are tuple with 1st and 2nd nn. Each can be either a number or a Ns*Ns matrix of values for site dependent case."""
    if order=='canted_Neel':
        return (-J[0]*(np.cos(theta)**2+D[0]*np.sin(theta)**2), -J[1]*(np.cos(theta)**2+D[1]*np.sin(theta)**2) )
#        return (-J[0]*(np.cos(theta)**2*np.cos(phi)**2-np.sin(phi)**2+D[0]*np.sin(theta)**2*np.cos(phi)**2),
#                J[1]*(np.cos(theta)**2*np.cos(phi)**2+np.sin(phi)**2+D[1]*np.sin(theta)**2*np.cos(phi)**2))
def get_p_yy(theta,phi,J,D,order='canted_Neel'):
    if order=='canted_Neel':
        return (J[0], J[1])
#        return (-J[0]*(np.cos(theta)**2*np.sin(phi)**2-np.cos(phi)**2+D[0]*np.sin(theta)**2*np.sin(phi)**2),
#                J[1]*(np.cos(theta)**2*np.sin(phi)**2+np.cos(phi)**2+D[1]*np.sin(theta)**2*np.sin(phi)**2))
def get_p_zz(theta,phi,J,D,order='canted_Neel'):
    if order=='canted_Neel':
        return (-J[0]*(np.sin(theta)**2+D[0]*np.cos(theta)**2), -J[1]*(np.sin(theta)**2+D[1]*np.cos(theta)**2) )
#        return (-J[0]*(np.sin(theta)**2+D[0]*np.cos(theta)**2),
#                J[1]*(np.sin(theta)**2+D[1]*np.cos(theta)**2))
def get_p_xy(theta,phi,J,D,order='canted_Neel'):
    if order=='canted_Neel':
        return (0,0)
#        return (J[0]/2*np.sin(2*phi)*(np.cos(theta)**2+1+D[0]*np.sin(theta)**2),
#                -J[1]/2*np.sin(2*phi)*(np.cos(theta)**2-1+D[1]*np.sin(theta)**2))

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

def get_angles(S,J_i,D_i,h_i):
    """Compute angles theta and phi of quantization axis depending on Hamiltonian parameters.
    For site-dependent Hamiltonian parameters we take the average.
    """
    if type(J_i[0]) in [float,int,np.float64]:   #if we give a single number for J1,J2,H etc.. -> static_dispersion.py
        J = J_i
        D = D_i
        h = h_i
    else:   #f we give a site dependent value of J1, j2 etc.., we need an average -> static_ZZ_*.py
        J = []
        D = []
        for i in range(2):
            if not (J_i[i] == np.zeros(J_i[i].shape)).all():
                J.append(abs(float(np.sum(J_i[i])/(J_i[i][np.nonzero(J_i[i])]).shape)))
            else:
                J.append(0)
            if not (D_i[i] == np.zeros(D_i[i].shape)).all():
                D.append(float(np.sum(D_i[i])/(D_i[i][np.nonzero(D_i[i])]).shape))
            else:
                D.append(0)
        if J[0]!=0:
            D[0] = D[0]/J[0]        #As we defined in notes
        if J[1]!=0:
            D[1] = D[1]/J[1]        #As we defined in notes
        if not (h_i == np.zeros(h_i.shape)).all():
            h_av = float(np.sum(h_i)/(h_i[np.nonzero(h_i)]).shape)
            h_stag = np.absolute(h_i[np.nonzero(h_i)]-h_av)
            h = float(np.sum(h_stag)/(h_stag[np.nonzero(h_stag)]).shape)
        else:
            h = 0
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
            gridk[i1,i2,0] = dx*(1+i1) #- np.pi
            gridk[i1,i2,1] = dy*(1+i2) #- np.pi
    return gridk

def get_Hamiltonian_rs(*parameters):
    """
    Compute the real space Hamiltonian -> (2Ns x 2Ns).
    Conventions for the real space wavefunction and other things are in the notes.
    SECOND NEAREST-NEIGHBOR NOT IMPLEMENTED.
    """
    S,Lx,Ly,h_i,theta,phi,J_i,D_i = parameters
    Ns = Lx*Ly
    ham = np.zeros((2*Ns,2*Ns),dtype=complex)
    #
    p_zz = get_p_zz(theta,phi,J_i,D_i)
    p_xx = get_p_xx(theta,phi,J_i,D_i)
    p_yy = get_p_yy(theta,phi,J_i,D_i)
    p_xy = get_p_xy(theta,phi,J_i,D_i)
    #diagonal
    ham[:Ns,:Ns] = abs(h_i)/2*np.cos(theta) - S/2*np.diag(np.sum(p_zz[0],axis=0))
    ham[Ns:,Ns:] = abs(h_i)/2*np.cos(theta) - S/2*np.diag(np.sum(p_zz[0],axis=0))
    #off_diag 1 - nn
    off_diag_1_nn = S/4*(p_xx[0]+p_yy[0])
    ham[:Ns,:Ns] += off_diag_1_nn
    ham[Ns:,Ns:] += off_diag_1_nn
    #off_diag 2 - nn
    off_diag_2_nn = S/4*(p_xx[0]-p_yy[0]+2*1j*p_xy[0])
    ham[:Ns,Ns:] += off_diag_2_nn
    ham[Ns:,:Ns] += off_diag_2_nn.T.conj().T
    return ham

def get_correlator(ts_i,ts_j,S,G,H,A,B):
    """Compute real space zz correlator as in notes."""
    t_zz_i,t_xz_i,t_yz_i = ts_i
    t_zz_j,t_xz_j,t_yz_j = ts_j
    return 2*1j*( t_zz_i*t_zz_j*np.imag(G*H+A*B)
                 + S/2*(t_xz_i*t_xz_j + t_yz_i*t_yz_j)*np.imag(G+H) + S/2*(t_xz_i*t_xz_j - t_yz_i*t_yz_j)*np.imag(A+B)
                 + S/2*(t_xz_i*t_yz_j + t_yz_i*t_xz_j)*np.real(A-B) + S/2*(t_xz_i*t_yz_j - t_yz_i*t_xz_j)*np.real(H-G)
                )

def get_ts(theta,phi):
    """Compute the parameters t_zz, t_xz and t_yz as in notes for sublattice A and B.
    Sublattice A has negative magnetic feld.
    """
    result = [(np.cos(theta),-np.sin(theta),0),
              (-np.cos(theta),np.sin(theta),0) ]
    return result

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
    Takes the i,j,t matrix of correlators, computes the Fourier transform and then plots it.
    corr has dimension [n_stop_ratio,Lx,Ly,Nt].
    """
    n_sr,Lx,Ly,Nt = corr.shape
    N_omega = 2000      #default from Jeronimo
    omega_list = np.linspace(-250,250,N_omega)
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
        a = corr[p] # insert matrix for stop ratio of shape (x,y,Nt), here (6,7,401)
        #Fourier transform x,y->kx,ky with fft2 for each time t
        ks_ts = np.zeros((Lx,Ly,Nt), dtype=complex)
        for t in range(Nt):
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
#            print(i,idx[i],idx[i+1],ks_m[i])
            ks_ws_plot.append(val_c)
            ks_m_plot.append(ks_m[idx[i]])
#            input()
        vma = np.amax(ks_ws_plot)
        #Plot 0 background
        ax.pcolormesh(X0, Y0, np.zeros((1,1)), cmap='magma', vmin=0, vmax=vma)

        ks_ws_plot.reverse()    ###################################################

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
        #plt.gca().invert_xaxis()
        ax.set_title('Stop ratio :$'+"{:.1f}".format(0.1*(p+1))+'$')
    plt.tight_layout()

    plt.subplots_adjust(wspace=0.2, hspace=0.3, left=0.04, right=0.97)

    plt.show()

def extract_experimental_parameters(fn):
    """Import experimental Hamiltonian parameters"""
    with open(fn,'rb') as f:
        data = pickle.load(f)
    #extract Lx and Ly from filename
    Lx = int(fn[fn.index('/')+10])
    Ly = int(fn[fn.index('/')+12])
    Ns = Lx*Ly
    #Extract sites names and give them an index in the Lx*Ly 1D setting
    dic_key = {}
    x0,y0 = list(data['z'].keys())[0]
    for i in range(Ns):
        key = list(data['z'].keys())[i]
        dic_key[key] = (key[1]-y0) + (key[0]-x0)*Ly
    #On-site 'z' terms
    h = np.zeros((Ns,Ns))
    keys = list(data['z'].keys())
    for i in range(Ns):
        ind = dic_key[keys[i]]
        h[ind,ind] = data['z'][keys[i]]*1000     #MHz
    #Nearest-neighbor terms: 
    g1 = np.zeros((Ns,Ns))
    keys = list(data['xx'].keys())
    for i in range(len(keys)):
        ind_a = dic_key[keys[i][0]]
        ind_b = dic_key[keys[i][1]]
        g1[ind_a,ind_b] = g1[ind_b,ind_a] = data['xx'][keys[i]]*1000     #MHz
    d1 = np.zeros((Ns,Ns))
    keys = list(data['zz'].keys())
    for i in range(len(keys)):
        ind_a = dic_key[keys[i][0]]
        ind_b = dic_key[keys[i][1]]
        d1[ind_a,ind_b] = d1[ind_b,ind_a] = data['zz'][keys[i]]*1000     #MHz
    #Next-nearest-neighbor terms: 
    g2 = np.zeros((Ns,Ns))
    keys = list(data['xix'].keys())
    for i in range(len(keys)):
        ind_a = dic_key[keys[i][0]]
        ind_b = dic_key[keys[i][1]]
        g2[ind_a,ind_b] = g2[ind_b,ind_a] = data['xix'][keys[i]]*1000     #MHz
    return Lx,Ly,g1,g2,d1,h

def get_Hamiltonian_parameters(time_steps,g1_in,g2_in,d1_in,h_in,g1_fin,g2_fin,d1_fin,h_fin):
    """Compute g1(t), g2(t), d1(t) and h(t) for each time and site of the ramp."""
    t_values = np.linspace(0,1,time_steps).reshape(time_steps,1,1)
    g1_t_i = (1-t_values)*g1_in + t_values*g1_fin
    g2_t_i = (1-t_values)*g2_in + t_values*g2_fin
    d1_t_i = (1-t_values)*d1_in + t_values*d1_fin
    h_t_i = (1-t_values)*h_in + t_values*h_fin
    return g1_t_i,g2_t_i,d1_t_i,h_t_i
























