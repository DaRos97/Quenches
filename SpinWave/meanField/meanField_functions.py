import numpy as np

def get_parameters(Lx,Ly,time_steps,g_val,h_val):
    """
    Define Hamiltonian parameters at all time steps.
    """
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
            h_in[iy+ix*Ly,iy+ix*Ly] = -(-1)**(ix+iy)*    h_val
    h_fin = np.zeros((Ns,Ns))
    g1_t_i,g2_t_i,d1_t_i,h_t_i = get_time_parameters(time_steps,g1_in,g2_in,d1_in,h_in,g1_fin,g2_fin,d1_fin,h_fin)   #parameters of Hamiltonian which depend on time
    return g1_t_i,g2_t_i,d1_t_i,h_t_i

def get_time_parameters(time_steps,g1_in,g2_in,d1_in,h_in,g1_fin,g2_fin,d1_fin,h_fin):
    """
    Compute g1(t), g2(t), d1(t) and h(t) for each time and site of the ramp.
    """
    t_values = np.linspace(0,1,time_steps).reshape(time_steps,1,1)
    g1_t_i = (1-t_values)*g1_in + t_values*g1_fin
    g2_t_i = (1-t_values)*g2_in + t_values*g2_fin
    d1_t_i = (1-t_values)*d1_in + t_values*d1_fin
    h_t_i = (1-t_values)*h_in + t_values*h_fin
    return g1_t_i,g2_t_i,d1_t_i,h_t_i

def get_angles(S,J_i,D_i,h_i):
    """
    Compute angles theta and phi of quantization axis depending on Hamiltonian parameters.
    For site-dependent Hamiltonian parameters we take the average.
    Just for c-Neel phase.
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
    #
    phi = 0
    return (theta,phi)

def get_ts(theta,phi):
    """Compute the parameters t_z, t_x and t_y as in notes for sublattice A and B.
    Sublattice A has negative magnetic feld.
    """
    result = [
        [   #sublattice A
            (np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)),  #t_zx,t_zy,t_zz
            (np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),-np.sin(theta)), #t_xx,t_xy,t_xz
            (-np.sin(phi),np.cos(phi),0)  ],                                      #t_yx,t_yy,t_yz
        [   #sublattice B
            (-np.sin(theta)*np.cos(phi),-np.sin(theta)*np.sin(phi),-np.cos(theta)),  #t_zx,t_zy,t_zz
            (-np.cos(theta)*np.cos(phi),-np.cos(theta)*np.sin(phi),np.sin(theta)),   #t_xx,t_xy,t_xz
            (-np.sin(phi),np.cos(phi),0)  ],                                         #t_yx,t_yy,t_yz
    ]
    return result

def get_ps(alpha,beta,ts,J,D,order='c-Neel'):
    """
    Compute coefficient p_gamma^{alpha,beta} for a given classical order.
    alpha,beta=0,1,2 -> z,x,y like for ts.
    J and D are tuple with 1st and 2nd nn. Each can be either a number or a Ns*Ns matrix of values for site dependent case.
    """
    if order=='c-Neel': #nn: A<->B, nnn: A<->A
        #Nearest neighor
        nn =  J[0]*ts[0][alpha][0]*ts[1][beta][0] + J[0]*ts[0][alpha][1]*ts[1][beta][1] + J[0]*D[0]*ts[0][alpha][2]*ts[1][beta][2]
        nnn = J[1]*ts[0][alpha][0]*ts[0][beta][0] + J[1]*ts[0][alpha][1]*ts[0][beta][1] + J[1]*D[1]*ts[0][alpha][2]*ts[0][beta][2]
    return (nn,nnn)


def get_Hamiltonian_rs(*parameters):
    """
    Compute the real space Hamiltonian -> (2Ns x 2Ns).
    Conventions for the real space wavefunction and other things are in the notes.
    SECOND NEAREST-NEIGHBOR NOT IMPLEMENTED.
    """
    S,Lx,Ly,h_i,ts,theta,phi,J_i,D_i = parameters
    Ns = Lx*Ly
    #
    p_zz = get_ps(0,0,ts,J_i,D_i)
    p_xx = get_ps(1,1,ts,J_i,D_i)
    p_yy = get_ps(2,2,ts,J_i,D_i)
    p_xy = get_ps(1,2,ts,J_i,D_i)
#    print(np.sum(p_zz[0],axis=0))
#    input()
    fac0 = 1#2      #Need to change this in notes -> counting of sites from 2 to 1 sites per UC
    fac1 = 1#2
    fac2 = 2#4
    ham = np.zeros((2*Ns,2*Ns),dtype=complex)
    #diagonal
    ham[:Ns,:Ns] = abs(h_i)/fac0*np.cos(theta) - S/fac1*np.diag(np.sum(p_zz[0],axis=0))
    ham[Ns:,Ns:] = abs(h_i)/fac0*np.cos(theta) - S/fac1*np.diag(np.sum(p_zz[0],axis=0))
    #off_diag 1 - nn
    off_diag_1_nn = S/fac2*(p_xx[0]+p_yy[0])
    ham[:Ns,:Ns] += off_diag_1_nn
    ham[Ns:,Ns:] += off_diag_1_nn
    #off_diag 2 - nn
    off_diag_2_nn = S/fac2*(p_xx[0]-p_yy[0]+2*1j*p_xy[0])
    ham[:Ns,Ns:] += off_diag_2_nn
    ham[Ns:,:Ns] += off_diag_2_nn.T.conj().T
    return ham

