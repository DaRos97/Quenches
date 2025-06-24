import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, LogNorm
import pickle
from scipy.fft import fftfreq, fftshift, fft, fft2, dstn
from collections import Counter     #for Bogoliubov momentum

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

def get_N_11(*pars):
    """Compute N_11 as in notes."""
    S,Gamma,h,ts,theta,phi,J,D = pars
    p_zz = get_ps(0,0,ts,J,D)
    p_xx = get_ps(1,1,ts,J,D)
    p_yy = get_ps(2,2,ts,J,D)
    result = h/2*np.cos(theta)
    for i in range(2):
        result += S*(Gamma[i]*(p_xx[i]+p_yy[i])/2-2*p_zz[i])
    return result
def get_N_12(*pars):
    """Compute N_12 as in notes."""
    S,Gamma,h,ts,theta,phi,J,D = pars
    p_xx = get_ps(1,1,ts,J,D)
    p_yy = get_ps(2,2,ts,J,D)
    p_xy = get_ps(1,2,ts,J,D)
    result = 0
    for i in range(2):
        result += S/2*Gamma[i]*(p_xx[i]-p_yy[i]-2*1j*p_xy[i])
    return result

def get_rk(*pars):
    """
    Compute rk as in notes.
    Controls are neded for ZZ in k.
    """
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

def get_E_0(*pars):
    """Compute E_0 as in notes."""
    S,Gamma,h,ts,theta,phi,J,D = pars
    p_zz = get_ps(0,0,ts,J,D)
    result = -h*(S+1/2)*np.cos(theta)
    for i in range(2):
        result += 2*S*(S+1)*p_zz[i]
    return result

def get_epsilon(*pars):
    """
    Compute dispersion epsilon as in notes.
    Controls are neded for ZZ in k.
    """
    N_11 = get_N_11(*pars)
    N_12 = get_N_12(*pars)
    result = np.sqrt(N_11**2-np.absolute(N_12)**2,where=(N_11**2>=np.absolute(N_12)**2))
#    result[N_11**2<np.absolute(N_12)**2] = 0
    return result

def get_E_GS(*pars):
    """Compute ground state energy as in notes."""
    E_0 = get_E_0(*pars)
    epsilon = get_epsilon(*pars)
    Ns = epsilon.shape[0]*epsilon.shape[1]
#    return E_0 + np.sum(epsilon[~np.isnan(epsilon)])/Ns
    return E_0 + np.sum(epsilon)/Ns

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

def BZgrid(Lx,Ly):
    """
    Compute BZ coordinates of points.
    """
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

def correlator_ze(S,Lx,Ly,ts,site0,ind_i,ind_j,A,B,G,H,exclude_list=[]):
    """
    Compute real space <[Z_i(t),E_j(0)]> correlator.
    """
    ts_i = ts[(site0+ind_i//Ly+ind_i%Ly)%2]
    ts_j = ts[(site0+ind_j//Ly+ind_j%Ly)%2]
    ZE = np.zeros(A[0,0].shape,dtype=complex)
    ind_nn_j = get_nn(ind_j,Lx,Ly)
    for ind_s in ind_nn_j:
        ts_s = ts[(site0+ind_s//Ly+ind_s%Ly)%2]
        ts_list = [ts_i, ts_j, ts_s]
        for ops in ['ZXX','XZX','XXZ','ZZZ','ZYY']:
            original_op = 'ZXX'
            if ops=='ZYY':
                original_op = 'ZYY'
            list_terms = compute_combinations(ops,[ind_i,ind_j,ind_s],'t00',S)
            coeff_t = compute_coeff_t(ops,original_op,ts_list)
            for t in list_terms:
                ZE += coeff_t * t[0] * compute_contraction(t[1],t[2],t[3],A,B,G,H,exclude_list)
    return 2*1j/len(ind_nn_j)*np.imag(ZE)

def correlator_zz(S,Lx,Ly,ts,site0,ind_i,ind_j,A,B,G,H,exclude_list=[]):
    """
    Compute real space <[Z_i(t),Z_j(0)]> correlator.
    """
    ts_i = ts[(site0+ind_i//Ly+ind_i%Ly)%2]
    ts_j = ts[(site0+ind_j//Ly+ind_j%Ly)%2]
    ts_list = [ts_i, ts_j]
    ZZ = np.zeros(A[0,0].shape,dtype=complex)
    for ops in ['XX','ZZ']:
        original_op = 'ZZ'
        list_terms = compute_combinations(ops,[ind_i,ind_j],'t0',S)
        coeff_t = compute_coeff_t(ops,original_op,ts_list)
        for t in list_terms:
            ZZ += coeff_t * t[0] * compute_contraction(t[1],t[2],t[3],A,B,G,H,exclude_list)
    return 2*1j*np.imag(ZZ)

def correlator_xx(S,Lx,Ly,ts,site0,ind_i,ind_j,A,B,G,H,exclude_list=[]):
    """
    Compute real space <[X_i(t),X_j(0)]> correlator.
    """
    ts_i = ts[(site0+ind_i//Ly+ind_i%Ly)%2]
    ts_j = ts[(site0+ind_j//Ly+ind_j%Ly)%2]
    ts_list = [ts_i, ts_j]
    XX = np.zeros(A[0,0].shape,dtype=complex)
    for ops in ['XX','ZZ']:
        original_op = 'XX'
        list_terms = compute_combinations(ops,[ind_i,ind_j],'t0',S)
        coeff_t = compute_coeff_t(ops,original_op,ts_list)
        for t in list_terms:
            XX += coeff_t * t[0] * compute_contraction(t[1],t[2],t[3],A,B,G,H,exclude_list)
    return 2*1j*np.imag(XX)

def correlator_ez(S,Lx,Ly,ts,site0,ind_i,ind_j,A,B,G,H,exclude_list=[]):
    """
    Compute real space <[E_i(t),Z_j(0)]> correlator.
    """
    ts_i = ts[(site0+ind_i//Ly+ind_i%Ly)%2]
    ts_j = ts[(site0+ind_j//Ly+ind_j%Ly)%2]
    EZ = np.zeros(A[0,0].shape,dtype=complex)
    ind_nn_i = get_nn(ind_i,Lx,Ly)
    for ind_r in ind_nn_i:
        ts_r = ts[(site0+ind_r//Ly+ind_r%Ly)%2]
        ts_list = [ts_i, ts_r, ts_j]
        for ops in ['XXZ','XZX','ZXX','ZZZ','YYZ']:
            original_op = 'XXZ'
            if ops=='YYZ':
                original_op = 'YYZ'
            list_terms = compute_combinations(ops,[ind_i,ind_r,ind_j],'tt0',S)
            coeff_t = compute_coeff_t(ops,original_op,ts_list)
            for t in list_terms:
                EZ += coeff_t * t[0] * compute_contraction(t[1],t[2],t[3],A,B,G,H,exclude_list)
    return 2*1j/len(ind_nn_i)*np.imag(EZ)

def correlator_ee(S,Lx,Ly,ts,site0,ind_i,ind_j,A,B,G,H,exclude_list=[]):
    """
    Compute real space <[E_i(t),E_j(0)]> correlator.
    Site j is where the E perturbation is applied -> we assume it is
    somewhere in the middle which has all 4 nearest neighbors and average
    over them.
    """
    ts_i = ts[(site0+ind_i//Ly+ind_i%Ly)%2]
    ts_j = ts[(site0+ind_j//Ly+ind_j%Ly)%2]
    EE = np.zeros(A[0,0].shape,dtype=complex)
    ind_nn_i = get_nn(ind_i,Lx,Ly)
    ind_nn_j = [ind_j,]
#    ind_nn_j = get_nn(ind_j,Lx,Ly)
    for ind_r in ind_nn_i:
        ts_r = ts[(site0+ind_r//Ly+ind_r%Ly)%2]
        for ind_s in ind_nn_j:
            ts_s = ts[(site0+ind_s//Ly+ind_s%Ly)%2]
            ts_list = [ts_i,ts_r,ts_j,ts_s]
            for ops in ['XXXX','ZZZZ','XXZZ','ZZXX','XZXZ','ZXZX','ZXXZ','XZZX','XXYY','ZZYY','YYXX','YYZZ','YYYY']:
                original_op = 'XXXX'
                if ops in ['XXYY','ZZYY']:
                    original_op = 'XXYY'
                if ops in ['YYXX','YYZZ']:
                    original_op = 'YYXX'
                if ops == 'YYYY':
                    original_op = 'YYYY'
                list_terms = compute_combinations(ops,[ind_i,ind_r,ind_j,ind_s],'tt00',S)
                coeff_t = compute_coeff_t(ops,original_op,ts_list)
                for t in list_terms:
                    contraction = compute_contraction(t[1],t[2],t[3],A,B,G,H,exclude_list)
                    EE += coeff_t * t[0] * contraction
    return 2*1j/len(ind_nn_i)/len(ind_nn_j)*np.imag(EE)

def generate_pairings(elements):
    """
    Here we get all the possible permutation lists for the Wick contraction -> perfect matchings.
    """
    if len(elements) == 0:
        return [[]]
    pairings = []
    a = elements[0]
    for i in range(1, len(elements)):
        b = elements[i]
        rest = elements[1:i] + elements[i+1:]
        for rest_pairing in generate_pairings(rest):
            pairings.append([(a, b)] + rest_pairing)
    return pairings

permutation_lists = {}
for i in range(2,16,2):
    permutation_lists[i] = generate_pairings(list(range(i)))

def compute_contraction(op_list,ind_list,time_list,A,B,G,H,exclude_list=[]):
    """
    Here we compute the contractions using Wick decomposition of the single operator list `op_list`, with the given sites and times.
    First we compute all the 2-operator terms.
    len(op) = 2 -> 1 term
    len(op) = 4 -> 3 terms
    len(op) = 6 -> 15 terms
    len(op) = 8 -> 105 terms
    etc..
    """
    ops_dic = {'aa':B,'bb':A,'ab':H,'ba':G}
    if len(op_list) in [0,]+exclude_list:
        return 0
    perm_list = permutation_lists[len(op_list)]
    result = 0
    for i in range(len(perm_list)):
        temp = 1
        for j in range(len(perm_list[i])):
            op_ = op_list[perm_list[i][j][0]]+op_list[perm_list[i][j][1]]
            ind_ =  [ ind_list[perm_list[i][j][0]], ind_list[perm_list[i][j][1]] ]
            time_ = time_list[perm_list[i][j][0]]!=time_list[perm_list[i][j][1]]
            op = ops_dic[op_][ind_[0],ind_[1]]
            temp *= op if time_ else op[0]
        result += temp
    return result

def compute_coeff_t(op_t,op_o,ts_list):
    """
    Compute the product of t-coefficients given the original operator `op_o` and the transformed one `op_t`
    """
    ind_t_dic = {'Z':0, 'X':1, 'Y':2}
    ind_o_dic = {'X':0, 'Y':1, 'Z':2}
    coeff = 1
    for i in range(len(op_t)):
        ind_transformed = ind_t_dic[op_t[i]]
        ind_original = ind_o_dic[op_o[i]]
        coeff *= ts_list[i][ind_transformed][ind_original]
    return coeff

def compute_combinations(op_list,ind_list,time_list,S):
    """
    Here we compute symbolically all terms of the HP expansion of the operator list `op`.
    a -> a, a^dag -> b.
    Need also to keep sites information
    X_j = sqrt(S/2)(a_j+b_j)
    Y_j = -i*sqrt(S/2)(a_j-b_j)
    Z_j = S-b_ja_j
    return a list of 3-tuple, with first element a coefficient (pm (i) S**n) second element an operator list ('abba..') and third element a list of sites ([ind_i,ind_j,..]) of same length as the operator string
    """
    op_dic = {'X':[np.sqrt(S/2),'a'], 'Y':'(a-b)', 'Z':'S-ba'}
    coeff_dic = {'X':np.sqrt(S/2), 'Y':1j*np.sqrt(S/2), 'Z':1}
    terms = []
    coeff = 1
    for i in range(len(op_list)):
        if op_list[i]=='X':
            terms.append([ [np.sqrt(S/2),'a',[ind_list[i]], time_list[i]] , [np.sqrt(S/2),'b',[ind_list[i]], time_list[i] ]])
        if op_list[i]=='Y':
            terms.append([ [-1j*np.sqrt(S/2),'a',[ind_list[i]], time_list[i]] , [1j*np.sqrt(S/2),'b',[ind_list[i] ], time_list[i] ]])
        if op_list[i]=='Z':
            terms.append([ [S,'',[],''] , [-1,'ba',[ind_list[i],ind_list[i]], time_list[i]+time_list[i] ]])
    for i in range(len(op_list)-1): #n-1 multiplications
        new_terms = []
        mult = []
        for j in range(len(terms[0])):
            for l in range(len(terms[1])):
                mult.append( [terms[0][j][0]*terms[1][l][0], terms[0][j][1]+terms[1][l][1], terms[0][j][2]+terms[1][l][2], terms[0][j][3]+terms[1][l][3] ]  )
        new_terms.append(mult)
        #remaining part
        for j in range(2,len(terms)):
            new_terms.append(terms[j])
        terms = list(new_terms)
    return terms[0]

def get_nn(ind,Lx,Ly):
    """
    Compute indices of nearest neighbors of site ind.
    """
    result= []
    if ind+Ly<=(Lx*Ly-1):        #right neighbor
        result.append(ind+Ly)
    if ind-Ly>=0:
        result.append(ind-Ly)   #left neighbor
    if (ind+1)//Ly==ind//Ly:    #up neighbor
        result.append(ind+1)
    if (ind-1)//Ly==ind//Ly:    #bottom neighbor
        result.append(ind-1)
    return result

get_correlator = {'zz':correlator_zz,'xx':correlator_xx,'ze':correlator_ze,'ez':correlator_ez, 'ee':correlator_ee}

def get_parameters(use_experimental_parameters,Lx,Ly,time_steps,g_val,h_val):
    """
    Import Hamiltonian parameters either from file (experimental ones) or by uniform default values.
    """
    if not use_experimental_parameters:
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
    else:       #not implemented yet
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

def fourier_fft(correlator_xt,N_omega):
    """
    Compute the standard 2D Fourier transform.
    In time we always use fft.
    """
    n_sr, Lx, Ly, Nt = correlator_xt.shape
    correlator_kw = np.zeros((n_sr,Lx,Ly,N_omega),dtype=complex)
    for i_sr in range(n_sr):
        temp = np.zeros((Lx,Ly,Nt),dtype=complex)
        for it in range(Nt):
            temp[:,:,it] = fftshift(fft2(correlator_xt[i_sr,:,:,it],norm='ortho'))
        for ix in range(Lx):
            for iy in range(Ly):
                correlator_kw[i_sr,ix,iy] = fftshift(fft(temp[ix,iy],n=N_omega))
    # Momenta
    return correlator_kw

def fourier_dst(correlator_xt,N_omega=2000,type_dst=1):
    """
    Compute the Discrete Sin Transform since we have open BC.
    In time we always use fft.
    correlator_xt has shape (n_sr, Lx, Ly, Nt) with n_sr number of stop ratios (10) and Nt number of time steps (400) in the measurement
    """
    n_sr, Lx, Ly, Nt = correlator_xt.shape
    correlator_kw = np.zeros((n_sr,Lx,Ly,N_omega),dtype=complex)
    temp = np.zeros((n_sr,Lx,Ly,Nt),dtype=complex)
    for i_sr in range(n_sr):
        for it in range(Nt):
            temp[i_sr,:,:,it] = dstn(correlator_xt[i_sr,:,:,it], type=type_dst, norm='ortho')
        for ix in range(Lx):
            for iy in range(Ly):
                correlator_kw[i_sr,ix,iy] = fftshift(fft(temp[i_sr,ix,iy],n=N_omega))
    return correlator_kw

def fourier_dst2(correlator_xt,N_omega=2000):
    """
    Compute the Discrete Sin Transform explicitly using sin functions.
    In time we always use fft.
    correlator_xt has shape (n_sr, Lx, Ly, Nt) with n_sr number of stop ratios (10) and Nt number of time steps (400) in the measurement
    """
    n_sr, Lx, Ly, Nt = correlator_xt.shape
    kx = np.pi * np.arange(1, Lx + 1) / (Lx + 1)
    ky = np.pi * np.arange(1, Ly + 1) / (Ly + 1)
    X = np.arange(1,Lx+1)
    Y = np.arange(1,Ly+1)
    correlator_kw = np.zeros((n_sr,Lx,Ly,N_omega),dtype=complex)
    temp = np.zeros((n_sr,Lx,Ly,Nt),dtype=complex)
    for i_sr in range(n_sr):
        for it in range(Nt):
            for ikx in range(Lx):
                for iky in range(Ly):
                    sin_transf = np.outer(np.sin(kx[ikx]*X),np.sin(ky[iky]*Y))
                    temp[i_sr,ikx,iky,it] = np.sum(sin_transf*correlator_xt[i_sr,:,:,it])
        for ix in range(Lx):
            for iy in range(Ly):
                correlator_kw[i_sr,ix,iy] = fftshift(fft(temp[i_sr,ix,iy],n=N_omega))
    return correlator_kw

def fourier_dat(correlator_xt,U_,V_,N_omega=2000):
    """
    Compute the Discrete Amazing Transform with the Bogoliubov functions.
    In time we always use fft.
    correlator_xt has shape (n_sr, Lx, Ly, Nt) with n_sr number of stop ratios (10) and Nt number of time steps (400) in the measurement
    U_ and V_ are (Ns,Ns) matrices -> (x,n)
    """
    n_sr, Lx, Ly, Nt = correlator_xt.shape
    Am = np.reshape(U_+V_,shape=(n_sr,Lx,Ly,Lx*Ly))      #matrix (n_sx,x,y,n)
    correlator_kw = np.zeros((n_sr,Lx,Ly,N_omega),dtype=complex)
    temp = np.zeros((n_sr,Lx,Ly,Nt),dtype=complex)
    for i_sr in range(n_sr):
        k_bog = np.zeros((Lx*Ly,2),dtype=int)
        for i_n in range(Lx*Ly):
            k_bog[i_n] = np.array(get_momentum_Bogoliubov(np.real(np.reshape(U_[i_sr,:,i_n],shape=(Lx,Ly)))))
        for it in range(Nt):
            for i_n in range(Lx*Ly):
                temp[i_sr,k_bog[i_n,0],k_bog[i_n,1],it] = np.sum(Am[i_sr,:,:,i_n]*correlator_xt[i_sr,:,:,it])
        for ix in range(Lx):
            for iy in range(Ly):
                correlator_kw[i_sr,ix,iy] = fftshift(fft(temp[i_sr,ix,iy],n=N_omega))
    return correlator_kw

def sign_changes(v):
    """
    Computes how many times the vector changes sign.
    """
    indn0 = np.where(np.abs(v)>1e-10)[0]
    vecn0 = v[indn0]
    return len(np.where(vecn0[:-1] * vecn0[1:]<0)[0])

def get_momentum_Bogoliubov(U,disp=False):
    """
    Here we extract the momentum in x and y direction by extracting the number of times the Bogoliubov function U changes sign.
    We take one line in x and y direction
    """
    Lx,Ly = U.shape
    if np.sum(np.abs(U))<1e-8:
        if disp:
            print("zero mode")
        return Lx-1,Ly-1
    ix = Lx//2 # if Lx%2==1 else Lx//2-2
    iy = Ly//2+2 # if Ly%2==1 else Ly//2-2
    ly = U[ix,:]        #vector along y at specific x
    lx = U[:,iy]
    sxs = [sign_changes(U[:,i]) for i in range(Ly)]
    sys = [sign_changes(U[i,:]) for i in range(Lx)]
    sx, _ = Counter(sxs).most_common(1)[0]
    sy, _ = Counter(sys).most_common(1)[0]
    return [sx,sy]
    if disp:
        print("sx:",sxs)
        print("sy:",sys)
        X,Y = np.meshgrid(np.arange(Lx),np.arange(Ly))
        fig=plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(X,Y,U.T,cmap='plasma')
        plt.show()

class SqrtNorm(mcolors.Normalize):
    def __call__(self, value, clip=None):
        return (super().__call__(value, clip))**(1/2) #np.sqrt(super().__call__(value, clip))

def plot(correlator_kw, **kwargs):
    """
    Plot frequency over mod k for the different stop ratios.
    correlator_kw has shape (n_sr, Lx, Ly, Nomega) with n_sr number of stop ratios (10) and Nomega number of frequency stps (2000)
    """
    n_sr,Lx,Ly,N_omega = correlator_kw.shape

    # Momenta
    if kwargs['fourier_type'] in ['fft',]:
        kx = fftshift(fftfreq(Lx,d=1)*2*np.pi)
        ky = fftshift(fftfreq(Ly,d=1)*2*np.pi)
    elif kwargs['fourier_type'] in ['dst','dst2','dat']:
        kx = np.pi * np.arange(1, Lx + 1) / (Lx + 1)
        ky = np.pi * np.arange(1, Ly + 1) / (Ly + 1)

    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K_mag = np.sqrt(KX**2 + KY**2)
    freqs = fftshift(fftfreq(N_omega,0.8/400))

    K_flat = K_mag.ravel()
    # Define k bins
    if 'n_bins' in kwargs.keys():
        num_k_bins = kwargs['n_bins']
    else:
        num_k_bins = 100
    k_bins = np.linspace(0,np.sqrt(2)*np.pi, num_k_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    #
    K_mesh, W_mesh = np.meshgrid(k_centers, freqs, indexing='ij')
    # Figure
    fig = plt.figure(figsize=(20.8,8))
    if 'title' in kwargs.keys():
        plt.suptitle(kwargs['title'],fontsize=20)
    if 'fourier_type' in kwargs.keys():
        cbar_label = kwargs['fourier_type']
    else:
        cbar_label = ''
    P_k_omega_sr = np.zeros((n_sr,num_k_bins,N_omega))
    for i_sr in range(n_sr):
        corr_flat = correlator_kw[i_sr].reshape(Lx*Ly, N_omega)
        for i in range(num_k_bins):
            mask = (K_flat >= k_bins[i]) & (K_flat < k_bins[i+1])
            if np.any(mask):
                P_k_omega_sr[i_sr, i, :] = np.mean(np.abs(corr_flat[mask, :]), axis=0)
    vmax = np.max(P_k_omega_sr)
    for i_sr in range(n_sr):     # Plotting
        vmax = np.max(P_k_omega_sr[i_sr])
        P_k_omega = P_k_omega_sr[i_sr]
        ax = fig.add_subplot(2,5,i_sr+1)
        ax.set_facecolor('black')
        mesh = ax.pcolormesh(K_mesh, W_mesh, P_k_omega,
                             shading='auto',
                             cmap='inferno',
                             norm=SqrtNorm(vmin=0,vmax=vmax)
                            )
        ax.set_ylim(-70,70)
        ax.set_xlim(0,np.sqrt(2)*np.pi)
        #plt.title("Spectral power in $|k|$ vs $\\omega$")
        cbar = fig.colorbar(mesh, ax=ax)
        if i_sr in [4,9]:
            cbar.set_label(cbar_label,fontsize=15)
        if i_sr in [0,5]:
            ax.set_ylabel(r'$\omega$',fontsize=15)
        if i_sr in [5,6,7,8,9]:
            ax.set_xlabel(r'$|k|$',fontsize=15)
    #
    plt.subplots_adjust(wspace=0.112, hspace=0.116, left=0.035, right=0.982, bottom=0.076, top=0.94)

    if 'figname' in kwargs.keys():
        plt.savefig(kwargs['figname'])
    if 'showfig' in kwargs.keys():
        if kwargs['showfig']:
            plt.show()


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

def get_fn(*args):
    """
    Get filename for set of parameters.
    """
    fn = ''
    for i,a in enumerate(args):
        t = type(a)
        if t in [str,]:
            fn += a
        elif t in [int, np.int64]:
            fn += str(a)
        elif t in [float, np.float32, np.float64]:
            fn += "{:.7f}".format(a)
        if not i==len(args)-1:
            fn +='_'
    return fn

######################################################################################################
#       QUENCH FUNCTIONS
######################################################################################################
def Gamma1(k_grid):
    """cos(kx)+cos(ky)"""
    return (np.cos(k_grid[:,:,0]) + np.cos(k_grid[:,:,1])).flatten()

def system_function(t,y,*args):
    """The function to feed scipy.integrate.solve_ivp"""
    S,hi,J1f,full_time_ramp,Lx,Ly,Gamma_1 = args
    Ns = Lx*Ly
    h = hi*(1-t/full_time_ramp)
    J1 = J1f*t/full_time_ramp
    J = (J1,0)
    D = (0,0)
    #Values of y
    th = y[0]
    ph = y[1]
    Gk = y[2:Ns+2]
    Hk = y[Ns+2:]
    #Parateres
    pxx1 = get_p_xx(th,ph,J,D)[0]
    pyy1 = get_p_yy(th,ph,J,D)[0]
    pzz1 = get_p_zz(th,ph,J,D)[0]
    pxy1 = get_p_xy(th,ph,J,D)[0]
    pxz1 = get_p_xz(th,ph,J,D)[0]
    pyz1 = get_p_yz(th,ph,J,D)[0]
    #Composite values
    eps = np.sum(Gk)/Ns
    dG_1 = np.sum(Gamma_1*Gk)/Ns
    dH_1 = np.sum(Gamma_1*Hk)/Ns
    #
    result = np.zeros(y.shape,dtype=complex)  #2N+2
    #Theta
    result[0] = -8*pyz1*eps - 4*(np.imag(dH_1*(pxz1-1j*pyz1))+pyz1*dG_1)
    #Phi
    if th != 0 and 0:
        result[1] = 8/np.sin(th)*pxz1*eps + 4/np.sin(th)*(np.real(dH_1*(pxz1-1j*pyz1))+pxz1*dG_1)
    else:
        result[1] = 0
    #Gk
    result[2:2+Ns] = -4*S*Gamma_1*((pxx1-pyy1)*np.imag(Hk)-2*pxy1*np.real(Hk))
    #Hk
    result[2+Ns:] = -4*1j*S*Gk*Gamma_1*(pxx1-pyy1+2*1j*pxy1) + 2*1j*Hk*(2*S*(2*pzz1-Gamma_1*(pxx1+pyy1))-h*np.cos(th))

    return result

def initial_condition(h_i,Lx,Ly):
    """Compute the initial condition for the z-staggered Néel state."""
    Ns = Lx*Ly
    res = np.zeros(2+2*Ns,dtype=complex)
    res[0] = np.pi/100
    res[1] = np.pi/7*0
    if 1:
        res[2:2+Ns] = h_i/2*np.ones(Ns)
    elif 0:
        res[2:2+Ns] = np.random.rand(Ns)/10
    return res

















