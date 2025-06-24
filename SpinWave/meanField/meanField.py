import numpy as np
import scipy
import sys
import meanField_functions as fs

if len(sys.argv) not in [3,4]:
    print("Usage: py static_ZZ_r.py arg1 arg2 arg3(optional=1)")
    print("\targ1,arg2: Lx and Ly (int)")
    print("\targ3: stop ratio (float, 1 for XX model)")
    exit()
else:
    Lx = int(sys.argv[1])
    Ly = int(sys.argv[2])
    Ns = Lx*Ly
    if len(sys.argv)==4:
        stop_ratio = float(sys.argv[3])
    else:
        stop_ratio = 1

plot_wf = True

S = 0.5     #spin value
full_time_ramp = 0.5  #ramp time in ms
time_steps = 500        #of ramp
time_step = full_time_ramp/time_steps  #time step of ramp
g_val = 20      #factor of 2 from experiment due to s^xs^x -> s^+s^-
h_val = 15
g1_t_i,g2_t_i,d1_t_i,h_t_i = fs.get_parameters(Lx,Ly,time_steps,g_val,h_val)   #parameters of Hamiltonian which depend on time
indt = int(time_steps*stop_ratio)
if indt==time_steps:    indt -= 1
J_i = (g1_t_i[indt,:,:],np.zeros((Ns,Ns)))  #site-dependent hopping
D_i = (d1_t_i[indt,:,:],np.zeros((Ns,Ns)))
h_i = h_t_i[indt,:,:]
theta,phi = fs.get_angles(S,J_i,D_i,h_i)
ts = fs.get_ts(theta,phi)       #All t-parameters for A and B sublattice
parameters = (S,Lx,Ly,h_i,ts,theta,phi,J_i,D_i)

" Real space Hamiltonian in XX limit (h=0)"
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
    print("Negative or null eigenvalue in Hamiltonian: ",scipy.linalg.eigvalsh(A-B)[0])
    K = scipy.linalg.cholesky(A-B+np.identity(Ns)*1e-5)

lam2,chi_ = scipy.linalg.eigh(K@(A+B)@K.T.conj())
if theta!=0 and 0:   # When theta!=0 -> gapless spectrum -> do not consider the 0 energy eigenmode and eigenstate
    lam2[0] = 1
eps = np.sqrt(lam2)         #dispersion -> positive
#
chi = chi_ / eps**(1/2)     #normalized eigenvectors
phi_ = K.T.conj()@chi
psi_ = (A+B)@phi_/eps
U_ = 1/2*(phi_+psi_)
V_ = 1/2*(phi_-psi_)
if theta!=0 and 0:        # Remove the 0-energy eigenvalue
    U_[:,0] *= 0
    V_[:,0] *= 0


if plot_wf:
    import matplotlib.pyplot as plt
    J_inv = np.zeros((2*Ns,2*Ns),dtype=complex)
    J_inv[:Ns,:Ns] = U_
    J_inv[:Ns,Ns:] = V_
    J_inv[Ns:,:Ns] = V_
    J_inv[Ns:,Ns:] = U_
    #
    fig = plt.figure(figsize=(25,10))
    for ik in range(10):
        k = Lx*Ly-1-ik%5
        U = J_inv[:Ns,k].reshape(Lx,Ly)
        V = J_inv[Ns:,k].reshape(Lx,Ly)
        ax = fig.add_subplot(2,5,ik+1,projection='3d')
        X,Y = np.meshgrid(np.arange(Lx)+1,np.arange(Ly)+1)
        if ik<5:
            ax.plot_surface(X,Y,np.real(U).T,cmap='plasma')
        else:
            ax.plot_surface(X,Y,np.real(V).T,cmap='plasma')
        if 0:
            ax = fig.add_subplot(2,5,ik+6,projection='3d')
            sin = np.sin(np.pi*X*k/(Lx+1))*np.sin(np.pi*Y*k/(Ly+1))
            ax.plot_surface(X,Y,sin,cmap='plasma')
        title = "U" if ik<5 else "V"
        title += ", mode k=%d"%(ik%5)
        ax.set_title(title)
    fig.tight_layout()
    plt.show()
