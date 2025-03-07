import numpy as np

def BZgrid(nkx,nky,UC):
    """Compute BZ coordinates of points."""
    d = 2*np.pi/nky
    gridk = np.zeros((nkx,nky,2))
    if UC == 2:
        for i1 in range(nkx):
            for i2 in range(nky):
                gx = -np.pi + d*(1+i1+i2//2)
                gy = d*((1+i2)//2-i1)
                gridk[i1,i2] = np.array([gx,gy]) #- np.array([0,d/2])
    elif UC == 1:
        for i1 in range(nkx):
            for i2 in range(nky):
                gridk[i1,i2,0] = -np.pi + d*(1+i1)
                gridk[i1,i2,1] = -np.pi + d*(1+i2)
    return gridk

def system_function(t,y,*args):
    """The function to feed scipy.integrate.solve_ivp"""
    S,hi,J1f,tau,L,Gamma_1 = args
    h = hi*(1-t/tau)
    J1 = J1f*t/tau + 1
    #Values of y
    th = y[0]
    ph = y[1]
    Gk = y[2:L**2+2]
    Hk = y[L**2+2:]
    #Composite values
    eps = np.sum(Gk)/L**2
    dG_1 = np.sum(Gamma_1*Gk)/L**2
    dH_1 = np.sum(Gamma_1*Hk)/L**2
    #
    result = np.zeros(y.shape,dtype=complex)  #2N+2
    #Theta
    result[0] = -8*p_yz_1(th,ph,J1)*eps - 4*(np.imag(dH_1*(p_xz_1(th,ph,J1)-1j*p_yz_1(th,ph,J1)))+p_yz_1(th,ph,J1)*dG_1)
    #Phi
    if th != 0:
        result[1] = 8/np.sin(th)*p_xz_1(th,ph,J1)*eps + 4/np.sin(th)*(np.real(dH_1*(p_xz_1(th,ph,J1)-1j*p_yz_1(th,ph,J1)))+p_xz_1(th,ph,J1)*dG_1)
    else:
        result[1] = 0
    #Gk
    result[2:2+L**2] = -4*S*Gamma_1*((p_xx_1(th,ph,J1)-p_yy_1(th,ph,J1))*np.imag(Hk)-2*p_xy_1(th,ph,J1)*np.real(Hk))
    #Hk
    result[2+L**2:] = -4*1j*S*Gk*Gamma_1*(p_xx_1(th,ph,J1)-p_yy_1(th,ph,J1)+2*1j*p_xy_1(th,ph,J1)) + 2*1j*Hk*(2*S*(2*p_zz_1(th,ph,J1)-Gamma_1*(p_xx_1(th,ph,J1)+p_yy_1(th,ph,J1)))-h*np.cos(th))
    return result

def Gamma1(k_grid):
    """cos(kx)+cos(ky)"""
    return (np.cos(k_grid[:,:,0]) + np.cos(k_grid[:,:,1])).flatten()

def p_xx_1(th,ph,J1):
    return -J1*(np.cos(th)**2*np.cos(ph)**2+np.sin(ph)**2)
def p_yy_1(th,ph,J1):
    return -J1*(np.cos(th)**2*np.sin(ph)**2+np.cos(ph)**2)
def p_zz_1(th,ph,J1):
    return -J1*np.sin(th)**2
def p_xy_1(th,ph,J1):
    return -J1/2*np.sin(th)**2*np.sin(2*ph)
def p_xz_1(th,ph,J1):
    return J1/2*np.sin(2*th)*np.cos(ph)
def p_yz_1(th,ph,J1):
    return -J1/2*np.sin(2*th)*np.sin(ph)

def initial_condition(h_i,L):
    res = np.zeros(2+2*L**2,dtype=complex)
    res[2:2+L**2] = h_i/2*np.ones(L**2)
    return res

