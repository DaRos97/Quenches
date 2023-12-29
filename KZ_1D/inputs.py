import numpy as np

#Quantities to compute
quantities = ['time-evolution','fidelity','populations',
        #others just for closed case (for now)
#                'correlation_function_zz',
#                'correlation_function_xx',
#                'density_of_excitations',
                ]

#Names used in filenames
names = {   't-ev'  :   'time_evolved_',
            'fid'   :   'fidelity_',
            'pop'   :   'populations_',
            'CFzz'  :   'correlation_function_zz_',
        }
#Some named colors
cols = ['r','g','y','b','k','m','orange','forestgreen']
#Types of ramps
list_ramps = ['usd', 'exp']

def H_t(N_,h_,J_,t_):
    """Hamiltonian of free fermions with PBC.

    """
    H_ = np.zeros((N_,N_))
    for i in range(N_-1):
        H_[i,i+1] = J_[i][t_]/2
        H_[i+1,i] = J_[i][t_]/2
        H_[i,i] = h_[i][t_]*2
    H_[-1,-1] = h_[-1][t_]*2
    H_[-1,0] = H_[0,-1] = -J_[-1][t_]/2   #- for PBC
    return H_

def pars_name(tau,dt,gamma=0):
    """Parameters listed in string format for filename.

    """
    res = "{:.1f}".format(tau)
    if not gamma==0:
        res += "_"+"{:.5f}".format(gamma).replace('.',',')
    res += '_'+"{:.5f}".format(dt).replace('.',',')
    return res
