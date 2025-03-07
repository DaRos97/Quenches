import numpy as np
import Functions_dynamics as fs
from scipy.integrate import solve_ivp #as solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

save_res = True
compute_anyway = True

"""
Here we try to solve the system of diff equations to get the quench dynamics of XY+stag h model starting from z-Néel state.
Mainly to understand how the solve method works.
"""

S = 1/2
L = 10      #linear system size
h_i = 15  #MHz      -> divide by 2*np.pi ?
J1_f = 10 #MHz
tau = 0.5   #us (micro-seconds)
N_time = 500

time_list = np.linspace(0,tau,N_time)
k_grid = fs.BZgrid(L,L,1)       #last argument is UC size


res_fn = 'test.pkl'
if not Path(res_fn).is_file() or compute_anyway:
    """
    In our case we have 2N+2 equations: theta, phi and G,H for each k.
    Can be simplified maybe?
    """
    arguments = (S,h_i,J1_f,tau,L,fs.Gamma1(k_grid))
    result = solve_ivp(
        fun=fs.system_function,
        args=arguments,    #same arguments are passed to events and jac
        t_span=(0,tau),
        y0=fs.initial_condition(h_i,L),
        method='DOP853',
        t_eval=time_list,
#        first_step = 1e-8, #1 ns
    )
    print("Finished solution: ")
    print('fev: ',result.nfev)
    print('jev: ',result.njev)
    print('status: ',result.status)
    print('message: ',result.message)
    print('success: ',result.success)
    print("Size: ",result.y.shape)
    if save_res:    #save to file the result
        with open(res_fn,'wb') as f:
            pickle.dump(result,f)
else:   #load Bunch file
    with open(res_fn,'rb') as f:
        result = pickle.load(res_fn)

#Plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()
ax.plot(result.t,np.real(result.y[0]),'*b',label='theta')
ax.plot(result.t,np.real(result.y[1]),'^r',label='phi')
ax.plot(result.t,np.real(np.sum(result.y[2:L**2+2],axis=0))/L**2,'og',label='epsilon')


ax.legend()
plt.show()

