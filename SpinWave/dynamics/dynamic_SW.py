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
L = 20      #linear system size
h_i = 15*2*np.pi  #MHz      -> divide by 2*np.pi ?
J1_f = 20*2*np.pi #MHz
tau = 0.01#0.5   #us (micro-seconds) -> total ramp time
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
        method='RK45',
        t_eval=time_list,
#       first_step=1e-6, #1 ns
#       max_step=1e-4,
#       dense_output=True,
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

h_t = h_i*(1-time_list/tau)
J1_t = J1_f*time_list/tau
theta_t = fs.theta_canted_Neel(*(J1_t,0,0,0,h_t,S))
#Plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()
l1 = ax.plot(result.t,np.real(result.y[0]),'*b',label='theta')
l2 = ax.plot(result.t,np.real(result.y[1]),'^r',label='phi')
l3 = ax.plot(result.t,theta_t,'^y',label='theoretical theta')

ax_r = ax.twinx()
lr1 = ax_r.plot(result.t,np.real(np.sum(result.y[2:L**2+2],axis=0))/L**2,'og',label='epsilon')
ax_r.set_ylim(0,5)

labels = [l.get_label() for l in l1+l2+l3+lr1]
ax.legend(l1+l2+l3+lr1,labels)


plt.show()

