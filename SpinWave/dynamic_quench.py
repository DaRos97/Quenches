import numpy as np
import functions as fs
from scipy.integrate import solve_ivp #as solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys

save_res = False
compute_anyway = True

"""
Here we try to solve the system of diff equations to get the quench dynamics of XY+stag h model starting from z-Néel state.
Mainly to understand how the solve method works.
"""

S = 1/2
Lx = 20      #linear system size
Ly = 20
Ns = Lx*Ly
h_i = 30  #MHz
J1_f = 40 #MHz

full_time_ramp = 0.01 if len(sys.argv)<3 else float(sys.argv[2])/1000    #ramp time in ms
time_steps = 20        #of ramp
time_step = full_time_ramp/time_steps  #time step of ramp

time_list = np.linspace(0,full_time_ramp,time_steps)
k_grid = fs.BZgrid(Lx,Ly)       #last argument is UC size

res_fn = 'test.pkl'
if not Path(res_fn).is_file() or compute_anyway:
    """
    In our case we have 2Ns+2 equations: theta, phi and G,H for each k.
    Can be simplified maybe?
    """
    arguments = (S,h_i,J1_f,full_time_ramp,Lx,Ly,fs.Gamma1(k_grid))
    result = solve_ivp(
        fun=fs.system_function,
        args=arguments,    #same arguments are passed to events and jac
        t_span=(0,full_time_ramp),
        y0=fs.initial_condition(h_i,Lx,Ly),
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

theta_t = np.zeros(time_steps)
for it in range(time_steps):
    h_t = h_i*(1-time_list[it]/full_time_ramp)
    J1_t = J1_f*time_list[it]/full_time_ramp
    theta_t[it] = fs.get_angles(S,(J1_t,0),(0,0),h_t)[0]
#Plot
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()
l1 = ax.plot(result.t,np.real(result.y[0]),'*b',label='theta')
l2 = ax.plot(result.t,np.real(result.y[1]),'^r',label='phi')
l3 = ax.plot(result.t,theta_t,'^y',label='theoretical theta')

ax_r = ax.twinx()
lr1 = ax_r.plot(result.t,np.real(np.sum(result.y[2:Ns+2],axis=0))/Ns,'og',label='epsilon')
ax_r.set_ylim(0,5)

labels = [l.get_label() for l in l1+l2+l3+lr1]
ax.legend(l1+l2+l3+lr1,labels)


plt.show()

