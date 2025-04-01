"""
Here we just extract the energies at the endo of the ramp for different ramp times.
"""
import numpy as np
import functions as fs
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import scipy

plot = True
save = True

use_experimental_parameters = 1#False
txt_exp = 'expPars' if use_experimental_parameters else 'uniform'

ramp_times = [30,40,50,60,80,100,200,300,500]
stop_ratio_list = [1,] #np.linspace(0.1,1,10)     #ratios of ramp where we stop and measure
energies = np.zeros(len(ramp_times))
for i in range(len(ramp_times)):
    full_time_ramp = ramp_times[i]/1000
    time_steps = 500        #of ramp
    time_step = full_time_ramp/time_steps  #time step of ramp
    print(full_time_ramp*1000)

    if use_experimental_parameters:
        initial_parameters_fn = 'exp_input/20250324_42Q_1D_StaggeredFrequency_0MHz_5.89_.p'
        final_parameters_fn = 'exp_input/20250324_42Q_1D_IntFrequency_10MHz_5.89_.p'
        g_in,h_in = fs.extract_experimental_parameters(initial_parameters_fn)
        g_fin,h_fin = fs.extract_experimental_parameters(final_parameters_fn)
        N = len(g_in)
    else:
        N = 42          #chain sites
        g_in = np.zeros(N)
        g_fin = np.ones(N)*10 #MHz
        h_in = np.ones(N)*15 #MHz
        for i in range(N):
            h_in[i] *= (-1)**i
        h_fin = np.zeros(N)
    g_t_i,h_t_i = fs.get_Hamiltonian_parameters(time_steps,g_in,g_fin,h_in,h_fin)   #parameters of Hamiltonian which depend on time

    args_fn = [(N,0),(full_time_ramp,5),(time_step,5),(txt_exp,0)]
    energies_fn = fs.get_data_filename("energies",args_fn,'.npy')
    if Path(energies_fn).is_file():
        ens = np.load(energies_fn)
    else:
        print("ramp time ",full_time_ramp," ns missing")
        exit()

    energies[i] = ens[-1]/np.sum(g_fin)*N

if plot:
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(ramp_times,energies,marker='*',color='b')
    ax.set_xlabel("Ramp time (ns)")
    ax.set_title("Bond average energy <XX+YY>/2")
    plt.show()

if save:
    np.save('data/energy_jeronimo.npy',energies)
