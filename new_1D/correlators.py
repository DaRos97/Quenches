import numpy as np
import functions as fs
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pathlib import Path
import sys
import scipy

filling_txt = '1%6'     #Not correct yet
filling = int(filling_txt[0])/int(filling_txt[-1])

use_experimental_parameters = 1#False
txt_exp = 'expPars' if use_experimental_parameters else 'uniform'

save_time_evolved_data = 0#True
plot_fidelity = 0#True
plot_populations = 0#True
plot_energies = 0#True
savefig_fidelity = 0#True
savefig_populations = 0#True
savefig_energies = 0#True

use_time_evolved = True
#which correlator
correlator_type = 'zz' if len(sys.argv)<2 else sys.argv[1]
for i in correlator_type:
    if i not in ['z','e','j']:  #supported operators
        print("Not supported operator ",i,"\nAbort")
#correlator options
save_correlator_data = 0#True
plot_correlator = True
savefig_correlator = 0#True

#Parameters
full_time_ramp = 0.5 if len(sys.argv)<3 else float(sys.argv[2])/1000#ramp time in ms
time_steps = 500        #of ramp
time_step = full_time_ramp/time_steps  #time step of ramp
#
if use_experimental_parameters:
    initial_parameters_fn = 'exp_input/20250324_42Q_1D_StaggeredFrequency_0MHz_5.89_.p'
    final_parameters_fn = 'exp_input/20250324_42Q_1D_IntFrequency_10MHz_5.89_.p'
    g_in,h_in = fs.extract_experimental_parameters(initial_parameters_fn)
    g_fin,h_fin = fs.extract_experimental_parameters(final_parameters_fn)
    N = len(g_in)
    #
    if 0:
        h_in = np.ones(N)*15 #MHz
        for i in range(N):
            h_in[i] *= (-1)**i
        h_fin = np.zeros(N)
        txt_exp = 'expG_uniformH'
    if 0:
        g_in = np.zeros(N)
        g_fin = np.ones(N)*10 #MHz
        txt_exp = 'expH_uniformG'
else:
    N = 42          #chain sites
    g_in = np.zeros(N)
    g_fin = np.ones(N)*10 #MHz
    h_in = np.ones(N)*15 #MHz
    for i in range(N):
        h_in[i] *= (-1)**i
    h_fin = np.zeros(N)
g_t_i,h_t_i = fs.get_Hamiltonian_parameters(time_steps,g_in,g_fin,h_in,h_fin)   #parameters of Hamiltonian which depend on time
kx = np.fft.fftshift(np.fft.fftfreq(N,d=1))

full_time_measure = 0.8     #measure time in ms
Nt = 401        #time steps after ramp for the measurement
Nomega = 2000   #Number of frequency points in the Fourier transform
measure_time_list = np.linspace(0,full_time_measure,Nt)
omega_list = np.linspace(-250,250,Nomega)
stop_ratio_list = np.linspace(0.1,1,10)     #ratios of ramp where we stop and measure

print("Parameters of ramp: ")
print("Sites: ",N,"\nRamp time (ns): ",full_time_ramp*1000,"\nRamp time step (ns): ",time_step*1000)

"""Time evolution"""
args_fn = [(N,0),(full_time_ramp,5),(time_step,5),(txt_exp,0),(filling_txt,0)]
time_evolved_psi_fn = fs.get_data_filename("time_evolved_psi",args_fn,'.npy')
fidelity_fn = fs.get_data_filename("fidelity",args_fn,'.npy')
populations_fn = fs.get_data_filename("populations",args_fn,'.npy')
energies_fn = fs.get_data_filename("energies",args_fn,'.npy')
if not Path(time_evolved_psi_fn).is_file() and (use_time_evolved or plot_fidelity or plot_populations or plot_energies): #Time evolution and fidelity along the ramp
    time_evolved_psi, fidelity, populations, energies = fs.get_ramp_evolution(*(N,g_t_i,h_t_i,time_step,filling))
    if save_time_evolved_data:
        np.save(time_evolved_psi_fn,time_evolved_psi)
        np.save(fidelity_fn,fidelity)
        np.save(populations_fn,populations)
        np.save(energies_fn,energies)
else:
    time_evolved_psi = np.load(time_evolved_psi_fn)
    fidelity = np.load(fidelity_fn)
    populations = np.load(populations_fn)
    energies = np.load(energies_fn)

"""Correlator"""
txt_wf = 'time-evolved' if use_time_evolved else 'GS-wf'
args_corr_fn = args_fn + [(correlator_type,0),(txt_wf,0),(full_time_measure,3),(Nt,0),(Nomega,0)]
correlator_fn = fs.get_data_filename('correlator',args_corr_fn,'.npy')
correlator_spacetime_fn = fs.get_data_filename('correlator_spacetime',args_corr_fn,'.npy')
if not Path(correlator_fn).is_file():
    print("Computing correlation function ",correlator_type," with "+txt_wf+': ')
    correlator,correlator_st = fs.compute_correlator(correlator_type,*(N,omega_list,stop_ratio_list,measure_time_list,g_t_i,h_t_i,time_evolved_psi,use_time_evolved,filling))
    if save_correlator_data:
        np.save(correlator_fn,correlator)
        np.save(correlator_spacetime_fn,correlator_st)
else:
    correlator = np.load(correlator_fn)

#########################################################################################
#########################################################################################
"""Plots"""

if plot_correlator:
    #Plot
    fig = plt.figure(figsize=(17, 8))
    txt_title = 'time evolved wavefunction' if use_time_evolved else 'ground state wavefunction'
    plt.suptitle("Correlator "+correlator_type+", total ramp time: "+str(int(full_time_ramp*1000))+" ns, "+txt_title)
    for i_sr in range(len(stop_ratio_list)):
        stop_ratio = stop_ratio_list[i_sr]
        #Plot
        ax = fig.add_subplot(2,5,i_sr+1)
        pm = ax.pcolormesh(kx, omega_list, (np.abs(correlator[i_sr]).T)**(1), shading='auto', cmap='magma')
        if i_sr in [4,9]:
            label_cm = 'Magnitude of Fourier Transform'
        else:
            label_cm = ''
        plt.colorbar(pm,label=label_cm)
        ax.set_ylim(-50,50)
#        if i_sr>4:
#            ax.set_xlabel('Momentum ($k_x$)')
#        if i_sr in [0,5]:
#            ax.set_ylabel(r'Frequency $\omega$ (MHz)')
        ax.set_title('Stop ratio '+"{:.1f}".format(stop_ratio))
        if 0 and correlator_type=='zz': #plot fit       -> carefull when using experimental values of g and h
            time_steps = g_t_i.shape[0]
            i_t = int(time_steps*stop_ratio)-1
            en_m = np.sqrt(h_t[i_t]**2+4*g_t[i_t]**2*np.cos(kx*2*np.pi+np.pi/2)**2)+abs(h_t[i_t])
            en_M = 2*np.sqrt(h_t[i_t]**2+4*g_t[i_t]**2*np.cos(kx*np.pi+np.pi/2)**2)
            en_ex = en_m-2*abs(h_t[i_t])
#            ax.plot(kx,en_m,color='r')
            ax.plot(kx,-en_m,color='r')
#            ax.plot(kx,en_M,color='g')
            ax.plot(kx,-en_M,color='g')
#            ax.plot(kx,en_ex,color='b')
            ax.plot(kx,-en_ex,color='b')
    fig.tight_layout()
    if savefig_correlator:
        correlator_figfn = fs.get_data_filename("correlator",args_corr_fn,'.png')
        plt.savefig(correlator_figfn)

if plot_fidelity:   #Plot fidelity along the ramp and quenched values of h and g
    time_steps = g_t_i.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot()
    l1 = ax.plot(np.linspace(0,full_time_ramp*1000,time_steps),fidelity,color='r',label="fidelity")
    ax.set_ylabel("Fidelity wrt GS")
    ax.set_xlabel("time (ns)")
    ax_r = ax.twinx()
    l2 = ax_r.plot(np.linspace(0,full_time_ramp*1000,time_steps),g_t_i[:,0],color='g site 0',label="g")
    l3 = ax_r.plot(np.linspace(0,full_time_ramp*1000,time_steps),h_t_i[:,0],color='b',label="h site 0")
    ax_r.set_ylabel("coupling g and field h (MHz)")
    ax.set_title("Full ramp fidelity")
    ls = l1+l2+l3
    labels = [l.get_label() for l in ls]
    ax.legend(ls,labels)
    if savefig_fidelity:
        fidelity_figfn = fs.get_data_filename("fidelity",args_fn,'.png')
        plt.savefig(fidelity_figfn)

if plot_populations:
    time_steps = g_t_i.shape[0]
    E_GS,psi_GS = scipy.linalg.eigh(fs.compute_H(g_t_i[-1],h_t_i[-1],N,1))
    cmap = mpl.colormaps['plasma']
    colors = cmap(np.linspace(0,1,N//2))
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(N//2,N):#N//2-1):
        ax.plot(np.linspace(0,full_time_ramp*1000,time_steps),populations[:,i],color=colors[i-N//2])
    ax.set_ylabel('Modes occupations')
    ax.set_xlabel('time (ns)')
    norm = Normalize(vmin=E_GS[N//2], vmax=E_GS[-1])
    sm = ScalarMappable(cmap=cmap, norm=norm)  # Create a ScalarMappable for the colorbar
    sm.set_array([])  # The ScalarMappable needs an array, even if unused
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Energy of modes")
    if savefig_populations:
        populations_figfn = fs.get_data_filename("populations",args_fn,'.png')
        plt.savefig(populations_figfn)

if plot_energies:
    time_steps = g_t_i.shape[0]
    gs_energies = np.zeros(g_t_i.shape[0])
    for it in range(time_steps):
        E_GS = scipy.linalg.eigvalsh(fs.compute_H(g_t_i[it],h_t_i[it],N,1))
        gs_energies[it] = np.sum(E_GS[:N//2])/N
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(np.linspace(0,full_time_ramp*1000,time_steps),energies,color='r',label='time-evolved state energy')
#    ax.plot(np.linspace(0,full_time_ramp*1000,time_steps),gs_energies,color='r',label='time-evolved state energy')
    ax.set_ylabel('Energy')
    ax.set_xlabel('time (ns)')
    if savefig_energies:
        energies_figfn = fs.get_data_filename("energies",args_fn,'.png')
        plt.savefig(energies_figfn)

if plot_fidelity or plot_populations or plot_correlator or plot_energies:
    plt.show()

























