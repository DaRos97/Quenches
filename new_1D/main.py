import numpy as np
import functions as fs
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pathlib import Path
import sys

save_time_evolved_data = True
plot_fidelity = False
plot_populations = False
savefig_fidelity = False
savefig_populations = False

use_time_evolved = True
save_zz_correlator_data = True
plot_zz_correlator = True
savefig_zz_correlator = False

#Parameters
g_ = 10 #MHz
h_ = 15 #MHz
full_time_ramp = 0.5 if len(sys.argv)<2 else int(sys.argv[1])/1000#ramp time in ms
full_time_measure = 0.8     #measure time in ms
time_step = full_time_ramp/500  #time step of ramp

N = 42          #chain sites
Nt = 401        #time steps after ramp for the measurement
Nomega = 2000   #Number of frequency points in the Fourier transform
measure_time_list = np.linspace(0,full_time_measure,Nt)
omega_list = np.linspace(-250,250,Nomega)
stop_ratio_list = np.linspace(0.1,1,10)     #ratios of ramp where we stop and measure

print("Parameters of ramp: ")
print("Sites: ",N,"\nRamp time (ns): ",full_time_ramp*1000,"\nRamp time step (ns): ",time_step*1000)

g_t,h_t = fs.get_Hamiltonian_parameters(int(full_time_ramp//time_step),g_,h_)   #parameters of Hamiltonian which depend on time

"""Time evolution"""
args_fn = [(N,0),(full_time_ramp,3),(time_step,5)]
time_evolved_psi_fn = fs.get_data_filename("time_evolved_psi",args_fn,'.npy')
fidelity_fn = fs.get_data_filename("fidelity",args_fn,'.npy')
populations_fn = fs.get_data_filename("populations",args_fn,'.npy')
if not Path(time_evolved_psi_fn).is_file() and (use_time_evolved or plot_fidelity or plot_populations): #Time evolution and fidelity along the ramp
    time_evolved_psi, fidelity, populations = fs.get_ramp_evolution(*(N,g_t,h_t,time_step))
    if save_time_evolved_data:
        np.save(time_evolved_psi_fn,time_evolved_psi)
        np.save(fidelity_fn,fidelity)
        np.save(populations_fn,populations)
else:
    time_evolved_psi = np.load(time_evolved_psi_fn)
    fidelity = np.load(fidelity_fn)
    populations = np.load(populations_fn)

"""ZZ correlator"""
txt_wf = 'time-evolved' if use_time_evolved else 'GS-wf'
args_corr_fn = args_fn + [(txt_wf,0),(full_time_measure,3),(Nt,0),(Nomega,0)]
zz_correlator_fn = fs.get_data_filename('zz_correlator',args_corr_fn,'.npy')
if not Path(zz_correlator_fn).is_file():
    print("Computing correlation function with "+txt_wf+': ')
    zz_correlator = fs.compute_zz_correlator(*(N,omega_list,stop_ratio_list,measure_time_list,g_t,h_t,time_evolved_psi,use_time_evolved))
    if save_zz_correlator_data:
        np.save(zz_correlator_fn,zz_correlator)
else:
    zz_correlator = np.load(zz_correlator_fn)

#########################################################################################
#########################################################################################
"""Plots"""

if plot_zz_correlator:
    #Plot
    fig = plt.figure(figsize=(17, 8))
    txt_title = 'time evolved wavefunction' if use_time_evolved else 'ground state wavefunction'
    plt.suptitle("Total ramp time: "+str(int(full_time_ramp*1000))+" ns, "+txt_title)
    kx = np.fft.fftshift(np.fft.fftfreq(N,d=1))
    for i_sr in range(len(stop_ratio_list)):
        stop_ratio = stop_ratio_list[i_sr]
        #Plot
        ax = fig.add_subplot(2,5,i_sr+1)
    #    ax = fig.add_subplot()
        pm = ax.pcolormesh(kx, omega_list, np.abs(zz_correlator[i_sr]).T, shading='auto', cmap='magma')
        if i_sr in [4,9]:
            label_cm = 'Magnitude of Fourier Transform'
        else:
            label_cm = ''
        plt.colorbar(pm,label=label_cm)
        ax.set_ylim(-50,50)
        if i_sr>4:
            ax.set_xlabel('Momentum ($k_x$)')
        if i_sr in [0,5]:
            ax.set_ylabel('Frequency $\omega$ (MHz)')
        ax.set_title('Stop ratio '+"{:.1f}".format(stop_ratio))
        if 0: #plot fit
            i_t = int(time_steps*stop_ratio)-1
            en_m = np.sqrt(h_t[i_t]**2+4*g_t[i_t]**2*np.cos(kx*2*np.pi+np.pi/2)**2)+abs(h[i_t])
            en_M = 2*np.sqrt(h_t[i_t]**2+4*g_t[i_t]**2*np.cos(kx*np.pi+np.pi/2)**2)
            en_ex = en_m-2*abs(h[i_t])
            ax.plot(kx,en_m,color='r')
            ax.plot(kx,-en_m,color='r')
            ax.plot(kx,en_M,color='g')
            ax.plot(kx,-en_M,color='g')
            ax.plot(kx,en_ex,color='b')
    fig.tight_layout()
    if savefig_zz_correlator:
        plt.savefig("figures/"+corr_fn[5:-4]+'.png')

if plot_fidelity:   #Plot fidelity along the ramp and quenched values of h and g
    fig = plt.figure()
    ax = fig.add_subplot()
    l1 = ax.plot(np.linspace(0,ramp_time,time_steps),fidelity,color='r',label="fidelity")
    ax.set_ylabel("Fidelity wrt GS")
    ax.set_xlabel("time (ms)")
    ax_r = ax.twinx()
    l2 = ax_r.plot(np.linspace(0,ramp_time,time_steps),g,color='g',label="g")
    l3 = ax_r.plot(np.linspace(0,ramp_time,time_steps),h,color='b',label="h")
    ax_r.set_ylabel("coupling g and field h (MHz)")
    ax.set_title("Full ramp fidelity")
    ls = l1+l2+l3
    labels = [l.get_label() for l in ls]
    ax.legend(ls,labels)
    if savefig_fidelity:
        plt.savefig("figures/ramp_fidelity_"+str(N)+'_'+"{:.3f}".format(full_time_ramp)+'.png')

if plot_populations:
    E_GS,psi_GS = scipy.linalg.eigh(compute_H(g[-1],h[1],N))
    cmap = mpl.colormaps['plasma']
    colors = cmap(np.linspace(0,1,N))#N//2-1))
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(N//2,N):#N//2-1):
        ax.plot(np.linspace(0,ramp_time,time_steps),populations[:,i],color=colors[i-N//2])
    ax.set_ylabel('Modes occupations')
    ax.set_xlabel('time (ms)')
    norm = Normalize(vmin=E_GS[N//2], vmax=E_GS[N-1])
    sm = ScalarMappable(cmap=cmap, norm=norm)  # Create a ScalarMappable for the colorbar
    sm.set_array([])  # The ScalarMappable needs an array, even if unused
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Energy of modes")
    if savefig_populations:
        plt.savefig("figures/"+pop_fn[5:-4]+'.png')

if plot_fidelity or plot_populations or plot_zz_correlator:
    plt.show()

























