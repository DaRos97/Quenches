import numpy as np
import functions as fs
import matplotlib.pyplot as plt
from pathlib import Path
import sys

use_time_evolved = True
#which correlator
correlator1_type = 'zz' if len(sys.argv)<2 else sys.argv[1]
correlator2_type = 'zz' if len(sys.argv)<3 else sys.argv[2]

#Parameters
N = 42          #chain sites
kx = np.fft.fftshift(np.fft.fftfreq(N,d=1))
full_time_ramp = 0.5
time_steps = 500        #of ramp
time_step = full_time_ramp/time_steps  #time step of ramp

full_time_measure = 0.8     #measure time in ms
Nt = 401        #time steps after ramp for the measurement
Nomega = 2000   #Number of frequency points in the Fourier transform
measure_time_list = np.linspace(0,full_time_measure,Nt)
omega_list = np.linspace(-250,250,Nomega)
stop_ratio_list = np.linspace(0.1,1,10)     #ratios of ramp where we stop and measure

print("Parameters of ramp: ")
print("Sites: ",N,"\nRamp time (ns): ",full_time_ramp*1000,"\nRamp time step (ns): ",time_step*1000)

"""Time evolution"""
args_fn = [(N,0),(full_time_ramp,5),(time_step,5)]
txt_wf = 'time-evolved' if use_time_evolved else 'GS-wf'
args_corr1_fn = args_fn + [(correlator1_type,0),(txt_wf,0),(full_time_measure,3),(Nt,0),(Nomega,0)]
correlator1_fn = fs.get_data_filename('correlator',args_corr1_fn,'.npy')
args_corr2_fn = args_fn + [(correlator2_type,0),(txt_wf,0),(full_time_measure,3),(Nt,0),(Nomega,0)]
correlator2_fn = fs.get_data_filename('correlator',args_corr2_fn,'.npy')

if Path(correlator1_fn).is_file():
    correlator1 = np.load(correlator1_fn)
if Path(correlator2_fn).is_file():
    correlator2 = np.load(correlator2_fn)

#

fig = plt.figure(figsize=(22, 15))
txt_title = 'time evolved wavefunction' if use_time_evolved else 'ground state wavefunction'
plt.suptitle("Correlator up: "+correlator1_type+", correlator down: "+correlator2_type)
for i_sr in range(5):
    label_cm = 'Magnitude of Fourier Transform' if i_sr==4 else ''
    stop_ratio = stop_ratio_list[i_sr]

    ax = fig.add_subplot(3,5,i_sr+1)
    pm = ax.pcolormesh(kx, omega_list, (np.abs(correlator1[i_sr]).T), shading='auto', cmap='magma')
    plt.colorbar(pm,label=label_cm)
    if i_sr != 0:
        ax.set_yticks([])
    ax.set_ylim(-50,50)

    ax = fig.add_subplot(3,5,i_sr+6)
    new_corr = np.abs((correlator1[i_sr].T)*(1j*omega_list[:,None])/(np.exp(1j*kx[None,:])-1) )**(1)
    pm = ax.pcolormesh(kx, omega_list, new_corr, shading='auto', cmap='magma')
    plt.colorbar(pm,label=label_cm)
    if i_sr != 0:
        ax.set_yticks([])
    ax.set_ylim(-50,50)

    ax = fig.add_subplot(3,5,i_sr+11)
    pm = ax.pcolormesh(kx, omega_list, (np.abs(correlator2[i_sr]).T), shading='auto', cmap='magma')
    plt.colorbar(pm,label=label_cm)
    if i_sr != 0:
        ax.set_yticks([])
    ax.set_ylim(-50,50)

plt.show()
