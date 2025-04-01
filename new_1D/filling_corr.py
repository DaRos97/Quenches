"""
Here w plot correlators at the end of the ramp for diffeent fillings.
"""
import numpy as np
import functions as fs
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pathlib import Path
import sys


use_experimental_parameters = 1#False
txt_exp = 'expPars' if use_experimental_parameters else 'uniform'
use_time_evolved = True
#which correlator
correlator_type = 'zz' if len(sys.argv)<2 else sys.argv[1]
for i in correlator_type:
    if i not in ['z','e','j']:  #supported operators
        print("Not supported operator ",i,"\nAbort")
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
#
kx = np.fft.fftshift(np.fft.fftfreq(N,d=1))
full_time_measure = 0.8     #measure time in ms
Nt = 401        #time steps after ramp for the measurement
Nomega = 2000   #Number of frequency points in the Fourier transform
measure_time_list = np.linspace(0,full_time_measure,Nt)
omega_list = np.linspace(-250,250,Nomega)
#
corrs = []
list_fillings = ['1%6','2%6','3%6','4%6','5%6']
for filling_txt in list_fillings:
    filling = int(filling_txt[0])/int(filling_txt[-1])
    """Time evolution"""
    args_fn = [(N,0),(full_time_ramp,5),(time_step,5),(txt_exp,0),(filling_txt,0)]
    txt_wf = 'time-evolved' if use_time_evolved else 'GS-wf'
    args_corr_fn = args_fn + [(correlator_type,0),(txt_wf,0),(full_time_measure,3),(Nt,0),(Nomega,0)]
    correlator_fn = fs.get_data_filename('correlator',args_corr_fn,'.npy')
    corrs.append(np.load(correlator_fn)[-1])

#Plot
fig = plt.figure(figsize=(17, 4))
txt_title = 'time evolved wavefunction' if use_time_evolved else 'ground state wavefunction'
plt.suptitle("Correlator "+correlator_type+", total ramp time: "+str(int(full_time_ramp*1000))+" ns, "+txt_title)
for i_f in range(len(list_fillings)):
    #Plot
    ax = fig.add_subplot(1,5,i_f+1)
    pm = ax.pcolormesh(kx, omega_list, np.abs(corrs[i_f]).T, shading='auto', cmap='magma')
    if i_f == 4:
        label_cm = 'Magnitude of Fourier Transform'
    else:
        label_cm = ''
    plt.colorbar(pm,label=label_cm)
    ax.set_ylim(-50,50)
    ax.set_title('Filling '+list_fillings[i_f])
fig.tight_layout()
plt.show()
