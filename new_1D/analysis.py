import numpy as np
import functions as fs
import matplotlib.pyplot as plt
from pathlib import Path
import sys

if 0:
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

if 1:
    """Plot end of ramp z-e correlator for t-ev/GS and uniform/exp Hamiltonian parameters"""
    use_experimental_parameters_ = [False,True]
    use_time_evolved_ = [False,True]

    correlator_type = 'ez'
    full_time_ramp = 0.5
    filling_txt = '3%6'
    #Filling
    filling = int(filling_txt[0])/int(filling_txt[-1])
    #Parameters
    time_steps = 500        #of ramp
    time_step = full_time_ramp/time_steps  #time step of ramp
    full_time_measure = 0.8     #measure time in ms
    Nt = 401        #time steps after ramp for the measurement
    Nomega = 2000   #Number of frequency points in the Fourier transform
    measure_time_list = np.linspace(0,full_time_measure,Nt)
    omega_list = np.linspace(-250,250,Nomega)
    #
    data = []
    for i in range(4):
        use_experimental_parameters = use_experimental_parameters_[i%2]
        use_time_evolved = use_time_evolved_[i//2]
        #Experimental (disordered) or uniform parameters
        txt_exp = 'expPars' if use_experimental_parameters else 'uniform'
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

        """Time evolution"""
        args_fn = [(N,0),(full_time_ramp,5),(time_step,5),(txt_exp,0),(filling_txt,0)]
        """Correlator"""
        txt_wf = 'time-evolved' if use_time_evolved else 'GS-wf'
        args_corr_fn = args_fn + [(correlator_type,0),(txt_wf,0),(full_time_measure,3),(Nt,0),(Nomega,0)]
        correlator_fn = fs.get_data_filename('correlator',args_corr_fn,'.npy')
        data.append(np.load(correlator_fn)[-1])

    fig = plt.figure(figsize=(20, 20))
    plt.suptitle("ze commutator at stop ratio 1",size=30)
    s_ = 30
    for i in range(4):
        ax = fig.add_subplot(2,2,i+1)
        if i==0:
            ax.set_title("Uniform g and h",size=s_)
            ax.set_ylabel("Groud state WF",size=s_)
        if i==1:
            ax.set_title("Experimental g and h",size=s_)
        if i==2:
            ax.set_ylabel("Time-Evolved WF",size=s_)
        pm = ax.pcolormesh(kx, omega_list, np.abs(data[i]).T, shading='auto', cmap='magma')
        plt.colorbar(pm)
        if i in [1,3]:
            ax.set_yticks([])
        ax.set_ylim(-50,50)


    plt.show()
