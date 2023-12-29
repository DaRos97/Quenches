import numpy as np
import pickle 
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline as RBS
from pathlib import Path
import h5py,os

cluster = False if os.getcwd()[6:11]=='dario' else True

home_dn = '/home/users/r/rossid/KZ_1D/' if cluster else'/home/dario/Desktop/git/Quenches/KZ_1D/'
result_dn = home_dn + 'results/'
exp_dn = home_dn + 'experimental_Data/'

hdf5_fn = exp_dn + 'experimental_data.hdf5'

z_dsname = 'z_data_suggested'
z_labels_dsname = 'z_labels'
xx_dsname = 'xx_data_suggested'
xx_labels_dsname = 'xx_labels'

if Path(hdf5_fn).is_file():
    with h5py.File(hdf5_fn,'r') as f:
        z_data = np.copy(f[z_dsname])
        z_labels = f[z_labels_dsname]
        xx_data = np.copy(f[xx_dsname])
        xx_labels = f[xx_labels_dsname]
else:
    print("Extracting parameters of experimental ramp...")
    params_fn = exp_dn + 'params_Dario'
    f_c_fn = exp_dn + 't_to_f_coupler'
    f_q_fn = exp_dn + 't_to_f_qubit'
    with open(params_fn, 'rb') as f:
        exp = pickle.load(f)
    #Change key names 
    st_keys = list(exp.keys())
    len_f = int(np.sqrt(len(st_keys))) #number of values of f_c/f_q for which a parameter ('xx', 'z', ecc..) is given in the data
    data = {}
    for i in range(len_f): #f_coupler
        for j in range(len_f): #f_quibit
            data[i*len_f+j] = exp[st_keys[i*len_f+j]]
    #
    with open(f_c_fn, 'rb') as f:
        tfc = pickle.load(f)
    with open(f_q_fn, 'rb') as f:
        tfq = pickle.load(f)
    n_tfc = len(tfc.keys())     #number of times for which f_c is given in the data
    n_tfq = len(tfq.keys())
    tcs = np.linspace(0,6,n_tfc,endpoint=True)
    tqs = np.linspace(0,6,n_tfq,endpoint=True)
    #Interpolate fc, fq as a function of time
    fc = np.array(list(tfc.values()))
    fq = np.array(list(tfq.values()))
    #
    fun_fc = interp1d(tcs,fc)
    fun_fq = interp1d(tqs,fq)
    #
    def find_z_f(point,*args):
        data, len_f = args
        z_f_data = np.zeros((len_f,len_f))
        for i in range(len_f):
            for j in range(len_f):
                z_f_data[i,j] = data[i*len_f+j]['z'][point]
        f_c_values = np.linspace(0,1,len_f,endpoint=True)
        f_q_values = np.linspace(0,1,len_f,endpoint=True)
        fun_z_f = RBS(f_c_values,f_q_values,z_f_data)
        return fun_z_f
    def find_xx_f(point,*args):
        data, len_f = args
        xx_f_data = np.zeros((len_f,len_f))
        for i in range(len_f):
            for j in range(len_f):
                xx_f_data[i,j] = data[i*len_f+j]['xx'][point]
        f_c_values = np.linspace(0,1,len_f,endpoint=True)
        f_q_values = np.linspace(0,1,len_f,endpoint=True)
        fun_xx_f = RBS(f_c_values,f_q_values,xx_f_data)
        return fun_xx_f
    #
    args = (data,len_f)
    N = len(data[0]['z'].keys())
    z_data = np.zeros((N,n_tfc))
    z_labels = list(data[0]['z'].keys())
    xx_data = np.zeros((N,n_tfc))
    xx_labels = list(data[0]['xx'].keys())
    for i in range(N):
        if 0:   #use first value of z (actually first 2 since it is staggered) and xx for all sites -> translational invariant
            fun_z_f = find_z_f(z_labels[int(0.5*(1+(-1)**i))],*args)
            fun_xx_f = find_xx_f(xx_labels[0],*args)
        else:
            fun_z_f = find_z_f(z_labels[i],*args)#int(0.5*(1+(-1)**i))],*args)
            fun_xx_f = find_xx_f(xx_labels[i],*args)
        for t in range(n_tfc):
            xx_data[i,t] = fun_xx_f(fun_fc(tcs[t]),fun_fq(tqs[t]))/2   #1/2 since j_xx = xx/2
            z_data[i,t] = fun_z_f(fun_fc(tcs[t]),fun_fq(tqs[t]))
    #Save for future use
    with h5py.File(hdf5_fn,'w') as f:
        f.create_dataset(z_dsname,data=z_data)
        f.create_dataset(z_labels_dsname,data=z_labels)
        f.create_dataset(xx_dsname,data=xx_data)
        f.create_dataset(xx_labels_dsname,data=xx_labels)

def find_parameters(tau,steps,plot=False):
    N, steps_0 = z_data.shape   #steps_0 is the number of time steps in the datafile of f_c/f_cq, times 2
    #Fit to get array of functions
    ttt = np.linspace(0,6,steps_0)
    times_0 = np.linspace(0,6,steps)   #steps is the number of steps I want to compute in each ramp
    #For each site interpolate z and xx in time and compute it in the number of steps we want
    h_t = []
    J_t = []
    for i in range(N):
        h_t.append(interp1d(ttt,z_data[i])(times_0))
        J_t.append(interp1d(ttt,xx_data[i])(times_0))
    if plot: #plot ramp profiles
        import matplotlib.pyplot as plt
        for n in range(len(list_Tau)):
            if 0:   #plot h
                for i in range(0,N,2):
                    plt.plot(times_dic[list_Tau[n]],h_t[i],'r',label=str(list_Tau[n]))
                    plt.plot(times_dic[list_Tau[n]],h_t[i+1],'b',label=str(list_Tau[n]))
            else:   #plot xx
                for i in range(N):
                    plt.plot(times_dic[list_Tau[n]],J_t[i],'r',label=str(list_Tau[n]))
#        plt.legend()
        plt.show()
        exit()
    #
    return h_t, J_t







