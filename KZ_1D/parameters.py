import numpy as np
import pickle 
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline as RBS

homedir = '/home/dario/Desktop/git/Quenches/KZ_1D/'
result_dir = homedir+'Data/'
datadir = homedir + 'extracted_exp_Data/'

z_name = datadir+'z_data_suggested.npy'
z_labels_name = datadir + 'z_labels.txt'
xx_name = datadir+'xx_data_suggested.npy'
xx_labels_name = datadir + 'xx_labels.txt'
try:
    z_data = np.load(z_name)
    with open(z_labels_name,'r') as f:
        zz = f.read().split(';')
    z_labels = zz[:-1]
    xx_data = np.load(xx_name)
    with open(xx_labels_name,'r') as f:
        xx = f.read().split(';')
    xx_labels = xx[:-1]
except:
    print("Extracting parameters of experimental ramp...")
    exp_dirname = homedir + 'experimental_Data/'
    params_dataname = exp_dirname + 'params_Dario'
    f_c_dataname = exp_dirname + 't_to_f_coupler'
    f_q_dataname = exp_dirname + 't_to_f_qubit'

    with open(params_dataname, 'rb') as f:
        exp = pickle.load(f)
    #Change key names 
    st_keys = list(exp.keys())
    len_f = int(np.sqrt(len(st_keys))) #number of values of f_c/f_q for which a parameter ('xx', 'z', ecc..) is given in the data
    data = {}
    for i in range(len_f): #f_coupler
        for j in range(len_f): #f_quibit
            data[i*len_f+j] = exp[st_keys[i*len_f+j]]
    #
    with open(f_c_dataname, 'rb') as f:
        tfc = pickle.load(f)
    with open(f_q_dataname, 'rb') as f:
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
    xx_data = np.zeros((N,n_tfc))
    xx_labels = list(data[0]['xx'].keys())
    z_data = np.zeros((N,n_tfc))
    z_labels = list(data[0]['z'].keys())
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
    np.save(z_name,z_data)
    np.save(xx_name,xx_data)
    with open(z_labels_name, 'w') as f:
        for i in range(len(z_labels)):
            f.write(str(z_labels[i])+';')
    with open(xx_labels_name, 'w') as f:
        for i in range(len(xx_labels)):
            f.write(str(xx_labels[i])+';')
    #

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
    #Compute times of each quench
    times = np.linspace(0,tau,steps)
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
    return h_t, J_t, times







