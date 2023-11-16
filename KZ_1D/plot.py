import numpy as np
import sys,getopt
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,})
import closed_functions as cfs
import open_functions as ofs
import parameters as ps
s_ = 15

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "N:",["ramp=","steps=","q="])
    #Default values
    ramp = 'exp'
    steps = 100
    quantity = 'fid'
except:
    print("Error in input parameters, sys.argv = ",argv)
    exit()
for opt, arg in opts:
    if opt=='--ramp':
        ramp = arg
        if ramp not in gn.list_ramps:
            print("Ramp type not accepted")
            exit()
    if opt=='--steps':
        steps = int(arg)
    if opt=='--q':
        quantity = arg

result_dir = '/home/dario/Desktop/git/Quenches/KZ_1D/Data/'
list_tau = [100, 200, 400, 1000]
list_gamma = [0., 0.001, 0.005, 0.01, 0.05, 0.1]

cols = ['r','g','y','b','k','m','orange','forestgreen']
style = ['-','--','-.',':',(0,(1,10)),(0,(1,8,3,8,5,8))]

if quantity=='fid':
    plt.figure(figsize=(15,7))
    X = np.linspace(0,1,steps)
    for t,tau in enumerate(list_tau):
        h_t,J_t,times = ps.find_parameters(tau,steps)
        dt = times[1]-times[0]
        for g,gamma in enumerate(list_gamma):
            if gamma == 0:
                filename = result_dir+cfs.names['fid']+cfs.pars_name(tau,dt)+'.npy'
            else:
                filename = result_dir+ofs.names['fid']+ofs.pars_name(tau,gamma,dt)+'.npy'
            try:
                fid = np.load(filename)
            except:
                continue
            plt.plot(X,fid,color=cols[t%len(cols)],ls=style[g%len(style)],label=r'$\gamma$='+"{:.4f}".format(list_gamma[g])+r', $\tau$='+str(tau))
    plt.xlim(-0.02,1.02)
    plt.ylabel(r'$|\langle\psi_{GS}|\psi(t)\rangle|^2$',size=s_)
    plt.xticks([0,1/2,1],['$t_i$','$t_{CP}$','$t_f$'],size=s_)
    plt.legend()
    plt.show()
elif quantity=='pop':
    plt.figure(figsize=(15,7))
    plt.subplots_adjust(right=0.85)
    for t,tau in enumerate(list_tau):
        h_t,J_t,times = ps.find_parameters(tau,steps)
        N = len(h_t)
        X = np.arange(0,N)
        dt = times[1]-times[0]
        for g,gamma in enumerate(list_gamma):
            if gamma == 0:
                filename = result_dir+cfs.names['pop']+cfs.pars_name(tau,dt)+'.npy'
            else:
                filename = result_dir+ofs.names['pop']+ofs.pars_name(tau,gamma,dt)+'.npy'
            try:
                pop = np.load(filename)
            except:
                continue
            plt.plot(X,pop,color=cols[t%len(cols)],ls=style[g%len(style)],label=r'$\gamma$='+"{:.4f}".format(gamma)+r', $\tau$='+str(tau))
    plt.ylabel(r'$n_q$',size=s_)
    plt.xlabel(r'$q$',size=s_)
    plt.legend(loc='upper left',bbox_to_anchor=(1,1))
    plt.show()











