import numpy as np
import sys,getopt
import matplotlib.pyplot as plt
lt.rcParams.update({"text.usetex": True,})
import closed_functions as cfs
import open_functions as ofs
import parameters as ps
import general as gn
from scipy.optimize import curve_fit
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
                filename = result_dir+gn.names[quantity]+'wf_'+gn.pars_name(tau,dt)+'.npy'
            else:
                filename = result_dir+gn.names[quantity]+'DM_'+gn.pars_name(tau,dt,gamma)+'.npy'
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
                filename = result_dir+gn.names[quantity]+'wf_'+gn.pars_name(tau,dt)+'.npy'
            else:
                filename = result_dir+gn.names[quantity]+'DM_'+gn.pars_name(tau,dt,gamma)+'.npy'
            try:
                pop = np.load(filename)
            except:
                continue
            plt.plot(X,pop,color=cols[t%len(cols)],ls=style[g%len(style)],label=r'$\gamma$='+"{:.4f}".format(gamma)+r', $\tau$='+str(tau))
    plt.ylabel(r'$n_q$',size=s_)
    plt.xlabel(r'$q$',size=s_)
    plt.legend(loc='upper left',bbox_to_anchor=(1,1))
    plt.show()
elif quantity=='CFzz':
    fig, ax = plt.subplots()
    end_ = 0
    step = 1
    for n,tau in enumerate(list_tau):
        h_t,J_t,times = ps.find_parameters(tau,steps)
        N = len(h_t)
        X = np.arange(0,N)
        dt = times[1]-times[0]
        for g,gamma in enumerate(list_gamma):
            if gamma == 0:
                filename = result_dir+gn.names[quantity]+'wf_'+gn.pars_name(tau,dt)+'.npy'
            else:
                filename = result_dir+gn.names[quantity]+'DM_'+gn.pars_name(tau,dt,gamma)+'.npy'
            try:
                CFzz = np.load(filename)
            except:
                if input("Compute ",quantity," for tau=",tau," and gamma=",gamma," ?(y/N)")=='y':
                    args = (h_t,J_t,times,tau,gamma,ps.result_dir,save)
                    fun = cfs.compute_CF_zz if gamma==0 else ofs.compute_CF_zz
                    CFzz = fun(args)
                else:
                    continue
            ax.plot(np.arange(1,N-end_,step),CFzz[1:N-end_:step],'*',color=cols[n%len(cols)],ls=style[g%len(style)],label=r'$\gamma$='+"{:.4f}".format(gamma)+r', $\tau$='+str(tau))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r"$\langle S^z_iS^z_{i+r}\rangle-\langle S^z_i\rangle\langle S^z_{i+r}\rangle$",size=s_)
    ax.set_xlabel("$r$",size=s_)
    try:
        popt,pcov = curve_fit(gn.pow_law,np.arange(1,N-end_,step),CFzz[1:N-end_:step],p0=[1,-2])
        ax.plot(np.linspace(1,N-end_,1000),pow_law(np.linspace(1,N-end_,1000),*popt),'k-',label='a='+"{:.4f}".format(popt[1]))
    except:
        print("Error in fitting")
    plt.legend()











