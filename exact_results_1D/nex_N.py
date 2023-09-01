import numpy as np
import matplotlib.pyplot as plt
import functions as fs
from scipy.optimize import curve_fit

s_ = 20
dt = 0.1
list_Tau = list(np.arange(5,60,step=5))
list_N = list(np.arange(10,80,step=10))
BC = "open"
type_of_quench = "real"
dirname = "Real_Quench/"
cols = ['r','g','y','b','k','m','forestgreen']
for N in list_N:
    nex = []
    for Tau in list_Tau:
        name_pars = str(N)+'_'+str(Tau)+'-'+str(dt)+'_'+BC
        nex_name = dirname + 'nex_'+name_pars+'.npy'
        nex.append(np.load(nex_name)/N)
    #fit
    popt, pcov = curve_fit(fs.pow_law,list_Tau,nex,p0=[1,-0.5])
    X = np.linspace(list_Tau[0],list_Tau[-1],100)
    plt.plot(X,fs.pow_law(X,*popt),color=cols[N%len(cols)])
    #
    plt.plot(list_Tau,nex,'*',color=cols[N%len(cols)],label='$N=$'+str(N)+': '+"{:4.3f}".format(popt[1]))
    plt.legend()
plt.xlabel(r'$\tau_Q$',size=s_)
plt.ylabel(r'$n_{ex}$',size=s_)
plt.show()
