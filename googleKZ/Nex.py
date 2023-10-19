import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,})
import functions as fs
import parameters as ps

#
list_Tau = [6,12,24,100] #List of quench times -> in ns (total time of quench->exp one is 12ns)
list_Tau = np.arange(1,1000,100)
plot_ramp = False
save_data = True
s_ = 20 #fontsize
steps = 200
#
h_t,J_t,times_dic = ps.find_parameters(list_Tau,steps)
#
args = (h_t, J_t, times_dic, list_Tau, ps.datadir, save_data)

#
nex = fs.compute_nex(-1,args)       #first argument is the time at which nex is computed -> t_i=0, t_CP=-1
#Fit nex
list_Tau = np.array(list_Tau,dtype=float)
fun = fs.pow_law
try:
    popt, pcov = curve_fit(fun,list_Tau,nex,p0=[1,-0.5])
    X = np.linspace(list_Tau[0],list_Tau[-1],100)
    plt.plot(X,fun(X,*popt))
    print(popt)
#    plt.text(list_Tau[len(list_Tau)//2],nex[0]*3/4,'exponent='+"{:3.4f}".format(popt[1]),size=s_)
except:
    print("Fit nex not found")

for i in range(len(list_Tau)):
    plt.plot(list_Tau[i],nex[i],'*',color=fs.cols[i%len(fs.cols)])
plt.ylabel(r'$n_{ex}$',size=s_)
#    plt.yscale('log')
plt.xlabel(r'$\tau_Q$',size=s_)

plt.show()


