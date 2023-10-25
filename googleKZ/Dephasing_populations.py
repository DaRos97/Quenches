import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,})
import dephasing_functions as fs
import parameters as ps

#
list_Tau = [100,400,1000] #List of quench times -> in ns (total time of quench->exp one is 12ns)
save_data = False
s_ = 20 #fontsize
steps = 20
#
h_t,J_t,times_dic = ps.find_parameters(list_Tau,steps)
N = len(h_t)
#
list_gamma = [0,]#np.linspace(0,1,11)
pop_ = []
for gamma in list_gamma:
    args = (h_t,J_t,gamma, times_dic, list_Tau, ps.homedir, save_data)
    pop_.append(fs.compute_populations(args))
#
list_Tau = np.array(list_Tau,dtype=float)
X = np.arange(0,N)
for i in range(len(list_gamma)):
    for t,Tau in enumerate(list_Tau):
        plt.plot(X,pop_[i][t],color=fs.cols[t%len(fs.cols)],label=r'$\gamma$='+"{:.4f}".format(list_gamma[i])+r', $\tau$='+str(Tau))
plt.ylabel(r'$n_q$',size=s_)
plt.xlabel(r'$q$',size=s_)
plt.legend()
plt.show()


