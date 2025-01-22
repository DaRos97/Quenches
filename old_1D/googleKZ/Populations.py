import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,})
import functions as fs
import parameters as ps

#
list_Tau = [100,400,1000] #List of quench times -> in ns (total time of quench->exp one is 12ns)
save_data = 0#True
s_ = 20 #fontsize
steps = 20
#
h_t,J_t,times_dic = ps.find_parameters(list_Tau,steps)
N = len(h_t)
#
args = (h_t, J_t, times_dic, list_Tau, ps.homedir, save_data)

#
n_q = fs.compute_populations(args)
list_Tau = np.array(list_Tau,dtype=float)
X = np.arange(0,N)
for i in range(len(list_Tau)):
    Tau = list_Tau[i]
    plt.plot(X,n_q[i],color=fs.cols[i%len(fs.cols)],label=r"$\tau_Q=$"+str(Tau))
plt.ylabel(r'$n_q$',size=s_)
plt.xlabel(r'$q$',size=s_)
plt.legend()
plt.show()


