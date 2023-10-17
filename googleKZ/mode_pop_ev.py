import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,})
import functions as fs
import parameters as ps

#
list_Tau = [0.5,1,6,12,24,50] #List of quench times -> in ns (total time of quench->exp one is 12ns)
plot_ramp = False
save_data = True
s_ = 20 #fontsize
steps = 400
#
h_t,J_t,times_dic = ps.find_parameters(list_Tau,steps)
N = len(h_t)
#
args = (h_t, J_t, times_dic, list_Tau, ps.datadir, save_data)
cols = ['r','g','y','b','k','m','orange','forestgreen']

#
rg = N//2 #number of modes from center --> all modes are N//2
n_q = fs.compute_pop_ev(args,rg)
list_Tau = np.array(list_Tau,dtype=float)
X = np.linspace(0,1,steps)
for i in range(len(list_Tau)):
    plt.subplot(2,3,i+1)
    Tau = list_Tau[i]
    for k in range(13,16):#,N):#-rg,N//2+rg):
        plt.plot(X,n_q[i][k,:],'-',label=str(k))
    plt.ylabel(r'$n_q$',size=s_)
    plt.xlabel(r'$t$',size=s_)
plt.show()


