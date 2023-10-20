import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,})
import dephasing_functions as fs
import parameters as ps
#
#Additional parameters
Tau = 100
plot_ramp = False
save_data = False
s_ = 20 #fontsize
steps = 50

list_gamma = np.linspace(1e-3,1,10)

h_t,J_t,times_dic = ps.find_parameters([Tau,],steps)

args = (h_t,J_t,list_gamma, times_dic,Tau, ps.homedir, save_data)

fid = fs.compute_fidelity(args)

#
fig = plt.figure(figsize=(16,8))
X = np.linspace(0,1,steps)
plt.title("Tau = ",Tau)
for i in range(len(list_gamma)):
    plt.plot(X,fid[i],color=fs.cols[i%len(fs.cols)],label=r'$\tau$='+str(list_gamma[i]))
plt.xlim(-0.02,1.02)
plt.ylabel(r'$|\langle\psi_{GS}|\psi(t)\rangle|^2$',size=s_)
plt.xticks([0,1/2,1],['$t_i$','$t_{CP}$','$t_f$'],size=s_)
plt.legend()

plt.show()





