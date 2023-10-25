import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,})
import dephasing_functions as fs
import parameters as ps
#
#Additional parameters
list_Tau = [100,200,]
save_data = False
s_ = 20 #fontsize
steps = 20

h_t,J_t,times_dic = ps.find_parameters(list_Tau,steps)

list_gamma = [0,]#np.linspace(0,1,11)
fid_ = []
for gamma in list_gamma:
    args = (h_t,J_t,gamma, times_dic, list_Tau, ps.homedir, save_data)
    fid_.append(fs.compute_fidelity(args))
#
fig = plt.figure(figsize=(16,8))
X = np.linspace(0,1,steps)
for i in range(len(list_gamma)):
    for t,Tau in enumerate(list_Tau):
        plt.plot(X,fid_[i][t],color=fs.cols[t%len(fs.cols)],label=r'$\gamma$='+"{:.4f}".format(list_gamma[i])+r', $\tau$='+str(Tau))
plt.xlim(-0.02,1.02)
plt.ylabel(r'$|\langle\psi_{GS}|\psi(t)\rangle|^2$',size=s_)
plt.xticks([0,1/2,1],['$t_i$','$t_{CP}$','$t_f$'],size=s_)
plt.legend()

plt.show()





