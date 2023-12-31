import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,})
import functions as fs
import parameters as ps
#
#Additional parameters
list_Tau = [100,200] #List of quench times -> in ns (total time of quench->exp one is 12ns)
save_data = True
s_ = 20 #fontsize
steps = 20

h_t,J_t,times_dic = ps.find_parameters(list_Tau,steps)

args = (h_t, J_t, times_dic, list_Tau, ps.homedir, save_data)
#
#Fidelity
fid = fs.compute_fidelity(args)
#
fig = plt.figure(figsize=(16,8))
X = np.linspace(0,1,steps)
for i in range(len(list_Tau)):
    Tau = list_Tau[i]
    plt.plot(X,fid[i],color=fs.cols[i%len(fs.cols)],label=r'$\tau$='+str(Tau))
plt.ylabel(r'$|\langle\psi_{GS}|\psi(t)\rangle|^2$',size=s_)
plt.xticks([0,1/2,1],['$t_i$','$t_{CP}$','$t_f$'],size=s_)
plt.legend()

plt.show()
































