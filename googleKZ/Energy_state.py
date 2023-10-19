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

energy_state = fs.compute_energy_state(args)
#
plt.figure()
plt.plot(list_Tau,energy_state,'*-')
#plt.yscale('log')
plt.xlabel(r'$\tau_Q$',size=s_)
plt.ylabel(r'$|(\langle H(t)\rangle-E_{GS}(t))/E_{GS}|$',size=s_)

plt.show()
