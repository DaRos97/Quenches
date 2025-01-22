import numpy as np
import functions as fs_c
import dephasing_functions as fs_o
import parameters as ps
import matplotlib.pyplot as plt

Tau = 100
gamma = 0
steps = 20
h_t,J_t,times_dic = ps.find_parameters([Tau,],steps)
dt = times_dic[Tau][1]-times_dic[Tau][0]

#closed
name_pars = str(Tau)+'_'+"{:.4f}".format(dt).replace('.',',')
dirname = 'closed_Data/'
try:
    fid_closed = np.load(dirname+'fidelity_'+name_pars+'.npy')
except:
    args = (h_t, J_t, times_dic, [Tau,], ps.homedir, 0)
    fid_closed = fs_c.compute_fidelity(args)[0]
    
#open
name_pars = str(Tau)+'_'+"{:.4f}".format(dt).replace('.',',')+"_"+"{:.5f}".format(gamma).replace('.',',')
dirname = 'open_Data/'
try:
    fid_open = np.load(dirname+'fidelity_'+name_pars+'.npy')
except:
    args = (h_t, J_t, 0, times_dic, [Tau,], ps.homedir, 0)
    fid_open = fs_o.compute_fidelity(args)[0]

plt.figure()
X = np.linspace(0,1,steps)
plt.plot(X,fid_closed,color='b',label=r'closed $\tau$='+str(Tau))
plt.plot(X,fid_open,color='r',label=r'open $\tau$='+str(Tau))
plt.legend()
plt.show()

