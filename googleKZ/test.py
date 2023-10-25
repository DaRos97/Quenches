import numpy as np
import functions as fs
import parameters as ps
import matplotlib.pyplot as plt

Tau = 200
gamma = 0
steps = 20
h_t,J_t,times_dic = ps.find_parameters([Tau,],steps)
dt = times_dic[Tau][1]-times_dic[Tau][0]

#closed
name_pars = str(Tau)+'_'+"{:.4f}".format(dt).replace('.',',')
dirname = 'closed_Data/'
fid_closed = np.load(dirname+'fidelity_'+name_pars+'.npy')

#open
name_pars = str(Tau)+'_'+"{:.4f}".format(dt).replace('.',',')+"_"+"{:.5f}".format(gamma).replace('.',',')
dirname = 'open_Data/'
fid_open = np.load(dirname+'fidelity_'+name_pars+'.npy')

plt.figure()
X = np.linspace(0,1,steps)
plt.plot(X,fid_closed,color='b',label=r'$\tau$='+str(Tau))
plt.plot(X,fid_open,color='r',label=r'$\tau$='+str(gamma))
plt.show()

