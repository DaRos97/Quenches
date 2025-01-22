import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,})
import dephasing_functions as fs
import parameters as ps
#
#Additional parameters
list_Tau = [100,250,500,750,1000]
gamma = 0
save_data = 0#True
steps = 200

h_t,J_t,times_dic = ps.find_parameters(list_Tau,steps)

if 0:   #use trivial couplings
    for i in range(len(h_t)):
        for t in range(steps):
            h_t[i][t] = (-1)**i
            J_t[i][t] = 1

args = (h_t,J_t,gamma, times_dic, list_Tau, ps.homedir, save_data)

rho = fs.time_evolve(args)

type_p = 'random'
fs.compute_zphases(rho,h_t,J_t,type_p)


s_ = 20 #fontsize
