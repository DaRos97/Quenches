import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import functions as fs

save_figs = 0#True

Lx = 401     #Linear size of lattice
Ly = 401
Ns = Lx*Ly    #number of sites
gridk = fs.BZgrid(Lx,Ly)#BZ grid
Gamma1 = np.cos(gridk[:,:,0])+np.cos(gridk[:,:,1])  #cos(kx) + cos(ky)
Gamma2 = np.cos(gridk[:,:,0]+gridk[:,:,1])+np.cos(gridk[:,:,0]-gridk[:,:,1])  #cos(kx+ky) + cos(kx-ky)
Gamma = (Gamma1,Gamma2)
#Parameters of the ramp
S = 0.5     #spin value
full_time_ramp = 0.5  #ramp time in ms
time_steps = 101        #of ramp
times = np.linspace(0,1,time_steps)
stop_ratio_list = np.linspace(0.1,1,10)     #ratios of full_time_ramp where we stop and measure

e_gs = np.zeros(time_steps)
thetas = np.zeros(time_steps)
gaps = np.zeros(time_steps)
#
fig = plt.figure(figsize=(20.8,8))
i_sr = 2
for i_t in range(time_steps):
    #Parameters Hamiltonian
    J1 = 40*(i_t/(time_steps-1))
    J2 = 0
    D1 = 0
    D2 = 0
    J = (J1,J2)
    D = (D1,D2)
    h = 30*(1-i_t/(time_steps-1))
    theta,phi = fs.get_angles(S,J,D,h)
    ts = fs.get_ts(theta,phi)
    parameters = (S,Gamma,h,ts,theta,phi,J,D)
    epsilon = fs.get_epsilon(*parameters)
    thetas[i_t] = theta
    e_gs[i_t] = fs.get_E_GS(*parameters)
    gaps[i_t] = np.min(epsilon)
    #
    if i_t in [10,30,50,60,80,100]:
        if i_t==60:
            i_sr = 7
        i_sr += 1
        ax = fig.add_subplot(2,5,i_sr,projection='3d')
        ax.plot_surface(gridk[:,:,0],gridk[:,:,1],epsilon,cmap='plasma')
        ax.set_aspect('equalxy')
        ax.set_title("stop ratio="+"{:.1f}".format(i_t/(time_steps-1)))
        n_i = 6
        ax.set_xticks([i*2*np.pi/n_i for i in range(n_i+1)],["{:.2f}".format(i*2*np.pi/n_i) for i in range(n_i+1)],size=8)
        ax.set_yticks([i*2*np.pi/n_i for i in range(n_i+1)],["{:.2f}".format(i*2*np.pi/n_i) for i in range(n_i+1)],size=8)
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")

plt.subplots_adjust(left=0.043,right=1,wspace=0.06,hspace=0.135)
gs = gridspec.GridSpec(2, 5, figure=fig)

gs = gridspec.GridSpec(2, 5, figure=fig)
ax = fig.add_subplot(gs[:, 0:2])
l1 = ax.plot(times,thetas,'b*-',label=r'$\theta$')
ax.set_yticks([i/6*np.pi/2 for i in range(7)],["{:.1f}".format(i/6*90)+'°' for i in range(7)],size=15,color='b')

ax_r = ax.twinx()
l2 = ax_r.plot(times,e_gs,'r*-',label=r'$E_{GS}$')
ax_r.tick_params(axis='y',colors='r')

ax_r = ax.twinx()
l3 = ax_r.plot(times,gaps,'g*-',label='Gap')
ax_r.tick_params(axis='y',colors='g')

ax.set_xlabel("stop ratio",size=20)

#Legend
labels = [l.get_label() for l in l1+l2+l3]
ax.legend(l1+l2+l3,labels,fontsize=20,loc=(0.4,0.1))

if save_figs:
    plt.savefig('Figures/theta_E_gap_dispersion.png')
plt.show()















