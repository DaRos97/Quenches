import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,})
from mpl_toolkits.axes_grid1.inset_locator import inset_axes,InsetPosition,mark_inset
from scipy.optimize import curve_fit
import sys, getopt
import functions as fs

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "N:",["type=","dt=","PBC"])
    N = 100
    dt = 0.1
    list_Tau = [
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
            1,
            1.5,
            2,
#            3,
#            4,
#            5,
#            7,8,9,
#            10,
#            15,
#            20,
#            25,
#            30,
#            40,
#            50,
#            60,
#            70,
#            80,90,
#            100,
#            200,
#            300,
#            400,
#            800,
#            2000
            ]
    list_Tau = [10,20,30,40,50,60]#,100]#,200]
    BC = "open"
    type_of_quench = "real" #as opposed to linear
    #
    plot_ramp = False
    compute_fid = 0
    compute_nex = 1
    if compute_nex:
        list_Tau = list(np.arange(10,61,step=10))
    compute_pop_ev = 0
    compute_pop_T = 0
    compute_Enex = 0
    compute_en_CP = 0
    compute_S = 0
    compute_CF_zz = 0
    compute_CF_pm = 0
    tot_figs = 3*compute_pop_T + compute_en_CP + compute_fid + compute_nex + compute_Enex + compute_S + compute_CF_zz + compute_CF_pm
    if compute_pop_ev:
        tot_figs += len(list_Tau)
    list_dims = [(1,1), (1,2), (2,2), (2,2), (2,3), (2,3), (3,3), (3,3), (3,3)]
    fig_x,fig_y = list_dims[tot_figs-1]
    s_ = 20         #text size
except:
    print("Error")
    exit()
for opt, arg in opts:
    if opt in ['-N']:
        N = int(arg)
    if opt=='--dt':
        dt = float(arg)
    if opt=='--PBC':
        BC = "periodic"
    if opt=='--type':
        type_of_quench = arg
if type_of_quench not in ["linear","real"]:
    print("errorrr")
    exit()
args = (N, dt, list_Tau, BC, type_of_quench)
h_t = fs.h_t_dic[type_of_quench]
J_t = fs.J_t_dic[type_of_quench]
time_span = fs.time_span_dic[type_of_quench]
if plot_ramp:
    plt.figure()
    plt.title("1D ramp")
    times_inset = time_span(1,100)
    plt.plot(times_inset,h_t(times_inset,1),'r-',label='h')
    plt.plot(times_inset,J_t(times_inset,1),'g-',label='J')
    plt.hlines(0,times_inset[0],times_inset[-1],color='k',alpha=0.3)
    plt.vlines(0,-1,1,color='k',alpha=0.3)
    plt.xticks([times_inset[0],0,times_inset[-1]],["$t_i$","$0$","$t_f$"])
    plt.yticks([-1,0,1/2,1],["-1","0","1/2","1"])
    plt.legend()
    plt.show()
    exit()
print("Arguments: ",*args)
#
#fs.compute_gap(args)
#exit()
#Time evolution
times_, psi_ = fs.time_evolve(args)
#
fig = plt.figure(figsize=(16,8))
#plt.axis('off')
E_ = np.linalg.eigvalsh(fs.H_t(N,J_t(0,1),h_t(0,1),BC))   #energies of H at time 0 -> gapless point.
Gap = E_[N//2]-E_[N//2-1]
cols = ['r','g','y','b','k','m','orange','forestgreen']
plt.suptitle(r'$N=$'+str(N)+', time-step='+str(dt),size=s_+10)#+', Gap='+'{:5.4f}'.format(Gap),size=s_+10)
n_fig = 0
if compute_fid:
    n_fig += 1
    #Fidelity
    fid = fs.compute_fidelity(args)
    #
    ax1 = fig.add_subplot(fig_x,fig_y,n_fig) 
    for i in range(len(list_Tau)):
        Tau = list_Tau[i]
        ax1.plot(h_t(times_[i],Tau),fid[i],color=cols[i%len(cols)],label=r'$\tau$='+str(list_Tau[i]))
    ax1.set_ylabel(r'$|\langle\psi_{GS}|\psi(t)\rangle|^2$',size=s_)
    ax1.set_xlabel(r'$h_z$',size=s_)
    #Inset
#    Y_ax2 = [0,0,0.5,0.5,0.3,0.3,0.3]
#    ax2 = plt.axes([0,Y_ax2[tot_figs-1],0.5,0.5])
    coord = ax1.get_position().get_points()
    ax2 = plt.axes([coord[0,0],coord[0,1],0.5,0.5])
    ip = InsetPosition(ax1, [0.1,0.1,0.25,0.25])
    ax2.set_axes_locator(ip)
#    mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec='0.5')
    times_inset = time_span(1,100)
    ax2.plot(times_inset,h_t(times_inset,1),'r-',label='h')
    ax2.plot(times_inset,J_t(times_inset,1),'g-',label='J')
    ax2.hlines(0,times_inset[0],times_inset[-1],color='k',alpha=0.3)
    ax2.vlines(0,-1,1,color='k',alpha=0.3)
    ax2.set_xticks([times_inset[0],0,times_inset[-1]],["$t_i$","$0$","$t_f$"])
    ax2.set_yticks([-1,0,1/2,1],["-1","0","1/2","1"])
#    ax2.legend(loc='lower left')
    #
    ax1.legend(loc='upper right')

if compute_nex:
    n_fig += 1
    #Fit nex
    ind_T = -1
    nex = fs.compute_nex(ind_T,args)
    plt.subplot(fig_x,fig_y,n_fig)
    plt.title("$N=$"+str(N),size=s_)
    list_Tau = np.array(list_Tau,dtype=float)
    fun = fs.pow_law
    try:
        popt, pcov = curve_fit(fun,list_Tau,nex,p0=[1,-0.5])
        X = np.linspace(list_Tau[0],list_Tau[-1],100)
        plt.plot(X,fun(X,*popt))
        plt.text(list_Tau[len(list_Tau)//2],nex[0]*3/4,'exponent='+"{:3.4f}".format(popt[1]),size=s_)
    except:
        print("Fit nex not found")

    for i in range(len(list_Tau)):
        plt.plot(list_Tau[i],nex[i],'*',color=cols[i%len(cols)])
    plt.ylabel(r'$n_{ex}$',size=s_)
#    plt.yscale('log')
    plt.xlabel(r'$\tau_Q$',size=s_)

if compute_pop_ev:
    #Fit nex
    plt.suptitle(r"$N=$"+str(N),size=s_)
    rg = 3 #number of modes from center --> all modes are N//2
    n_q = fs.compute_pop_ev(args,rg)
    list_Tau = np.array(list_Tau,dtype=float)
    for i in range(len(list_Tau)):
        Tau = list_Tau[i]
        n_fig += 1
        plt.subplot(fig_x,fig_y,n_fig)
        plt.text(-1,0.2,r'$\tau=$'+str(int(Tau)),size=s_)
        plt.ylabel(r'$n_q$',size=s_)
        if n_fig > 6:
            plt.xlabel(r'$h_z$',size=s_)
        for k in range(N//2-rg,N//2+rg):
            plt.plot(h_t(times_[i],Tau),n_q[i][k,:],'-',label=str(k))
        #plt.legend()

if compute_pop_T:
    #Fit nex
    plt.suptitle(r"$N=$"+str(N),size=s_)
    n_q = fs.compute_pop_T(args)
    list_Tau = np.array(list_Tau,dtype=float)
    in_ = 10
    time_tt = [r"$t_{CP}$",r"$t_f$"]
    for t in range(2):
        n_fig += 1
        plt.subplot(fig_x,fig_y,n_fig+t)
        plt.title("time="+time_tt[t],size=s_)
        #plt.text(-1,0.2,r'$\tau=$'+str(int(Tau)),size=s_)
        for i in range(len(list_Tau)):
            Tau = list_Tau[i]
            plt.plot(np.arange(in_,N-in_),n_q[i][in_:N-in_,t],label=r'$\tau=$'+str(int(Tau)),color=cols[i%len(cols)])
        plt.ylabel(r'$n_q$',size=s_)
        if t:
            plt.xlabel(r'$q$',size=s_)
    plt.legend()

if compute_en_CP:
    nnn = 0
    for N in [20,40,60,80]:
        nnn += 1
        args = (N, dt, list_Tau, BC, type_of_quench)
        energies = fs.compute_en_CP(args)
        #
        plt.subplot(2,2,nnn)
        plt.title('$N=$'+str(N),size=s_)
        labels_t = ['$t=t_{CP}$', '$t=t_{final}$']
        for i in range(energies.shape[1]):
            plt.plot(list_Tau,energies[:,i],'*-',label=labels_t[i])
        plt.legend(fontsize=s_)
        #plt.xticks(list_Tau,size=s_)
        plt.yticks(size=s_)
        #plt.yscale('log')
        if N in [60,80]:
            plt.xlabel(r'$\tau_Q$',size=s_)
        if N in [20,60]:
            #plt.ylabel(r'$\langle\psi(t)|H(t)|\psi(t)\rangle-E_{GS}(t)$',size=s_)
            plt.ylabel(r'$|(\langle H(t)\rangle-E_{GS}(t))/E_{GS}|$',size=s_)

if compute_Enex:    #compute energy density
    n_fig += 1
    #Fit nex
    Enex = fs.compute_Enex(args)
    plt.subplot(fig_x,fig_y,n_fig)
    list_Tau = np.array(list_Tau,dtype=float)
    for i in range(len(list_Tau)):
        Tau = list_Tau[i]
        total_ev_time = 2*np.tan(1)*Tau if type_of_quench == "real" else 2*Tau
        steps = int(total_ev_time/dt)
        times = time_span(Tau,steps)
        h = h_t(times,Tau)
        plt.plot(h,Enex[i],'-',color=cols[i%len(cols)])
    plt.gca().invert_xaxis()
    plt.ylabel(r'$E_{ex}$',size=s_)
    plt.xlabel(r'$h_z$',size=s_)

if compute_S:
    n_fig += 1
    ind_T = -1
    n_in = 1
    Step = 2
    X = np.arange(n_in,N,step=Step)
    S = fs.compute_S(ind_T,args,X)
    plt.subplot(fig_x,fig_y,n_fig)
    for n,Tau in enumerate(list_Tau):
        plt.plot(X,S[n][n_in:N:Step],color=cols[n%len(cols)])


if compute_CF_zz:
    ind_T = 2                  #-1 is final time ait end of quench
    t_text = r"$t_{CP}$" if ind_T==2 else r"$t_f$"
    in_ = 1         #first distance plotted
    out_ = 9             #removed sites at borders of the chain
    end_ = 2*out_#11#N//6     #removed distances from end
    step = 2        #1--> all distances, 2--> only even/odd distances
    #
#    wf = "gs"
    wf = "t-ev"
    CF = fs.compute_CF_zz(ind_T,args,wf,in_,out_,step)
    n_fig += 1
    ax = fig.add_subplot(fig_x,fig_y,n_fig) 
    for n,Tau in enumerate(list_Tau):
        ax.plot(np.arange(in_,N-end_,step),CF[n][in_:N-end_:step],'*',label=r"$\tau_Q=$"+str(Tau),color=cols[n%len(cols)])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r"$\langle S^z_iS^z_{i+r}\rangle-\langle S^z_i\rangle\langle S^z_{i+r}\rangle$",size=s_)
    ax.set_xlabel("$r$",size=s_)
    try:
        popt,pcov = curve_fit(fs.pow_law,np.arange(in_,N-end_,step),CF[-1][in_:N-end_:step],p0=[1,-2])
        ax.plot(np.linspace(in_,N-end_,1000),pow_law(np.linspace(in_,N-end_,1000),*popt),'k-',label='a='+"{:.4f}".format(popt[1]))
    except:
        print("Error in fitting")
    ax.text(1,1e-6,wf+" wavefunction, t="+t_text,size=s_)
    plt.legend()

if compute_CF_pm:
    ind_T = 2                  #-1 is final time ait end of quench
    t_text = r"$t_{CP}$" if ind_T==2 else r"$t_f$"
    in_ = 1         #first distance plotted
    out_ = 9             #removed sites at borders of the chain
    end_ = 2*out_#11#N//6     #removed distances from end
    end_distance = 15
    step = 2        #1--> all distances, 2--> only even/odd distances
    #
#    wf = "gs"
    wf = "t-ev"
    CF = fs.compute_CF_pm(ind_T,args,wf,in_,out_,step,end_distance)
    #
    n_fig += 1
    ax = fig.add_subplot(fig_x,fig_y,n_fig) 
    for n,Tau in enumerate(list_Tau):
        ax.plot(np.arange(in_,N-end_,step),-CF[n][in_:N-end_:step],'*',label=r"$\tau_Q=$"+str(Tau),color=cols[n%len(cols)])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r"$\langle S^+_iS^-_{i+r}\rangle$",size=s_)
    ax.set_xlabel("$r$",size=s_)
    def corr_pm(x,c3,c4):
        for i in range(len(x)):
            x[i] = int(x[i])
        return c3*x**(-5/2) + c4*(-1)**x*x**(-1/2)
    if 1:
        y_ind_end = np.nonzero(np.isnan(CF[-1][in_:N-end_:step]))[0][0]
        y_fit = CF[-1][in_:y_ind_end*step:step]
        popt,pcov = curve_fit(fs.pow_law,np.arange(in_,y_ind_end*step,step),y_fit,p0=[0.48,-0.5])
        print(popt)
        #ax.plot(np.linspace(in_,y_ind_end*step,1000),fs.pow_law(np.linspace(in_,y_ind_end*step,1000),*popt),'k-',label='a='+"{:.4f}".format(popt[1]))
    else:
#    except:
        print("Error in fitting")
    ax.text(1,1e-6,wf+" wavefunction, t="+t_text,size=s_)
    plt.legend()
plt.show()
































