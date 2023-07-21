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
    N = 20
    dt = 0.1
    list_Tau = [
#            1,2,3,4,
#            5,
#            7,8,9,
            10,
#            15,
            20,
#            25,
            30,
            40,
            50,
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
#    list_Tau = [200,]
    BC = "open"
    type_of_quench = "real"
    #
    compute_fid = 1
    compute_nex = 1
    compute_Enex = 0
    compute_S = 0
    compute_CL = 0
    CL_add = 1 if compute_CL else 0
    tot_figs = compute_fid + compute_nex + compute_Enex + compute_S + compute_CL
    list_dims = [(1,1+CL_add), (1+CL_add,2), (2,2), (2,2)]
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
h_t = fs.h_t_real if type_of_quench == "real" else fs.h_t_linear
J_t = fs.J_t_real if type_of_quench == "real" else fs.J_t_linear
time_span = fs.time_span_real if type_of_quench == "real" else fs.time_span_linear 
print("Arguments: ",*args)
#
#fs.compute_GSE(args)
#exit()
#Time evolution
times_, psi_ = fs.time_evolve(args)
#
fig = plt.figure(figsize=(16,8))
E_ = np.linalg.eigvalsh(fs.H_t(N,J_t(0,1),h_t(0,1),BC))   #energies of H at time 0.
Gap = E_[N//2]-E_[N//2-1]
cols = ['r','g','y','b','k','m','orange','forestgreen']
plt.suptitle(r'$N=$'+str(N)+', time-step='+str(dt)+', Gap='+'{:5.4f}'.format(Gap),size=s_+10)
n_fig = 0
if compute_fid:
    n_fig += 1
    #Fidelity
    fid = fs.compute_fidelity(args)
    #
    ax1 = fig.add_subplot(fig_x,fig_y,n_fig) 
    for i in range(len(list_Tau)):
        X = np.linspace(-1,1,len(times_[i]),endpoint=False)
#        plt.plot(X,fid[i],color=cols[i%len(cols)],label=r'$\tau$='+str(list_Tau[i]))
        ax1.plot(X,fid[i],color=cols[i%len(cols)],label=r'$\tau$='+str(list_Tau[i]))
        #plt.vlines(-(list_Tau[i])**(-1/2),0,1,color=cols[i%len(cols)])
    ax1.set_ylabel('Fidelity',size=s_)
    ax1.set_xlabel(r'$h(t)$')#=-tg^{-1}(t/\tau),\quad J=exp(-(t/\sigma)^2)$',size=s_)
    #Inset
    Y_ax2 = [0,0,0.5,0.5]
    ax2 = plt.axes([0,Y_ax2[tot_figs-1],0.5,0.5])
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
    ax2.legend(loc='lower left')
    #
    ax1.legend(loc='upper right')

if compute_nex:
    n_fig += 1
    #Fit nex
    ind_T = -1
    nex = fs.compute_nex(ind_T,args)
    print(list_Tau,nex)
    plt.subplot(fig_x,fig_y,n_fig)
    list_Tau = np.array(list_Tau,dtype=float)
    fun = fs.pow_law
    try:
        popt, pcov = curve_fit(fun,list_Tau,nex,p0=[1,-0.5])
        print(popt)
        X = np.linspace(list_Tau[0],list_Tau[-1],100)
        plt.plot(X,fun(X,*popt))
        plt.text(list_Tau[len(list_Tau)//2],nex[0]*3/4,'exponent='+"{:3.4f}".format(popt[1]),size=s_)
    except:
        print("Fit nex not found")

    for i in range(len(list_Tau)):
        plt.plot(list_Tau[i],nex[i],'*',color=cols[i%len(cols)])
    plt.ylabel(r'$n_{ex}$',size=s_)
    plt.xlabel(r'$\tau_Q$',size=s_)

if compute_Enex:    #compute energy density
    n_fig += 1
    #Fit nex
    Enex = fs.compute_Enex(args)
    plt.subplot(fig_x,fig_y,n_fig)
    list_Tau = np.array(list_Tau,dtype=float)
    time_span = fs.time_span_real if type_of_quench == "real" else fs.time_span_linear 
    h_t = fs.h_t_real if type_of_quench == "real" else fs.h_t_linear
    for i in range(len(list_Tau)):
        Tau = list_Tau[i]
        total_ev_time = 2*np.tan(1)*Tau if type_of_quench == "real" else 2*Tau
        steps = int(total_ev_time/dt)
        times = time_span(Tau,steps)
        h = h_t(times,Tau)
        plt.plot(h,Enex[i],'-',color=cols[i%len(cols)])
    plt.gca().invert_xaxis()
    plt.ylabel(r'$E_{ex}$',size=s_)
    plt.xlabel(r'$h$',size=s_)

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


if compute_CL:
    n_fig += 1
    ind_T = -1                  #-1 is final time at end of quench
    sub_lat = 1
    CF = fs.compute_CF(ind_T,args,"t-ev")
#    CF_gs = fs.compute_CF(ind_T,args,"gs")
    #
    fit_CL = []
    fit_CL_gs = []
    #Plot CF
    plt.subplot(fig_x,fig_y,n_fig)
    for n,Tau in enumerate(list_Tau):
        n_init = 1
        n_final = 12#N-10
        if sub_lat:
            X_fit = np.arange(n_init,n_final,2)
            CFz = CF[n][2,n_init:n_final:2]
#            CFz_gs = CF_gs[n][2,n_init:n_final:2]
        else:
            X_fit = np.arange(n_init,n_final)
            CFz = CF[n][2,n_init:n_final]
#            CFz_gs = CF_gs[n][2,n_init:n_final]
        #Compute fit and extract CL
        X = np.linspace(X_fit[0],X_fit[-1],100)
        plt.plot(X_fit,CFz,'*',color=cols[n%len(cols)],label='zz-t')
#        plt.plot(X_fit,CFz_gs,'^',color=cols[n%len(cols)],label='zz-GS')
        fun = fs.exp_dec
#        try:
#            popt, pcov = curve_fit(fun,X_fit,CFz_gs,p0=[CFz_gs[0],1])
#            fit_CL_gs.append(popt[1])
#            plt.plot(X,fs.exp_dec(X,*popt),'-',color=cols[n%len(cols)],label = 'fit gs')
#        except:
#            fit_CL.append(0)
#            print("aaa")
        try:
            popt, pcov = curve_fit(fun,X_fit,CFz,p0=[CFz_gs[0],1])
            fit_CL.append(popt[1])
#            plt.plot(X,fs.exp_dec(X,*popt),'-',color=cols[n%len(cols)],label = 'fit t-ev')
        except:
            fit_CL.append(0)
            print("aaa")
    plt.xticks(X_fit)
    plt.legend()
    plt.yscale('log')
    plt.ylabel("Correlation Function",size=s_)
    plt.xlabel(r"$r$",size=s_)
    print(fit_CL)
    plt.show()
    exit()
    #Plot CL
    plt.subplot(fig_x,fig_y,n_fig+1)
    for n,Tau in enumerate(list_Tau):
        plt.plot(Tau,fit_CL[n],'*',color=cols[n%len(cols)])
    popt, pcov = curve_fit(fs.pow_law,list_Tau,fit_CL,p0=[1,-0.5])
    X = np.linspace(list_Tau[0],list_Tau[-1],100)
    plt.plot(X,fs.pow_law(X,*popt))
    plt.text(list_Tau[len(list_Tau)//2],fit_CL[1],'exponent='+"{:3.4f}".format(popt[1]),size=s_)
    plt.ylabel("Correlation length",size=s_)
    plt.xlabel(r"$\tau_Q$",size=s_)
plt.show()
































