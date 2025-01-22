import numpy as np
import sys, getopt
import inputs
import parameters as ps

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "N:",["ramp=","steps=","tau=","gamma="])
    #Default values
    ramp = 'exp' #'exp'erimental or user-defined ('usd')
    N = 32      #number of sites. Needed just if ramp is not 'exp'
    # Important parameters
    steps = 5     #ramp time steps
    tau = 100       #ramp total time
    gamma = 0.0012       #on-site dephasing strength
except:
    print("Error in input parameters, sys.argv = ",argv)
    exit()
for opt, arg in opts:
    if opt in ['-N']:
        N = int(arg)
    if opt=='--ramp':
        ramp = arg
        if ramp not in inputs.list_ramps:
            print("Ramp type not accepted")
            exit()
    if opt=='--steps':
        steps = int(arg)
    if opt=='--tau':
        tau = float(arg)
    if opt=='--gamma':
        gamma = float(arg)
if ramp=='exp':
    h_t,J_t = ps.find_parameters(tau,steps)
else:
    pass
N = len(h_t)
print("Using parameters: ","\n\t-Ramp: ",ramp,"\n\t-Quench time: ",tau,"\n\t-Dephasing: ",gamma,"\n\t-Number of sites: ",N)

if gamma == 0:
    import closed_functions as fs
    name_fn = 'wf_'
else:
    import open_functions as fs
    name_fn = 'DM_'

compute = { 'time-evolution':fs.time_evolve,
            'fidelity':fs.compute_fidelity,
            'populations':fs.compute_populations,
            'correlation_function_zz':fs.compute_CF_zz,
            'correlation_function_xx':fs.compute_CF_xx,
            'density_of_excitations':fs.compute_nex,
          }

for quant in inputs.quantities:
    args = (h_t,J_t,steps,tau,gamma,ps.result_dn,ps.cluster,name_fn)
    result_fn =  ps.result_dn+inputs.names[quant]+name_fn+inputs.pars_name(tau,dt,gamma)+'.npy'
    if not Path(result_fn).is_file():
        compute[quant](args,result_fn)



#?????
if 0:
    ind_T = 2                  #-1 is final time -> end of quench
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














    
