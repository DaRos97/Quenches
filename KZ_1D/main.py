import numpy as np
import sys, getopt
import general as gn
import parameters as ps

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "N:",["ramp=","steps=","tau=","gamma=","save"])
    #Default values
    ramp = 'exp'
    N = 32
    steps = 100
    tau = 100
    save = True
    gamma = 0
except:
    print("Error in input parameters, sys.argv = ",argv)
    exit()
for opt, arg in opts:
    if opt in ['-N']:
        N = int(arg)
    if opt=='--ramp':
        ramp = arg
        if ramp not in gn.list_ramps:
            print("Ramp type not accepted")
            exit()
    if opt=='--steps':
        steps = int(arg)
    if opt=='--tau':
        tau = float(arg)
    if opt=='--gamma':
        gamma = float(arg)
    if opt=='--save':
        save = True

if ramp=='exp':
    h_t,J_t,times = ps.find_parameters(tau,steps)
else:
    pass

if gamma == 0:
    import closed_functions as fs
else:
    import open_functions as fs
compute = { 'time-evolution':fs.time_evolve,
            'fidelity':fs.compute_fidelity,
            'populations':fs.compute_populations,
          }

for quant in gn.quantities:
    args = (h_t,J_t,times,tau,gamma,ps.result_dir,save)
    results = compute[quant](args)















    
