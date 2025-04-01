# New simulation for 1D system in 2nd experiment

## Time evolution and correlators
`correlators.py`

We compute the time evolution of the system along a linear ramp and  then compute the correlators as the commutator at different time
and lattice position of operators: Z, E and J (spin current).
The ZZ correlator for example is the Fourier transform of `<Z_i(t)Z_0(0) - Z_0(0)Z_i(t)>`.
We have PBC, 42 sites and measuring times between 0 and 800 ns.
States are obtained through a linear ramp of total time 500 ns starting from the staggered state and arriving to the gappless state.
Model is XX+YY + staggered Z.

## Analysis
`analysis.py`

Here we compute some things on the correlators.

`energy_jeronimo.py`
Extract the final energy (end of ramp) for different ramp times and divide by nn coupling J to get a comparison with experimental energy.

`filling_corr.py` 
Extract and plot correlators at the end of the ramp at different fillings.

## Experimental parameters

In `exp_input` are saved the measured parameters of the Hamiltonian of the experiment.
We use the values of nn interaction and on-site field, the zz interaction cannot be included in this free fermion picture.
The format is:
