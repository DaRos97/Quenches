# New simulation for 1D system in 2nd experiment

## ZZ correlator

We want to measure the ZZ correlator on a certain state psi. 
Measured quantity is the Fourier transform of `<Z_i(t)Z_0(0) - Z_0(0)Z_i(t)>`.
We have PBC, 42 sites and measuring times between 0 and 800 ns.
States are obtained through a linear ramp of total time 500 ns starting from the staggered state and arriving to the gappless state.
Model is still XX+YY + staggered Z.

We compute also the ramp to get to the final measured state by time evolving.

