# Exact results in 1D quench

Here we evaluate some quantities if the 1D quench. The physical setup is a XX chain with staggered magnetic field which is ramped down to 0 in order to cross a phase transition.
In this quench excitations are created and the state no longer follows the GS of the system. 

We study the problem using the Jordan-Wigner transformation which maps the system from spins to free fermions. We want to compare this kind of quench with the Kibble-Zurek predictions.

## Quantities

The quantities we evaluate are:
- Fidelity of the state with the istantaneous GS --> *compute_fid*
- Density of excitations at the end of the quench -> KZ prediction --> *compute_nex*
- Single mode occupation as a function of time during the quench --> *compute_pop_ev*
- Occupation of single modes at half quench and at the end of the evolution for different quench times --> *compute_pop_T* (not ready)
- Energy of excitations as a function of time during the quench --> *compute_Enex*
- Normalized energy of the state at the critical point and at the end of the quench as a function of quench time --> *compute_en_CP*
- Compute entropy --> *compute_S* (NOT READY)
- Compute the correlation length by fitting the correlation function --> *compute_CL* (NOT READY) 
