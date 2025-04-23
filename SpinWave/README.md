## Spin wave calculations for 2D XX + staggered h model

# Classical configuration
`j2_energy.py` computes the classical spin configuration when second nn coupling is on.

# Static spin wave analysis
In `static_dispersion.py` we compute the dispersion for different fields and J1-D1-J2-D2.
In `static_ZZ_k.py` we compute the ZZ correlator using the Bogoliubov transformation and momentum space, so in the periodic case.
in `static_ZZ_r.py` we compute the ZZ correlator using the real space wavefunction which works better for direct comparison with experiments (finite system size).


#Introduction

Here with 2sw.py we compute the GS energy of -J\sum (XX+YY) + h(staggered) using spin waves (Holstein-Primakof).

For each value of J,h we compute the GS energy for all values of theta, which defines the quantization axis.

For theta = 0 we expand in spin-waves around the AFM configuration, while for theta=pi/2 the Q-axis is along x.

# To Do
- Make EE commutator in real-space
- Chage way to do Fourier transform
- Higher order correlators in free fermions ?
