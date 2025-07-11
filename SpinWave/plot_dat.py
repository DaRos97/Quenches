import numpy as np
import scipy
import functions as fs
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from pathlib import Path
import pickle

exclude_zero_mode = False#True

fn = 'Input/commutators_x_t_6x7_qubits_XX_10_Z_15_MHz.pkl'

data = pickle.load(open(fn,'rb'))

Ntimes, Lx, Ly = data[1.0].shape

txt_pars = 'uniform'
args_fn = (Lx,Ly,txt_pars)
transformation_fn = 'Data/rs_bogoliubov_' + fs.get_fn(*args_fn) + '.npz'
U_ = np.load(transformation_fn)['amazingU']
V_ = np.load(transformation_fn)['amazingV']

if exclude_zero_mode:
    for i_sr in range(2,10):
        U_[i_sr,:,0] *= 0
        V_[i_sr,:,0] *= 0

ind_j = 17 #2*Ly+3
N_omega = 2000
correlator = np.zeros((10,Lx,Ly,Ntimes),dtype=complex)
correlator[-1] = np.transpose(data[1.0], axes=(1,2,0))

fourier_type = 'dat'

if fourier_type=='dct':
    type_dct = 2
    correlator_kw = fs.fourier_dct(correlator,N_omega,type_dct)
if fourier_type=='dat':
    correlator_kw = fs.fourier_dat(correlator,U_,V_,ind_j,N_omega)

print("Plotting")
fs.plot(
    correlator_kw,
    n_bins=50,
    fourier_type=fourier_type,
    title='TN',
    figname='Figures/TN_'+fourier_type+'.png',
    showfig=True,
)

