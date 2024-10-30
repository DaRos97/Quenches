import numpy as np
import matplotlib.pyplot as plt

J_nn = 1
nkx = 100
nky = 2*nkx
Ns = nkx*nky
list_kx = np.linspace(0,np.pi,nkx,endpoint=False)
list_ky = np.linspace(-np.pi,np.pi,nky,endpoint=False)
Kx,Ky = np.meshgrid(list_ky,list_kx)
n_th = 201
list_th = np.linspace(np.pi/2,0,n_th)
n_h = 26
H_list = np.linspace(2.5,0,n_h)
E_h = np.zeros(n_h)
th_h = np.zeros(n_h)
gap_h = np.zeros(n_h)
for ih in range(n_h):
    h = H_list[ih]
    fn0 = "results/energy0_J"+"{:.3f}".format(J_nn)+"_h"+"{:.3f}".format(h)+"_nkx"+str(nkx)+"_nky"+str(nky)+"_th"+str(n_th)+".npy"
    fn1 = "results/dispersion_J"+"{:.3f}".format(J_nn)+"_h"+"{:.3f}".format(h)+"_nkx"+str(nkx)+"_nky"+str(nky)+"_th"+str(n_th)+".npy"
    fn2 = "results/minimaNk_J"+"{:.3f}".format(J_nn)+"_h"+"{:.3f}".format(h)+"_nkx"+str(nkx)+"_nky"+str(nky)+"_th"+str(n_th)+".npy"
    E0_th = np.load(fn0)
    w_th = np.load(fn1)
    min_th = np.load(fn2)
    i_best = n_th-1
    for i in range(n_th):
        if np.min(min_th[i]) > -1e-7:
            continue
        i_best = i
        break
    E_h[ih] = E0_th[i_best] + np.sum(w_th[i_best][~np.isnan(w_th[i_best])])/Ns/2
    th_h[ih] = list_th[i_best]
    gap_h[ih] = np.min(min_th[i_best])
print(th_h)
#
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot()
l1 = ax.plot(H_list,th_h,label=r"$\theta$ of quantization axis for GS",color='b',marker='*',ls='')
ax.set_yticks([np.pi/12*i for i in range(7)],["$0°=z-AFM$",r"$15°$",r"$30°$",r"$45°$",r"$60°$",r"$75°$",r"$90°=\hat{x}$"])
ax.yaxis.set_tick_params(labelsize=20,color='b')
ax.tick_params(axis='y',colors='b')
#
ax_r = ax.twinx()
l2 = ax_r.plot(H_list,E_h,label="GS energy",color='r')
ax_r.yaxis.set_tick_params(labelsize=20)
ax_r.tick_params(axis='y',colors='r')

ax_rr = ax.twinx()
l3 = ax_rr.plot(H_list,gap_h,color='g',label="Gap")
ax_rr.tick_params(axis='y',colors='g')

ax.xaxis.set_tick_params(labelsize=20)
ax.set_xlabel(r"staggered field $h$",size=20)

ls = l1+l2+l3
labs = [l.get_label() for l in ls]
ax.legend(ls,labs,fontsize=20)

plt.show()
