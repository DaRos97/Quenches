import pickle 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline as RBS

#
with open('experimental_data/params_Dario', 'rb') as f:
    exp = pickle.load(f)
#Change keys to usable ones
stupid_keys = list(exp.keys())
if 0:
    #print(stupid_keys)
    print(exp[stupid_keys[0]].keys())
    exit()
len_f = int(np.sqrt(len(stupid_keys)))
data = {}
for i in range(len_f): #f_coupler
    for j in range(len_f): #f_quibit
        #create new entry with same value but different key
        data[i*len_f+j] = exp[stupid_keys[i*len_f+j]]
if 0:   #Plot circuit geometry
    pts = []
    labels = list(data[0]['z'].keys())
    labels.append(labels[0])
    plt.figure(figsize=(15,15))
    plt.gca().set_aspect('equal')
    plt.axis('off')
    n_axis = 13
    for i in range(n_axis+1):
        if 1!=0:
            plt.vlines(i,0,13,linewidth=0.1,color='k')
            plt.hlines(i,0,13,linewidth=0.1,color='k')
    for i in range(len(labels)-1):
        pt = [float(tuple(labels[i])[0]),float(tuple(labels[i])[1])]
        pt2 = [float(tuple(labels[i+1])[0]),float(tuple(labels[i+1])[1])]
        #Arrows
        dd = 0.15
        d = (0,dd) if pt2[0]-pt[0] == 0 else (2*dd,0)
        plt.arrow(pt[0]+d[0]*np.sign(pt2[0]-pt[0]),
                pt[1]+d[1]*np.sign(pt2[1]-pt[1]),
                pt2[0]-pt[0]-2*d[0]*np.sign(pt2[0]-pt[0]),
                pt2[1]-pt[1]-2*d[1]*np.sign(pt2[1]-pt[1]),
                length_includes_head=True,head_width=0.1,head_length=0.2,fill=True)
        plt.text(pt[0]-0.25,pt[1]-0.05,str(labels[i]),size=13)
#    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()
    exit()
#
with open('experimental_data/t_to_f_coupler', 'rb') as f:
    tfc = pickle.load(f)
with open('experimental_data/t_to_f_qubit', 'rb') as f:
    tfq = pickle.load(f)
#
n_tfc = len(tfc.keys())
n_tfq = len(tfq.keys())
tcs = np.linspace(0,6,n_tfc,endpoint=True)
tqs = np.linspace(0,6,n_tfq,endpoint=True)
#Interpolate fc, fq as a function of time
fc = np.array(list(tfc.values()))
fq = np.array(list(tfq.values()))
#
fun_fc = interp1d(tcs,fc)
fun_fq = interp1d(tqs,fq)
#
if 0:#Plot tfc and tfq
    plt.figure()
    plt.plot(tcs,fun_fc(tcs),'r-',label='f coupler')
    plt.plot(tqs,fun_fq(tqs),'g-',label='f qubit')
    plt.legend()
    plt.xlabel('time')
    plt.show()
    exit()

def find_z_f(point,*args):
    data, len_f = args
    z_f_data = np.zeros((len_f,len_f))
    for i in range(len_f):
        for j in range(len_f):
            z_f_data[i,j] = data[i*len_f+j]['z'][point]
    f_c_values = np.linspace(0,1,len_f,endpoint=True)
    f_q_values = np.linspace(0,1,len_f,endpoint=True)
    fun_z_f = RBS(f_c_values,f_q_values,z_f_data)
    if 0:#plot 2d z as function of fc and fq
        fig,ax = plt.subplots(subplot_kw={"projection": "3d"})
        from matplotlib import cm
        X,Y = np.meshgrid(f_c_values,f_q_values)
        ax.plot_surface(Y,X,fun_z_f(f_c_values,f_q_values),cmap=cm.plasma)
        ax.set_xlabel('f_c')
        ax.set_ylabel('f_q')
        ax.set_zlabel('z')
        plt.show()
        exit()
    return fun_z_f
def find_xx_f(point,*args):
    data, len_f = args
    xx_f_data = np.zeros((len_f,len_f))
    for i in range(len_f):
        for j in range(len_f):
            xx_f_data[i,j] = data[i*len_f+j]['xx'][point]
    f_c_values = np.linspace(0,1,len_f,endpoint=True)
    f_q_values = np.linspace(0,1,len_f,endpoint=True)
    fun_xx_f = RBS(f_c_values,f_q_values,xx_f_data)
    if 1:#plot 2d xx as function of fc and fq
        fig,ax = plt.subplots(subplot_kw={"projection": "3d"})
        from matplotlib import cm
        X,Y = np.meshgrid(f_c_values,f_q_values)
        ax.plot_surface(Y,X,fun_xx_f(f_c_values,f_q_values),cmap=cm.plasma)
        ax.set_xlabel('f_c')
        ax.set_ylabel('f_q')
        ax.set_zlabel('xx')
        plt.show()
        exit()
    return fun_xx_f

if 1:   #2D case
    args = (data,len_f)
    xx_t_data = np.zeros(n_tfc)
    z1_t_data = np.zeros(n_tfc)
    z2_t_data = np.zeros(n_tfc)
    xx_label = ((3,3),(3,4))
    z1_label = (3,3)
    z2_label = (3,4)
    for i in range(n_tfc):
        xx_t_data[i] = find_xx_f(xx_label,*args)(fun_fc(tcs[i]),fun_fq(tqs[i]))
        z1_t_data[i] = find_z_f(z1_label,*args)(fun_fc(tcs[i]),fun_fq(tqs[i]))
        z2_t_data[i] = find_z_f(z2_label,*args)(fun_fc(tcs[i]),fun_fq(tqs[i]))
    if 1:   #plot
        plt.figure(figsize=(10,5))
#        plt.subplot(1,3,1)
#        plt.plot(tcs,xx_t_data,'r',label='xx '+str(xx_label))
#        plt.xlabel('time')
#        plt.legend()
        plt.subplot(1,2,1)
        plt.plot(tcs,z1_t_data,'b',label='z '+str(z1_label))
        plt.plot(tcs,z2_t_data,'r',label='z '+str(z2_label))
        plt.xlabel('time')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(tcs,z1_t_data-z2_t_data,'b',label='z '+str(z1_label)+'-z '+str(z2_label))
        plt.xlabel('time')
        plt.legend()
        plt.show()
        exit()
if 0:
    #We start by extracting the grid of xx and z
    xx_f_data = np.zeros((len_f,len_f))
    z_f_data = np.zeros((len_f,len_f))
    for i in range(len_f):
        for j in range(len_f):
            xx_f_data[i,j] = data[i*len_f+j]['xx'][(3,3),(3,4)]
            z_f_data[i,j] = data[i*len_f+j]['z'][(3,3)]
    #Interpolate xx and z as a function of f
    f_c_values = np.linspace(0,1,len_f,endpoint=True)
    f_q_values = np.linspace(0,1,len_f,endpoint=True)
    fun_xx_f = RBS(f_c_values,f_q_values,xx_f_data)
    fun_z_f = RBS(f_c_values,f_q_values,z_f_data)
    #create dataset of xx and z as a function of time
    xx_t_data = np.zeros(n_tfc)
    z_t_data = np.zeros(n_tfc)
    for i in range(n_tfc):
        xx_t_data[i] = fun_xx_f(fun_fc(tcs[i]),fun_fq(tqs[i]))
        z_t_data[i] = fun_z_f(fun_fc(tcs[i]),fun_fq(tqs[i]))

    #Interpolate xx and z as a function of t in the 2 directions?


if 0:#1D case
    D1 = {}
    D1[0] = data[0]
    for i in range(1,20):
        D1[i] = data[i*20+19]       #take only f_q = 1 -> last of js
    xx = np.zeros(20)
    z1 = np.zeros(20)
    z2 = np.zeros(20)
    for i in range(20):
        xx[i] = D1[i]['xx'][((3,3),(3,4))]      #take random bond
        z1[i] = D1[i]['z'][(3,3)]               #take random site
        z2[i] = D1[i]['z'][(3,4)]               #take site next to previous one
    #interpolate xx as a function of time
    fun_xx_f = interp1d(np.linspace(0,20,20,endpoint=True),xx)
    xx_t = np.zeros(n_tfc)
    for i in range(n_tfc):
        xx_t[i] = fun_xx_f(fun_fc(tcs[i]))
    fun_xx_t = interp1d(tcs,xx_t)
    #interpolate z as a function of time
    fun_z1_f = interp1d(np.linspace(0,20,20,endpoint=True),z1)
    fun_z2_f = interp1d(np.linspace(0,20,20,endpoint=True),z2)
    z1_t = np.zeros(n_tfc)
    z2_t = np.zeros(n_tfc)
    for i in range(n_tfc):
        z1_t[i] = fun_z1_f(fun_fc(tcs[i]))
        z2_t[i] = fun_z2_f(fun_fc(tcs[i]))
    fun_z1_t = interp1d(tcs,z1_t)
    fun_z2_t = interp1d(tcs,z2_t)

    if 1:   #plot xx and z as a function of time
        plt.figure()
    #    plt.plot(tt,fun_xx_t(tt),'r')
        plt.subplot(1,2,1)
        plt.plot(tcs,fun_z1_t(tcs),'g')
        plt.subplot(1,2,2)
        plt.plot(tcs,fun_z2_t(tcs),'r')
        plt.show()






























