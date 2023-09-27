import numpy as np
import functions as fs
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import getopt
from scipy.interpolate import RectBivariateSpline as RBS

#input arguments
argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "N:S:K:", ['disp','Nq='])
    S = '50'
    K = 30          #cuts in BZ of minimization
    Nq = 30         #number of points to evaluate the SF in
    N = 0           #index of h value --> not needed
    disp = True
except:
    print("Error")
for opt, arg in opts:
    if opt in ['-S']:
        S = arg
    if opt in ['-K']:
        K = int(arg)
    if opt in ['-N']:
        N = int(arg)
    if opt == '--disp':
        disp = True
    if opt == '--Nq':
        Nq = int(arg)
#Parameters
J_nn = 1
hi = 10
hf = 0
hpts = 100
H = []
for i in range(hpts):
    H.append(hi+(hf-hi)/(hpts-1)*i)
h = H[N]
################################
savenameSFzz = "data_SF/SFzz_J1_"+str(J_nn)+'_h_'+'{:5.4f}'.format(h).replace('.','')+'_S_'+S+'_Nq_'+str(Nq)+'.npy'
savenameSFxy = "data_SF/SFxy_J1_"+str(J_nn)+'_h_'+'{:5.4f}'.format(h).replace('.','')+'_S_'+S+'_Nq_'+str(Nq)+'.npy'

#SFzz = np.load(savenameSFzz)
SFxy = np.load(savenameSFxy)
Nx,Ny = SFxy.shape

Kxg = np.linspace(0,2*np.pi,Nx)
Kyg = np.linspace(0,np.pi,Ny)
K = np.zeros((2,Nx,Ny))
for i in range(Nx):
    for j in range(Ny):
        K[:,i,j] = np.array([Kxg[i],Kyg[j]])
if 1:
    #
    plt.figure(figsize=(12,12))
    plt.rcParams.update({
        "text.usetex": True,
    })
    plt.subplot(1,2,1)
    plt.gca().set_aspect('equal')
    plt.title("SF xy")
    #
    plt.scatter(K[0],K[1],c=SFxy,cmap = cm.get_cmap('plasma_r'),s=70)
    #
    cbar = plt.colorbar()
    plt.axis('off')
    plt.xlabel(r'$K_x$',size=15)
    plt.ylabel(r'$K_y$',size=15,rotation='horizontal')
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    
    plt.show()
    exit()
else:
    #
    plt.figure(figsize=(12,12))
    plt.rcParams.update({
        "text.usetex": True,
    #    "font.family": "Helvetica"
    })
    #plt.subplot(1,2,1)
    plt.gca().set_aspect('equal')
    #title = 'S='+S+', DM='+DM+', (J2,J3)=('+str(J2)+','+str(J3)+'), ansatz: '+ans
    #plt.title(title)
    #
    #BZ
    plt.plot(fs.X1,fs.fu1(fs.X1),'k-')
    plt.hlines(2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')#, linestyles = 'dashed')
    plt.plot(fs.X2,fs.fu3(fs.X2),'k-')
    plt.plot(fs.X1,fs.fd1(fs.X1),'k-')
    plt.hlines(-2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')#, linestyles = 'dashed')
    plt.plot(fs.X2,fs.fd3(fs.X2),'k-')
    #EBZ
    plt.plot(fs.X3,fs.Fu1(fs.X3),'k-')
    plt.hlines(4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
    plt.plot(fs.X4,fs.Fu3(fs.X4),'k-')
    plt.plot(fs.X3,fs.Fd1(fs.X3),'k-')
    plt.hlines(-4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
    plt.plot(fs.X4,fs.Fd3(fs.X4),'k-')
    #Middle lines
    plt.plot(fs.X2,np.sqrt(3)*fs.X2,'k--')
    plt.plot(fs.X2,-np.sqrt(3)*fs.X2,'k--')
    plt.plot(fs.X1,np.sqrt(3)*fs.X1,'k--')
    plt.plot(fs.X1,-np.sqrt(3)*fs.X1,'k--')
    plt.hlines(0, -8*np.pi/3, -4*np.pi/3, color = 'k', linestyles = 'dashed')
    plt.hlines(0, 4*np.pi/3, 8*np.pi/3, color = 'k', linestyles = 'dashed')
    #
    plt.scatter(K[0],K[1],c=SFxy+SFzz,cmap = cm.get_cmap('plasma_r'),s=260)
    #High symmetry points
    ccc = 40
    dd = 0.1
    sss = 20
    plt.scatter(0,0,s=sss,color='k')
    plt.text(0+dd,0+dd,r'$\Gamma$',size=ccc)
    plt.scatter(2*np.pi/3,2*np.pi/np.sqrt(3),s=sss,color='k')
    plt.text(2*np.pi/3+2*dd,2*np.pi/np.sqrt(3)-dd,r'$K$',size=ccc)
    plt.scatter(np.pi,np.pi/np.sqrt(3),s=sss,color='k')
    plt.text(np.pi+dd,np.pi/np.sqrt(3)+dd,r'$M$',size=ccc)
    #cbar = plt.colorbar()
    #cbar.ax.tick_params(labelsize=20)
    plt.axis('off')
    plt.xlabel(r'$K_x$',size=15)
    plt.ylabel(r'$K_y$',size=15,rotation='horizontal')
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

    if 1:
        namefile = '../../../../Figs_SB_paper/SSF_LRO_'+ans+'_'+S+'_'+DM+'_'+'{:3.2f}'.format(J2).replace('.','')+'_'+'{:3.2f}'.format(J3).replace('.','')+'.pdf'
        plt.savefig(namefile,bbox_inches='tight')
    else: 
        plt.show()
    
    exit()









###
plt.subplot(2,2,1)
plt.gca().set_aspect('equal')
#
plt.plot(fs.X1,fs.fu1(fs.X1),'k-')
plt.hlines(2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(fs.X2,fs.fu3(fs.X2),'k-')
plt.plot(fs.X1,fs.fd1(fs.X1),'k-')
plt.hlines(-2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(fs.X2,fs.fd3(fs.X2),'k-')
#
plt.plot(fs.X3,fs.Fu1(fs.X3),'k-')
plt.hlines(4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(fs.X4,fs.Fu3(fs.X4),'k-')
plt.plot(fs.X3,fs.Fd1(fs.X3),'k-')
plt.hlines(-4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(fs.X4,fs.Fd3(fs.X4),'k-')
#

plt.scatter(K[0],K[1],c=SFxy,cmap = cm.get_cmap('plasma_r'))
plt.colorbar()
###
plt.subplot(2,2,2)
plt.gca().set_aspect('equal')
#
plt.plot(fs.X1,fs.fu1(fs.X1),'k-')
plt.hlines(2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(fs.X2,fs.fu3(fs.X2),'k-')
plt.plot(fs.X1,fs.fd1(fs.X1),'k-')
plt.hlines(-2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(fs.X2,fs.fd3(fs.X2),'k-')
#
plt.plot(fs.X3,fs.Fu1(fs.X3),'k-')
plt.hlines(4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(fs.X4,fs.Fu3(fs.X4),'k-')
plt.plot(fs.X3,fs.Fd1(fs.X3),'k-')
plt.hlines(-4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(fs.X4,fs.Fd3(fs.X4),'k-')
#

plt.scatter(K[0],K[1],c=SFzz,cmap = cm.get_cmap('plasma_r'))
plt.colorbar()
###
plt.subplot(2,2,3)
plt.gca().set_aspect('equal')
#
plt.plot(fs.X1,fs.fu1(fs.X1),'k-')
plt.hlines(2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(fs.X2,fs.fu3(fs.X2),'k-')
plt.plot(fs.X1,fs.fd1(fs.X1),'k-')
plt.hlines(-2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(fs.X2,fs.fd3(fs.X2),'k-')
#
plt.plot(fs.X3,fs.Fu1(fs.X3),'k-')
plt.hlines(4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(fs.X4,fs.Fu3(fs.X4),'k-')
plt.plot(fs.X3,fs.Fd1(fs.X3),'k-')
plt.hlines(-4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(fs.X4,fs.Fd3(fs.X4),'k-')
#

plt.scatter(K[0],K[1],c=SFxy+SFzz,cmap = cm.get_cmap('plasma_r'))
plt.colorbar()

plt.show()


exit()


############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################
#figManager = plt.get_current_fig_manager()
#figManager.window.showMaximized()
title = 'coool'#ans+'_'+DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+')'
plt.suptitle(title)
#plt.axis('off')
plt.subplot(2,2,1)
plt.title(title+'--Sxy')
#hexagons
plt.plot(fs.X1,fs.fu1(fs.X1),'k-')
plt.hlines(2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(fs.X2,fs.fu3(fs.X2),'k-')
plt.plot(fs.X1,fs.fd1(fs.X1),'k-')
plt.hlines(-2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(fs.X2,fs.fd3(fs.X2),'k-')

plt.plot(fs.X3,fs.Fu1(fs.X3),'k-')
plt.hlines(4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(fs.X4,fs.Fu3(fs.X4),'k-')
plt.plot(fs.X3,fs.Fd1(fs.X3),'k-')
plt.hlines(-4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(fs.X4,fs.Fd3(fs.X4),'k-')
#
plt.scatter(K[0],K[1],c=SFxy,cmap = cm.get_cmap('plasma_r'))
plt.colorbar()
#
plt.subplot(2,2,2)
plt.title("Szz")
#hexagons
plt.plot(fs.X1,fs.fu1(fs.X1),'k-')
plt.hlines(2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(fs.X2,fs.fu3(fs.X2),'k-')
plt.plot(fs.X1,fs.fd1(fs.X1),'k-')
plt.hlines(-2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(fs.X2,fs.fd3(fs.X2),'k-')

plt.plot(fs.X3,fs.Fu1(fs.X3),'k-')
plt.hlines(4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(fs.X4,fs.Fu3(fs.X4),'k-')
plt.plot(fs.X3,fs.Fd1(fs.X3),'k-')
plt.hlines(-4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(fs.X4,fs.Fd3(fs.X4),'k-')
#
plt.scatter(K[0],K[1],c=SFzz,cmap = cm.get_cmap('plasma_r'))
plt.colorbar()
#
plt.subplot(2,2,3)
plt.title(title+'--S_tot')
#hexagons
plt.plot(fs.X1,fs.fu1(fs.X1),'k-')
plt.hlines(2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(fs.X2,fs.fu3(fs.X2),'k-')
plt.plot(fs.X1,fs.fd1(fs.X1),'k-')
plt.hlines(-2*np.pi/np.sqrt(3), -2*np.pi/3,2*np.pi/3, color = 'k')
plt.plot(fs.X2,fs.fd3(fs.X2),'k-')

plt.plot(fs.X3,fs.Fu1(fs.X3),'k-')
plt.hlines(4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(fs.X4,fs.Fu3(fs.X4),'k-')
plt.plot(fs.X3,fs.Fd1(fs.X3),'k-')
plt.hlines(-4*np.pi/np.sqrt(3), -4*np.pi/3,4*np.pi/3, color = 'k')
plt.plot(fs.X4,fs.Fd3(fs.X4),'k-')
#
plt.scatter(K[0],K[1],c=SFxy+SFzz,cmap = cm.get_cmap('plasma_r'))
plt.colorbar()


plt.show()
