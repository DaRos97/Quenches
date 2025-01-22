import numpy as np
import matplotlib.pyplot as plt


def BZgrid(nkx,nky,UC):
    """Compute BZ coordinates of points."""
    d = 2*np.pi/nky
    gridk = np.zeros((nkx,nky,2))
    if UC == 2:
        for i1 in range(nkx):
            for i2 in range(nky):
                gx = -np.pi + d*(1+i1+i2//2)
                gy = d*((1+i2)//2-i1)
                gridk[i1,i2] = np.array([gx,gy]) #- np.array([0,d/2])
    elif UC == 1:
        for i1 in range(nkx):
            for i2 in range(nky):
                gridk[i1,i2,0] = -np.pi + d*(1+i1)
                gridk[i1,i2,1] = -np.pi + d*(1+i2)
    return gridk

def minus_k_index(gridk,b1,b2,UC):
    """Get index of -k point in BZ, considering the periodicity."""
    nkx,nky = gridk.shape[:2]
    ind_mk = np.zeros((nkx,nky,2),dtype=int)
    for ikx in range(nkx):
        for iky in range(nky):
            if np.max(abs(gridk[ikx,iky]-np.array([0,np.pi])))<1e-8:   #top point of square
                ind_mk[ikx,iky] = np.array([ikx,iky])
                continue
            if np.max(abs(gridk[ikx,iky]-np.array([np.pi,np.pi])))<1e-8:   #top-right point of square
                ind_mk[ikx,iky] = np.array([ikx,iky])
                continue
            indexes = np.argwhere(np.all(abs(gridk+gridk[ikx,iky])<1e-7,axis=2))
            if len(indexes)>0:
                ind_mk[ikx,iky] = indexes[0]
            else:
                if UC == 2:
                    new_k = -gridk[ikx,iky]+b1 if gridk[ikx,iky,0]>0 else -gridk[ikx,iky]+b2-b1
                elif UC == 1:
                    new_k = -gridk[ikx,iky]+b2 if abs(gridk[ikx,iky,1]-np.pi)<1e-8 else -gridk[ikx,iky]+b1
                indexes = np.argwhere(np.all(abs(gridk-new_k)<1e-7,axis=2))
                ind_mk[ikx,iky] = indexes[0]
    return ind_mk

def plot_gridBZ(ax,UC):
    """Plot BZ axes and borders."""
    ax.axis('off')
    ax.set_aspect('equal')
    f = 1.2
    v = 0.15
    ax.arrow(-np.pi*f,0,2*np.pi*f,0,color='k',head_width=0.1)
    ax.arrow(0,-np.pi*f,0,2*np.pi*f,color='k',head_width=0.1)
    ax.text(np.pi*f,v,r'$k_x$',size=20)
    ax.text(v,np.pi*f,r'$k_y$',size=20)
    if UC == 2:
        ax.plot([-np.pi,0],[0,np.pi],color='orange',lw=1)
        ax.plot([0,np.pi],[np.pi,0],color='orange',lw=1)
        ax.plot([np.pi,0],[0,-np.pi],color='orange',lw=1)
        ax.plot([0,-np.pi],[-np.pi,0],color='orange',lw=1)
    elif UC ==1:
        ax.plot([-np.pi,-np.pi],[-np.pi,np.pi],color='g',lw=1)
        ax.plot([-np.pi,np.pi],[np.pi,np.pi],color='g',lw=1)
        ax.plot([np.pi,np.pi],[-np.pi,np.pi],color='g',lw=1)
        ax.plot([-np.pi,np.pi],[-np.pi,-np.pi],color='g',lw=1)

def plot_BZ(ax,gridk,UC):
    """Plot BZ."""
    plot_gridBZ(ax,UC)
    ax.scatter(gridk[:,:,0],gridk[:,:,1])
    ax.set_title("Brillouin zone",size=20)

def plot_real_space(ax,gridk,a1,a2,UC):
    """Plot lattice points in real space."""
    nkx, nky = gridk.shape[:2]
    L = nky
    ax.axis('off')
    ax.set_aspect('equal')
    d = -0.5    #axes
    ax.arrow(d,d,0,0.8,color='k',head_width=0.05)
    ax.arrow(d,d,0.8,0,color='k',head_width=0.05)
    ax.text(d+0.8,d+0.1,r'$x$',size=20)
    ax.text(d+0.1,d+0.8,r'$y$',size=20)
    for i in range(L):  #grid
        ax.plot([i,i],[0,L-1],color='k',lw=0.2,zorder=0)
        ax.plot([0,L-1],[i,i],color='k',lw=0.2,zorder=0)
    for i1 in range(nkx):   #points
        for i2 in range(nky):
            vA = i1*a1 + i2*a2
            vB = i1*a1 + i2*a2 + np.array([1,0])
            if vA[0]<0:
                vA += L//2*a1
            if vB[0]<0:
                vB += L//2*a1
            ax.scatter(vA[0],vA[1],color='k',marker='o',s=70)
            if UC == 2:
                ax.scatter(vB[0],vB[1],color='r',marker='o',s=70)
    ax.set_title("Real space",size=20)

def plot_opposite_BZ(k,mk,UC):
    """Plot points in BZ with their negative counterpoint."""
    fig = plt.figure()
    ax = fig.add_subplot()
    plot_gridBZ(ax,UC)
    ax.scatter(k[0],k[1],c='r')
    ax.scatter(mk[0],mk[1],c='b')
    plt.show()


