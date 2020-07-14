from scipy import stats
from scipy.special import binom
import numpy as np
import matplotlib.pyplot as plt


def f(x,y,h,n):
    return stats.betabinom.pmf(h,n,y,x)

maxmu = 50
steps= 75

# H = [1,2,1]
# T = [7,6,7]
# H = [4,4,4]
# T = [4,4,4]

H = [2,5,3]
T = [6,3,5]
colrs = 30

fig,axs = plt.subplots(2,len(H))

mask = np.zeros((steps,steps),dtype=bool)

for i in range(len(axs[0,:])):
    h = H[i]
    t = T[i]
    ch = np.sum(H[:i])
    ct = np.sum(T[:i])
    n = h+t
    mu = np.linspace(1+ch,maxmu+ch,steps)
    nu = np.linspace(1+ct,maxmu+ct,steps)
    X,Y = np.meshgrid(mu,nu)
    
    z= f(X,Y,h,n)
    # Z = z/np.max(z)
    Z = np.ma.array(z/np.max(z))
    if i >0:
        h_old = H[i-1]
        t_old = T[i-1]
        n_old = h_old+t_old
        z_old = f(X,Y,h_old,n_old)
        Z_old = z_old/np.max(z_old)
        new_mask = Z_old < 0.1
        mask = mask | new_mask
        Z.mask = mask
    
    l = mu*(h/t)
    true = mu*(4/6)
    ctp = axs[0,i].contourf(X,Y,Z,colrs,cmap='viridis')
    axs[0,i].plot(mu,l)
    axs[0,i].plot(mu,true)
    axs[0,i].set_ylim(bottom=1+ct,top=maxmu+ct)
    axs[0,i].set_xlim(left = 1+ch,right = maxmu+ch)
    axs[0,i].title.set_text("H = {}, T = {}".format(h,t))

axs[0,len(H)-1].set_ylabel("Iterative update")
axs[0,len(H)-1].yaxis.set_label_position("right")

for i in range(len(axs[1,:])):
    h = np.sum(H[:i+1])
    t = np.sum(T[:i+1])
    n = h+t
    mu = np.linspace(1,maxmu,steps)
    nu = np.linspace(1,maxmu,steps)
    X,Y = np.meshgrid(mu,nu)
    z= f(X,Y,h,n)
    Z = z/np.max(z)
    l = mu*(h/t)
    true = mu*(4/6)
    axs[1,i].contourf(X,Y,Z,colrs,cmap='viridis')
    axs[1,i].plot(mu,l)
    axs[1,i].plot(mu,true)
    axs[1,i].set_ylim(bottom=1,top=maxmu)
    axs[1,i].set_xlim(left = 1,right = maxmu)

    axs[1,i].title.set_text("H = {}, T = {}".format(h,t))

axs[1,len(H)-1].set_ylabel("Total evidence update")
axs[1,len(H)-1].yaxis.set_label_position("right")

plt.tight_layout()
fig.colorbar(ctp,ax=axs,orientation = 'horizontal')
#plt.savefig("contour.png")
plt.show()
