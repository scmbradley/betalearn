
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

numlines = 10
eps = 1e-6

mup  = np.arange(0.1,1,0.1)
nup = 1-mup
phi = 8

arr = np.transpose(np.array((mup,nup)))

fig,axs = plt.subplots()
x = np.linspace(0,1,128)
for p in arr:
    y = stats.beta.pdf(x,phi*p[0],phi*p[1])
    axs.plot(x,y,label = r'$\mu^\prime = {:.2f}$, $\nu^\prime={:.2f}$'.format(p[0],p[1]))
axs.axes.get_yaxis().set_visible(False)
axs.set_title(r"Beta distributions for $\phi = {}$".format(phi))
for spine in ["left", "top", "right"]:
    axs.spines[spine].set_visible(False)
axs.xaxis.set_ticks_position('bottom')
axs.set_xlabel("Chance of heads")

axs.legend(loc="best")

plt.savefig("fix-phi.png")


mu = 0.4
nu = 1-mu

phip = np.arange(3,24,3)

fig,axs = plt.subplots()
for p in phip:
    y = stats.beta.pdf(x, p*mu,p*nu)
    axs.plot(x,y, label = r"$\phi = {}$".format(p))
axs.axes.get_yaxis().set_visible(False)
axs.set_title(r"Beta distributions for $\mu' = {}, \nu' = {}$".format(mu,nu))
for spine in ["left", "top", "right"]:
    axs.spines[spine].set_visible(False)
axs.xaxis.set_ticks_position('bottom')
axs.set_xlabel("Chance of heads")

axs.legend(loc="best")

plt.savefig("fix-mu.png")
#plt.show()

plt.close('all')

# We're fixing mu'


H = [2,5,3]
T = [6,3,5]

# we only use the first two of these.

figg,axs = plt.subplots(3,1,figsize=(5,16))
#fig.set_size_inches(10,3)
#,figsize=(8,2)
mu = 0.9
nu=1-mu
for p in phip:
    y = stats.beta.pdf(x, p*mu,p*nu)
    axs[0].plot(x,y, label = r"$\phi = {}$".format(p))
    h = H[0]
    t = T[0]
    n = h+t
    phi_1 = p +n
    mu_1 = (h+p*mu)/phi_1
    nu_1 = 1-mu_1
    z = stats.beta.pdf(x,phi_1*mu_1, phi_1*nu_1)
    axs[1].plot(x,z,label = r"$\phi = {}$".format(phi_1))
    h = H[1]
    t = T[1]
    n= h+t
    phi_2 = phi_1 +n
    mu_2 = (h +p*mu_1)/phi_2
    nu_2 = 1-mu_2
    w = stats.beta.pdf(x,phi_2*mu_2, phi_2*nu_2)
    axs[2].plot(x,w,label = r"$\phi = {}$".format(phi_2))

axs[0].set_title("Prior",x=0.3)
axs[1].set_title("Updated on H={},T={}".format(H[0],T[0]),y=0.95,x=0.3)
axs[2].set_title("Updated on H={},T={}".format(H[1],T[1]),y=0.9,x=0.3)
axs[2].set_xlabel("Chance of heads")
    
for i in [0,1,2]:
    axs[i].axes.get_yaxis().set_visible(False)
    # axs[i].set_title(r"Beta distributions for $\mu' = {}, \nu' = {}$".format(mu,nu))
    for spine in ["left", "top", "right"]:
        axs[i].spines[spine].set_visible(False)
    axs[i].xaxis.set_ticks_position('bottom')
    axs[i].legend(loc="center left")
    # axs[i].axis('square')
    # axs[i].set_aspect(np.diff(axs[i].get_xlim())/np.diff(axs[i].get_ylim()))

fig.tight_layout()

plt.savefig("fix-mu-learn.png")
#plt.close('all')



plt.show()
