
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

axs.legend(loc="best")

plt.savefig("fix-mu.png")
    
plt.close('all')


#plt.show()
