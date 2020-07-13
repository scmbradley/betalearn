import betalearn as bl
print(bl)

from numpy.random import seed 
seed(seed=42)

from time import monotonic as tmr
import matplotlib.pyplot as plt

start = tmr()

# First up, we generate some graphs to demonstrate that the fast methods
# are a reasonable approximation.

print("Running smaller slower simulation")
ls = bl.LearningSequence(
    bl.BetaPrior(8,fillers=True),
    bl.EvidenceStream(0.3,16,8),
    totev_alpha=0.1,
    totev_alpha_fast=0.1)

ls.two_graph_totev_totev_fast()
plt.savefig("two-graph-fast.png")
print("two graph discrepancy fast saved")
plt.clf()

ls.discrepancy()
plt.savefig("discrepancy.png")
print("discrepancy graph saved")
plt.clf()

ls.plot_dists(7)
plt.savefig("plots-dists.png")
print("plot distributions saved")
plt.clf()

# Then we move on to a longer and larger simulation using only the fast simulations.

print("Running bigger faster simulation")
bigg = bl.LearningSequence(
    bl.BetaPrior(8,fillers=True,stubborns=[20,5]),
    bl.EvidenceStream(0.3,32,4),
    totev_alpha_fast= 0.01,
    iter_alpha_fast = 0.01,
    permuted_evidence_fast= True)

bigg.commutativity(fast=True)
plt.savefig("commutativity.png")
print("commutativity graph saved")
plt.clf()

bigg.all_spread(root_n=True)
plt.savefig("spread.png")
print("spread graph saved")
plt.clf()

bigg.graph_iter_fast_v_GC()
plt.savefig("iter-v-GC.png")
print("iter v GC saved")
plt.clf()

bigg.graph_totev_fast_v_GC()
plt.savefig("totev-v-GC.png")
print("totev v GC saved")
plt.clf()
    
duration = tmr() - start
print("time elapsed in seconds:", duration)

# ls.graph_iter_v_GC()
# plt.savefig("alpha_v_gc.png")
# print("iter_v_GC saved")
# plt.clf()

# ls.graph_iter_v_totev()
# plt.savefig("iter_totev.png")
# print("iter_v_totev saved")
# plt.clf()

# ls.commutativity(fast=False)
# plt.savefig("commutativity.png")
# print("commutativity saved")
# plt.clf()

# ls.all_spread(root_n = True)
# plt.savefig("all_spread.png")
# print("all_spread saved")
# plt.clf()
