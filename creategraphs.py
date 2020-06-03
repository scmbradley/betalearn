import betalearn as bl

from numpy.random import seed 
seed(seed=42)

# First up, we generate some graphs to demonstrate that the fast methods
# are a reasonable approximation.

ls = bl.LearningSequence(
    bl.BetaPrior(8,stubborns=[100,30],fillers=True),
    bl.EvidenceStream(0.3,16,8),
    totev_alpha=0.1,
    totev_alpha_fast=0.1
    permuted_evidence=True)

ls.two_graph_iter_iter_fast()
plt.savefig("two_graph_fast")
print("two graph discrepancy fast saved")
plt.clf()




# ls.graph_iter_v_GC()
# plt.savefig("alpha_v_gc.pdf")
# print("iter_v_GC saved")
# plt.clf()

# ls.graph_iter_v_totev()
# plt.savefig("iter_totev.pdf")
# print("iter_v_totev saved")
# plt.clf()

# ls.commutativity(fast=False)
# plt.savefig("commutativity.pdf")
# print("commutativity saved")
# plt.clf()

# ls.all_spread(root_n = True)
# plt.savefig("all_spread.pdf")
# print("all_spread saved")
# plt.clf()
