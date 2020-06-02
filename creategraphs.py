import betalearn as bl

from numpy.random import seed 
seed(seed=42)

ls = bl.LearningSequence(
    bl.BetaPrior(1,randoms=True,stubborns=True,fillers=True),
    bl.EvidenceStream(0.3,32,8),
    iter_alpha = 0.001,
    totev_alpha=0.001,
    permuted_evidence=True)


ls.graph_iter_v_GC()
plt.savefig("alpha_v_gc.pdf")
print("iter_v_GC saved")

ls.graph_iter_v_totev()
plt.savefig("iter_totev.pdf")
print("iter_v_totev saved")

ls.commutativity(fast=False)
plt.savefig("commutativity.pdf")
print("commutativity saved")
