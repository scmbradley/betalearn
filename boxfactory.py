from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


# This is based on EvidenceStream from betalearn,
# but isn't a subclass, because I basically redefine everything,
# so what's the point. In future, I'll make EvidenceStream
# more flexible to cover this use case.


# Plan: take 2 or 4 prob values, run probs for length and face propositions.

class BoxEvidence():
    def __init__(length=8,size=8,true_probs=[0.2,0.8]):
        self.length=length
        self.size=size
        self.true_probs=true_probs
        evarr = stats.multinomial.rvs(size,true_probs,size=length)
        self.evidence = evarr
        self.cumulative = np.cumsum(evarr,axis=0)
        self.given_len = len(true_probs)

    if self.given_len == 2:
        self.other_len = 4
    elif self.given_len==4:
        self.other_len = 2

    
