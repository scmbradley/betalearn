from scipy import stats
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# For testing purposes:
np.random.seed(seed=42)

class BetaArray:
    def __init__(self,arr):
        assert isinstance(arr,np.ndarray), "Input object is not a numpy array"
        assert arr.shape[1] == 2, "Array is the wrong shape"
        self.array=arr
        self.array_size = self.array.shape[0]
        self.prob_of_heads=self.array[:,0]/np.sum(self.array,axis=1)
        self.masker= False

    def __getitem__(self,key):
        return self.array[key]

    
    # Define some private functions that are helpers for prob_of_evidence

    ####################
    #  Functions need to take individual params,
    #  because integrate won't work over an array
    ####################
    # # This one returns an array of pdf values for theta
    # def pdf_at(self,theta):
    #     return stats.beta.pdf(theta,self.array[:,0],self.array[:,1])
    #
    # #This one returns an array of pdf values for probability of evidence at theta
    # def prob_evidence_at(self,theta,evidence):
    #     heads, tails = evidence
    #     size = heads + tails
    #     return stats.binom.pmf(heads, size, theta)* self.pdf_at(theta)
    ####################

    
    # Take an array of booleans, check it's the right length, manipulate it into an array of pairs
    # Make the array of params a masked array, add mask
    def set_mask(self,bools):
        assert len(bools) == self.array_size, "Boolean array is the wrong length for set_mask"
        masker = np.transpose(np.concatenate((bools,bools)).reshape(2,self.array_size))
        self.masker = masker

    # Returns a masked array of params
    def mask_array(self,bools):
        masker = np.transpose(np.concatenate((bools,bools)).reshape(2,self.array_size))
        self.array = np.where(masker,self.array,np.nan)
        self.prob_of_heads= np.where(bools,self.prob_of_heads,np.nan)

    def masked_prob_heads(self,bools):
        np.where(bools,self.prob_of_heads, np.nan)
        
    # Helper functions for prob_of_evidence
    # This one returns the pdf at theta of a a distribution (a pair of parameters)
    def pdf_at(self,theta,params):
        return stats.beta.pdf(theta,params[0],params[1])

    # This one returns the probability of the evidence at a distribution
    def prob_evidence_at(self,theta,evidence,params):
        heads, tails = evidence
        size = heads+tails
        return stats.binom.pmf(heads,size,theta)*self.pdf_at(theta,params)

    
    # This function will take evidence in the form of a pair of heads and tails values
    # and output the probability of that evidence for each prior in the array.
    
    def prob_of_evidence(self,evidence):
        assert isinstance(evidence, np.ndarray), "Evidence object is not a numpy ndarray"
        assert evidence.shape[0] == 2, "Evidence array is the wrong shape"
        output = []
        for prior in self.array:
            val, err = integrate.quad(lambda x: self.prob_evidence_at(x, evidence, prior),0,1)
            output.append(val)
        return np.array(output)

    # This should probably be tidied up so we have a helper function that wraps
    # either the above or the below function that returns val.
    def prob_evidence_fast(self,evidence):
        assert isinstance(evidence, np.ndarray), "Evidence object is not a numpy ndarray"
        assert evidence.shape[0] == 2, "Evidence array is the wrong shape"
        heads, tails = evidence
        size = heads+tails
        output = []
        for prob in self.prob_of_heads:
            #val, err = integrate.quad(lambda x: self.prob_evidence_at(x, evidence, prior),0,1)
            val = stats.binom.pmf(heads,size,prob)
            output.append(val)
        return np.array(output)


    # GC update works by simply adding the params from the evidence to the array
    def GC_update(self,evidence):
        assert isinstance(evidence, np.ndarray), "Evidence object is not a numpy ndarray"
        assert evidence.shape[0] == 2, "Evidence array is the wrong shape"
        return BetaArray(self.array.data + evidence)

    def alpha_cut(self,evidence,alpha):
        updated_array = self.GC_update(evidence)
        probs = updated_array.prob_of_evidence(evidence)
        bools = probs >= alpha*np.nanmax(probs)
        updated_array.mask_array(bools)
        return updated_array

    # Likewise, this should involve a wrapper.
    def alpha_cut_fast(self,evidence,alpha):
        updated_array = self.GC_update(evidence)
        probs = updated_array.prob_evidence_fast(evidence)
        bools = probs >= alpha*np.nanmax(probs)
        updated_array.mask_array(bools)
        return updated_array

# The BetaPrior is a subclass of the BetaArray: the one you start with
class BetaPrior(BetaArray):
    def __init__(self,size,stubborns=False, fillers=False):
        # self.size = size
        # create an array pairs (i,j) for i,j <= size
        index_array = np.transpose(np.indices((size,size)) + 1).reshape(size**2,2)
        self.array = index_array
        
        # This leaves the (0,1/size) range empty (and likewise at the other end.
        # If we want to fill in this range set fillers=True
        if fillers:
            filler_vals = np.arange(size,10*size,size)
            filler_ones = np.ones(len(filler_vals),dtype=int)
            top_range = np.transpose(np.array((filler_vals,filler_ones)))
            bottom_range = np.transpose(np.array((filler_ones,filler_vals)))
            self.array = np.concatenate((self.array,top_range,bottom_range))
        
        # We still don't have any really stubborn beta priors in here.
        # If we want some priors which converge really slowly, set stubborns=True
        if stubborns:
            stub_array = []
            for x in np.arange(2,101,3):
                stub_array.append(x*np.array([4,1]))
            self.array = np.concatenate((self.array,stub_array))

        self.array_size = self.array.shape[0]
        self.prob_of_heads=self.array[:,0]/np.sum(self.array,axis=1)
            



# Create an evidence class that we can iterate over.

class EvidenceStream:
    def __init__(self,true_theta,length,number_samples):
        evarr = stats.binom.rvs(number_samples,true_theta,size=length)
        self.evidence= np.transpose(np.append(evarr,(np.ones(length,dtype=int)*number_samples)-evarr).reshape(2,length))
        self.evidence_words= ["prior"]
        for x in self.evidence:
            s=""
            e = [str(x[0]),"H",str(x[1]),"T"]
            self.evidence_words.append(s.join(e))
            
        # This returns an iterable already
        self.permuted = np.random.permutation(self.evidence)
        self.evidence_words_permuted = ["prior"]
        for x in self.permuted:
            s=""
            e = [str(x[0]),"H",str(x[1]),"T"]
            self.evidence_words_permuted.append(s.join(e))
        self.evidence_length = length
        # Cumulative sum of evidence for totev update
        self.cumulative = np.cumsum(self.evidence,axis=0)

    # define __getitem__ to allow iterating over the object
    def __getitem__(self,key):
        return self.evidence[key]

    
            

# LearningSequence produces a list of BetaArrays produced by successive learning.
# Several learning outputs can be produced.
# GC (default)
# total evidence alpha cut (for specified alpha)
# iterative alphat cut (ditto)
# approximate (i.e. fast) alpha cuts for above (produced using binom for the mean of theta,
# rather than by integration)
# set the alpha values to zero (meaning, False). If they're set to a value,
# Run integrations
class LearningSequence:
    def __init__(self,prior,evidence_stream,
                 iter_alpha = 0, iter_alpha_fast=0,
                 totev_alpha = 0, totev_alpha_fast = 0,
                 permuted_evidence=False):
        assert isinstance(prior,BetaArray), "prior is not a BetaArray"
        assert isinstance(evidence_stream,EvidenceStream), "evidence is not an EvidenceStream"
#        assert evidence_stream.shape[1] == 2, "evidence stream is wrong shape"
        self.prior = prior
        self.evidence_stream = evidence_stream
        self.iter_alpha = iter_alpha
        self.iter_alpha_fast = iter_alpha_fast
        self.totev_alpha = totev_alpha
        self.totev_alpha_fast = totev_alpha_fast
        self.evidence_length = evidence_stream.evidence_length
        self.evidence_words = evidence_stream.evidence_words
        # Generate GC update
        self.GC_list = [prior]
        for evidence in evidence_stream:
            self.GC_list.append(self.GC_list[-1].GC_update(evidence))
            
        # We should do some wrapping here too.
        if iter_alpha:
            print("Calculating iterative alpha cut")
            self.iter_alpha_list = [prior]
            round=1
            for evidence in evidence_stream:
                print("Round", round, "of", self.evidence_length)
                round += 1
                self.iter_alpha_list.append(self.iter_alpha_list[-1].alpha_cut(evidence, iter_alpha))

        if iter_alpha_fast:
            print("Calculating iterative alpha cut (fast)")
            self.iter_alpha_fast_list = [prior]
            round=1
            for evidence in evidence_stream:
                print("Round", round, "of", self.evidence_length)
                round += 1
                self.iter_alpha_fast_list.append(self.iter_alpha_fast_list[-1].alpha_cut_fast(evidence, iter_alpha_fast))

        if totev_alpha:
            print("Calculating total evidence alpha cut")
            self.totev_alpha_list=[prior]
            round = 1
            for evidence in evidence_stream.cumulative:
                print("Round", round, "of", self.evidence_length)
                round += 1
                self.totev_alpha_list.append(self.totev_alpha_list[0].alpha_cut(evidence, totev_alpha))

        
                

    # Helper function to create time series of probs of heads
    def _time_series_heads(self,arr,idx):
        ts = []
        for a in arr:
            ts.append(a.prob_of_heads[idx])
        return np.array(ts)

    def time_series_GC(self,idx):
        return self._time_series_heads(self.GC_list,idx)
    
    def time_series_iter_alpha(self,idx):
        assert self.iter_alpha != 0, "Error: no iter_alpha array"
        return self._time_series_heads(self.iter_alpha_list,idx)

    def time_series_iter_alpha_fast(self,idx):
        assert self.iter_alpha_fast != 0, "Error: no iter_alpha array"
        return self._time_series_heads(self.iter_alpha_fast_list,idx)
        

    # This TS doesn't work, because the mask isn't getting applied to the prob of heads.
    def time_series_totev_alpha(self,idx):
        assert self.totev_alpha != 0, "Error: no totev_alpha array"
        return self._time_series_heads(self.totev_alpha_list,idx)


    # Graphing as a method of LearningSequence
    def _red_grey(self,ts_red,ts_grey):
        fig,axs=plt.subplots()
        # +1 here for the prior
        x = np.arange(0,self.evidence_length+1)
        for i in np.arange(self.prior.array_size):
            y = ts_grey(i)
            axs.plot(x,y,color='0.4',linewidth=1)
        for i in np.arange(self.prior.array_size):
            y = ts_red(i)
            axs.plot(x,y,color='r',linewidth=1,marker=".")
        axs.set_xticks(np.arange(0,len(self.evidence_words)))
        axs.set_xticklabels(self.evidence_words,rotation="vertical")
        #    axs.margins(0.2)
        plt.subplots_adjust(bottom=0.15)

    def graph_iter_v_GC(self):
        self._red_grey(self.time_series_iter_alpha,self.time_series_GC)

    def graph_totev_v_GC(self):
        self._red_grey(self.time_series_totev_alpha,self.time_series_GC)

    def graph_iter_v_totev(self):
        self._red_grey(self.time_series_totev_alpha,self.time_series_iter_alpha)

    # Two graphs
    def _two_graphs(self,ts_top,ts_bottom):
        fig,axs=plt.subplots(2,1)
        x = np.arange(0,self.evidence_length+1)
        for i in np.arange(self.prior.array_size):
            y = ts_bottom(i)
            axs[0].plot(x,y,color='0.4', linewidth=1)
        for i in np.arange(self.prior.array_size):
            y = ts_top(i)
            axs[1].plot(x,y,color='0.4', linewidth=1)
        for i in np.arange(self.prior.array_size):
            y = ts_top(i)
            axs[0].plot(x,y,color='r', linewidth=1,marker=".")
        for i in np.arange(self.prior.array_size):
            y = ts_bottom(i)
            axs[1].plot(x,y,color='b', linewidth=1,marker=".")
        axs[0].tick_params(axis='x', labelbottom=False, labeltop=True)
        axs[0].set_xticks(np.arange(0,len(self.evidence_words)))
        axs[1].set_xticks(np.arange(0,len(self.evidence_words)))
        axs[1].set_xticklabels(self.evidence_words,rotation="vertical")
        axs[0].set_xticklabels(self.evidence_words,rotation="vertical")
        plt.subplots_adjust(hspace=0.2)
        #plt.savefig("commutativity.pdf")
        print("Figure saved")

    def two_graph_iter_iter_fast(self):
        self._two_graphs(self.time_series_iter_alpha,self.time_series_iter_alpha_fast)


# TODO:
# two graphs: evidence/evidence permuted labels
# measure P^(H) - P_(H) as a function of "time"
# random prior
# multiple alpha values
# approx: totev
        
        
def test():
    return LearningSequence(BetaPrior(5), EvidenceStream(0.3,4,8), iter_alpha=0.5,iter_alpha_fast=0.5)
    





