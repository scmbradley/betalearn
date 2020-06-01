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
    ####################
    # Not actually used. I'm not using masked arrays, because they don't behave quite how I want
    # so I'm kind of bodging together a replacement using np.where and n.nan
    ####################
    # def set_mask(self,bools):
    #     assert len(bools) == self.array_size, "Boolean array is the wrong length for set_mask"
    #     masker = np.transpose(np.concatenate((bools,bools)).reshape(2,self.array_size))
    #     self.masker = masker
    ####################

    # Returns a masked array of params
    def mask_array(self,bools):
        masker = np.transpose(np.concatenate((bools,bools)).reshape(2,self.array_size))
        self.array = np.where(masker,self.array,np.nan)
        self.prob_of_heads= np.where(bools,self.prob_of_heads,np.nan)

    ####################
    # This is folded in to mask_array
    ####################
    # def masked_prob_heads(self,bools):
    #     np.where(bools,self.prob_of_heads, np.nan)
    ####################
    
    # Helper functions for prob_of_evidence
    # This one returns the pdf at theta of a a distribution (a pair of parameters)
    def _pdf_at(self,theta,params):
        return stats.beta.pdf(theta,params[0],params[1])

    # This one returns the probability of the evidence at a distribution
    def _prob_evidence_at(self,theta,evidence,params):
        heads, tails = evidence
        size = heads+tails
        return stats.binom.pmf(heads,size,theta)*self._pdf_at(theta,params)

    
    # This function will take evidence in the form of a pair of heads and tails values
    # and output the probability of that evidence for each prior in the array.
    def _prob_of_evidence(self,evidence, likelihood_fn,loop_array):
        assert isinstance(evidence, np.ndarray), "Evidence object is not a numpy ndarray"
        assert evidence.shape[0] == 2, "Evidence array is the wrong shape"
        output = []
        for element in loop_array:
            val = likelihood_fn(evidence,element)
            output.append(val)
        return np.array(output)

    def _likelihood_slow(self,evidence,param):
        val, err = integrate.quad(lambda x: self._prob_evidence_at(x, evidence, param),0,1)
        return val

    def _likelihood_fast(self,evidence,prob):
        heads, tails = evidence
        size = heads+tails
        val = stats.binom.pmf(heads,size,prob)
        return val

    def prob_of_evidence(self,evidence):
        return self._prob_of_evidence(evidence,self._likelihood_slow, self.array)

    def prob_of_evidence_fast(self,evidence):
        return self._prob_of_evidence(evidence, self._likelihood_fast, self.prob_of_heads)
    

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
        probs = updated_array.prob_of_evidence_fast(evidence)
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
        if stubborns != False:
            stub_list = [self.array]
            try:
                stub_max = stubborns[0]
                stub_step = stubborns[1]
            except TypeError as err:
                print("Error creating stubborns.")
                print(err)
                print("stubborns should be a pair of ints setting the max and step for the loop.")
                stub_max = 10
                stub_step = 2
                print("Defaulting to stub_max = {}, stub_step = {}".format(stub_max,stub_step))
                
            for x in range(1,stub_max,stub_step):
                stub_list.append(self.array*x)
                stub_array = np.concatenate(stub_list)
            self.array = np.unique(stub_array,axis=0)


        self.array_size = self.array.shape[0]
        self.prob_of_heads=self.array[:,0]/np.sum(self.array,axis=1)
            



# Create an evidence class that we can iterate over.

class EvidenceStream:
    def __init__(self,true_theta,length,number_samples):
        evarr = stats.binom.rvs(number_samples,true_theta,size=length)
        self.evidence= np.transpose(np.append(evarr,(np.ones(length,dtype=int)*number_samples)-evarr).reshape(2,length))
        self.permuted = np.random.permutation(self.evidence)
        # Words are generated using the function defined below.
        self.evidence_words = self.make_words(self.evidence)
        self.evidence_words_permuted = self.make_words(self.permuted)

        self.evidence_length = length
        # Cumulative sum of evidence for totev update
        self.cumulative = np.cumsum(self.evidence,axis=0)

    # define __getitem__ to allow iterating over the object
    def __getitem__(self,key):
        return self.evidence[key]

    def _evidence_into_words(self,arr):
        """
        Takes a pair of numbers and translates them into a words e.g. "4H4T"
        """
        e = [str(x[0]),"H",str(x[1]),"T"]
        return "".join(e)
    
    def make_words(self,evidence):
        words= ["prior"]
        for x in evidence:
            s=""
            e = [str(x[0]),"H",str(x[1]),"T"]
            words.append(s.join(e))
        return words


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

        # # We should do some wrapping here too.
        # if iter_alpha:
        #     print("Calculating iterative alpha cut")
        #     self.iter_alpha_list = [prior]
        #     round=1
        #     for evidence in evidence_stream:
        #         print("Round", round, "of", self.evidence_length)
        #         round += 1
        #         self.iter_alpha_list.append(self.iter_alpha_list[-1].alpha_cut(evidence, iter_alpha))

        # if iter_alpha_fast:
        #     print("Calculating iterative alpha cut (fast)")
        #     self.iter_alpha_fast_list = [prior]
        #     round=1
        #     for evidence in evidence_stream:
        #         print("Round", round, "of", self.evidence_length)
        #         round += 1
        #         self.iter_alpha_fast_list.append(self.iter_alpha_fast_list[-1].alpha_cut_fast(evidence, iter_alpha_fast))

        # if totev_alpha:
        #     print("Calculating total evidence alpha cut")
        #     self.totev_alpha_list=[prior]
        #     round = 1
        #     for evidence in evidence_stream.cumulative:
        #         print("Round", round, "of", self.evidence_length)
        #         round += 1
        #         self.totev_alpha_list.append(self.totev_alpha_list[0].alpha_cut(evidence, totev_alpha))

        # if totev_alpha_fast:
        #     print("Calculating total evidence alpha cut")
        #     self.totev_alpha_fast_list=[prior]
        #     round = 1
        #     for evidence in evidence_stream.cumulative:
        #         print("Round", round, "of", self.evidence_length)
        #         round += 1
        #         self.totev_alpha_fast_list.append(self.totev_alpha_fast_list[0].alpha_cut_fast(evidence, totev_alpha_fast))

        if iter_alpha:
            self.iter_alpha_list  = self._gen_array_list(
                self.evidence_stream, iter_alpha, self.prior, iterative=True,fast=False)

        if iter_alpha_fast:
            self.iter_alpha_fast_list = self._gen_array_list(
                self.evidence_stream, iter_alpha_fast, self.prior, iterative=True, fast=True)
        if totev_alpha:
            self.totev_alpha_list = self._gen_array_list(
                self.evidence_stream, totev_alpha, self.prior, iterative=False, fast=False)
        if totev_alpha_fast:
            self.totev_alpha_fast_list = self._gen_array_list(
                self.evidence_stream, totev_alpha_fast, self.prior, iterative=False, fast=True)


    def _gen_array_list(self, evidence_stream, alpha, prior, fast=False, iterative=False):
        """ 
        Generates a list of parameter values for updated distributions.

        Required parameters: an evidence_stream, an alpha value, and a prior to start with.
        Defaults to slow total evidence updating.
        """
        # Set the index to -1 (last item of the list) or 0 (first item)
        # depending on whether we are iterative or total evidence updating
        if iterative:
            idx = -1
            stream = self.evidence_stream
            method = "iterative alpha cut, alpha = {}".format(alpha)
        else:
            idx = 0
            stream = self.evidence_stream.cumulative
            method = "total evidence alpha cut, alpha = {}".format(alpha)
        length = self.evidence_length
        the_list = [prior]
        if fast:
            update_fn = lambda x: the_list[x].alpha_cut_fast
            fast_word = "(fast)"
        else:
            update_fn = lambda x: the_list[x].alpha_cut
            fast_word = ""
        round=1
        print("Generating updated priors using {} {}".format(method, fast_word))
        for evidence in stream:
            print("Generating updated priors. Round {} of {}".format(round, length))
            round += 1
            # OK. Look, there's a little bit of currying going on in this next line.
            # I needed to do it this way because when I set update_fn
            # above, it doesn't know yet what idx will be.
            the_list.append(update_fn(idx)(evidence,alpha))
        return the_list


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

    def time_series_totev_alpha(self,idx):
        assert self.totev_alpha != 0, "Error: no totev_alpha array"
        return self._time_series_heads(self.totev_alpha_list,idx)

    # def time_series_totev_alpha_fast(self,idx):
    #     assert self.totev_alpha_fast != 0, "Error: no totev_alpha array"
    #     return self._time_series_heads(self.totev_alpha_fast_list,idx)


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
        axs.text(1.05,0.5,"One", transform=axs.transAxes)
        print(axs.get_xlim())
        axs.set_xticklabels(self.evidence_words,rotation="vertical")
        #    axs.margins(0.2)
        plt.subplots_adjust(bottom=0.15)

    def graph_iter_v_GC(self):
        self._red_grey(self.time_series_iter_alpha,self.time_series_GC)

    def graph_iter_fast_v_GC(self):
        self._red_grey(self.time_series_iter_alpha_fast,self.time_series_GC)

    def graph_totev_v_GC(self):
        self._red_grey(self.time_series_totev_alpha,self.time_series_GC)

    def graph_iter_v_totev(self):
        self._red_grey(self.time_series_totev_alpha,self.time_series_iter_alpha)

    # Two graphs
    # NOTE: permuted label appears on the bottom, call the function accordingly
    def _two_graphs(self,ts_top,ts_bottom,label_permuted=False,top_label="",bottom_label=""):
        fig,axs=plt.subplots(2,1)
        fig.set_tight_layout(True)
        # fig.subplots_adjust(right=0.8)
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
        
        if label_permuted:
            axs[1].set_xticklabels(self.evidence_words_permuted,rotation="vertical")
        else:
            axs[1].set_xticklabels(self.evidence_words,rotation="vertical")

        axs[0].text(1.05,0.5,top_label, transform=axs[0].transAxes)
        axs[1].text(1.05,0.5,bottom_label,transform=axs[1].transAxes)

        axs[0].set_xticklabels(self.evidence_words,rotation="vertical")
        # axs[0].set_title("One",horizontalalignment="right")
        # axs[1].ylabel("Two")
        plt.subplots_adjust(hspace=0.2)
        #plt.savefig("commutativity.pdf")
        print("Figure saved")

    def two_graph_iter_iter_fast(self):
        self._two_graphs(
            self.time_series_iter_alpha,self.time_series_iter_alpha_fast,
            top_label="Iterative alpha cut",bottom_label="Fast iterative alpha cut")

    # def commutativity(self):
    #     self._two_graphs(self.

# TODO:
# two graphs: evidence/evidence permuted labels
# measure P^(H) - P_(H) as a function of "time"
# random prior
# multiple alpha values
# docstrings
        
        
def test():
    return LearningSequence(BetaPrior(4), EvidenceStream(0.3,4,8),iter_alpha_fast=0.5,iter_alpha=0.5)

def graph_test():
    test().two_graph_iter_iter_fast()
    plt.show()
    
    
