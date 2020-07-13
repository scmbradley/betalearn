from scipy import stats
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# For testing purposes:


class BetaArray:
    """
    Main class for the array of beta distributions

    Takes an array as an argument, and defines a bunch of ways of manipulating it.
    Normally, you'd generate BetaArray using BetaPrior subclass.
    """
    def __init__(self,arr):
        assert isinstance(arr,np.ndarray), "Input object is not a numpy array"
        assert arr.shape[1] == 2, "Array is the wrong shape"
        self.array=arr
        self.array_size = self.array.shape[0]
        self.prob_of_heads=self.array[:,0]/np.sum(self.array,axis=1)
        self.masker= False
        self._set_spread()

    def __getitem__(self,key):
        return self.array[key]

    def _set_spread(self):
        self.spread = np.nanmax(self.prob_of_heads) - np.nanmin(self.prob_of_heads)
        
    # Returns a masked array of params
    def mask_array(self,bools):
        masker = np.transpose(np.concatenate((bools,bools)).reshape(2,self.array_size))
        self.array = np.where(masker,self.array,np.nan)
        self.prob_of_heads= np.where(bools,self.prob_of_heads,np.nan)
        self._set_spread()
    
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
        probs = self.prob_of_evidence(evidence)
        bools = probs >= alpha*np.nanmax(probs)
        updated_array.mask_array(bools)
        return updated_array, probs

    # Likewise, this should involve a wrapper.
    def alpha_cut_fast(self,evidence,alpha):
        updated_array = self.GC_update(evidence)
        probs = self.prob_of_evidence_fast(evidence)
        bools = probs >= alpha*np.nanmax(probs)
        updated_array.mask_array(bools)
        return updated_array , probs

# The BetaPrior is a subclass of the BetaArray: the one you start with
class BetaPrior(BetaArray):
    """
    Generates a BetaArray object with a range of values for mu and nu.
    
    size: determines how many distributions to generate. 
    stubborns: adds additional distributions that are slower to converge.
    should be set to a pair of ints, for the max and the step.
    fillers: boolean that fills in the (0,1/size) and (1/1-size,1) ranges.
    randoms: pair of values setting the size and maximum for random distributions
    """
    def __init__(self,size,
                 stubborns=False, fillers=False,
                 randoms=False):
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

        if randoms != False:
            try:
                randoms_size = randoms[0]
                randoms_max = randoms[1]
            except TypeError as err:
                print("Error creating randoms")
                print(err)
                print("Randoms should be a pair of ints setting the size and max for random")
                randoms_size=50
                randoms_max=20
                print("Defaulting to randoms_size = {}, and randoms_max = {}".format(
                    randoms_size,randoms_max))
            rands= np.random.randint(1,high=randoms_max,size=2*randoms_size)
            rands.shape= [randoms_size,2]
            self.array = np.unique(np.concatenate((self.array,rands)),axis=0)

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
        self._set_spread()
            



# Create an evidence class that we can iterate over.

class EvidenceStream:
    """
    Generates a stream of evidence to be learned by a BetaPrior in a LearningSequence

    true_theta: the chance of heads
    length: how many instances of evidence
    number_samples: how big each instance of evidence is.

    So: you get length pieces of evidence each of which tell you how number_samples flips landed.
    """
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
    """
    Produces a sequence of BetaArrays, using an EvidenceStream, and switches for various updates.

    prior: a BetaArray object that is the prior for the update
    evidence_stream: an EvidenceStream object that yields the evidence for learning
    iter_alpha, iter_alpha_fast, totev_alpha, totev_alpha_fast should be either False, or in (0,1)
    permuted_evidence, permuted_evidence_fast are booleans for whether to produce permuted evidence time series
    """
    def __init__(self,prior,evidence_stream,
                 iter_alpha = 0, iter_alpha_fast=0,
                 totev_alpha = 0, totev_alpha_fast = 0,
                 permuted_evidence=False,
                 permuted_evidence_fast=False):
        assert isinstance(prior,BetaArray), "prior is not a BetaArray"
        assert isinstance(evidence_stream,EvidenceStream), "evidence is not an EvidenceStream"
#        assert evidence_stream.shape[1] == 2, "evidence stream is wrong shape"
        self.prior = prior
        self.evidence_stream = evidence_stream
        self.evidence_stream_permuted = evidence_stream.permuted
        self.iter_alpha = iter_alpha
        self.iter_alpha_fast = iter_alpha_fast
        self.totev_alpha = totev_alpha
        self.totev_alpha_fast = totev_alpha_fast
        self.evidence_length = evidence_stream.evidence_length
        self.evidence_words = evidence_stream.evidence_words
        self.evidence_words_permuted = evidence_stream.evidence_words_permuted
        # This empty list gets populated as the spread ts get created
        self.existing_spread_ts = []
        # Generate GC update
        self.GC_list = [prior]
        for evidence in evidence_stream:
            self.GC_list.append(self.GC_list[-1].GC_update(evidence))
        self.GC_spread_ts = self._ts_spread(self.GC_list, name="GC")

        if iter_alpha:
            self.iter_alpha_list, self.iter_alpha_lik_list = self._gen_array_list(
                self.evidence_stream, iter_alpha, self.prior, iterative=True,fast=False)
            if permuted_evidence:
                self.iter_alpha_perm_list, self.iter_alpha_perm_lik_list = self._gen_array_list(
                    self.evidence_stream_permuted, iter_alpha, self.prior, iterative=True, fast=False)
            self.iter_alpha_spread_ts = self._ts_spread(self.iter_alpha_list,name="Iterative")

        if iter_alpha_fast:
            self.iter_alpha_fast_list, self.iter_alpha_fast_lik_list = self._gen_array_list(
                self.evidence_stream, iter_alpha_fast, self.prior, iterative=True, fast=True)
            if permuted_evidence_fast:
                self.iter_alpha_fast_perm_list, self.iter_alpha_fast_perm_lik_list = self._gen_array_list(
                    self.evidence_stream_permuted, iter_alpha_fast,
                    self.prior, iterative=True, fast=True)
            self.iter_alpha_fast_spread_ts = self._ts_spread(self.iter_alpha_fast_list,name = "Iterative  (fast)")

        try:
            disc = np.abs(self.iter_alpha_lik_list - self.iter_alpha_fast_lik_list)
            self.iter_alpha_disc_list = np.nanmean(disc, axis=1)
        except AttributeError as err:
            print("Couldn't generate iter discrepancy array: missing information")
            print(err)
        # Permuted evidence for iter only, since totev is obviously commutative.

        if totev_alpha:
            self.totev_alpha_list, self.totev_alpha_lik_list = self._gen_array_list(
                self.evidence_stream, totev_alpha, self.prior, iterative=False, fast=False)
            self.totev_alpha_spread_ts = self._ts_spread(self.totev_alpha_list,name = "Total evidence")

        if totev_alpha_fast:
            self.totev_alpha_fast_list, self.totev_alpha_fast_lik_list = self._gen_array_list(
                self.evidence_stream, totev_alpha_fast, self.prior, iterative=False, fast=True)
            self.totev_alpha_fast_spread_ts = self._ts_spread(self.totev_alpha_fast_list,name= "Total evidence (fast)")
            
        try:
            disc = np.abs(self.totev_alpha_lik_list - self.totev_alpha_fast_lik_list)
            self.totev_alpha_disc_list = np.nanmean(disc, axis=1)
        except AttributeError as err:
            print("Couldn't generate totev discrepancy array: missing information")
            print(err)
        # make a wrapper to allow iter discrepancy, and max rather than mean disc.
        

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
            stream = evidence_stream
            method = "iterative alpha cut, alpha = {}".format(alpha)
        else:
            idx = 0
            stream = evidence_stream.cumulative
            method = "total evidence alpha cut, alpha = {}".format(alpha)
        length = self.evidence_length
        arr_list = [prior]
        lik_list = []
        if fast:
            update_fn = lambda x: arr_list[x].alpha_cut_fast
            fast_word = "(fast)"
        else:
            update_fn = lambda x: arr_list[x].alpha_cut
            fast_word = ""
        round=1
        # Currently doesn't report whether it's using permuted evidence or not.
        print("Generating updated priors using {} {}".format(method, fast_word))
        for evidence in stream:
            print("Generating updated priors. Round {} of {}".format(round, length))
            print(evidence)
            round += 1
            # OK. Look, there's a little bit of currying going on in this next line.
            # I needed to do it this way because when I set update_fn
            # above, it doesn't know yet what idx will be.
            arr, lik = update_fn(idx)(evidence,alpha)
            lik_list.append(lik)
            arr_list.append(arr)
        return arr_list , np.array(lik_list)


    # Helper function to create time series of probs of heads
    def _ts_heads(self,arr,idx):
        ts = []
        for a in arr:
            ts.append(a.prob_of_heads[idx])
        return np.array(ts)

    def ts_GC(self,idx):
        return self._ts_heads(self.GC_list,idx)
    
    def ts_iter_alpha(self,idx):
        assert self.iter_alpha != 0, "Error: no iter_alpha array"
        return self._ts_heads(self.iter_alpha_list,idx)

    def ts_iter_alpha_fast(self,idx):
        assert self.iter_alpha_fast != 0, "Error: no iter_alpha_fast array"
        return self._ts_heads(self.iter_alpha_fast_list,idx)

    def ts_iter_alpha_perm(self,idx):
        return self._ts_heads(self.iter_alpha_perm_list,idx)

    def ts_iter_alpha_fast_perm(self,idx):
        return self._ts_heads(self.iter_alpha_fast_perm_list,idx)

    def ts_totev_alpha(self,idx):
        assert self.totev_alpha != 0, "Error: no totev_alpha array"
        return self._ts_heads(self.totev_alpha_list,idx)

    def ts_totev_alpha_fast(self,idx):
        assert self.totev_alpha_fast != 0, "Error: no totev_alpha_fast array"
        return self._ts_heads(self.totev_alpha_fast_list,idx)

    def _ts_spread(self,array_list,name=""):
        spread_list = []
        for arr in array_list:
            spread_list.append(arr.spread)
        obj = np.array(spread_list)
        self.existing_spread_ts.append([name,obj])
        return obj


    # Graphing as a method of LearningSequence
    def _red_grey(self,ts_red,ts_grey):
        fig,axs=plt.subplots()
        fig.set_tight_layout(True)
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
        self._red_grey(self.ts_iter_alpha,self.ts_GC)

    def graph_iter_fast_v_GC(self):
        self._red_grey(self.ts_iter_alpha_fast,self.ts_GC)

    def graph_totev_v_GC(self):
        self._red_grey(self.ts_totev_alpha,self.ts_GC)

    def graph_totev_fast_v_GC(self):
        self._red_grey(self.ts_totev_alpha_fast,self.ts_GC)

    def graph_iter_v_totev(self):
        self._red_grey(self.ts_totev_alpha,self.ts_iter_alpha)

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

    def two_graph_iter_iter_fast(self):
        self._two_graphs(
            self.ts_iter_alpha,self.ts_iter_alpha_fast,
            top_label="Iterative\n alpha cut",bottom_label="Fast iterative\n alpha cut")
        
    def two_graph_totev_totev_fast(self):
        self._two_graphs(
            self.ts_totev_alpha,self.ts_totev_alpha_fast,
            top_label="Total evidence\n alpha cut",bottom_label="Fast total evidence\n alpha cut")

    def commutativity(self,fast=False):
        if fast:
            ts_one,ts_two = self.ts_iter_alpha_fast, self.ts_iter_alpha_fast_perm
        else:
            ts_one, ts_two = self.ts_iter_alpha,self.ts_iter_alpha_perm
        self._two_graphs(
            ts_one,ts_two,
            top_label="Original\n evidence series",
            bottom_label = "Permuted\n evidence series",
            label_permuted=True)

    def spread_graph(self,spread_ts):
        fig,axs=plt.subplots()
        fig.set_tight_layout(True)
        x = np.arange(len(spread_ts))
        y = spread_ts
        z = self.GC_spread_ts
        axs.plot(x,z,color='b',linewidth=2)
        axs.plot(x,y,color='r',linewidth=1,marker=".")
        axs.set_xticks(np.arange(0,len(self.evidence_words)))
        axs.set_xticklabels(self.evidence_words,rotation="vertical")
        #    axs.margins(0.2)

    # root_n currently doesn't do anything.
    # The plan is that it will fit a curve of shape 1/sqrt n
    def all_spread(self,root_n=False):
        fig,axs = plt.subplots()
        fig.set_tight_layout(True)
        x = np.arange(len(self.existing_spread_ts[0][1]))
        for ts in self.existing_spread_ts:
            y = ts[1]
            axs.plot(x,y,linewidth=1,label =ts[0])
        if root_n:
            z = 1/np.sqrt(x)
            axs.plot(x,z,linewidth = 0.5,label = r"n^-2")
        axs.set_xticks(np.arange(0,len(self.evidence_words)))
        axs.set_xticklabels(self.evidence_words,rotation="vertical")
        axs.legend(loc='best')

    # currently the iter_ts switch does nothing.
    # in the future, it will allow iterative alpha cut discrepancy graphs too
    def discrepancy(self,iter_ts=False):
        fig,axs = plt.subplots()
        fig.set_tight_layout(True)
        # fig.subplots_adjust(right=0.8)
        x = np.arange(0,self.evidence_length)
        y = self.totev_alpha_disc_list
        plt.plot(x,y,linewidth=1)

    # likewise, currently vestigial iter_ts switch
    def plot_dists(self,idx, iter_ts = False,stop = 8):
        x = np.linspace(0,1,128)
        fig,axs = plt.subplots()
        i = 0
        for arr in self.GC_list:
            a,b = arr.array[idx]
            axs.plot(x,stats.beta.pdf(x,a,b),label=arr.array[idx],lw=2)
            i+=1
            if i >= stop:
                break
        axs.axes.get_yaxis().set_visible(False)
        for spine in ["left", "top", "right"]:
            axs.spines[spine].set_visible(False)
        axs.xaxis.set_ticks_position('bottom')
        axs.legend(loc='best')

        

# todo:
# multiple alpha values
# IDM? (throw out all priors with high t value?)
# Discrepancy : log plots
# Use scipy betabinom.
        
def spread_test():
    foo = LearningSequence(BetaPrior(8), EvidenceStream(0.3,8,8),
                           totev_alpha=0.5, totev_alpha_fast=0.5)
    # foo.all_spread(root_n=False)
    # foo.spread_graph(foo.totev_alpha_fast_spread_ts)
    # plt.show()

def test(fast=True):
    if fast:
        return LearningSequence(
            BetaPrior(4), EvidenceStream(0.3,16,8),totev_alpha_fast = 0.5)

    else:
        return LearningSequence(
            BetaPrior(4,randoms=[50,20]), EvidenceStream(0.3,8,8),iter_alpha = 0.5,  iter_alpha_fast=0.5,permuted_evidence=True)

def graph_test():
    foo = test()
    foo.spread_graph(foo.iter_alpha_fast_spread_ts)
    plt.show()
    
    
