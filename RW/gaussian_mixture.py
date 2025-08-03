
import random
import numpy as np
import matplotlib.pyplot as plt

class GaussianMixtureSampling:
    """Sample from a n-state Gaussian mixture
    Dictionary example:\n data = dict(a=[[0.2],[5,2]],b=[[0.7],[9,1]],c=[[0.1],[13,4]])"""
    def __init__(self,dict_state_prob_params, seed=None):
        """Initialize the attributes."""
        
        self.dict = dict_state_prob_params
        self.states = list(dict_state_prob_params.keys())
        self.state_prob = {}
        self.state_params = {}
        
        for name in self.states:
            self.state_prob[name] = dict_state_prob_params[name][0]
            self.state_params[name] = dict_state_prob_params[name][1]
        
        #Initialize 
        self.samples = []
        
        #Random seeds
        if seed != None:
            random.seed(seed)
            np.random.seed(seed)
    
        
    
    def sampling(self,num):
        """sample n from the distribution""" 
        self.num = num
        prob = np.array(list(self.state_prob.values()))
        cum_prob = prob.cumsum()
        cum_prob = cum_prob.tolist()
        
        
        for i in range(num):
            draw = random.random()
           
            """cnt = 0
            for i in range(len(cum_prob)):
                if draw > cum_prob[i]:
                    cnt += 1
                else:
                    break    """
            cnt = np.searchsorted(cum_prob,draw)        
            
            sel_state = self.states[cnt]
            sel_params = self.state_params[sel_state]
            self.samples.append(np.random.normal(sel_params[0],sel_params[1]))
            

    def show_samples(self):
        """Plot the distribution"""
        fig, ax = plt.subplots()
        ax.hist(self.samples, bins=200)
        ax.set_title("Gaussian Mixture Samples")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        plt.show()
        
        
        
#data = dict(a=[[0.2],[5,2]],b=[[0.6],[9,1]],c=[[0.2],[13,4]])
data = dict(a=[[0.5],[5,1]],b=[[0.5],[5,4]])

gg = GaussianMixtureSampling(data,42)
gg.sampling(10000)
gg.show_samples()


                

"""EQUIVALENT WAY
cnt = 0
for i in range(len(cum_prob)):
    if draw > cum_prob[i]:
        cnt += 1
    else:
        break
    
    
cnt = 0
while draw > cum_prob[i]:
    cnt += 1


np.searchsorted(array, value)"""