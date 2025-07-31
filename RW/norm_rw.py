
import numpy as np

class LogNormPriceRW:
    """Define the behavior of a Price process which follows a log-normal RW"""
    def __init__(self,num=50):
        """Initialize the attributes"""
        #Initiate starting point
        self.num = num
        self.p = [100]
        #self.seed = 42
        #np.random.seed(self.seed)
    
    def fill_rw(self):
        """Create the path for the RW"""
        while len(self.p) <= self.num:
            x = np.random.normal(0.00,0.03)     #set drift=0 and sigma almost 20% annuallly
            p = self.p[-1] * np.exp(x)
            self.p.append(p)
    
        
