

import numpy as np
import matplotlib.pyplot as plt

from norm_rw import LogNormPriceRW

#Random colors


#Create a graph with 100 price process
n = 100
fig, ax = plt.subplots()

for i in range(n):
    #Random colors
    col = tuple([round(x, 1) for x in np.random.uniform(0, 1, 3)])
    #Create process
    p = LogNormPriceRW()
    p.fill_rw()
    ax.plot(p.p, color =col)

plt.show()