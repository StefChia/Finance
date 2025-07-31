
from norm_rw import LogNormPriceRW
import matplotlib.pyplot as plt

price_process = LogNormPriceRW(100)
price_process.fill_rw()

fig, ax = plt.subplots()
ax.plot(price_process.p, linewidth = 2)

plt.show()