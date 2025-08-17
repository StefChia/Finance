import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.express as px

from framework import download_historical_prices, compute_returns, portf_optim, compute_prices_paths, compute_drawdowns, compute_sample_statistics, print_summary_statistics
from framework import compute_ewma, update_ewma_in_test_sample, launch_portf_opt_on_sample_test



#Downloaad and prepare the data

tickers = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "GOOGL", # Alphabet (Class A)
    "GOOG",  # Alphabet (Class C)
    "META",  # Meta Platforms
    "NVDA",  # NVIDIA
    "TSLA",  # Tesla
    "BRK-B", # Berkshire Hathaway (Class B)
    "JPM",   # JPMorgan Chase
    "V",     # Visa
    "MA",    # Mastercard
    "JNJ",   # Johnson & Johnson
    "PG",    # Procter & Gamble
    "HD",    # Home Depot
    "UNH",   # UnitedHealth
    "PFE",   # Pfizer
    "BAC",   # Bank of America
    "DIS",   # Walt Disney
    "NFLX",  # Netflix
    "KO",    # Coca-Cola
    "PEP",   # PepsiCo
    "WMT",   # Walmart
    "CVX",   # Chevron
    "XOM",   # ExxonMobil
    "MRK",   # Merck
    "CSCO",  # Cisco Systems
    "ABT",   # Abbott Laboratories
    "ORCL",  # Oracle
    "INTC"   # Intel
]


data = pd.DataFrame()

prices = download_historical_prices(tickers)
returns = compute_returns(prices)

l = int(len(returns) * 0.75)
index_test = returns.index[l:]
ret_train = returns[:l]
ret_test = returns[l:]




#SAMPLE VARIANCE

#Expected values
m = np.zeros((30,1))

#Varcov from Sample Covariance
varcov = ret_train.cov()
#print(varcov)


#MINIMAL VARIANCE PORTFOLIO

res = portf_optim(m,varcov)
print(res['status'])
print(res['objective'])
w = res['weights']


port_ret = ret_test.values @ w
port_ret = pd.DataFrame(port_ret,index=index_test, columns=['Portf'])
port_pri = compute_prices_paths(port_ret,False)

fig, ax = plt.subplots()
ax.plot(port_pri)
ax.set_title('Cumulative returns')
plt.show()

draw = compute_drawdowns(port_pri.values,True)
#print(draw)

basic_stats,basic_stats_annual = compute_sample_statistics(port_pri,port_ret)
print_summary_statistics(basic_stats_annual)






#EWMA

#VARCOV with EWMA
m_estim_test = [np.zeros((30,1)) for i in range(len(index_test))]

varcov_ewma_0 = compute_ewma(ret_train)
varcov_estim_test = update_ewma_in_test_sample(varcov_ewma_0, ret_test)
#print(varcov_estim_test[:2])

portf_ret = launch_portf_opt_on_sample_test(m_estim_test,varcov_estim_test,ret_test)
portf_ret = pd.DataFrame(portf_ret,index=index_test,columns=['Portf ret'])


port_pri = compute_prices_paths(portf_ret,False)

fig, ax = plt.subplots()
ax.plot(port_pri)
ax.set_title('Cumulative returns')
plt.show()

draw = compute_drawdowns(port_pri.values,True)
#print(draw)

basic_stats,basic_stats_annual = compute_sample_statistics(port_pri,portf_ret)
print_summary_statistics(basic_stats_annual)

