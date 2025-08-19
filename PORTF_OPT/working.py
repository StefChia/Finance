import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.express as px

from arch import arch_model

from framework import download_historical_prices, compute_returns, portf_optim, compute_prices_paths, compute_drawdowns, compute_sample_statistics, print_summary_statistics
from framework import compute_ewma_train_sample, update_ewma_in_test_sample, launch_portf_opt_on_sample_test, risk_parity_portf, launch_risk_parity_opt_on_sample_test, compute_sample_varcov_via_PCs_model



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
returns = compute_returns(prices,log_ret=True)

n = len(tickers)
l = int(len(returns) * 0.75)
index_test = returns.index[l:]
ret_train = returns[:l]
ret_test = returns[l:]



#COMPUTE A BENCHMARK
bench_ticker = ['SPY']

bench_prices = download_historical_prices(bench_ticker)
bench_returns = compute_returns(bench_prices,log_ret=True)

bench_prices_test = bench_prices[l:]
bench_returns_test = bench_returns[l:]

bench_pri = compute_prices_paths(bench_returns_test,log_ret=True)
bench_draw = compute_drawdowns(bench_pri.values,False)

basic_stats,basic_stats_annual = compute_sample_statistics(bench_prices_test,bench_returns_test)
print_summary_statistics(basic_stats_annual)










#MEAN VARIANCE

#SAMPLE VARIANCE

#Expected values
m = np.zeros((n,1))
#m = ret_train.mean().to_numpy().reshape(-1,1)

#Varcov from Sample Covariance
#varcov = ret_train.cov()
varcov = compute_sample_varcov_via_PCs_model(ret_train)


#MINIMAL VARIANCE PORTFOLIO/ TARGET VARIANCE / TARGET EXPECTED VALUE PORTFOLIO

res = portf_optim(m,varcov)
#print(res['status'])
#print(res['objective'])
w = res['weights']


port_ret = ret_test.values @ w
port_ret = pd.DataFrame(port_ret,index=index_test, columns=['Portf'])
port_ret['Bench'] = bench_returns_test                          #Compare with the benchmark
port_pri = compute_prices_paths(port_ret,False)

fig, ax = plt.subplots()
ax.plot(port_pri['Portf'],label='Portf',color='blue')
ax.plot(port_pri['Bench'],label='Bench',color='green')
ax.set_title('Cumulative returns')
plt.legend()
plt.show()

draw = compute_drawdowns(port_pri['Portf'].values,True)
#print(draw)

basic_stats,basic_stats_annual = compute_sample_statistics(port_pri,port_ret)
print_summary_statistics(basic_stats_annual)






"""
#MEAN VARIANCE

#EWMA

#Expected values
m = np.zeros((n,1))
#m = ret_train.mean().to_numpy().reshape(-1,1)

#VARCOV with EWMA
m_estim_test = [m for i in range(len(index_test))]
means = ret_train.mean()

varcov_ewma_0 = compute_ewma_train_sample(ret_train)
varcov_estim_test = update_ewma_in_test_sample(varcov_ewma_0, ret_test)
#print(varcov_estim_test[:2])

portf_ret = launch_portf_opt_on_sample_test(m_estim_test,varcov_estim_test,ret_test)
portf_ret = pd.DataFrame(portf_ret,index=index_test,columns=['Portf'])
portf_ret['Bench'] = bench_returns_test                          #Compare with the benchmark

port_pri = compute_prices_paths(portf_ret,False)

fig, ax = plt.subplots()
ax.plot(port_pri['Portf'],label='Portf',color='blue')
ax.plot(port_pri['Bench'],label='Bench',color='green')
ax.set_title('Cumulative returns')
plt.legend()
plt.show()

draw = compute_drawdowns(port_pri['Portf'].values,True)
#print(draw)

basic_stats,basic_stats_annual = compute_sample_statistics(port_pri,portf_ret)
print_summary_statistics(basic_stats_annual)
"""









"""
#RISK PARITY PORTFOLIO

#SAMPLE VARIANCE

#Expected values
m = np.zeros((n,1))
#m = ret_train.mean().to_numpy().reshape(-1,1)

#Varcov from Sample Covariance
#varcov = np.cov(ret_train,rowvar=False)
varcov = compute_sample_varcov_via_PCs_model(ret_train)
 
 
risk_parity = risk_parity_portf(varcov)
w_rp = risk_parity['weights']

port_ret = ret_test.values @ w_rp.reshape(-1,1)
port_ret = pd.DataFrame(port_ret,index=index_test, columns=['Portf'])
port_ret['Bench'] = bench_returns_test                          #Compare with the benchmark
port_pri = compute_prices_paths(port_ret,False)

fig, ax = plt.subplots()
ax.plot(port_pri['Portf'],label='Portf',color='blue')
ax.plot(port_pri['Bench'],label='Bench',color='green')
ax.set_title('Cumulative returns')
plt.legend()
plt.show()

draw = compute_drawdowns(port_pri['Portf'].values,True)
#print(draw)

basic_stats,basic_stats_annual = compute_sample_statistics(port_pri,port_ret)
print_summary_statistics(basic_stats_annual)
"""






"""
#RISK-PARITY


#EWMA

#Expected values
m = np.zeros((n,1))
#m = ret_train.mean().to_numpy().reshape(-1,1)

#VARCOV with EWMA
m_estim_test = [m for i in range(len(index_test))]
means = ret_train.mean()

varcov_ewma_0 = compute_ewma_train_sample(ret_train)
varcov_estim_test = update_ewma_in_test_sample(varcov_ewma_0, ret_test)
#print(varcov_estim_test[:2])

portf_ret = launch_risk_parity_opt_on_sample_test(varcov_estim_test,ret_test)
portf_ret = pd.DataFrame(portf_ret,index=index_test,columns=['Portf'])
portf_ret['Bench'] = bench_returns_test                          #Compare with the benchmark

port_pri = compute_prices_paths(portf_ret,False)

fig, ax = plt.subplots()
ax.plot(port_pri['Portf'],label='Portf',color='blue')
ax.plot(port_pri['Bench'],label='Bench',color='green')
ax.set_title('Cumulative returns')
plt.legend()
plt.show()

draw = compute_drawdowns(port_pri['Portf'].values,True)
#print(draw)

basic_stats,basic_stats_annual = compute_sample_statistics(port_pri,portf_ret)
print_summary_statistics(basic_stats_annual)
"""


















"""
#VARCOV VIA GARCH(1,1)

#EXpected values
m = np.zeros((n,1))
#m = ret_train.mean().to_numpy().reshape(-1,1)

m_estim_test = [m for i in range(len(index_test))]

#FIT THE GARCH ON THE TRAINING SAMPLE
model = arch_model(ret_train['AAPL'], vol='GARCH',p=1,q=1, mean='Zero')
res = model.fit(disp="off")

print(res.summary())
params = res.params
forecast = res.forecast(horizon=1)
v_0 = forecast.variance[-1:].values
print(v_0)

#
varcov_estim_test = [v_0]
for i in range(len(ret_test)-1):
    v_new = params['omega']+ params['alpha'] * ret_test[i]**2 + params['beta'] * varcov_estim_test[-1]
    varcov_estim_test.append(v_new)

"""

