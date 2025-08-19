
import bt
import yfinance as yf

# Get price data
data = yf.download("BTC-USD", start="2021-01-01")['Close']

# Define a simple momentum strategy: buy when price > 50-day moving average
def strategy_logic(data):
    signal = data > data.rolling(50).mean()
    return signal.astype(int)

signal = strategy_logic(data)

# Create strategy and backtest
strategy = bt.Strategy('Momentum50',
                       [bt.algos.WeighTarget(signal),
                        bt.algos.Rebalance()])

portfolio = bt.Backtest(strategy, data)
result = bt.run(portfolio)

# Analyze results
result.plot()
print(result.display())