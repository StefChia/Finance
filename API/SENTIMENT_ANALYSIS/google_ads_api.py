
import json
from pathlib import Path
import yfinance as yf
import pandas as pd

from pytrends.request import TrendReq

pytrends = TrendReq(hl='en-US', tz=360)
#pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25), proxies=['https://34.203.233.13:80',], retries=2, backoff_factor=0.1, requests_args={'verify':False})

kw_list = ['Bitcoin','Ethereum','Solana','Crypto']
pytrends.build_payload(kw_list, cat=0, timeframe='today 5-y', geo='', gprop='')


"""
pytrends.interest_over_time()

pytrends.multirange_interest_over_time()

pytrends.get_historical_interest(kw_list, year_start=2018, month_start=1, day_start=1, hour_start=0, year_end=2018, month_end=2, day_end=1, hour_end=0, cat=0, geo='', gprop='', sleep=0)

pytrends.related_topics()

pytrends.trending_searches(pn='united_states') # trending searches in real time for United States
pytrends.realtime_trending_searches(pn='US') # realtime search trends for United States
pytrends.suggestions(keyword)"""

path = Path('/Users/stefanochiapparini/Desktop/PYTHON/Finance/API/SENTIMENT_ANALYSIS/google_searches_dataset.json')



# Fetch the interest over time
data1 = pytrends.interest_over_time()
print(data1)

data1.index = data1.index.strftime('%Y-%m-%d')

#3 months daily
pytrends.build_payload(kw_list, cat=0, timeframe='today 3-m', geo='', gprop='')
data2 = pytrends.interest_over_time()
#print(data2)
data2.index = data2.index.strftime('%Y-%m-%d')


#1 week hourly
pytrends.build_payload(kw_list, cat=0, timeframe='now 7-d', geo='', gprop='')
data3 = pytrends.interest_over_time()
#print(data3)
data3.index = data3.index.strftime('%Y-%m-%d %H:%M')

#1 day minute
pytrends.build_payload(kw_list, cat=0, timeframe='now 1-d', geo='', gprop='')
data4 = pytrends.interest_over_time()
#print(data4)
data4.index = data4.index.strftime('%Y-%m-%d %H:%M')


data = [data1.to_dict(),data2.to_dict(),data3.to_dict(),data4.to_dict()]



path.write_text(json.dumps(data))





#DOWNLOAD RETURNS

#DOWNLOAD RETURNS
# Example: Ethereum Classic in USD
ticker = "ETH-USD"
asset = yf.Ticker(ticker)

# Fetch weekly historical data
df = asset.history(period="3mo", interval="1d")  # last 1 year, weekly
p_v = df[['Close','Volume']]
p_v.index = pd.to_datetime(df.index, format= '%Y-%m-%d')

p_v.to_csv('returns_dataset.csv')