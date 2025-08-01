
import requests
"""url = 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd'
response = requests.get(url)
print(response.json())"""

name = 'bitcoin'
url = f'https://api.coingecko.com/api/v3/simple/price?ids={name}&vs_currencies=usd'
response = requests.get(url)
print(response.json())