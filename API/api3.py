
#API WITH WebSockets
# https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams

import asyncio
import websockets
import json

async def listen_last_price():
    url = "wss://stream.binance.com:9443/ws/btcusdt@ticker"

    async with websockets.connect(url) as ws:
        print("Connected to Binance trade stream.")

        while True:
            message = await ws.recv()
            data = json.loads(message)
            #print(data)
            print(f'Last price: {data['c']} | Time: {data['E']}')

asyncio.run(listen_last_price())



"""async def listen_trades():
    url = "wss://stream.binance.com:9443/ws/btcusdt@trade"

    async with websockets.connect(url) as ws:
        print("Connected to Binance trade stream.")

        while True:
            message = await ws.recv()
            data = json.loads(message)

            print(f"Price: {data['p']} | Quantity: {data['q']} | Time: {data['T']}")


asyncio.run(listen_trades())"""




"""async def listen_bid_ask():
    url = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"

    async with websockets.connect(url) as ws:
        print("Connected to Binance trade stream.")

        while True:
            message = await ws.recv()
            data = json.loads(message)
            #print(data)
            print(f"Bid: {data['b']} | Quantity: {data['B']} | Ask: {data['a']} | Quantity: {data['A']}")

asyncio.run(listen_bid_ask())"""

