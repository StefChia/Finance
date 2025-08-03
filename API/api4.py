
import asyncio
import websockets
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation

prices =[]
timestamp = []


# Figure setup
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)

async def listen_last_price():
    url = "wss://stream.binance.com:9443/ws/btcusdt@ticker"

    async with websockets.connect(url) as ws:
        print("Connected to Binance trade stream.")

        while True:
            message = await ws.recv()
            data = json.loads(message)
            #print(data)
            #print(f'Last price: {data['c']} | Time: {data['E']}')
            prices.append(float(data['c']))
            timestamp.append(int(data['E']))


# Plot updater (sync, used by matplotlib)
def update_plot(frame):
    if prices:
        line.set_data(range(len(prices)), prices)
        ax.relim()
        ax.autoscale_view()
    return line,

# Launch async task in background
def start_listener():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(listen_last_price())

# Run WebSocket in a thread
import threading
thread = threading.Thread(target=start_listener, daemon=True)
thread.start()

# Animate the chart
ani = animation.FuncAnimation(fig, update_plot, interval=500)
plt.title("BTC/USDT Live Price")
plt.xlabel("Ticks")
plt.ylabel("Price")
plt.tight_layout()
plt.show()