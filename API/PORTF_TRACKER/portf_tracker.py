
import requests 
import pandas as pd
import json

from tracker import Tracker

port = Tracker()

while True:
    port.trans_pull_from_database()
    port.show_transactions_database()
    add = input('Do you want to buy/sell positions? (yes/no) ')
    if add.lower() == 'yes':
        port.fill()
        port.trans_write_on_disk()
    else: 
        break
    
    