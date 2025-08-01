
import requests 
import pandas as pd
import json

from tracker import Tracker

trans = Tracker()

while True:
    trans.trans_pull_from_database()
    trans.show_transactions_database()
    add = input('Do you want to buy/sell positions? (yes/no) ')
    if add.lower() == 'yes':
        trans.fill()
        trans.trans_write_on_disk()
    else: 
        break
    
#port.reinitiate_trans_database()
    