
import requests 
import pandas as pd
import json

from tracker import Tracker

trans = Tracker()

trans.trans_pull_from_database()
print('\nThis is the current transactions database:')
trans.show_transactions_database()

while True:
    
    #check if I want to reinitiate
    rein = input('Do you want to reinitiate the database? (yes/no) ')
    if rein.lower() == 'yes':
        trans.reinitiate_trans_database()
        print('Now you have to buy/sell positions.')
        trans.fill()
        trans.trans_write_on_disk()
        print('\nThis is the current transactions database that has been saved:')
        trans.show_transactions_database()
        
    #Add transactions
    add = input('\nDo you want to buy/sell positions? (yes/no) ')
    if add.lower() == 'yes':
        trans.fill()
        trans.trans_write_on_disk()
    else:
        print('\nThis is the current transactions database that has been saved:')
        trans.show_transactions_database()
        excel = input('\nDo you want an excel copy? (yes/no) ') 
        if excel.lower() == 'yes':
            trans.download_trans_excel()
        break

#trans.trans_pull_from_database()
trans.get_live_portfolio()
trans.show_current_portfolio()
excel = input('\nDo you want an excel copy? (yes/no) ') 
if excel.lower() == 'yes':
            trans.download_port_excel()



