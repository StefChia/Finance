
import pandas as pd

from tracker_adv import Tracker

from portfolio_mgmt import PortMngmt
#from var_compute import VAR

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


advance_flag = input('Do you want the advanced analytics? (yes/no) ')
if advance_flag.lower() == 'yes':
    adv = PortMngmt(trans)
    prices = adv.download_historical_prices()
    returns = adv.compute_returns()
    basic_stats,basic_stats_annaulized,corr_matrix = adv.compute_sample_statistics(prices,returns)
    #adv.show_corr_matrix(corr_matrix)

    port_ret = adv.get_port_dyn_ret()
    #print(port_ret)
    port_pri = adv.compute_prices_paths()
    #print(port_pri)
    basic_stats_portf,basic_stats_annaulized_portf,corr_matrix_portf = adv.compute_sample_statistics(port_pri,port_ret)
    adv.show_prices_paths(adv.port_value)

"""
advance_flag_2 = input('Do you want the VAR? (yes/no) ')
if advance_flag_2.lower() == 'yes':
    trans.compute_var()"""