
import requests

def get_price(ticker,name=''):
        """Get the price via API"""
        
        
        #Get via API the Current Price
        url = "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/quote"

        querystring = {"ticker":ticker,"type":"STOCKS"}

        headers = {
            "x-rapidapi-key": "6487ea259dmsh1cf031f35e0b4e1p17e5a5jsn1a70867302f5",
            "x-rapidapi-host": "yahoo-finance15.p.rapidapi.com"
        }
        try:
            response = requests.get(url, headers=headers, params=querystring)
        except ValueError:
            print(f'{ticker} is not a valid ticker. We have used')
        else:
            """if response.status_code == 200:
                print(f'{name} Price successfully found.')"""
            content = response.json()
            #curr_p = content['body']['primaryData']['lastSalePrice']
            return content
    
gg = get_price('AAPL')
print(gg)

#{'message': 'You have exceeded the MONTHLY quota for Requests on your current plan, BASIC. Upgrade your plan at https://rapidapi.com/sparior/api/yahoo-finance15'}