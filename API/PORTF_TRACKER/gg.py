import pandas as pd

g = {'ggg':45, 'hddgd':474}

data = pd.DataFrame()
data.columns = g.keys()
print(data.head())
print(data.columns)