import pandas as pd

g = {'ggg':45, 'hddgd':474}

data = pd.DataFrame()
data.col = g.keys()
print(data.head())
print(data.columns)