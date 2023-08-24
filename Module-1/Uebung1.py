import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

url = 'https://raw.githubusercontent.com/sigvehaug/CAS-Applied-Data-Science/master/Module-1/iris.csv'
df = pd.read_csv(url, names=['slength','swidth', 'plength', 'pwidth', 'species'])


df.iloc[[38, 48, 99],0] = np.nan

mean = df.mean()
df.fillna((mean),inplace= True)

#print(mean)
#print(df.iloc[38])

