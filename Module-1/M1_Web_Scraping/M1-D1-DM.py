import pandas as pd
import matplotlib.pyplot as plt
import time
url = 'https://raw.githubusercontent.com/sigvehaug/CAS-Applied-Data-Science/master/Module-1/iris.csv'
df = pd.read_csv(url, names=['slength','swidth', 'plength', 'pwidth', 'species'])
df_iris = df[df['species'] == 'Iris-setosa']
#df_mean_iris =[df_iris['plength']*2, df_iris['plength']]
#print(df_mean_iris)

#df = df_iris.iloc[0:100:5,1:4]
print(df_iris['plength'].mean())
print(df_iris)



ax = plt.gca()
df_iris.plot(kind='line',
        y='slength',
        color='green', ax=ax)
df_iris.plot(kind='line',
             y='swidth',
             color='red',ax=ax)
df_iris.plot(kind='line',
             y='pwidth',
             color='blue', ax=ax)
plt.show()

gdf = df.groupby('species')
print(gdf.get_group('Iris-setosa'))
print(gdf.groups)

#print(df.sort_values('plength'))

print(df)
print(df.shape)


#TODO try to measure the time fpr an operation

df_iris.to_csv('cleaned_df.csv')