import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import umap
from IPython.display import display
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.mixture import GaussianMixture

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)


dfPubliBikeAvailability = pd.read_csv("data/publi-e-bike-availability-bern.csv", encoding='latin-1', sep=';')

dfPubliBikeAvailability = dfPubliBikeAvailability.rename(columns={"filename": "timestamp"})
dfPubliBikeAvailability["timestamp"] = pd.to_datetime(dfPubliBikeAvailability["timestamp"])
dfPubliBikeAvailability["dayofweek"] = dfPubliBikeAvailability["timestamp"].dt.weekday
dfPubliBikeAvailability["hourofday"] = dfPubliBikeAvailability["timestamp"].dt.hour
dfPubliBikeAvailability["minuteofday"] = dfPubliBikeAvailability["timestamp"].dt.minute
dfPubliBikeAvailability['station_id'] = dfPubliBikeAvailability['stationsname'].astype('category').cat.codes
dfPubliBikeAvailability['available_y1_n0'] = [1 if (i > 0) else 0 for i in dfPubliBikeAvailability['anzahl_e_bikes']]



print(dfPubliBikeAvailability['available_y1_n0'])
print(dfPubliBikeAvailability)
print("MAX")
print(dfPubliBikeAvailability['anzahl_e_bikes'].max())


#############################################################################################
######## Analyse Data for just one Station. In this case station Breitenrainstrasse##########
#############################################################################################

#dfPubliBikeAvailability = dfPubliBikeAvailability.sort_values(by='timestamp', ascending=True)
#dfBreitenrainstrasse = dfPubliBikeAvailability[dfPubliBikeAvailability["stationsname"] == "Breitenrainstrasse"]
#fig = px.line(y=dfBreitenrainstrasse["anzahl_e_bikes"], x=dfBreitenrainstrasse["timestamp"])
#fig.show()
#print(dfBreitenrainstrasse)





#############################################################################################
######  Preparing Data fpr Training. Split in Train and Test Data  ##########################
######  Feature: [dayofweek, hour, Min, Stationname] Label:  anzahl_e_bikes     ##########################
#############################################################################################

# percentage to split the data into train and test data
percentTrain = 0.8
rangeTrain= int(np.round(dfPubliBikeAvailability.shape[0] * percentTrain))

#split data tran and test
dfFeatureTrainPB = dfPubliBikeAvailability[['station_id','dayofweek','hourofday','minuteofday']][:rangeTrain]
#dfLabelsTrainPB = dfPubliBikeAvailability[['anzahl_e_bikes']][:rangeTrain]
dfLabelsTrainPB = dfPubliBikeAvailability[['available_y1_n0']][:rangeTrain]

dfFeaturesTestPB =  dfPubliBikeAvailability[['station_id','dayofweek','hourofday','minuteofday']][rangeTrain:]
#dfLabelsTestPB =  dfPubliBikeAvailability[['anzahl_e_bikes']][rangeTrain:]
dfLabelsTestPB =  dfPubliBikeAvailability[['available_y1_n0']][rangeTrain:]
print(dfFeatureTrainPB.shape)
print(type(dfFeatureTrainPB))



dfFeatureTrainPB = dfFeatureTrainPB.to_numpy()
dfLabelsTrainPB = dfLabelsTrainPB.to_numpy()

dfFeaturesTestPB = dfFeaturesTestPB.to_numpy()
dfLabelsTestPB = dfLabelsTestPB.to_numpy()
print("klhgluhl")
print(dfLabelsTrainPB.shape)
print(type(dfLabelsTrainPB))


print("rangeTrain: ", rangeTrain)
print("sizePB: ", dfPubliBikeAvailability.shape)
print("sizeTrain: ", dfFeatureTrainPB.shape)
print("sizeTest: ", dfFeaturesTestPB.shape)
#print(dfPubliBikeAvailability.size)
print(dfFeatureTrainPB)
print(dfLabelsTrainPB)


#### Cluster Ansatz. Gruppen sind die Anzahl Bikes und input ist TimeStamp + Location Name #######
umap_model = umap.UMAP(n_neighbors=10, n_components=2, random_state=1000)
umap_mnist = umap_model.fit_transform(dfFeatureTrainPB)
plt.scatter(umap_mnist[:, 0], umap_mnist[:, 1], c=dfLabelsTrainPB, s=2)
plt.show()