import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from IPython.display import display
pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)


dfPubliBikeAvailability = pd.read_csv("data/publi-e-bike-availability-bern.csv", encoding='latin-1', sep=';')

dfPubliBikeAvailability = dfPubliBikeAvailability.rename(columns={"filename": "timestamp"})
dfPubliBikeAvailability["timestamp"] = pd.to_datetime(dfPubliBikeAvailability["timestamp"])
dfPubliBikeAvailability["dayofweek"] = dfPubliBikeAvailability["timestamp"].dt.weekday
dfPubliBikeAvailability["hourofday"] = dfPubliBikeAvailability["timestamp"].dt.hour
dfPubliBikeAvailability["minuteofday"] = dfPubliBikeAvailability["timestamp"].dt.minute
dfPubliBikeAvailability['station_id'] = dfPubliBikeAvailability['stationsname'].astype('category').cat.codes
print(dfPubliBikeAvailability)



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
dfLabelsTrainPB = dfPubliBikeAvailability[['anzahl_e_bikes']][:rangeTrain]

dfFeaturesTestPB =  dfPubliBikeAvailability[['station_id','dayofweek','hourofday','minuteofday']][rangeTrain:]
dfLabelsTestPB =  dfPubliBikeAvailability[['anzahl_e_bikes']][rangeTrain:]
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