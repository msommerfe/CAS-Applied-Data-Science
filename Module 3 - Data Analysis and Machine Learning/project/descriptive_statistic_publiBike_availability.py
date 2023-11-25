import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
import pandas as pd
import umap
import plotly.express as px
from IPython.display import display
from sklearn.model_selection import train_test_split
from calendar import day_name
import matplotlib.dates as mdates

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', True)


#Reading Publi-e-bike availability data 1 Year
dfPubliBikeAvailability = pd.read_csv("data/bike-availability-All-Stations_hourly.csv", encoding='latin-1', sep=';')
dfPubliBikeAvailability["timestamp"] = pd.to_datetime(dfPubliBikeAvailability["Abfragezeit"])
dfPubliBikeAvailability.set_index('timestamp')
dfPubliBikeAvailability["dayofweek"] = dfPubliBikeAvailability["timestamp"].dt.weekday
dfPubliBikeAvailability['dayofweek_name'] = dfPubliBikeAvailability['dayofweek'].apply(lambda w:day_name[w])
dfPubliBikeAvailability["hourofday"] = dfPubliBikeAvailability["timestamp"].dt.hour
dfPubliBikeAvailability['station_id'] = dfPubliBikeAvailability['id']
dfPubliBikeAvailability['anzahl_e_bikes'] = dfPubliBikeAvailability['EBike']
dfPubliBikeAvailability['anzahl_bikes'] = dfPubliBikeAvailability['Bike']
dfPubliBikeAvailability["continuous_week_hours"] = dfPubliBikeAvailability['dayofweek'] * 24 + dfPubliBikeAvailability['hourofday']





#prepare and Clean Data
#Choose only the Station 230 = "Sattler-Gelateria"
dfPubliBikeAvailability = dfPubliBikeAvailability[dfPubliBikeAvailability["station_id"] == 230 ]


#Select the relevant data for the basi season (15.05 - 15.09)
dfPubliBikeAvailability = dfPubliBikeAvailability[pd.to_datetime('2023-05-15 00:00:00') < dfPubliBikeAvailability["timestamp"]]
dfPubliBikeAvailability = dfPubliBikeAvailability[dfPubliBikeAvailability["timestamp"] < pd.to_datetime('2023-09-16 00:00:00') ]

#Stündlicher Mean eines Tages über alle Daten im dfPubliBikeAvailability
fig = px.scatter(dfPubliBikeAvailability.groupby(dfPubliBikeAvailability["timestamp"].dt.hour)['anzahl_e_bikes'].mean(),trendline="lowess", trendline_options=dict(frac=0.05))
fig.show()

#Täglicher Mean einer Woche über alle Daten im dfPubliBikeAvailability
fig = px.scatter(dfPubliBikeAvailability.groupby(dfPubliBikeAvailability["timestamp"].dt.weekday)['anzahl_e_bikes'].mean(),trendline="lowess", trendline_options=dict(frac=0.05))
fig.show()



#Assining the availability into 3 Groups;  Group "0" --> Available bikes = 0; Group "1" --> Available bikes = 1-2
# Following line helped me to fine 3 Groups that has a almoat equal distribution (0-q33, q33-q66, q66-q100) --> pd.qcut(dfPubliBikeAvailability['anzahl_e_bikes'],3, precision=0 )
dfPubliBikeAvailability['lable_availability'] = dfPubliBikeAvailability['anzahl_e_bikes']
dfPubliBikeAvailability['lable_availability'] = [0 if (i<2) else i for i in dfPubliBikeAvailability['lable_availability']]
dfPubliBikeAvailability['lable_availability'] = [1 if (1<i<6) else i for i in dfPubliBikeAvailability['lable_availability']]
dfPubliBikeAvailability['lable_availability'] = [2 if (i>5) else i for i in dfPubliBikeAvailability['lable_availability']]

#Plot availability Groups over Time
fig = px.line(y=dfPubliBikeAvailability["lable_availability"], x=dfPubliBikeAvailability["timestamp"])
fig.show()

# Ploting Trendlinie for one Week 1. with Number of bikes 2. With availability Groups
fig = px.scatter(x = dfPubliBikeAvailability["continuous_week_hours"], y = dfPubliBikeAvailability['anzahl_e_bikes'], trendline="lowess", trendline_options=dict(frac=0.02))
fig.show()
fig = px.scatter(x = dfPubliBikeAvailability["continuous_week_hours"], y = dfPubliBikeAvailability['lable_availability'], trendline="lowess", trendline_options=dict(frac=0.02))
fig.show()

#Plot the histogram of availability just to see if the distribution over the three availavility groups is simular
#plt.hist(dfPubliBikeAvailability['lable_availability'], fill=False, histtype='step', label="length", density="True", bins = 100)
#plt.show()



#############################################################################################
######  Preparing Data for Training. Split in Train and Test Data  ##########################
######  Feature: [dayofweek, hour] Label:  anzahl_e_bikes     ##############
#############################################################################################

dfFeatureTrainPB, dfFeaturesTestPB, dfLabelsTrainPB, dfLabelsTestPB = train_test_split(
    dfPubliBikeAvailability[['dayofweek','hourofday']],
    dfPubliBikeAvailability[['lable_availability']],
    test_size=0.2, random_state=42)


#Convert to numpy array because tensorflow just accepts numpy arrays
dfFeatureTrainPB = dfFeatureTrainPB.to_numpy()
dfLabelsTrainPB = dfLabelsTrainPB['lable_availability'].to_numpy()
dfFeaturesTestPB = dfFeaturesTestPB.to_numpy()
dfLabelsTestPB = dfLabelsTestPB['lable_availability'].to_numpy()



#Set up a model. here we can play a lot
#@Todo: erste Layer mal Probieren ohne Flatten. Flatten bei mehrdimensional sinnvoll aber sind ja hier schon 1-dim
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(2,)),
    tf.keras.layers.Dense(10000, activation=tf.keras.layers.LeakyReLU()),
    tf.keras.layers.Dense(1000, activation=tf.keras.layers.LeakyReLU()),
    tf.keras.layers.Dense(500, activation=tf.keras.layers.LeakyReLU()),
    tf.keras.layers.Dense(3, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

#Train the model
availability = model.fit(dfFeatureTrainPB, dfLabelsTrainPB, epochs=500, batch_size=2000, validation_data=(dfFeaturesTestPB, dfLabelsTestPB))


#Ploting the error over epochs
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].plot(availability.epoch, availability.history['loss'])
axs[0].plot(availability.epoch, availability.history['val_loss'])
axs[0].legend(('training loss', 'validation loss'), loc='lower right')
#axs[1].plot(availability.epoch, availability.history['mse'])
#axs[1].plot(availability.epoch, availability.history['val_mse'])
axs[1].plot(availability.epoch, availability.history['accuracy'])
axs[1].plot(availability.epoch, availability.history['val_accuracy'])
axs[1].legend(('training accuracy', 'validation accuracy'), loc='lower right')
plt.show()

#Shows the quality of the trained model
print(model.evaluate(dfFeaturesTestPB,  dfLabelsTestPB, verbose=2))


#Calculating the availability for the test data
predict = model.predict(dfFeaturesTestPB)
predict = pd.DataFrame(predict)


#Preparing data for plotting
dfTrained= pd.DataFrame()
dfTrained['x'] = dfFeaturesTestPB[:,0]*24 +dfFeaturesTestPB[:,1]
dfTrained['yTest'] = dfLabelsTestPB
dfTrained['yPredict'] = predict.idxmax(axis='columns').values

# Plotting Testdata and Preidcted Data over Week inkluding a trendline
fig = px.scatter(dfTrained, x = 'x', y = ['yTest', 'yPredict'],trendline="lowess", trendline_options=dict(frac=0.04))
fig.show()
