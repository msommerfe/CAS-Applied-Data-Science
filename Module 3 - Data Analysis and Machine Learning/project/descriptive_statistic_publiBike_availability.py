import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
import pandas as pd
import umap
import plotly.express as px
from IPython.display import display


pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', True)

#Reading Publi-e-bike availability data
dfPubliBikeAvailability = pd.read_csv("data/publi-e-bike-availability-bern.csv", encoding='latin-1', sep=';')

#Prepareing Data
# TODO---> Vielleicht auch nur mal genau eine Woche nutzen zum trainieren der Daten, da eine Woche ein recht gutes Pattern ist.
dfPubliBikeAvailability = dfPubliBikeAvailability.rename(columns={"filename": "timestamp"})
dfPubliBikeAvailability["timestamp"] = pd.to_datetime(dfPubliBikeAvailability["timestamp"])
dfPubliBikeAvailability["dayofweek"] = dfPubliBikeAvailability["timestamp"].dt.weekday
dfPubliBikeAvailability["hourofday"] = dfPubliBikeAvailability["timestamp"].dt.hour
dfPubliBikeAvailability["minuteofday"] = dfPubliBikeAvailability["timestamp"].dt.minute
dfPubliBikeAvailability['station_id'] = dfPubliBikeAvailability['stationsname'].astype('category').cat.codes





# Want to have 3 Groups with simular distribution. First Group "0" --> Available bikes = 0; Group "1" --> Available bikes = 1-2
dfPubliBikeAvailability['lable_availability'] = dfPubliBikeAvailability['anzahl_e_bikes']

#First Train just one Station. If you want do train all, just comment the following line
dfPubliBikeAvailability = dfPubliBikeAvailability[dfPubliBikeAvailability["stationsname"] == "Sattler-Gelateria"]

#Genau eine Woche als INput nehmen. Einfach auskommentieren falls alle Daten (2 Wochen) verwendet werden sollen
dfPubliBikeAvailability = dfPubliBikeAvailability['2023-05-02T00:00:00+00:00' < dfPubliBikeAvailability["timestamp"]]
dfPubliBikeAvailability = dfPubliBikeAvailability[dfPubliBikeAvailability["timestamp"] < '2023-05-09T00:00:00+00:00' ]

#Assining the availability into 3 Groups;  Group "0" --> Available bikes = 0; Group "1" --> Available bikes = 1-2
dfPubliBikeAvailability['lable_availability'] = [0 if (i<2) else i for i in dfPubliBikeAvailability['lable_availability']]
dfPubliBikeAvailability['lable_availability'] = [1 if (1<i<5) else i for i in dfPubliBikeAvailability['lable_availability']]
dfPubliBikeAvailability['lable_availability'] = [2 if (i>4) else i for i in dfPubliBikeAvailability['lable_availability']]

#Plot availability Groups over Time
fig = px.line(y=dfPubliBikeAvailability["lable_availability"], x=dfPubliBikeAvailability["timestamp"])
fig.show()

#Plot the histogram of availability just to see if the distribution over the three availavility groups is simular
plt.hist(dfPubliBikeAvailability['lable_availability'], fill=False, histtype='step', label="length", density="True", bins = 100)
plt.show()



#############################################################################################
######  Preparing Data for Training. Split in Train and Test Data  ##########################
######  Feature: [dayofweek, hour, Min, Stationname] Label:  anzahl_e_bikes     ##############
#############################################################################################

# First shuffle rows then split into train and test data based on percentTrain
dfPubliBikeAvailability = sklearn.utils.shuffle(dfPubliBikeAvailability)
print(dfPubliBikeAvailability)
percentTrain = 0.8
rangeTrain= int(np.round(dfPubliBikeAvailability.shape[0] * percentTrain))

#split data tran and test
dfFeatureTrainPB = dfPubliBikeAvailability[['station_id','dayofweek','hourofday','minuteofday']][:rangeTrain]
dfLabelsTrainPB = dfPubliBikeAvailability[['lable_availability']][:rangeTrain]
dfFeaturesTestPB =  dfPubliBikeAvailability[['station_id','dayofweek','hourofday','minuteofday']][rangeTrain:]
dfLabelsTestPB =  dfPubliBikeAvailability[['lable_availability']][rangeTrain:]


#Convert to numpy array because tensorflow just accepts numpy arrays
dfFeatureTrainPB = dfFeatureTrainPB.to_numpy()
dfLabelsTrainPB = dfLabelsTrainPB['lable_availability'].to_numpy()
dfFeaturesTestPB = dfFeaturesTestPB.to_numpy()
dfLabelsTestPB = dfLabelsTestPB['lable_availability'].to_numpy()




#### Cluster Ansatz. Gruppen sind die Anzahl Bikes und input ist TimeStamp + Location Name But somehow werde ich nicht schlau aus dem Ergebnis,...Uncomment to test it #######
'''umap_model = umap.UMAP(n_neighbors=15, n_components=2, random_state=100)
umap_PB = umap_model.fit_transform(dfFeatureTrainPB)
plt.scatter(umap_PB[:, 0], umap_PB[:, 1], c=dfLabelsTrainPB, s=2)
plt.show()
'''




#Set up a model. here we can play a lot
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(4,)),
    tf.keras.layers.Dense(1000, activation=tf.keras.layers.LeakyReLU()),
    tf.keras.layers.Dense(10000, activation=tf.keras.layers.LeakyReLU()),
    tf.keras.layers.Dense(12, activation=tf.keras.layers.LeakyReLU()),
#    tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU()),
#    tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU()),
#    tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU()),
#    tf.keras.layers.Dense(8, activation=tf.keras.layers.LeakyReLU()),
    tf.keras.layers.Dense(3, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

#Train the model
availability = model.fit(dfFeatureTrainPB, dfLabelsTrainPB, epochs=400, batch_size=500, validation_data=(dfFeaturesTestPB, dfLabelsTestPB))


#Ploting the error over epochs
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].plot(availability.epoch, availability.history['loss'])
axs[0].plot(availability.epoch, availability.history['val_loss'])
axs[0].legend(('training loss', 'validation loss'), loc='lower right')
axs[1].plot(availability.epoch, availability.history['accuracy'])
axs[1].plot(availability.epoch, availability.history['val_accuracy'])
axs[1].legend(('training accuracy', 'validation accuracy'), loc='lower right')
plt.show()

#Shows the quality of the trained model
print(model.evaluate(dfFeaturesTestPB,  dfLabelsTestPB, verbose=2))


#Calculating the availability for the test data
predict = model.predict(dfFeaturesTestPB)
predict = pd.DataFrame(predict)

#plotting the calculated availability over hours
fig = px.scatter(x=dfFeaturesTestPB[:,2], y = predict.idxmax(axis='columns').values, )
fig.show()


# Hoi Viki,...habe jetzt recht viel mit den Parametern rumgespielt aber irgendwie finde ich die Ergebnisse nicht so befriedigend. Vielleicht müssen wir uns mal die Timeseries anschauen (https://www.tensorflow.org/tutorials/structured_data/time_series)
# Vielleicht fällt Dir ja noch etwas ein. Ansonsten spiele gerne mal mit den Parametern und den Modell rum. Vielleicht kannst auch noch was an den input/output daten optimieren falls dir eine Idee kommt.
#Ich würde mal für heute abschliessen

