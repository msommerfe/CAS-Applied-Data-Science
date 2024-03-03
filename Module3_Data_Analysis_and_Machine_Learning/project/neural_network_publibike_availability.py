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
import data_preparation_publibike
import matplotlib.dates as mdates


dfPubliBikeAvailability = data_preparation_publibike.getCleanPublibikeDataframe()

#plots some plots to get an overview of the data
data_preparation_publibike.plotDataOverviewPubliBikeAvailability(dfPubliBikeAvailability)




#############################################################################################
######  Preparing Data for Training. Split in Train and Test Data  ##########################
######  Feature: [dayofweek, hour] Label:  anzahl_e_bikes     ##############
#############################################################################################

dfFeatureTrainPB, dfFeaturesTestPB, dfLabelsTrainPB, dfLabelsTestPB = train_test_split(
    dfPubliBikeAvailability[['dayofweek','hourofday','tempPD_grad', 'sunPD_min',
                             'PrecipitationPD_mm']],
    dfPubliBikeAvailability[['availability_group']],
    test_size=0.2, random_state=42)


#Convert to numpy array because tensorflow just accepts numpy arrays
dfFeatureTrainPB = dfFeatureTrainPB.to_numpy()
dfLabelsTrainPB = dfLabelsTrainPB['availability_group'].to_numpy()
dfFeaturesTestPB = dfFeaturesTestPB.to_numpy()
dfLabelsTestPB = dfLabelsTestPB['availability_group'].to_numpy()



#Set up a model. here we can play a lot
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(5,)),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(10000, activation='relu'),
    #tf.keras.layers.Dense(1000, activation='relu'),
    #tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(3, activation='sigmoid'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

#Train the model
availability = model.fit(dfFeatureTrainPB, dfLabelsTrainPB, epochs=1000, batch_size=1000, validation_data=(dfFeaturesTestPB, dfLabelsTestPB))


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
fig = px.scatter(dfTrained, x = 'x', y = ['yTest', 'yPredict'],trendline="lowess", trendline_options=dict(frac=0.08))
fig.show()


fig = px.scatter(dfTrained, x = 'x', y = ['yTest', 'yPredict'],
                 labels={"x": "Continous hours of a week","value": "Availability", "yTest": "hhh", "yPredict": "jökhökjb"},
                 title = "Testdata vs predicted data",
                 trendline="lowess", trendline_options=dict(frac=0.08))
fig.show()