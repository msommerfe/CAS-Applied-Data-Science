from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn import metrics

from matplotlib import  pyplot as plt
import numpy as np
import os
import pandas as pd
from time import time as timer

import tensorflow as tf


from matplotlib import animation
from IPython.display import HTML
from sklearn.model_selection import train_test_split
from calendar import day_name
import data_preparation_publibike
import matplotlib.dates as mdates


dfPubliBikeAvailability = data_preparation_publibike.getCleanPublibikeDataframe()

#plots some plots to get an overview of the data
#data_preparation_publibike.plotDataOverviewPubliBikeAvailability(dfPubliBikeAvailability)


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
x_train = dfFeatureTrainPB.to_numpy()
y_train = dfLabelsTrainPB['availability_group'].to_numpy()
x_test = dfFeaturesTestPB.to_numpy()
y_test = dfLabelsTestPB['availability_group'].to_numpy()

rfr = ensemble.RandomForestRegressor(max_depth=10, n_estimators=30)
rfr.fit(x_train, y_train)
rfr.score(x_test, y_test)

y_p_train = rfr.predict(x_train)
y_p_test = rfr.predict(x_test)


# 4. plot y vs predicted y for test and train parts
plt.figure(figsize=(10,10))
plt.plot(y_train, y_p_train, 'b.', label='train')
plt.plot(y_test, y_p_test, 'r.', label='test')
plt.plot(y_train,y_train,'--',color='black')

plt.plot([0], [0], 'w.')  # dummy to have origin
plt.xlabel('true')
plt.ylabel('predicted')
plt.gca().set_aspect('equal')
plt.legend()
plt.show()

#Plot Importance of Feature
p_importances = permutation_importance(rfr, x_test, y_test, n_repeats=10, n_jobs=-1)
plt.figure(figsize=(8, 10))
plt.barh(list(dfPubliBikeAvailability[['dayofweek','hourofday','tempPD_grad', 'sunPD_min','PrecipitationPD_mm']].columns),
         p_importances.importances_mean,
         yerr=p_importances.importances_std, )
plt.ylabel('feature')
plt.xlabel('importance')
plt.xlim(0, 1)
plt.show()