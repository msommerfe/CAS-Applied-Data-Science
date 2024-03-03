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

#Badi season (15.05 - 15.09)
#PERIOD_FROM = '2023-05-15 00:00:00'
#PERIOD_TO = '2023-09-16 00:00:00'
PERIOD_FROM = '2020-05-15 00:00:00'
PERIOD_TO = '2024-09-16 00:00:00'

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', True)


def getWeatherDataBern():
    #load Weather data
    dfWeatherBernCurrentYear = pd.read_csv("data/nbcn-daily_BER_current.csv", encoding='latin-1', sep=';')
    dfWeatherBernPreviousYears = pd.read_csv("data/nbcn-daily_BER_previous.csv", encoding='latin-1', sep=';')

    # Weather Data were in two files so here we concat them into one data frame
    df_weatherDataBern = pd.concat([dfWeatherBernCurrentYear, dfWeatherBernPreviousYears])
    df_weatherDataBern["date"] = pd.to_datetime(df_weatherDataBern['date'], format='%Y%m%d').dt.date


    #Select the relevant data for the relevant PERIOD
    df_weatherDataBern = df_weatherDataBern[pd.to_datetime(PERIOD_FROM).date() <= df_weatherDataBern["date"]]
    df_weatherDataBern = df_weatherDataBern[df_weatherDataBern["date"] <= pd.to_datetime(PERIOD_TO).date() ]


    # convert Column: Lufttemperatur 2 m über Boden; Tagesmittel to Int
    df_weatherDataBern['tempPD_grad'] = df_weatherDataBern['tre200d0'].astype(int)

    # convert Column: Sonnenscheindauer; Tagessumme to Int
    df_weatherDataBern['sunPD_min'] = df_weatherDataBern['sre000d0'].astype(int)

    # convert Column: Niederschlag; Tagessumme 6 UTC - 6 UTC Folgetag to Int
    df_weatherDataBern['PrecipitationPD_mm'] = df_weatherDataBern['rre150d0'].astype(int)

    # Drop unnecessary columns
    df_weatherDataBern = df_weatherDataBern.drop(columns=['gre000d0', 'hto000d0', 'nto000d0','prestad0', 'rre150d0', 'sre000d0', 'tre200d0', 'tre200dn', 'tre200dx','ure200d0'])
    return df_weatherDataBern



def getPubliBikeAvailability():
    #Reading Publi-e-bike availability data
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
    dfPubliBikeAvailability["date"] = dfPubliBikeAvailability['timestamp'].dt.date
    dfPubliBikeAvailability = dfPubliBikeAvailability.drop(columns=['Abfragezeit', 'id', 'EBike','Bike'])

    #Choose only the Station 230 = "Sattler-Gelateria"
    dfPubliBikeAvailability = dfPubliBikeAvailability[dfPubliBikeAvailability["station_id"] == 230 ]
    #dfPubliBikeAvailability = dfPubliBikeAvailability[dfPubliBikeAvailability["station_id"] == 315 ] #315 Marzilli

    #Select the relevant data for the basi season (15.05 - 15.09)
    dfPubliBikeAvailability = dfPubliBikeAvailability[pd.to_datetime(PERIOD_FROM).date() <= dfPubliBikeAvailability["timestamp"].dt.date]
    dfPubliBikeAvailability = dfPubliBikeAvailability[dfPubliBikeAvailability["timestamp"].dt.date <= pd.to_datetime(PERIOD_TO).date() ]

    #Assining the availability into 3 Groups;  Group "0" --> Available bikes = 0; Group "1" --> Available bikes = 1-2
    # Following line helped me to fine 3 Groups that has a almoat equal distribution (0-q33, q33-q66, q66-q100) --> pd.qcut(dfPubliBikeAvailability['anzahl_e_bikes'],3, precision=0 )
    dfPubliBikeAvailability['availability_group'] = dfPubliBikeAvailability['anzahl_e_bikes']
    dfPubliBikeAvailability['availability_group'] = [0 if (i<2) else i for i in dfPubliBikeAvailability['availability_group']]
    dfPubliBikeAvailability['availability_group'] = [1 if (1<i<6) else i for i in dfPubliBikeAvailability['availability_group']]
    dfPubliBikeAvailability['availability_group'] = [2 if (i>5) else i for i in dfPubliBikeAvailability['availability_group']]
    return dfPubliBikeAvailability


def plotDataOverviewPubliBikeAvailability(dfPubliBikeAvailability):
    #Täglicher Mean einer Woche über alle Daten im dfPubliBikeAvailability
    #fig = px.scatter(dfPubliBikeAvailability.groupby(dfPubliBikeAvailability["timestamp"].dt.weekday)['anzahl_e_bikes'].mean(),trendline="lowess", trendline_options=dict(frac=0.05))
    #fig.show()


    fig = px.scatter(dfPubliBikeAvailability, x = 'timestamp', y = ['anzahl_e_bikes', 'availability_group'],
                     labels={"timestamp": "Timestamp","value": "Availability"},
                     title = "Availability Data 15.05.2023 - 15.09.2023",
                     trendline="lowess", trendline_options=dict(frac=0.04))
    fig.show()

    # Ploting Trendlinie for one Week 1. with Number of bikes 2. With availability Groups
    fig = px.scatter(dfPubliBikeAvailability, x = 'continuous_week_hours', y = ['anzahl_e_bikes', 'availability_group'],
                     labels={"continuous_week_hours": "Continous hours of a week","value": "Availability"},
                     title = "Smoothed median 15.05.2023 - 15.09.2023 reduced to a Week",
                     trendline="lowess", trendline_options=dict(frac=0.04))
    fig.show()
    return


def getCleanPublibikeDataframe():
    dfPubliBikeAvailability = getPubliBikeAvailability()
    dfWeatherDataBern = getWeatherDataBern()
    return dfPubliBikeAvailability.merge(dfWeatherDataBern, on=['date'], how = 'left')