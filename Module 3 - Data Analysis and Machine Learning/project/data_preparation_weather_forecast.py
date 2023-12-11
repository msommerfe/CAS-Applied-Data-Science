import pandas as pd


def getWeatherForecastBern():
    #load Weather data
    dfWeatherForecast = pd.read_csv("data/COSMO-E-all-stations.csv", skiprows = 24, encoding='latin-1', sep=';')
    dfReturn = dfWeatherForecast[['stn','time', 'leadtime','T_2M.1', 'TOT_PREC.1', 'DURSUN.1']]
    dfReturn= dfReturn.drop([0, 1])
    dfReturn["date"] = pd.to_datetime(dfReturn['time']).dt.date
    dfReturnBern = dfReturn[dfReturn['stn'] == 'BER']
    dfReturnBern = dfReturnBern.replace(['-999.0'], '0.0')

    dfReturnBern['T_2M.1'] = dfReturnBern['T_2M.1'].astype(float)
    #Dieser Request liefert den MEan über einen Tag. Muss aber noch in ein neues DF aufgebaut werden
    # dfReturnBern.groupby(dfReturnBern["date"])['T_2M.1'].mean()

    return dfWeatherForecast

dfWeatherForecast = getWeatherForecastBern()
print(dfWeatherForecast)



dfWeatherForecast = 1

