import pandas as pd
import requests
import plotly.express as px

r = requests.get('https://rest.publibike.ch/v1/public/partner/stations')

dictPubliAll = r.json()
publiStations = dictPubliAll['stations']
df_publiStations =pd.DataFrame(publiStations)

numberBikes = [len(vehicles)for vehicles in df_publiStations["vehicles"]]
df_publiStations.insert(0,'numberBikes',numberBikes,True)
print(df_publiStations)


fig = px.scatter_mapbox(df_publiStations, lat="latitude", lon="longitude", hover_name='name', hover_data=["vehicles", "capacity"], color='numberBikes', size='numberBikes',
                        zoom=8)

fig.update_layout(mapbox_style="open-street-map")
fig.show()
