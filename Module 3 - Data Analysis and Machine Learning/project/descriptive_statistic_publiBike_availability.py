import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from IPython.display import display
pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)

import io
import matplotlib.dates as mdates

dfPubliBikeAvailability = pd.read_csv("data/publi-e-bike-availability-bern.csv", encoding='latin-1', sep=';')
dfPubliBikeAvailability = dfPubliBikeAvailability.rename(columns={"filename": "timestamp"})
print(dfPubliBikeAvailability)
dfPubliBikeAvailability = dfPubliBikeAvailability.sort_values(by='timestamp', ascending=True)

dfBreitenrainstrasse = dfPubliBikeAvailability[dfPubliBikeAvailability["stationsname"] == "Breitenrainstrasse"]
dfBreitenrainstrasse["timestamp"] = pd.to_datetime(dfBreitenrainstrasse["timestamp"])
print(dfBreitenrainstrasse)


fig = px.line(y=dfBreitenrainstrasse["anzahl_e_bikes"], x=dfBreitenrainstrasse["timestamp"])
fig.show()


# Cluster Ansatz. Gruppen sind die Anzahl Bikes und input ist TimeStamp + Location Name


