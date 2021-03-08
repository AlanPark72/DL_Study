from datetime import datetime
import pandas as pd
from meteostat import Point, Daily

start = datetime(2018,1,1)
end = datetime(2021,3,8)

YT = Point(37.251494, 127.071288)

data = Daily(YT, start, end)
data = data.fetch()

df = pd.DataFrame(data)
df.to_csv('weather.csv',index=True)

test = 1