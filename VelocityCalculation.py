import pandas as pd
from math import sin, cos, sqrt, atan2, radians
df = pd.read_csv('location_log.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y %H:%M')

def getDistanceFromLatLonInKm(lat1,lon1,lat2,lon2):
    R = 6371 # Radius of the earth in km
    dLat = radians(lat2-lat1)
    dLon = radians(lon2-lon1)
    rLat1 = radians(lat1)
    rLat2 = radians(lat2)
    a = sin(dLat/2) * sin(dLat/2) + cos(rLat1) * cos(rLat2) * sin(dLon/2) * sin(dLon/2)
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    d = R * c # Distance in km
    return d

def calc_velocity(dist_km, time_start, time_end):
    """Return 0 if time_start == time_end, avoid dividing by 0"""
    return dist_km / (time_end - time_start).seconds if time_end > time_start else 0

def isVelocityRealistic(velocity):
    return velocity < 1000

# Sort
df = df.sort_values(by=['ID', 'timestamp'])

# Group the sorted dataframe by ID, and grab the initial value for lat, lon, and time.
df['lat0'] = df.groupby('ID')['latitude'].transform(lambda x: x.iat[0])
df['lon0'] = df.groupby('ID')['longitude'].transform(lambda x: x.iat[0])
df['t0'] = df.groupby('ID')['timestamp'].transform(lambda x: x.iat[0])

df['dist_km'] = df.apply(
    lambda row: getDistanceFromLatLonInKm(
        lat1=row['latitude'],
        lon1=row['longitude'],
        lat2=row['lat0'],
        lon2=row['lon0']
    ),
    axis=1
)

# create a new column for velocity
df['velocity_kmph'] = df.apply(
    lambda row: calc_velocity(
        dist_km=row['dist_km'],
        time_start=row['t0'],
        time_end=row['timestamp']
    )*3600,
    axis=1
)

# create a new column for velocity realistic
df['velocity_real'] = df.apply(
    lambda row: isVelocityRealistic(
        velocity=row['velocity_kmph'],
    ),
    axis=1
)

print(df)
