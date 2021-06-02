import numpy
import pymongo
import pandas as pd
from math import sin, cos, sqrt, atan2, radians
import json
from flask_cors import CORS
from flask import Flask
from flask import request
from bson import ObjectId, json_util
from datetime import datetime
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
from bson.json_util import dumps, loads

def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Initializing model object
model = KerasRegressor(build_fn=create_baseline, epochs=10, batch_size=10, verbose=1)

# Loading model
model.model = load_model('model1.h5')
# Test prediction to define prediction
prediction = model.predict(numpy.array([[16,1,1,0,0]]))


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["PollingSystem"]
bot_activity_collection = mydb["bot_activity"]
mycol = mydb["userlocations"]

def getDistanceFromLatLonInKm(lat1,lon1,lat2,lon2):
    R = 6371 # Radius of the earth in km
    dLat = radians(float(lat2)-float(lat1))
    dLon = radians(float(lon2)-float(lon1))
    rLat1 = radians(float(lat1))
    rLat2 = radians(float(lat2))
    a = sin(dLat/2) * sin(dLat/2) + cos(rLat1) * cos(rLat2) * sin(dLon/2) * sin(dLon/2)
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    d = R * c # Distance in km
    return d

def calc_velocity(dist_km, time_start, time_end):
    """Return 0 if time_start == time_end, avoid dividing by 0"""
    return dist_km / (time_end - time_start).seconds if time_end > time_start else 0

def isVelocityRealistic(velocity):
    return velocity < 1000


def myconverter(o):
    if isinstance(o, datetime):
        return o.__str__()

app = Flask(__name__)
cors = CORS(app)

# Get a prediction from the model
@app.route("/predictScore",methods =['GET'])
def predictScore():
    data = pd.DataFrame(list(mycol.find()))
    print(data['date'])
    data['timestamp'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M')
    # Sort
    data = data.sort_values(by=['userId', 'timestamp'])

    data['lat0'] = data.groupby('userId')['latitude'].transform(lambda x: x.iat[0])
    data['lon0'] = data.groupby('userId')['longitude'].transform(lambda x: x.iat[0])
    data['t0'] = data.groupby('userId')['timestamp'].transform(lambda x: x.iat[0])


    data['dist_km'] = data.apply(
        lambda row: getDistanceFromLatLonInKm(
            lat1=row['latitude'],
            lon1=row['longitude'],
            lat2=row['lat0'],
            lon2=row['lon0']
        ),
        axis=1
    )

    # create a new column for velocity
    data['velocity_kmph'] = data.apply(
        lambda row: calc_velocity(
            dist_km=row['dist_km'],
            time_start=row['t0'],
            time_end=row['timestamp']
        )*3600,
        axis=1
    )

    # create a new column for velocity realistic
    data['velocity_real'] = data.apply(
        lambda row: isVelocityRealistic(
            velocity=row['velocity_kmph'],
        ),
        axis=1
    )

    # print(data.to_dict(orient='records'))


    for index, row in data.iterrows():
        row['_id'] = str(row['_id'])
        data['_id'].values[index] = row['_id']

    data['date'] = data['date'].astype(str)
    return json.dumps(data.to_dict(orient='records'), default = myconverter)


@app.route("/uploadKsTimes",methods =['POST'])
def uploadKsTimes():
    data = numpy.array(request.get_json())
    prediction = model.predict(numpy.array(data))
    print(request.args.get('username'))
    if float(prediction) < 0.8:
        print("Bot")
        now = datetime.now()
        bot_activity_collection.insert_one(
            {
                "username": request.args.get('username'),
                "date": now.strftime("%m/%d/%Y, %H:%M:%S")
            }
        )
    return str(prediction)


@app.route("/getBotActivities",methods =['GET'])
def getBotActivities():
    return dumps(list(bot_activity_collection.find()))

if __name__ == '__main__':
    app.run(debug=True, port=5002)
