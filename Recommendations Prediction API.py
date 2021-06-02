import pymongo
import json
from flask_cors import CORS
import numpy
from flask import Flask
from flask import request
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
import RecommenderDataSet

def larger_model():
    # create model
    model = Sequential()
    model.add(Dense(81, input_dim=26, kernel_initializer='normal', activation='relu'))
    model.add(Dense(89, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Initializing model object
model = KerasRegressor(build_fn=larger_model, epochs=10, batch_size=10, verbose=1)

# Loading model
model.model = load_model('model2.h5')
# Test prediction to define prediction
prediction = model.predict(numpy.array([[16,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0]]))

# MongoDB connection init
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
# Select database
mydb = myclient["PollingSystem"]
botActivity = mydb["customers"]

app = Flask(__name__)
cors = CORS(app)

# Get a prediction from the model
@app.route("/predictScore",methods =['POST'])
def predictScore():
    data = numpy.array(request.get_json())
    print(data)
    prediction = model.predict(data)
    return str(prediction)

@app.route("/getRecommendScore",methods =['POST'])
def getRecommendScore():
    data = request.get_json()
    modelData = []
    modelData.append(data['age'])
    modelData.append(RecommenderDataSet.getGenderIndex(data['gender']))
    modelData.append(RecommenderDataSet.getOccupationIndex(str(data['occupation'])))
    pollNameStr = str(data['pollName']) + " " + str(data['votedPollNames'])
    print(pollNameStr.count('product'))
    modelData.append(pollNameStr.count('product'))
    modelData.append(pollNameStr.count('brand'))
    modelData.append(pollNameStr.count('positive'))
    modelData.append(pollNameStr.count('negative'))
    modelData.append(pollNameStr.count('quality'))
    modelData.append(pollNameStr.count('price'))
    modelData.append(pollNameStr.count('software'))
    modelData.append(pollNameStr.count('application'))
    modelData.append(pollNameStr.count('app'))
    modelData.append(pollNameStr.count('language'))
    modelData.append(pollNameStr.count('library'))
    modelData.append(pollNameStr.count('location'))
    modelData.append(pollNameStr.count('program'))
    modelData.append(pollNameStr.count('business'))
    modelData.append(pollNameStr.count('school'))
    modelData.append(pollNameStr.count('food'))
    modelData.append(pollNameStr.count('event'))
    modelData.append(pollNameStr.count('design'))
    modelData.append(pollNameStr.count('plant'))
    modelData.append(pollNameStr.count('paint'))
    modelData.append(pollNameStr.count('car'))
    modelData.append(pollNameStr.count('travel'))
    modelData.append(pollNameStr.count('system'))
    print(modelData)
    predData = []
    predData.append(modelData)
    prediction = model.predict(predData)
    return str(prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
