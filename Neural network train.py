import pandas
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.models import load_model
import numpy
import matplotlib.pyplot as plt

# load train dataset
dataframe = pandas.read_csv("recommender_final_dataset.csv")
dataset = dataframe.values
X = dataset[:,0:26]
Y = dataset[:,26:27]

def larger_model():
    # create model
    model = Sequential()
    model.add(Dense(81, input_dim=26, kernel_initializer='normal', activation='relu'))
    model.add(Dense(89, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

# evaluate model
estimator = KerasRegressor(build_fn=larger_model, epochs=100, batch_size=75, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

history = estimator.fit(X, Y, validation_split=0.33)
estimator.model.save('model2.h5')


model = KerasRegressor(build_fn=larger_model, epochs=10, batch_size=10, verbose=1)

model.model = load_model('model2.h5')
# Test prediction to define prediction
prediction = model.predict(numpy.array([[16,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0]]))

# load train dataset
testDataframe = pandas.read_csv("test_data.csv")
testDataset = testDataframe.values
X_test = testDataset[:,0:26]
Y_test = testDataset[:,26]

accumilativeAccuracy = 0

for i in range(0, 10):
    prediction = model.predict(numpy.array([X_test[i]]))
    print(float(prediction))
    accuracy = (1 - (abs(Y_test[i]-prediction)))*100
    print("Test "+str(i+1)+" Accuracy: ", accuracy)
    accumilativeAccuracy += accuracy

accumilativeAccuracy = accumilativeAccuracy/10
print("Overall accuracy: ", accumilativeAccuracy)

print(history.history)
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
