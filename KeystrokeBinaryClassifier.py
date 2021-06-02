import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# load dataset
dataframe = pandas.read_csv("keystroke_data.csv")
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:5].astype(float)
Y = dataset[:,5]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

history = estimator.fit(X, Y, validation_split=0.33)
estimator.model.save('model1.h5')

print(estimator.predict([[2,30,25,0,15]]))

# load train dataset
testDataframe = pandas.read_csv("keystroke_test_data.csv")
testDataset = testDataframe.values
X_test = testDataset[:,0:5]
Y_test = testDataset[:,5]

accumilativeAccuracy = 0

print(Y_test)
for i in range(0, len(Y_test)):
    prediction = estimator.predict(numpy.array([X_test[i]]))
    # print(float(prediction[0]) - Y_test[i][0])
    print("Pred: " + str(prediction[0][0]))
    if int(prediction[0][0]) == int(Y_test[i]):
        accuracy = 100
    else:
        accuracy = 0
    print("Test "+str(i+1)+" Accuracy: ", accuracy)
    accumilativeAccuracy += accuracy

accumilativeAccuracy = accumilativeAccuracy/len(Y_test)
print("Overall accuracy: ", accumilativeAccuracy)


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
