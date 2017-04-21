from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load song dataset
dataset = numpy.loadtxt("train_abridged.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:6]
Y = dataset[:,6]
# create model
model = Sequential()
model.add(Dense(12, input_dim=6, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10, verbose=1)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
#print(rounded)

# save the model parameters
model.save('songs_model.h5')

