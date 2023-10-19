from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

dataset = loadtxt('diabetes_dataset.csv', delimiter=',')
# split into input (x) and output (y) variables
x = dataset[:,0:8]
y = dataset[:,8]
# printing
print('value of X are:', x)
print('value of Y are:', y)

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# compile the keras model
model.compile(loss='binary_crossentropy', metrics = ['accuracy'])
# make class predictions with the model
predictions = model.predict_classes(x)
for i in range(5):
	print(f'{x[i].tolist()} => {predictions[i]} expected {y[i]}')