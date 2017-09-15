from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

x_data = [1,2,3]
y_data = [1,2,3]

# create model for linear regression
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Activation('linear'))
# Pass SGD as an optimizer with learning rate of 0.1
model.compile(loss='mse', optimizer=SGD(lr=0.1))

# prints summary of the model to the terminal
model.summary()

# feed the data to the model
model.fit(x_data, y_data, epochs=100)

# test the data
y_predict = model.predict(x_data)
print(y_data)