""" hello_sine.py: fit a noisy sinus with a neural network """
import keras
import matplotlib.pyplot as plt
import numpy as np

# data configuration
# number of training points
NTRAIN = 100000
# number of test points
NTEST = 1000
# frequency of sin function
FREQ = 2 * np.pi * 440
# variance of additive noise
NOISE_STDDEV = 0.1

# seed the generator to get reproducible results
np.random.seed(750)

def gen_data(n, test=False):
    """ generating function for the data """
    data = np.zeros((n,2))
    data[:,0] = np.linspace(0, 2*np.pi, n)
    data[:,1] = np.sin(data[:,0])
    if not test:
        data[:,1] += np.random.normal(scale=NOISE_STDDEV, size=n)
        np.random.shuffle(data)
    return data[:,0], data[:,1]

# generate the datasets
x_train, y_train = gen_data(NTRAIN)
x_test, y_test = gen_data(NTEST, test=True)

# normalize the training input to zero mean and unit variance
mean = np.mean(x_train)
std = np.std(x_train)
x_train -= mean
x_train /= std

# build the neural network
model = keras.models.Sequential()
# let's use 1 hidden layer with 300 hidden units
model.add(keras.layers.Dense(output_dim=300, input_dim=1))
# the classical sigmoid activation
model.add(keras.layers.Activation('sigmoid'))
model.add(keras.layers.Dense(output_dim=1))
# linear output layer because we are performing regression
model.add(keras.layers.Activation('linear'))

# configure the training to use 'mean squared error' and stochastic
# gradient descent
model.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=0.01))

# train the neural network, saving the best model
# according to loss on validation set
checkpoint = keras.callbacks.ModelCheckpoint('fitted_weights.h5', verbose=1, save_best_only=True)
history = model.fit(x_train, y_train, nb_epoch=200, batch_size=128, validation_data=((x_test-mean)/std,y_test), callbacks=[checkpoint], verbose=2)

# load the best weights
model.load_weights('fitted_weights.h5')

# prepare the test data
y_pred = model.predict((x_test - mean)/std)
y_pred_train = model.predict(x_train)

# plot the results
plt.plot(x_test, y_test,color='black', label='sin(2*pi*440)')
plt.plot(x_test, y_pred, color='blue', label='fit result')
plt.scatter(std*x_train + mean, y_train, color='red', label='training data')
plt.axis([0,2*np.pi,-1.5,1.5])
plt.legend(loc='best')
plt.savefig('fit_result.png')
plt.close()

plt.plot(np.arange(0,200), history.history['loss'], label='training loss')
plt.plot(np.arange(0,200), history.history['val_loss'], label='validation_loss')
plt.legend(loc='best')
plt.savefig('loss.png')
