import keras
import numpy as np
import matplotlib.pyplot as plt

def __main():
    # get the data
    x,y = np.loadtxt('sine.training.txt', unpack=True)

    # normalize the data
    mean = np.mean(x)
    std = np.std(x)
    x -= mean
    x /= std

    # build the network
    network = keras.models.Sequential()
    network.add(keras.layers.Dense(output_dim=300, input_dim=1))
    network.add(keras.layers.Activation('sigmoid'))
    network.add(keras.layers.Dense(output_dim=1))
    network.add(keras.layers.Activation('linear'))
    network.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=0.01))
    checkpoint = keras.callbacks.ModelCheckpoint(
        'nn_keras_weights.h5',
        verbose=1,
        save_best_only=True
    )
    network.fit(
        x,
        y,
        nb_epoch=200,
        batch_size=128,
        validation_split=0.1,
        callbacks=[checkpoint],
        verbose=2
    )

    network.load_weights('nn_keras_weights.h5')
    x_test, _ = np.loadtxt('sine.test.txt', unpack=True)
    y_pred = network.predict((x_test - mean) / std)
    np.savetxt('nn_keras_prediction.txt', y_pred)

if __name__ == '__main__':
    __main()
