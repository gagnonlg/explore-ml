import os

import keras
import numpy as np
import matplotlib.pyplot as plt

def __main():
    # get the data
    t, x, dx, y, dy = np.loadtxt('orbite.txt', unpack=True)
    print t


    train_data = np.empty((x.size*10000,3))
    for i, (t_,x_, dx_, y_, dy_) in enumerate(zip(t,x,dx,y,dy)):
        train_data[i*10000:(i+1)*10000,0] = t_
        train_data[i*10000:(i+1)*10000,1:] = np.random.multivariate_normal([x_,y_], [[dx_,0],[0,dy_]], 10000)

    # shuffle
    ishuffle = np.arange(0, train_data.shape[0])
    np.random.shuffle(ishuffle)
    train_data = train_data[ishuffle]
    print train_data.shape

    # normalize the data
    xtr = train_data[:,0]
    mean = np.mean(x)
    std = np.std(x)
    xtr -= mean
    xtr /= std
    ytr = train_data[:,1:]

    # build the network
    network = keras.models.Sequential()
    network.add(keras.layers.Dense(output_dim=300, input_dim=1))
    network.add(keras.layers.Activation('sigmoid'))
    network.add(keras.layers.Dense(output_dim=2))
    network.add(keras.layers.Activation('linear'))
    network.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=0.01))
    checkpoint = keras.callbacks.ModelCheckpoint(
        'nn_keras_orbite_weights.h5',
        verbose=1,
        save_best_only=True
    )
    network.fit(
        xtr,
        ytr,
        nb_epoch=200,
        batch_size=128,
        validation_split=0.1,
        callbacks=[checkpoint],
        verbose=2
    )

    network.load_weights('nn_keras_orbite_weights.h5')

    plt.errorbar(x,y,xerr=dx,yerr=dy,fmt='o')

    tspace = np.linspace(t[0], t[-1], 1000)
    pred = network.predict((tspace - mean)/std)
    plt.plot(pred[:,0], pred[:,1])
    plt.show()

    

if __name__ == '__main__':
    __main()
