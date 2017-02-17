import array
import gzip
import logging
import os
import struct
import urllib

import keras
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)


def download_mnist_dataset():
    logging.info('downloading the MNIST dataset')

    def fetch(path):
        if os.path.exists(path):
            logging.info('{} already present, skipping'.format(path))
        else:
            logging.info('downloading {}'.format(path))
            urllib.urlretrieve(
                url='http://yann.lecun.com/exdb/mnist/{}'.format(path),
                filename=path
            )

    fetch('train-images-idx3-ubyte.gz')
    fetch('train-labels-idx1-ubyte.gz')
    fetch('t10k-images-idx3-ubyte.gz')
    fetch('t10k-labels-idx1-ubyte.gz')


def load_images(train=True):
    logging.info('loading %s images', 'training' if train else 'test')

    if train:
        path = 'train-images-idx3-ubyte.gz'
        nexpected = 60000
    else:
        path = 't10k-images-idx3-ubyte.gz'
        nexpected = 10000

    with gzip.GzipFile(path, 'rb') as dfile:
        magic, = struct.unpack('>i', dfile.read(4))
        logging.debug('magic number: %d', magic)
        assert magic == 2051

        nimages, = struct.unpack('>i', dfile.read(4))
        logging.debug('n. of images: %d', nimages)
        assert nimages == nexpected

        nrows, = struct.unpack('>i', dfile.read(4))
        logging.debug('n. of rows: %d', nrows)
        assert nrows == 28

        ncols, = struct.unpack('>i', dfile.read(4))
        logging.debug('n. of columns: %d', ncols)
        assert ncols == 28

        logging.debug('reading data into numpy array')
        py_array = array.array('I')
        py_array.fromstring(dfile.read())
        darray = np.frombuffer(py_array, dtype=np.uint8)
        logging.debug('array.shape=%s', darray.shape)
        assert darray.shape[0] == nimages * nrows * ncols

        darray = darray.astype(np.float32)
        return darray.reshape((nimages, nrows*ncols))


def load_labels(train=True):
    logging.info('loading %s labels', 'training' if train else 'test')

    if train:
        path = 'train-labels-idx1-ubyte.gz'
        nexpected = 60000
    else:
        path = 't10k-labels-idx1-ubyte.gz'
        nexpected = 10000

    with gzip.GzipFile(path, 'rb') as dfile:
        magic, = struct.unpack('>i', dfile.read(4))
        logging.debug('magic number: %d', magic)
        assert magic == 2049

        nlabels, = struct.unpack('>i', dfile.read(4))
        logging.debug('n. of labels: %d', nlabels)
        assert nlabels == nexpected

        logging.debug('reading data into numpy array')
        darray = np.frombuffer(dfile.read(), dtype=np.uint8)
        logging.debug('array.shape=%s', darray.shape)
        assert darray.shape[0] == nlabels

        labels = np.zeros((nlabels, 10))
        labels[np.arange(labels.shape[0]), darray] = 1

        return labels

def normalize(dataset):
    mean = np.mean(dataset, axis=0)
    std = np.std(dataset, axis=0)
    std[std == 0] = 1
    dataset -= mean
    dataset /= std
    return mean,std


def main():
    download_mnist_dataset()
    x_train = load_images(train=True)
    y_train = load_labels(train=True)
    x_test  = load_images(train=False)
    y_test = load_labels(train=False)

    logging.info('normalizing data')
    mean, std = normalize(x_train)

    logging.info('building model')
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(
        input_dim=x_train.shape[1],
        output_dim=300,
        activation='sigmoid'
    ))
    model.add(keras.layers.Dense(
        output_dim=10,
        activation='softmax'
    ))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.SGD(lr=0.01),
        metrics=['accuracy']
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        'mnist_weights.h5',
        verbose=1,
        save_best_only=True
    )

    logging.info('training model')
    history = model.fit(
        x_train,
        y_train,
        nb_epoch=100,
        batch_size=128,
        validation_split=0.1,
        callbacks=[checkpoint],
        verbose=2
    )

    model.load_weights('mnist_weights.h5')

    y_pred = model.predict((x_test - mean)/std)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    y_test = np.argmax(y_test, axis=0)
    diff = y_pred - y_test
    nbad = np.count_nonzero(diff)
    accuracy = 1 - float(nbad) / y_pred.shape[0]
    logging.info('accuracy: %f', accuracy)

    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.legend(loc='best')
    plt.savefig('mnist_loss.png')
    plt.close()

    plt.plot(history.history['acc'], label='training accuracy')
    plt.plot(history.history['val_acc'], label='validation accuracy')
    plt.plot([accuracy]*len(history.history(['acc'])), label='test accuracy')
    plt.legend(loc='best')
    plt.savefig('mnist_accuracy.png')


    return 0

if __name__ == '__main__':
    try:
        rc = main()
    except:
        logging.exception('uncaught exception')
        rc = 1
    exit(rc)

