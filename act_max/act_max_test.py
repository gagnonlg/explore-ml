import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def generate_data(size):
    x = np.random.uniform(-1, 1, (size, 2))

    y = np.prod(x, axis=1)
    y[y < 0] = -1
    y[y >= 0] = 1

    return x, y

def activation_hist(network, neuron, n=10000, data=None):
    # network: Keras model
    # neuron: the neuron indices. E.g. (0, 0) == first layer, first neuron, i.e. L0N0
    l_idx, n_idx = neuron

    if data is None:
        x, y = generate_data(n)
    elif len(data) == 2:
        x, y = data
    else:
        x = data
        y = np.prod(x, axis=1)
        y[y < 0] = -1
        y[y >= 0] = 1

    def f(x):
        tmp = x
        i_dense = 0
        for layer in network.layers:
            tmp = layer(tmp)
            if type(layer) == tf.keras.layers.Dense:
                if i_dense == l_idx:
                    break
                i_dense += 1
        return tmp.numpy()[:, n_idx]

    plt.hist(
        [f(x[y == -1]), f(x[y == +1])],
        label=['Negative product', 'Positive product'],
        histtype='step',
        log=True
    )
    plt.legend(loc='best')
    return plt


def activation_maximisation(network, neuron, threshold, l_rate=0.01, n_seed=100000, maxiter=1000):
    # network: Keras model
    # neuron: the neuron indices. E.g. (0, 0) == first layer, first neuron, i.e. L0N0
    # threshold: activation threshold after which iterations are stopped

    l_idx, n_idx = neuron

    x, _ = generate_data(n_seed)

    x_synth = tf.Variable(x)

    x_mask = np.ones((x.shape[0],1)).astype(np.float32)
    over_threshold_mask = np.ones_like(x_mask)
    no_gradient_mask = np.ones_like(x_mask)

    i_iter = 0

    while np.any(x_mask) and i_iter < maxiter:
        i_iter += 1
        with tf.GradientTape() as tape:
            tmp = x_synth
            i_dense = 0
            for layer in network.layers:
                tmp = layer(tmp)
                if type(layer) == tf.keras.layers.Dense:
                    if i_dense == l_idx:
                        break
                    i_dense += 1
            act = tmp[:, n_idx]
        dact_dx = tape.gradient(act, x_synth)
        over_threshold_mask[act >= threshold] = 0
        no_gradient_mask[np.all(dact_dx == 0, axis=1)] = 0
        x_mask = over_threshold_mask * no_gradient_mask
        x_synth = tf.Variable(x_synth + x_mask * l_rate * dact_dx)

    return x_synth[over_threshold_mask[:, 0] == 0]

#############################################################################################

path = 'product_network_v1.tf'

network = tf.keras.models.load_model(path)
network.summary()

activation_hist(network, (0, 0))
plt.show()

x_synth = activation_maximisation(network, (0,0), 0.5)
plt.hist2d(x_synth[:, 0], x_synth[:, 1], bins=25, range=((-1, 1), (-1,1)), cmap='inferno')
plt.show()
