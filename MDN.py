import matplotlib.pyplot as plt
import numpy as np
np.random.seed(12122)
import keras
import keras.backend as K


def mixture_density(nb_components, target_dimension=1):

    """ The Mixture Density output layer. Use with the keras functional api:

        inputs = Inputs(...)
        net = ....
        model = Model(input=[inputs], output=[mixture_density(2)(net)])
    """

    def layer(X):
        pi = keras.layers.Dense(2, activation='softmax')(X)
        mu = keras.layers.Dense(2, activation='linear')(X)
        prec = keras.layers.Dense(2, activation=K.abs)(X)
        return keras.layers.Merge(mode='concat')([pi,mu,prec])

    return layer


def mixture_density_loss(nb_components, target_dimension=1):

    """ Compute the mixture density loss:
        \begin{eqnarray}
          P(Y|X) = \sum_i P(C_i) N(Y|mu_i(X), beta_i(X)) \\
          Loss(Y|X) = - log(P(Y|X))
        \end{eqnarray}
    """

    def loss(y_true, y_pred):

        batch_size = K.shape(y_pred)[0]

        # Each row of y_pred is composed of (in order):
        # 'nb_components' prior probabilities
        # 'nb_components'*'target_dimension' means
        # 'nb_components'*'target_dimension' precisions
        priors = y_pred[:,:nb_components]

        m_i0 = nb_components
        m_i1 = m_i0 + nb_components * target_dimension
        means = y_pred[:,m_i0:m_i1]

        p_i0 = m_i1
        p_i1 = p_i0 + nb_components * target_dimension
        precs = y_pred[:,p_i0:p_i1]

        # Now, compute the (x - mu) vector. Have to reshape y_true and
        # means such that the subtraction can be broadcasted over
        # 'nb_components'
        means = K.reshape(means, (batch_size , nb_components, target_dimension))
        x = K.reshape(y_true, (batch_size, 1, target_dimension)) - means


        # Compute the dot-product over the target dimensions. There is
        # one dot-product per component per example so reshape the
        # vectors such that a batch_dot product can be carried over
        # the axis of target_dimension
        x = K.reshape(x, (batch_size * nb_components, target_dimension))
        precs = K.reshape(precs, (batch_size * nb_components, target_dimension))
        # reshape the result into the natural structure
        expargs = K.reshape(K.batch_dot(-0.5 * x * precs, x, axes=1), (batch_size, nb_components))

        # There is also one determinant per component per example
        dets = K.reshape(K.abs(K.prod(precs, axis=1)), (batch_size, nb_components))
        norms = K.sqrt(dets/np.power(2*np.pi,target_dimension)) * priors

        # LogSumExp, for enhanced numerical stability
        x_star = K.max(expargs, axis=1, keepdims=True)
        logprob = - x_star - K.log(K.sum(norms * K.exp(expargs - x_star), axis=1))

        return logprob

    return loss

########################################################################
# Sanity test

def gen_data(N):

    """ Generate a 2 component distribution by
        adding noise to sigmoid(x) and (1 - sigmoid(x)) """

    def component_1(N):
        x = np.random.uniform(-10, 10, N)
        y = 1.0 / (1.0 + np.exp(-x))
        z = np.random.normal(scale=0.05, size=N)
        return x, y + z
    def component_2(N):
        x = np.random.uniform(-10, 10, N)
        y = 1 - 1.0 / (1.0 + np.exp(-x))
        z = np.random.normal(scale=0.05, size=N)
        return x, y + z

    n1 = N / 2#np.random.randint(N+1)
    n2 = N - n1

    x1, y1 = component_1(n1)
    x2, y2 = component_2(n2)

    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])

    ishuffle = np.arange(N)
    np.random.shuffle(ishuffle)

    return x[ishuffle], y[ishuffle]


def main():

    trainX, trainY = gen_data(10000)
    validX, validY = gen_data(1000)
    testX, testY = gen_data(1000)

    # The target distribution has 2 components, so a 2 component
    # mixture should model it very well
    inputs = keras.layers.Input(shape=(1,))
    h = keras.layers.Dense(300, activation='relu')(inputs)
    model = keras.models.Model(input=[inputs], output=[mixture_density(2)(h)])

    # The gradients can get very large when the estimated precision
    # gets very large (small variance) which makes training
    # unstable. If this happens, look into the "clipvalue" or
    # "clipnorm" parameter of the keras optimizers to limit the size
    # of the gradients
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=mixture_density_loss(nb_components=2),
    )

    model.fit(
        x=trainX,
        y=trainY,
        batch_size=32,
        nb_epoch=100,
        validation_data=(validX, validY),
        callbacks=[
            keras.callbacks.ModelCheckpoint('mdn.h5', verbose=1, save_best_only=True)
        ],
        verbose=2
    )

    if keras.__version__.split('.')[0] == '1':
        saved = h5.File('mdn.h5', 'r+')
        if 'optimizer_weights' in saved.keys():
            del saved['optimizer_weights']
        saved.close()

    keras.activations.abs = K.abs
    model = keras.models.load_model(
        'mdn.h5',
        custom_objects={
            'loss': mixture_density_loss(nb_components=2)
        }
    )

    y_pred = model.predict(testX)
    y_smp = np.zeros(y_pred.shape[0])
    for i in range(y_pred.shape[0]):
        priors = y_pred[i,:2]
        means = y_pred[i,2:4]
        precs = y_pred[i,4:6]

        # Sample a component of the mixture according to the priors
        cpn = np.random.choice([0, 1], p=priors)
        # Sample a data point for the chosen mixture
        y_smp[i] = np.random.normal(loc=means[cpn], scale=1.0/np.sqrt(precs[cpn]))

    plt.scatter(testX, testY, label='True data')
    plt.scatter(testX, y_smp, label='Generated data')
    plt.legend(loc='best')
    # The distributions should match very well!
    plt.show()

if __name__ == '__main__':
    main()
