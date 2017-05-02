import matplotlib.pyplot as plt
import numpy as np
np.random.seed(12122)
import keras
import keras.backend as K

class MixtureDensity(keras.layers.Dense):

    """ The Mixture Density output layer is implemented as a type of Dense
        layer in which the output vector is actually a concatenation
        of the predicted priors, means and precisions """

    def __init__(self, nb_components, target_dimension=1, **kwargs):
        """ MixtureDensity output layer

        Arguments:
          -- nb_components: the number of components used to model the data
          -- target_dimension: the dimensionality of the data
          -- **kwargs: Any arguments to a "Dense" keras layer, apart from
            "units" and "activation"
        """

        self.nb_components = nb_components
        self.target_dimension = target_dimension

        output_dim = self.nb_components * (1 + 2 * self.target_dimension)

        # Ensure that we control these settings
        kwargs.pop('activation', None)
        kwargs.pop('units', None)

        super(MixtureDensity, self).__init__(
            units=output_dim,
            **kwargs
        )

    def call(self, x):

        # This first block is the regular Dense layer output
        output = K.dot(x, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)

        # Apply softmax to the component probabilities subvectors
        cpn_probs = K.softmax(output[:, :self.nb_components])

        # No activations for the component means subvectors
        m_i0 = self.nb_components
        m_i1 = m_i0 + self.nb_components * self.target_dimension
        cpn_means = output[:, m_i0:m_i1]

        # Take absolute value of the component precisions subvector to
        # ensure that they are positive
        p_i0 = m_i1
        p_i1 = p_i0 + self.nb_components * self.target_dimension
        cpn_precs = K.abs(output[:, p_i0:p_i1])

        return K.concatenate([cpn_probs, cpn_means, cpn_precs], axis=1)

    def get_config(self):

        config = {
            'nb_components' : self.nb_components,
            'target_dimension' : self.target_dimension
        }

        base_config = super(MixtureDensity, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(input_dim=1, units=300))
    model.add(keras.layers.Activation('relu'))
    model.add(MixtureDensity(nb_components=2))

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
        epochs=100,
        validation_data=(validX, validY),
        callbacks=[
            keras.callbacks.ModelCheckpoint('mdn.h5', verbose=1, save_best_only=True)
        ],
        verbose=2
    )

    model = keras.models.load_model(
        'mdn.h5',
        custom_objects={
            'MixtureDensity': MixtureDensity,
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
