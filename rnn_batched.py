import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

np.random.seed(17430)


theano.config.floatX = 'float32'

#theano.config.optimizer='fast_compile'
#theano.config.exception_verbosity='high'

# generate data
def gen_sample():
    mean = [
        [0.0, 0.5],
        [0.5, 0.0]
    ]
    scale = [
        [0.15, 0.15],
        [0.25, 0.25]
    ]

    n = np.random.randint(2,7)

    i = np.random.randint(2)
    
    return np.random.normal(mean[i], scale[i], (n,2)), i

def show_distribution():
    for _ in range(500):
        smp, cls = gen_sample()
        plt.scatter(smp[:,0], smp[:,1], color='k' if cls == 0 else 'b')
    plt.show()

def make_dataset(N):

    dsetX = np.zeros((N, 6, 2))
    dsetY = np.zeros((N, 1))
    dsetM = np.zeros((N, 6))
    for i in range(N):
        smp, cls = gen_sample()
        dsetX[i,:smp.shape[0]] = smp
        dsetY[i,0] = cls
        dsetM[i,:smp.shape[0]] = 1

    return dsetX.astype('float32'), dsetY.astype('float32'), dsetM.astype('float32')

# let's define some tensors 
n_features = 2
n_state = 5
n_out = 1
batch_size = 10

X = T.tensor3('X') # (batch_size, n_step, n_features)

U = theano.shared(
    np.random.uniform(
        size=(n_state, n_features),
    ).T.astype('float32')
)

W = theano.shared(
    np.random.uniform(
        size=(n_state, n_state),
    ).T.astype('float32')
)
b = theano.shared(
    np.zeros(n_state).astype('float32')
)
initial_state = theano.shared(
    np.zeros((batch_size,n_state)).astype('float32')
)

V = theano.shared(
    np.random.uniform(
        size=(n_out, n_state),
    ).T.astype('float32')
)

c = theano.shared(
    np.zeros(n_out).astype('float32')
)


def rnn_step(X, H, U, b, W):
    """ One RNN step for all examples in a batch in parallel

    X: shape (batch_size, n_features) -> features at same time step for all examples in batch
    H: shape (batch_size, n_state) -> state at previous time step for all examples in batch
    U, b, W: RNN parameters
    
    returns: (batch_size, n_state) -> new state value at time step for all examples in batch
    """
    return T.tanh(b + T.dot(X, U) + T.dot(H, W))

results, updates = theano.scan(
    fn=rnn_step,
    outputs_info=T.zeros_like(initial_state),
    sequences=X.dimshuffle(1, 0, 2),
    non_sequences=[U,b,W]
)
# results: (n_step, batch_size, n_state)

def pred_step(H, V, c):
    return T.nnet.sigmoid(c + T.dot(H, V))

preds, pupds = theano.scan(
    fn=pred_step,
    outputs_info=None,
    sequences=results,
    non_sequences=[V,c]
)
# preds: (n_step, batch_size, n_out)

# ## SGD machinery
Y = T.matrix('Y')
mask = T.matrix('M')

loss = T.mean(
    T.sum(
        mask.dimshuffle(1, 0, 'x') * T.nnet.binary_crossentropy(preds, Y.dimshuffle('x', 0, 1)),
        axis=0
    ),
)
params = [U, W, b, V, c]
gparams = [T.grad(loss, p) for p in params]
updates = [
    (param, param - 0.001 * gparam)
    for param, gparam in zip(params, gparams)
]

rnnfunc = theano.function(
    inputs=[X, Y, mask],
    outputs=[results, preds, loss],
    updates=updates
)

rnntest = theano.function(
    inputs=[X],
    outputs=preds[-1]
)

trainX, trainY, trainM = make_dataset(1000)
testX, testY, _ = make_dataset(1000)

if __name__ == '__main__':
    for epoch in range(100):
        losses = []
        for i in range(0, 1000, batch_size):
            states, preds, ls = rnnfunc(
                trainX[i:i+batch_size],
                trainY[i:i+batch_size],
                trainM[i:i+batch_size]
            )
            losses.append(ls)
        good = float(0.0)
        for i in range(0, 1000, batch_size):
            pred = rnntest(testX[i:i+batch_size])
            pcls = (pred > 0.5).astype('float32')
            good += np.count_nonzero(pcls == testY[i:i+batch_size])

        print 'epoch {}: loss: {} acc:{}'.format(
            epoch,
            np.mean(losses),
            (good/1000)
        )
