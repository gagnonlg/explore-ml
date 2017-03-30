import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

theano.config.floatX = 'float32'

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

    n = np.random.randint(2,6)

    i = np.random.randint(2)
    
    return np.random.normal(mean[i], scale[i], (n,2)), i

def show_distribution():
    for _ in range(500):
        smp, cls = gen_sample()
        plt.scatter(smp[:,0], smp[:,1], color='k' if cls == 0 else 'b')
    plt.show()

def make_dataset(N):

    dsetX = np.zeros((N, 5, 2))
    dsetY = np.zeros((N, 1))
    for i in range(N):
        smp, cls = gen_sample()
        dsetX[i,:smp.shape[0]] = smp
        dsetY[i,0] = cls

    return dsetX.astype('float32'), dsetY.astype('float32')

def rnn_step(x, h, U, b, W):
    return T.tanh(b + T.dot(U, x) + T.dot(W, h))


# x: n_features
n_features = 2
x = T.matrix('x')
# h: n_state
n_state = 5
h = theano.shared(np.random.uniform(size=n_state).astype('float32'))
# U*x -> U: n_state, n_features
U = theano.shared(
    np.random.uniform(
        size=(n_state, n_features),
    ).astype('float32')
)
# W*h -> W: n_state, n_state
W = theano.shared(
    np.random.uniform(
        size=(n_state, n_state),
    ).astype('float32')
)
# b: n_state
b = theano.shared(
    np.zeros(n_state).astype('float32')
)

initial_state = theano.shared(
    np.random.uniform(size=n_state).astype('float32')
)

results, updates = theano.scan(
    fn=rnn_step,
    outputs_info=initial_state,
    sequences=x,
    non_sequences=[U,b,W]
)

def pred_step(h, V, c):
    return T.nnet.sigmoid(c + T.dot(V, h))

# Vh + c = y
# V: n_out x n_state
# c: n_out
n_out = 1
V = theano.shared(np.random.uniform(size=(n_out,n_state)).astype('float32'))
c = theano.shared(np.random.uniform(size=(n_out,)).astype('float32'))

preds, pupds = theano.scan(
    fn=pred_step,
    outputs_info=None,
    sequences=results,
    non_sequences=[V, c]
)

y = T.vector('y')
loss = T.sum(T.nnet.binary_crossentropy(preds, y))


params = [U,W,b,V,c]
gparams = [T.grad(loss, p) for p in params]

gupdates = [
    (param, param - 0.001 * gparam)
    for param, gparam in zip(params, gparams)
]

rnnfunc = theano.function(
    inputs=[x, y],
    outputs=[results, preds, loss],
    updates=gupdates
)

rnntest = theano.function(
    inputs=[x],
    outputs=preds[-1]
)

trainX, trainY = make_dataset(1000)
testX, testY = make_dataset(1000)

for epoch in range(10):
    losses = []
    for i in range(1000):
        losses.append(rnnfunc(trainX[i], trainY[i])[-1])
    good = float(0.0)
    for i in range(1000):
        pred = rnntest(testX[i])
        pcls = (pred > 0.5).astype('float32')
        if pcls[0] == testY[i][0]:
            good += 1.0

    print 'epoch {}: loss: {} acc:{}'.format(
        epoch,
        np.mean(losses),
        (good/1000)
    )

    
