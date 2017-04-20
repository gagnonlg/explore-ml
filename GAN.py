import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T

##### Config

z_dim = 10
h_dim = 100
d_h_dim = 100
output_dim = 2

training_iterations = 10000
k_steps = 10
m = 32

theano.config.floatX = 'float32'

np.random.seed(1066)

g_lr = 0.01
d_lr = 0.001

##### Data

def sample_data(N):
    mean = [100.0, -100.0]
    cov = [
        [1.00, 0.20],
        [0.20, 0.25]
    ]
    return np.random.multivariate_normal(mean, cov, N).astype('float32')

def sample_z_prior(N):
    return np.random.uniform(size=(N,z_dim)).astype('float32')

##### Generator
print '-> Defining generator'

g_W0 = theano.shared(
    np.random.uniform(
        low=-1.0/np.sqrt(z_dim),
        high=1.0/np.sqrt(z_dim),
        size=(z_dim, h_dim)
    ).astype('float32')
)
g_b0 = theano.shared(np.zeros(h_dim).astype('float32'))
g_W1 = theano.shared(
    np.random.uniform(
        low=-1.0/np.sqrt(h_dim),
        high=1.0/np.sqrt(h_dim),
        size=(h_dim, output_dim)
    ).astype('float32')
)
g_b1 = theano.shared(np.zeros(output_dim).astype('float32'))

def G(Z):
    return T.dot(T.nnet.relu(T.dot(Z, g_W0) + g_b0), g_W1) + g_b1

##### Discriminator
print '-> Defining discriminator'

d_V0 = theano.shared(
    np.random.uniform(
        low=-1.0/np.sqrt(output_dim),
        high=1.0/np.sqrt(output_dim),
        size=(output_dim, d_h_dim)
    ).astype('float32')
)
d_c0 = theano.shared(np.zeros(d_h_dim).astype('float32'))
d_V1 = theano.shared(
    np.random.uniform(
        low=-1.0/np.sqrt(d_h_dim),
        high=1.0/np.sqrt(d_h_dim),
        size=(d_h_dim, 1)
    ).astype('float32')
)
d_c1 = theano.shared(np.zeros(1).astype('float32'))

def D(X):
    return T.nnet.sigmoid(
        T.dot(T.nnet.relu(T.dot(X, d_V0) + d_c0), d_V1) + d_c1
    )

##### Generator training function
print '-> Compiling generator'

Z = T.matrix('Z')
g_loss = T.mean(- T.log(D(G(Z))))

g_params = [g_W0, g_b0, g_W1, g_b1]
grad_g_params = [T.grad(g_loss, p) for p in g_params]
g_updates = [
    (param, param - g_lr * gparam)
    for param, gparam in zip(g_params, grad_g_params)
]

g_train = theano.function(
    inputs=[Z],
    outputs=[g_loss, G(Z)],
    updates=g_updates
)

g_sample_func = theano.function(
    inputs=[Z],
    outputs=[G(Z)],
)

##### Discriminator training function
print '-> Compiling discriminator'

x_true = T.matrix()
z_prior = T.matrix()
d_loss = -T.mean(T.log(D(x_true)) + T.log(1 - D(G(z_prior))))

d_params = [d_V0, d_c0, d_V1, d_c1]
grad_d_params = [T.grad(d_loss, p) for p in d_params]

d_updates = [
    (param, param - d_lr * gparam)
    for param, gparam in zip(d_params, grad_d_params)
]

d_train = theano.function(
    inputs=[x_true, z_prior],
    outputs=[d_loss, D(x_true), D(G(z_prior))],
    updates=d_updates
)

##### Training loop
print '-> Training'

for epoch in range(training_iterations):
    d_losses = []
    for k in range(k_steps):
        zs = sample_z_prior(m)
        xs = sample_data(m)
        loss, dx, dgz = d_train(xs, zs)
        d_losses.append(loss)
    # print dx
    # print dgz
    zs = sample_z_prior(m)
    g_loss, g_sample = g_train(zs)


    if epoch % 100 == 0:
    
        print 'epoch {}: d_loss={}, g_loss={}'.format(
            epoch,
            np.mean(d_losses),
            g_loss
        )

        print 'epoch {}: example g(z): {}'.format(
            epoch,
            g_sample[0]
        )
    
###### Tests
print '-> Evaluating results'

xtrue = sample_data(100)
plt.scatter(xtrue[:,0], xtrue[:,1], color='b')

xgen = g_sample_func(sample_z_prior(100))[0]
plt.scatter(xgen[:,0], xgen[:,1], color='r')

plt.show()

