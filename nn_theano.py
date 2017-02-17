import numpy as np
import theano
import theano.tensor as T

# uncomment for debugging
#theano.config.optimizer = 'None'

class Layer(object):

    def __init__(self, input, n_in, n_out, linear=False):

        self.W = theano.shared(
            value=np.asarray(
                np.random.uniform(
                    low=-np.sqrt(24. / (n_in + n_out)),
                    high=np.sqrt(24. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX,
            ),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        linear_output = T.dot(input, self.W) + self.b
        self.output = (
            linear_output if linear
            else T.nnet.sigmoid(linear_output)
        )

        self.params = [self.W, self.b]
            
    def mean_squared_error(self, y):
        return T.mean(T.pow(self.output - y, 2))

class NeuralNetwork(object):

    def __init__(self, input, n_in, n_hidden, n_out):

        self.hidden_layer = Layer(
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            linear=False
        )
        
        self.output_layer = Layer(
            input=self.hidden_layer.output,
            n_in=n_hidden,
            n_out=n_out,
            linear=True
        )

        self.mean_squared_error = self.output_layer.mean_squared_error

        self.params = self.hidden_layer.params + self.output_layer.params

        self.input = input

        self.output = self.output_layer.output

def main():

    print '-> getting the data'

    # get the data
    datax,datay = np.loadtxt('sine.training.txt', unpack=True)
    testx,_ = np.loadtxt('sine.test.txt', unpack=True)

    # datax = np.atleast_2d(datax)
    # datay = np.atleast_2d(datay)

    # normalize the data
    mean = np.mean(datax)
    std = np.std(datax)
    datax -= mean
    datax /= std
    testx -= mean
    testx /= std

    datax = np.atleast_2d(datax).T
    datay = np.atleast_2d(datay).T
    testx = np.atleast_2d(testx).T

    xdata = theano.shared(
        np.asarray(
            datax,
            dtype=theano.config.floatX
        ),
        borrow=True
    )

    ydata = theano.shared(
        np.asarray(
            datay,
            dtype=theano.config.floatX
        ),
        borrow=True
    )

    xtest = theano.shared(
        np.asarray(
            testx,
            dtype=theano.config.floatX
        ),
        borrow=True
    )


    print '-> building the model'

    index = T.lscalar()
    x = T.matrix('x')
    y = T.matrix('y')

    network = NeuralNetwork(
        input=x,
        n_in=1,
        n_hidden=300,
        n_out=1
    )

    loss = network.mean_squared_error(y)

    gparams = [T.grad(loss, param) for param in network.params]

    updates = [
        (param, param - 0.01 * gparam)
        for param, gparam in zip(network.params, gparams)
    ]

    batch_size=128
    n_train_batches=datax.size/batch_size


    train_model = theano.function(
        inputs=[index],
        outputs=loss,
        updates=updates,
        givens={
            x: xdata[index * batch_size: (index + 1) * batch_size],
            y: ydata[index * batch_size: (index + 1) * batch_size],
        }
    )

    test_model = theano.function(
        inputs=[],
        outputs=network.output,
        givens={
            x: xtest
        }
    )

    print '-> training the model'
    
    nepochs = 200
    epoch = 0
    while (epoch < nepochs):

        losses = np.zeros(n_train_batches)

        for minibatch_index in range(n_train_batches):

            losses[minibatch_index] = train_model(minibatch_index)

        print 'epoch {}: avg. loss = {}'.format(epoch, np.mean(losses))
        epoch += 1

    
    outputs = test_model()
    np.savetxt('nn_theano_prediction.txt', outputs)
            
if __name__ == '__main__':
    main()
