import numpy as np
import matplotlib.pyplot as plt

x,y = np.loadtxt('sine.test.txt', unpack=True)


isort = np.argsort(x)
x = x[isort]
y = y[isort]

keras = np.loadtxt('nn_keras_prediction.txt')
theano = np.loadtxt('nn_theano_prediction.txt')

keras = keras[isort]
theano = theano[isort]

plt.plot(x, keras, label='keras', color='black')
plt.plot(x, theano, label='theano', color='green')
plt.plot(x, np.sin(x), 'k--', label='sin(x)')
plt.legend(loc='best')
plt.scatter(x,y, color='red')
plt.show()
