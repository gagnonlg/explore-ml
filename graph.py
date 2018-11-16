import numpy as np
import matplotlib.pyplot as plt

x,y = np.loadtxt('sine.test.txt', unpack=True)


isort = np.argsort(x)
x = x[isort]
y = y[isort]

keras = np.loadtxt('nn_keras_prediction.txt')
tensorflow = np.loadtxt('nn_tensorflow_prediction.txt')

print keras.shape
print tensorflow.shape

keras = keras[isort]
tensorflow = tensorflow[isort]

#plt.plot(x, keras, label='keras', color='black')
plt.plot(x, tensorflow, label='tensorflow', color='green')
plt.plot(x, np.sin(x), 'k--', label='sin(x)')
plt.legend(loc='best')
plt.scatter(x,y, color='red')
plt.savefig('graph.png')
