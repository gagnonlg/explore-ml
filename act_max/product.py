import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras


def generate_data(size):
    x = np.random.uniform(-1, 1, (size, 2))

    y = np.prod(x, axis=1)
    y[y < 0] = -1
    y[y >= 0] = 1

    return x, y


n_train = 100000
n_test = 100000
x_training, y_training = generate_data(n_train)
x_test, y_test = generate_data(n_test)


##################################################################

network = keras.models.Sequential()

# **** Here, create and train the network using the keras api
network.add(keras.layers.Dense(20,activation='relu' ,input_dim=2))
network.add(keras.layers.Dense(1,activation='tanh'))
network.summary()
network.compile(loss='mse',optimizer='sgd')
history=network.fit(x_training,y_training,epochs=10,batch_size=200)
##################################################################

network.save('product_network_v1.tf')

y_pred = network.predict(x_test)

plt.hist(y_pred[y_test == -1], label='Negative product')
plt.hist(y_pred[y_test == +1], label='Positive product')
plt.legend(loc='best')
plt.xlabel('Network output')
plt.ylabel('N. of examples')
plt.savefig('product_test.png')
