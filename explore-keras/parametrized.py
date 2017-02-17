""" Implementation of the "toy example" in arXiv:1601.07913 """
import matplotlib.pyplot as plt
import numpy as np

means_training = np.arange(-2, 3, dtype=np.float32)
means_test = np.arange(-1.0, 1.5, 0.5, dtype=np.float32)
width = 0.25

n = 100000

x = np.random.uniform(-4,4,n)

n_train = (means_training.shape[0] + 1) * n
x_train = np.empty((n_train, 2))
y_train = np.empty(n_train)
i = 0
for m in means_training:
    x_train[i:i+n,0] = np.random.normal(m, width, n)
    x_train[i:i+n,1] = m
    y_train[i:i+n] = 1
    i += n

x_train[i:i+n,0] = np.random.uniform(-4,4,n)
x_train[i:i+n,1] = np.random.choice(means_training, n)
y_train[i:i+n] = 0

indices = np.arange(0, x_train.shape[0])
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]


import keras

# normalize
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
x_train -= mean
x_train /= std


model = keras.models.Sequential()
model.add(keras.layers.Dense(
    input_dim=2,
    output_dim=300,
    activation='relu'
))
model.add(keras.layers.Dense(
    input_dim=x_train.shape[1],
    output_dim=300,
    activation='relu'
))
model.add(keras.layers.Dense(
    output_dim=1,
    activation='sigmoid'
))

model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.SGD(lr=0.1, decay=1e-6, nesterov=True),
)

checkpoint = keras.callbacks.ModelCheckpoint(
    '1601.07913_toy_weights.h5',
    verbose=1,
    save_best_only=True
)


history = model.fit(
    x_train,
    y_train,
    nb_epoch=1,
    batch_size=128,
    validation_split=0.1,
    callbacks=[checkpoint],
    verbose=1
)

model.load_weights('1601.07913_toy_weights.h5')

for m in means_test:
    x = np.empty((1000, 2))
    x[:,0] = np.linspace(-4, 4, 1000)
    x[:,1] = m
    y = model.predict((x - mean)/std)

    if m in means_training:
        style='k-'
        plt.text(m-0.25, 1, str(m))

    else:
        style='r--'
        plt.text(m-0.25, 1.05, str(m))


    plt.plot(x[:,0], y, style)

plt.axis([-4,4,0,1.2])
plt.savefig('1601.07913_fit_result.png'.format(m))

