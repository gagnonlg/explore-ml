import numpy as np
import tensorflow as tf

BATCH_SIZE=128
N_EPOCHS = 200

# placeholder for the input tensor
# Batch dimension not specified
x = tf.placeholder(tf.float32, (None, 1))
# The hidden layer
h = tf.layers.Dense(units=100, activation=tf.nn.relu)(x)
# The output layer
y = tf.layers.Dense(units=1, activation=None)(h)

# Placeholder for the labels
y_true = tf.placeholder(tf.float32, (None, 1))

# The objective function
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y)
# The training algorithm
train = tf.train.AdamOptimizer().minimize(loss)

# Get the data
my_data = np.loadtxt('sine.training.txt')
my_data_train = my_data[:int(my_data.shape[0] * 0.9)]
my_data_valid = my_data[int(my_data.shape[0] * 0.9):]

dataset = tf.data.Dataset.from_tensor_slices(my_data_train)
dataset = dataset.shuffle(my_data_train.shape[0])
dataset = dataset.batch(BATCH_SIZE)
it_train = dataset.make_initializable_iterator()
next_batch = it_train.get_next()

my_test_data = np.loadtxt('sine.test.txt')


saver = tf.train.Saver()

best_val_loss = np.inf

# Graph is built, start the session
with tf.Session() as sess:

    # init everything
    sess.run(tf.global_variables_initializer())

    for i_epoch in range(N_EPOCHS):
        sess.run(it_train.initializer)
        epoch_losses = []
        while True:
            try:
                batch_data = sess.run(next_batch)
                _, this_loss = sess.run(
                    (train, loss),
                    feed_dict={
                        x: batch_data[:, [0]],
                        y_true: batch_data[:, [1]]
                    }
                )
                epoch_losses.append(this_loss)
            except tf.errors.OutOfRangeError:

                val_loss = sess.run(
                    loss,
                    feed_dict={
                        x: my_data_valid[:, [0]],
                        y_true: my_data_valid[:, [1]]
                    }
                )

                print(
                    "epoch #{}: training={}, validation={}, checkpoint={}".format(
                        i_epoch,
                        np.mean(epoch_losses),
                        val_loss,
                        'yes' if val_loss < best_val_loss else 'no'
                    )
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    saver.save(sess, '/tmp/model.ckpt')

                break


    print('Training done, evaluating test set')
    saver.restore(sess, '/tmp/model.ckpt')
    y_test = sess.run(y, feed_dict={x: my_test_data[:, [0]]})
    np.savetxt('nn_tensorflow_prediction.txt', y_test)
    print ('Done!')
