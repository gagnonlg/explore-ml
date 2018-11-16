import keras
import numpy as np


class EarlyStop_FScore0(keras.callbacks.Callback):

    def on_train_begin(self, *args, **kwargs):
        self.best = 0.0
        self.remain = 10

    def on_epoch_end(self, epoch, logs=None):
        y_true = self.validation_data[1][:, 0]
        y_pred = (np.argmax(
            self.model.predict(self.validation_data[0]),
            axis=-1
        ) == 0).astype(np.float32)

        isel = np.where(y_pred == 1)

        tp = float(np.count_nonzero(y_true[isel] == y_pred[isel]))
        fp = float(np.count_nonzero(1 - y_true[isel] == y_pred[isel]))
        p = float(np.count_nonzero(y_true))

        print tp, fp, p

        weights = self.validation_data[2]
        tp2 = np.sum(weights[np.where((y_pred==1)&(y_true == y_pred))])
        fp2 = np.sum(weights[np.where((y_pred==1)&(1 - y_true == y_pred))])
        p2 = np.sum(weights[np.where(y_true == 1)])
        print tp2, fp2, p2


        precision = tp / (tp + fp)
        recall = tp / p

        fscore = 2 * precision * recall / (precision + recall)
        print fscore

        if fscore > self.best:
            print 'fscore increased: %f -> %f' % (self.best, fscore)
            self.best = fscore
            self.remain = 10
        else:
            print 'fscore not increased: %f -> %f' % (self.best, fscore)
            self.remain -= 1
        print '%d epochs remaining' % self.remain
        if self.remain == 0:
            print 'stopping training'
            self.model.stop_training = True


def _data(n):
    x = np.random.normal(size=(n, 2)).astype(np.float32)

    y = np.empty_like(x)
    y[:, 0] = np.logical_xor(x[:, 0] >= 0, x[:, 1] >=0)
    y[:, 1] = np.logical_or(
        np.logical_and(x[:, 0] >= 0, x[:, 1] >=0),
        np.logical_and(x[:, 0] < 0, x[:, 1] < 0)
    )

    return x, y

x_train, y_train = _data(10000)

# print x_train
# print y_train
# exit()

input_node = keras.layers.Input((2,))
model = keras.layers.Dense(100, activation='relu')(input_node)
model = keras.layers.Dense(2, activation='softmax')(model)
model = keras.models.Model(inputs=input_node, outputs=model)
model.compile(optimizer='sgd', loss='categorical_crossentropy')
model.fit(
    x_train,
    y_train,
    epochs=100,
    verbose=2,
    callbacks=[EarlyStop_FScore0()],
    validation_split=0.1
)
