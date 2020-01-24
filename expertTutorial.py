from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


mdata = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mdata.load_data()

x_train = x_train / 255
x_test = x_test / 255

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(32)


class ExModel(tf.keras.Model):
    def __init__(self):
        super(ExModel, self).__init()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


model = ExModel()

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')


@tf.function
def train_step(images, labels):
    with tf.Gradientape() as tape:
        predictions = model(images, trainin=True)
        loss = loss_obj(labels, predictions)
    gradients = tape.gradients(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_acc(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_obj(labels, predictions)

    test_loss(t_loss)
    test_acc(labels, predictions)


epochs = 5

for epoch in range(epochs):
    train_loss.reset_states()
    train_acc.reset_states()
    test_loss.reset_states()
    test_acc.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'epoch {}, loss: {}, acc: {}, ' \
               'test loss: {}, test acc: {}'

    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_acc.result() * 100,
                          test_loss.result(),
                          test_acc.result() * 100))

print("end")
