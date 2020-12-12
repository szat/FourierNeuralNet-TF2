import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
print(tf.__version__)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

from config.config import *
from my_utils.my_utils import *

reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('coeff')[:N_TRAIN,::R,::R]
y_train = reader.read_field('sol')[:N_TRAIN,::R,::R]

S_ = x_train.shape[1]
grids = []
grids.append(np.linspace(0, 1, S_))
grids.append(np.linspace(0, 1, S_))
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
grid = grid.reshape(1,S_,S_,2)

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
grid = tf.convert_to_tensor(grid, dtype=tf.float32)
x_train = tf.expand_dims(x_train, axis=3)
grid = tf.repeat(grid, repeats = N_TRAIN, axis = 0)
x_train = tf.concat([x_train, grid], axis=3)
y_train = tf.expand_dims(y_train, axis=3)



linear_ = tf.keras.layers.Dense(32)
permute_ = tf.keras.layers.Permute((3, 1, 2))
reshape_ = tf.keras.layers.Reshape((49*49, 32))
conv1d_ = tf.keras.layers.Conv1D(32, 1)
reshape_2 = tf.keras.layers.Reshape((49,49, 32))

linear_2 = tf.keras.layers.Dense(128)
linear_3 = tf.keras.layers.Dense(1)

# input will be shape == [10, 32, 49, 49]
# outpout will be shape == [10, 32, 49, 49]
class FourierBlock(layers.Layer):
    def __init__(self, in_channels, out_channels, modes1, modes2, name=None):
        super(FourierBlock, self).__init__()

    # @tf.function
    def call(self, input, training=False):
        # need to reshape first, then reshape after
        # TODO:
        return input


class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc0 = tf.keras.layers.Dense(32)
        # self.perm0 = tf.keras.layers.Permute((3,1,2))
        # self.block0 = FourierBlock(32, 32, 12, 12)
        # self.perm1 = tf.keras.layers.Permute((2,3,1))
        self.fc1 = tf.keras.layers.Dense(128)
        self.fc2 = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, input):
        x = self.fc0(input)
        # print(x.shape + "\n")
        # x = self.perm0(x)
        # print(x.shape + "\n")
        # x = self.block0(x)
        # x = self.perm1(x)
        # print(x.shape + "\n")
        x = self.fc1(x)
        # print(x.shape + "\n")
        x = self.fc2(x)
        # print(x.shape + "\n")
        return x

model = MyModel()

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


model.build(x_train.shape)
model.summary()

model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=2)

plot_model(model, to_file='Example.png')

model.build()

base_input = model.layers[0].input
base_output = model.layers[2].output
output = layers.Dense(10)(layers.Flatten()(base_output))
model = keras.Model(base_input, output)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)
model.save("pretrained")