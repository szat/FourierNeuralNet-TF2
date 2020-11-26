from config.config import *
from my_utils.my_utils import *
# import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Activation, Layer
from tensorflow.keras.optimizers import Adam


class TrivialBlock(Layer):
    def __init__(self, name=None):
        super(TrivialBlock, self).__init__()

        self.conv1A = Conv2D(32, (3, 3), padding="same")
        # self.act1A = Activation("softmax")

    @tf.function
    def call(self, inputs):
        x = self.conv1A(inputs)
        # x = self.act1A(x)
        return x


class MyModel(Model):
    def __init__(self, name=None):
        super(MyModel, self).__init__(name=name)
        self.block = TrivialBlock()

    @tf.function
    def call(self, inputs):
        x = self.block(inputs)
        return x
