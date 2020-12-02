from config.config import *
from my_utils.my_utils import *
# import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv1D, Activation, Layer
from tensorflow.keras.optimizers import Adam


class FFT_W_Block(Layer):
    def __init__(self, name=None):
        super(FFT_W_Block, self).__init__()
        self.W = Conv1D()

    @tf.function
    def call(self, input):
        x = input
        return x


class MyModel(Model):
    def __init__(self, name=None):
        super(MyModel, self).__init__(name=name)
        self.block = FFT_W_Block()

    @tf.function
    def call(self, inputs):
        x = self.block(inputs)
        return x
