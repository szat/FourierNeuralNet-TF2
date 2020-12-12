from config.config import *
from my_utils.my_utils import *
# import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv1D, Activation, Layer
from tensorflow.keras.optimizers import Adam


class FourierBlock(Layer):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, name=None):
        super(FourierBlock, self).__init__()
        # self.W = Conv1D()

    @tf.function
    def call(self, input):
        x = input
        return x


# class AdditiveBlock(Layer):
#     def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, name=None):
#         super(AdditiveBlock, self).__init__()
#         self.fourier = FourierBlock()
#         self.W = Conv1D()
#         self.bn = BatchNorm2d()
#
#     @tf.function
#     def call(self, input):
#         x = input
#         return x


class MyModel(Model):
    def __init__(self, modes: int, width: int, name=None):
        super(MyModel, self).__init__(name=name)

        # self.fc_0 = Linear()
        # self.iter_0 = AdditiveBlock()
        # self.iter_1 = AdditiveBlock()
        # self.iter_2 = AdditiveBlock()
        # self.iter_3 = AdditiveBlock()
        # self.fc_1 = Linear()
        # self.fc_2 = Linear()

    @tf.function
    def call(self, inputs):
        x = self.block(inputs)
        return x
