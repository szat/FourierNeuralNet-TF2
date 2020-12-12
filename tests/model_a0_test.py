import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from models.model_a0 import *


class FourierBlockTest(tf.test.TestCase):

    def setUp(self):
        super(FourierBlockTest, self).setUp()
        self.seed = tf.random.set_seed(5)
        self.ch_i = 32
        self.ch_o = 32
        self.modes1 = 12
        self.modes2 = 12
        self.batch = 10
        self.w = 49

    def tearDown(self):
        pass

    def test_tensor_shape_FourierBlock(self):
        shape_in = [self.batch, self.ch_i, self.w, self.w]
        in_tensor = tf.random.uniform(shape=shape_in)
        block = FourierBlock(self.ch_i, self.ch_o, self.modes1, self.modes2)
        out_tensor = block(in_tensor)
        self.assertAllEqual(in_tensor.shape, out_tensor.shape)

    def test_dev_shape_FourierBlock(self):
        x_train = tf.random.uniform((self.batch, self.w, self.w, 3))
        fc0 = tf.keras.layers.Dense(32)
        perm0 = tf.keras.layers.Permute((3,1,2))
        # reshape_1 = tf.keras.layers.Reshape((49 * 49, 32))
        # conv1d_ = tf.keras.layers.Conv1D(32, 1)
        # reshape_2 = tf.keras.layers.Reshape((49, 49, 32))
        perm1 = tf.keras.layers.Permute((2,3,1))
        fc1 = tf.keras.layers.Dense(128)
        fc2 = tf.keras.layers.Dense(1)

        x_train = fc0(x_train)
        self.assertAllEqual(x_train.shape, tf.TensorShape([self.batch, self.w, self.w, self.ch_i]))

        x_train = perm0(x_train)
        self.assertAllEqual(x_train.shape, tf.TensorShape([self.batch,  self.ch_i, self.w, self.w]))

        x_train = perm1(x_train)
        self.assertAllEqual(x_train.shape, tf.TensorShape([self.batch, self.w, self.w, self.ch_i]))

        x_train = fc1(x_train)
        self.assertAllEqual(x_train.shape, tf.TensorShape([self.batch, self.w, self.w, 128]))

        x_train = fc2(x_train)
        self.assertAllEqual(x_train.shape, tf.TensorShape([self.batch, self.w, self.w, 1]))


    def test_trainable_var_FourierBlock(self):
        ...


if __name__ == '__main__':
    tf.test.main()
