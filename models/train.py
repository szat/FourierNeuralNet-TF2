from config.config import *
from my_utils.my_utils import *
from models.model_a0 import *
from tensorflow.keras.optimizers import Adam

reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('coeff')[:N_TRAIN,::R,::R]
# [:N_TRAIN,::R,::R][:,:S,:S]
y_train = reader.read_field('sol')[:N_TRAIN,::R,::R]
# [:N_TRAIN,::R,::R][:,:S,:S]

# reader = MatReader(TEST_PATH)
# x_test = reader.read_field('coeff')[:N_TEST,::R,::R]
# # [:N_TEST,::R,::R][:,:S,:S]
# y_test = reader.read_field('sol')[:N_TEST,::R,::R]
# [:N_TEST,::R,::R][:,:S,:S]

# Unit Gaussian Normalizer, what for?

S_ = x_train.shape[1]

# To each slice, add the X and Y grids, i.e. the positions of the
# data. Otherwise the neural network does not know what is where? Is this important?
grids = []
grids.append(np.linspace(0, 1, S_))
grids.append(np.linspace(0, 1, S_))
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
grid = grid.reshape(1,S_,S_,2)

# numpy -> tensor
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
grid = tf.convert_to_tensor(grid, dtype=tf.float32)
x_train = tf.expand_dims(x_train, axis=3)
grid = tf.repeat(grid, repeats = N_TRAIN, axis = 0)
x_train = tf.concat([x_train, grid], axis=3)

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
my_model = MyModel(name="the_model")
my_model.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])
my_model.build(input_shape=x_train.shape)
# Use fit if all of the data can fit into RAM
my_model.fit(x_train, y_train, batch_size=32, epochs=1)