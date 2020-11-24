from config.config import *
from my_utils.my_utils import *

reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('coeff')
# [:N_TRAIN,::R,::R][:,:S,:S]
y_train = reader.read_field('sol')
# [:N_TRAIN,::R,::R][:,:S,:S]

reader = MatReader(TEST_PATH)
x_test = reader.read_field('coeff')
# [:N_TEST,::R,::R][:,:S,:S]
y_test = reader.read_field('sol')
# [:N_TEST,::R,::R][:,:S,:S]

