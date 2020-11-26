import os
from pathlib import Path

PROJECT_PATH = Path(__file__).parent.parent.resolve().__str__()

TRAIN_PATH = PROJECT_PATH + '/data/piececonst_r241_N1024_smooth1.mat'
TEST_PATH = PROJECT_PATH + '/data/piececonst_r241_N1024_smooth2.mat'

N_TRAIN = 1000
N_TEST = 100

BATCH_SIZE = 20
L_RATE = 0.001

EPOCHS = 500
STEP_SIZE = 100
GAMMA = 0.5

MODES = 12
WIDTH = 32

# R is the subsampling
R = 5
# Was like this, I think this is incorrect
# H = int(((421 - 1) / R) + 1)
# H is the number of tics, I think
H = int(((241 - 1) / R) + 1)
S = H


