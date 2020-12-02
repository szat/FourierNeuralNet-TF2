import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# x_train.shape == [1000, 49, 49, 3]

width=32
batchsize=10
size_x = 49
size_y = 49
t = torch.rand(1000, 49, 49, 3)
# batch it
t = t[0:10, :, :, :]
fc0 = nn.Linear(3, width)

tt = fc0(t)
# torch.Size([10, 49, 49, 32])

ttt = tt.permute(0, 3, 1, 2)
# torch.Size([10, 32, 49, 49])

w0 = nn.Conv1d(width, width, 1)

ttt_w0 = w0(ttt.view(batchsize, width, -1))
# torch.Size([10, 32, 2401])

ttt_w0 = ttt_w0.view(batchsize, width, size_x, size_y)
x2 = ttt_w0
# torch.Size([10, 32, 49, 49])



bn0 = torch.nn.BatchNorm2d(width)
ttt_w0_bn0 = bn0(ttt_w0)
# torch.Size([10, 32, 49, 49])


ttt.view(batchsize, width, -1).shape

ttt_w0 = w0(ttt)