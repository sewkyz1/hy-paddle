
import paddle
from paddle import Tensor

from paddle import nn
from paddleseg.cvlibs import param_init
# import torch 
# from torch import nn as nnn
# import paddle.nn.functional as F



# 测试遍历各层参数权重 
paddle.device.set_device('gpu:0')

class conv(paddle.nn.Layer):
    def __init__(self):
        super(conv, self).__init__()
        self._conv = nn.Conv2D(6, 6, (3, 3))


class MyLayer(nn.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()
        self._conv = nn.Conv2D(4, 6, (3, 3),bias_attr=True )
        self._bn = nn.BatchNorm2D(6, momentum=0.99, epsilon=1e-3)
        self.lll = nn.Sequential(conv(), nn.Conv2D(6, 1, (3, 3),bias_attr=True ))

    def forward(self, input):
        temp = self._conv(input)
        temp = self._bn(temp)
        temp = self.lll(temp)           
        return temp


class test(nn.Layer):
    def __init__(self, model):
        super(test, self).__init__()
        self.model = model

model = MyLayer()
model = test(model)
optim_params = []
for n,m in model.named_sublayers():
    print(n,m)
for n,m in model.named_parameters():
    print(n,m)
    if not m.stop_gradient:
        optim_params.append(m)

adamw = paddle.optimizer.AdamW(parameters=optim_params)
current_lr = adamw._learning_rate
obj = {'model': model.state_dict(), 'opt': adamw.state_dict(), 'epoch': 100}
a = 1