import paddle
from paddle import Tensor

from paddle import nn
from paddleseg.cvlibs import param_init
import torch 
from torch import nn as nnn
import paddle.nn.functional as F



# # 测试遍历各层参数权重 
# paddle.device.set_device('gpu:0')
# class conv(paddle.nn.Layer):
#     def __init__(self):
#         super(conv, self).__init__()
#         self._conv = nn.Conv2D(4, 6, (3, 3))
#         self._linear = nn.Linear(1, 1)

# class MyLayer(nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self._conv = nn.Conv2D(4, 6, (3, 3),bias_attr=True )
#         self._linear = nn.Linear(1, 1)
#         self._bn = nn.BatchNorm2D(6, momentum=0.01, epsilon=1e-3)
#         self._dropout = nn.Dropout(p=0.5)
#         self.afda = nn.Linear(1,1)
#         self.lll = nn.Sequential(conv(), paddle.nn.Linear(1, 1), paddle.nn.Linear(1, 1))

#         for m in self.sublayers():
#             print(m)
#             print('\n')
#         for m in self.named_sublayers():
#             print(m)
#             print('\n')     
#         for m in self.children():
#             print(m)
#             print('\n')       
#         for m in self.named_children():
#             print(m)
#             print('\n')

#     def forward(self, input):
#         temp = self._linear(input)
#         temp = self._dropout(temp)
#         temp = self.lll(temp)           
#         return temp

# mylayer = MyLayer()

# a = torch.rand(2,2)
# b = paddle.rand([2,2])


# 冻结权重
# freeze_params = True
# if freeze_params:
#     for i, param in enumerate(mylayer.named_parameters()):  # 返回所有最底层的参数nn.Conv2D，nn.BatchNorm2D，nn.Linear
#         # mylayer.parameters()[i].training = False
#         print(param)
#         param[1].trainable = False
#         print(param)
#         print(param[1].trainable)
#     # 测试
#     for i, param in enumerate(mylayer.named_parameters()):
#         # mylayer.parameters()[i].training = False
#         print(param[1].trainable)

# mylayer.eval()

# class LLL(nnn.Module):
#     def __init__(self):
#         super(LLL, self).__init__()
#         self._conv1 = nnn.Conv2d(3, 6, (3, 3))
#         self._conv2 = nnn.Conv2d(6, 6, (3, 3))
    
#     def forward(self, input):
#         x = self._conv1(input)
#         x = self._conv2(x)
#         return x

# input = torch.rand(1,3,100,100)
# model = LLL()
# out = model(input)
# model.eval()
# for p in model.parameters():
#     print(p)

# # 初始化参数
# for layer in mylayer.named_sublayers():
#     # mylayer.parameters()[i].training = False
#     print(layer)
#     if isinstance(layer[1], nn.Conv2D):
#         # param_init.normal_init(layer[1].weight,mean=0.0, std=0.02)
#         param_init.kaiming_uniform(layer[1].weight, nonlinearity="relu")
#         # layer_init = nn.initializer.KaimingUniform(layer[1].weight, nonlinearity="relu")
#         # layer_init(layer[1].weight)
#         if layer[1].bias is not None:
#             param_init.constant_init(layer[1].bias,value=0)

#     elif isinstance(layer[1], nn.BatchNorm2D):
#         # nn.init.constant_(layer.weight, 1)
#         # nn.init.constant_(layer.bias, 0)
#         param_init.constant_init(layer[1].weight,value=1)
#         param_init.constant_init(layer[1].bias,value=0)

#     elif isinstance(layer[1], nn.Linear):
#         param_init.xavier_uniform(layer[1].weight)
#         # nn.init.xavier_uniform_(layer.weight)
#         if layer[1].bias is not None:
#             # nn.init.constant_(layer.bias, 0)
#             param_init.constant_init(layer[1].bias,value=0)
    

# 设置参数
# x = paddle.ones([2], dtype="float32")
# param = paddle.create_parameter(shape=x.shape,
#                         dtype=str(x.numpy().dtype),
#                         default_initializer=paddle.nn.initializer.Assign(x))
# paddle.sum
# w61 = paddle.ones([2], dtype="float32")
# pd_p6_w1 = paddle.create_parameter(shape=w61.shape,
#                         dtype=str(w61.numpy().dtype),
#                         default_initializer=paddle.nn.initializer.Assign(w61))
# pd_p6_w1.stop_gradient = False
# a = paddle.sum(pd_p6_w1, axis=0)
# weighta = pd_p6_w1 / (a + 1e-4)

# th_p6_w1 = nnn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
# b = torch.sum(th_p6_w1, dim=0)
# weightb = th_p6_w1 / (b + 1e-4)



# ModuleList Sequential 测试
# class net_modlist(nnn.Module):
#     def __init__(self):
#         super(net_modlist, self).__init__()
#         self.modlist = nnn.ModuleList([
#                        nnn.Conv2d(3, 20, 5),
#                        nnn.ReLU(),
#                         nnn.Conv2d(20, 64, 5),
#                         nnn.ReLU()
#                         ])

#     def forward(self, x):
#         for m in self.modlist:
#             x = m(x)
#         return x

# model = net_modlist()
# input = torch.rand([1, 3, 384, 640])
# output = model(input)

# class net_modlist(nnn.Module):
#     def __init__(self):
#         super(net_modlist, self).__init__()
#         self.modlist = nnn.Sequential(
#                        nnn.Conv2d(3, 20, 5),
#                        nnn.ReLU(),
#                         nnn.Conv2d(20, 64, 5),
#                         nnn.ReLU()
#                         )

#     def forward(self, x):
#         for m in self.modlist:
#             x = m(x)
#         return x

# model = net_modlist()
# input = torch.rand([1, 3, 384, 640])
# output = model(input)

# class net_modlist(nn.Layer):
#     def __init__(self):
#         super(net_modlist, self).__init__()
#         self.modlist = nn.Sequential(
#                        nn.Conv2D(3, 20, 5),
#                        nn.ReLU(),
#                         nn.Conv2D(20, 64, 5),
#                         nn.ReLU()
#                         )

#     def forward(self, x):
#         for m in self.modlist:
#             x = m(x)
#         return x

# model = net_modlist()
# input = paddle.rand([1, 3, 384, 640])
# output = model(input)

# 交换维度
# x = paddle.randn([2, 3, 4])
# x_transposed = paddle.transpose(x, perm=[1, 0, 2])
# print(x_transposed.shape)
# # [3L, 2L, 4L]
# x = paddle.randn([1, 36, 48, 80])
# feat = paddle.transpose(x, perm=[0, 2, 1, 3])
# feat = paddle.reshape(feat.clone(),[feat.shape[0] ,-1, 4])

# 测试F.interpolate 插值
# x = paddle.randn([1, 160, 6, 10])
# # x = F.interpolate(x, scale_factor=[2,2], mode="bilinear", align_corners=True)
# y = paddle.randn([1, 160, 6, 10])
# z = sum((x,y))
# view->unsqueeze+reshape
# x = paddle.randn([1, 9, 48, 80])
# feat = paddle.unsqueeze(x, axis=4)
# feat = paddle.reshape(feat, [1, -1, 1])
# feats = F.sigmoid(feat)



# 自定义层
# # from paddle.autograd import PyLayer
# # Inherit from PyLayer
# class cus_tanh(PyLayer):
#     @staticmethod
#     def forward(ctx, x, func1, func2=paddle.square):
#         # ctx is a context object that store some objects for backward.
#         ctx.func = func2
#         y = func1(x)
#         # Pass tensors to backward.
#         ctx.save_for_backward(y)
#         return y

#     @staticmethod
#     # forward has only one output, so there is only one gradient in the input of backward.
#     def backward(ctx, dy):
#         # Get the tensors passed by forward.
#         # y = ctx.saved_tensor()[0] 等价 y, = ctx.saved_tensor()
#         y, = ctx.saved_tensor()
#         grad = dy * (1 - ctx.func(y))
#         # forward has only one input, so only one gradient tensor is returned.
#         return grad

# data = paddle.randn([2, 3], dtype="float64")
# data.stop_gradient = False
# z = cus_tanh.apply(data, func1=paddle.tanh)
# z.mean().backward()

# print(data.grad)



# 测试shape
# a = torch.rand([2,2])
# b = torch.rand(2,2)
# c = paddle.rand([2,2])
# ih, iw = a.size()[-2:]
# a = nn.Conv2D(3, 20, 5)
# b = nnn.Conv2d(3, 20, 5)
# MaxPool2D = nn.MaxPool2D(kernel_size=2, stride=2, padding=0, return_mask=False)
# MaxPool2d = nnn.MaxPool2d(kernel_size=2, stride=2, padding=0)

# a = paddle.rand([2,2])
# b = paddle.rand([1])
# c = paddle.divide(a,b)


# class MyLayer(nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self._conv = nn.Conv2D(4, 6, (3, 3),bias_attr=True )
#         self._linear = nn.Linear(1, 1)
#         self._bn = nn.BatchNorm2D(6, momentum=0.01, epsilon=1e-3)
#         self._dropout = nn.Dropout(p=0.5)
#         self.afda = nn.Linear(1,1)
#         self.lll = nn.Sequential((paddle.nn.Linear(1, 1)), paddle.nn.Linear(1, 1))


#         for m in self.named_sublayers():
#             print(m)
#             print('\n')     


#     def forward(self, input):
#         temp = self._linear(input)
#         temp = self._dropout(temp)
#         temp = self.lll(temp)           
#         return temp

# mylayer = MyLayer()

# data = paddle.rand([1,3,200,200]),
# model2 = nn.Sequential(
#     (paddle.nn.Linear(10, 2)),(paddle.nn.Linear(2, 3))
# )

# for m in model2.named_sublayers():
#     print(m)
#     print('\n')    
# model2['l1']  # access l1 layer
# model2.add_sublayer('l3', paddle.nn.Linear(3, 3))  # add sublayer
# res2 = model2(data)  # sequential execution



# 最值
# a = paddle.randn([46035, 4])
# a = paddle.randn([1, 4])
# a = paddle.to_tensor(a,dtype='int32')
# c = paddle.randn([21, 4])
# c = paddle.randn([4, 1])
# c = paddle.to_tensor(c,dtype='int32')
# iw = paddle.minimum(a, c)
# iw = paddle.minimum(paddle.unsqueeze(a, axis=1), c)

# print(a)
# print(c)
# d = paddle.to_tensor([a,c])
# iw = paddle.minimum(a[:,3], c[:,2])
# iw = paddle.minimum(paddle.unsqueeze(a[:, 3], axis=1), paddle.unsqueeze(c[:, 2], axis=0))
# iw = paddle.minimum(paddle.unsqueeze(a[:, 3], axis=1), c[:, 2])
# iw = paddle.minimum(paddle.unsqueeze(a[:, 3], axis=1), c[:, 2]) - torch.maximum(torch.unsqueeze(a[:, 1], 1), c[:, 0])
# b = paddle.maximum(a,c)
# print(b)




# 数组索引
a = paddle.randn([10])
b = paddle.randn([10]) > 0
c = a[b]
a[b] = 1000

d = paddle.randn([10]) > 1000  # 全False
e = paddle.randn([10]) > 0
f = d[e]
# d[e] = True  # 不支持对bool型变量进行索引

a = 1
