from ef_model import efficientnet_b3
import paddle
encoder = efficientnet_b3()

input = paddle.rand([1,3,384,640])

output = encoder(input)

a=1