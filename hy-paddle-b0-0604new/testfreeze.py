import paddle
# from hybridnets.model import ModelWithLoss
# from backbone import HybridNetsBackbone

# model1 = HybridNetsBackbone(num_classes=1, compound_coef=0,
#                             ratios=[(0.62, 1.58), (1.0, 1.0), (1.58, 0.62)], scales=[1, 1.624504792712471, 2.4966610978032238],
#                             seg_classes=2,
#                             seg_mode='multiclass')  # 导入模型
# model2 = HybridNetsBackbone(num_classes=1, compound_coef=0,
#                             ratios=[(0.62, 1.58), (1.0, 1.0), (1.58, 0.62)], scales=[1, 1.624504792712471, 2.4966610978032238],
#                             seg_classes=2,
#                             seg_mode='multiclass')  # 导入模型
ckpt1 = paddle.load('checkpoints\\bdd100k\\hybridnets-d0_37_8750.pdparams')
ckpt2 = paddle.load('checkpoints\\bdd100k\\freeze_seg.pdparams')
# for key, value in ckpt1['model'].items():
#     print(key)
i = 0
for (key1, value1), (key2, value2) in zip(ckpt1['model'].items(), ckpt2['model'].items()):
    # print(key1)
    # print(key2)
    # if key1 != key2:
    #     print(key1,key2)
    a = paddle.equal_all(value1, value2)
    if a == False:
        print(key1,key2)
        i = i + 1
        

a = 1