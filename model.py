import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils
from torchvision import models
import torch
from thop import profile
import time


#在模型的最后一层中，输出的前两个节点用来预测性别，之后两个节点用来预测眼镜，之后两个节点用来预测帽子，后面所有节点用来预测年龄
def get_model(num_classes=12, pretrained='imagenet'):  # shuchu

    model = pretrainedmodels.nasnetamobile(num_classes=1000, pretrained=pretrained)

    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes) # 输入dim_feats, 输出num_classes
    return model

def main():
    model = get_model()
    print(model)
    myinput = torch.randn(1,3,224,224)
    flops,params = profile(model,inputs=(myinput,))
    print('flops is ',flops)
    print('params is ',params)
    model(myinput)
    start = time.time()
    for i in range(1000):
        model(myinput)
    end = time.time()
    print('time consume is ',(end-start)/1000.0)
if __name__ == '__main__':
    main()