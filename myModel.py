import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Net(torchvision.models.resnet50(True)):
    def __int__(self):
        super(Net,self).__init__()
        self.predict = nn.Linear(1000,2)
    def forward(self,x):
        x = super(Net,self).forward(x) #调用resNet50网络的前向传播函数
        x = F.relu(x)
        x = self.predict(x)
        return x
