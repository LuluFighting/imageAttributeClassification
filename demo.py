import argparse
from pathlib import Path
from model import get_model
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

def get_args():
    parser = argparse.ArgumentParser(description="multi attributes predict demo",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--testImg",type=str,default=None,help='testing Img')
    parser.add_argument("--resume",type=str,default=None,help="Model weight to be tested")
    args = parser.parse_args()
    return args
class AttrRecognition:
    def __init__(self,resume_path = './checkpoint/epoch006step680_0.02474_4.6562.pth'):
        print("=> creating model")
        self.model = get_model(pretrained=None)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        if Path(resume_path).is_file():
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path,map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'".format(resume_path))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(resume_path))
        if self.device == 'cuda':
            cudnn.benchmark = True
    def predict(self,img_path=None,img_size = 224):
        '''
        :param img_path:
        :return: gender,glasses,hat,age
        gender:1为男士，0为女士
        glasses:1为戴眼镜,0为不戴眼镜
        hat:1为戴帽子，0为不戴帽子
        age:预测年龄值
        '''
        assert(Path(img_path).is_file())
        img = cv2.imread(str(img_path),1)
        img = cv2.resize(img,(img_size,img_size)).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            input_img = torch.from_numpy(np.transpose(img,(2,0,1)))
            input_img = input_img.unsqueeze(0)
            input_img = input_img.to(self.device)
            outputs = self.model(input_img)
            # gender = outputs[:, 0:2]
            # glasses = outputs[:, 2:4]
            # hat = outputs[:, 4:6]
            output_age = F.softmax(outputs[:, 0:101],dim=1).cpu().numpy()
            # pred_gender = torch.max(gender,1)[1].cpu().numpy().item()
            # pred_glasses = torch.max(glasses,1)[1].cpu().numpy().item()
            # pred_hat = torch.max(hat,1)[1].cpu().numpy().item()
            ages = np.arange(0,101)
            pred_age = (output_age * ages).sum(axis=-1).item()

        return pred_gender,pred_glasses,pred_hat,int(pred_age)

def main():
    args = get_args()
    attri = AttrRecognition(args.resume) if args.resume is not None else AttrRecognition()
    pred_gender, pred_glasses, pred_hat, pred_age = attri.predict(args.testImg)
    print('gender is ',pred_gender)
    print('glasses is ',pred_glasses)
    print('hat is ',pred_hat)
    print('age is ',pred_age)
if __name__ == '__main__':
    main()


