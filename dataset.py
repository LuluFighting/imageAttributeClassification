import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
import os
from torch.utils.data import Dataset
from imgaug import augmenters as iaa
import dlib
import sys

DLIB_PREDICTOR = './shape_predictor_5_face_landmarks.dat'
FACE_ALIGNMENT = True
def face_aligment(img_path,padding):
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(DLIB_PREDICTOR)
    img = cv2.imread(str(img_path), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(img, 1)
    if len(dets)==0:
        return img
    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(sp(img,detection))
    image = dlib.get_face_chip(img,faces[0],size=224,padding=padding)
    return image
class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.OneOf([
                iaa.Sometimes(0.25,iaa.AdditiveGaussianNoise(scale=0.1 * 255)),
                iaa.Sometimes(0.25,iaa.GaussianBlur(sigma=(0,3.0)))
            ]),
            iaa.Affine(
                rotate=(-5,5),mode='edge',   #旋转正负20度
                scale={"x":(0.95,1.05),"y":(0.95,1.05)},  #图像缩放95%-105%
                translate_percent={'x':(-0.05,0.05),"y":(-0.05,0.05)} #平移+=20%
            ),
            iaa.AddToHueAndSaturation(value=(-10,10),per_channel=True),  #随机在某些像素上加值
            iaa.GammaContrast((0.3,2)),   #对对比度做自适应变化
            iaa.Fliplr(0.5),      #水平翻转
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        return img

class AgeDataset(Dataset):
    def __init__(self,data_dir,data_type,img_size=224,augment=False,age_stddev=1.0):
        assert(data_type in ('train','test','valid'))
        csv_path = Path(data_dir).joinpath(f"gt_avg_{data_type}.csv")

        img_dir = Path(data_dir).joinpath(data_type)
        self.img_size = img_size
        self.augment = augment
        self.age_stddev = age_stddev
        if augment:
            self.transform = ImgAugTransform()
        else:
            self.transform = lambda i: i
        if FACE_ALIGNMENT:
            self.detector = dlib.get_frontal_face_detector()
            self.sp = dlib.shape_predictor(DLIB_PREDICTOR)

        self.x = []
        self.y = []
        self.std = []
        # df = pd.read_csv(str(csv_path))
        df = pd.read_csv('./appa-real-release/gt_avg_train.csv')
        ignore_path = Path(__file__).resolve().parent.joinpath("ignore_list.csv")

        # ignore_img_names = list(pd.read_csv(str(ignore_path))['img_name'].values)  #读取ignore_list.csv中的文件
        ignore_img_names = []
        for _, row in df.iterrows():  #一行一行读csv文件
            img_name = row["file_name"]
            if img_name in ignore_img_names:
                continue
            img_path = img_dir.joinpath(img_name + '_face.jpg')
            assert(img_path.is_file())
            self.x.append(str(img_path))
            self.y.append(row["apparent_age_avg"])
            self.std.append(row["apparent_age_std"])

    def face_aligment(self,img_path,padding):
        img = cv2.imread(str(img_path), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = self.detector(img,1)
        if len(dets)==0:
            return img
        face = self.sp(img,dets[0])
        return dlib.get_face_chip(img,face,224,padding=padding)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        img_path = self.x[idx]
        age = self.y[idx]
        if self.augment:
            age+= np.random.randn() * self.std[idx] * self.age_stddev
        if FACE_ALIGNMENT:
            img = self.face_aligment(img_path,padding=0.25)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(str(img_path),1)
        img = cv2.resize(img,(self.img_size,self.img_size))
        img = self.transform(img).astype(np.float32)
        return torch.from_numpy(np.transpose(img,(2,0,1))) , np.clip(round(age),0,100)

class GenderDataset(Dataset):
    def __init__(self,data_dir,data_type,img_size=224,augment=False):
        assert(data_type in ("train","valid","test"))
        img_dir = Path(data_dir).joinpath(data_type)
        self.img_size = img_size
        self.augment = augment
        if FACE_ALIGNMENT:
            self.detector = dlib.get_frontal_face_detector()
            self.sp = dlib.shape_predictor(DLIB_PREDICTOR)
        if augment:
            self.transform = ImgAugTransform()
        else:
            self.transform = lambda i:i
        self.x = []
        self.y = []   #label标签，男士为1，女士为0
        man_dir = Path(img_dir).joinpath('man')
        woman_dir = Path(img_dir).joinpath('woman')
        for _,_,file_name_list in os.walk(man_dir):
            for file_name in file_name_list:
                img_path = man_dir.joinpath(file_name)
                assert(img_path.is_file())
                self.x.append(str(img_path))
                self.y.append(1)
        for _,_,file_name_list in os.walk(woman_dir):
            for file_name in file_name_list:
                img_path = woman_dir.joinpath(file_name)
                assert(img_path.is_file())
                self.x.append(str(img_path))
                self.y.append(0)
    def face_aligment(self,img_path,padding):
        img = cv2.imread(str(img_path), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = self.detector(img,1)
        if len(dets)==0:
            return img
        face = self.sp(img,dets[0])
        return dlib.get_face_chip(img,face,224,padding=padding)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        img_path = self.x[idx]
        label = self.y[idx]
        if FACE_ALIGNMENT:
            img = self.face_aligment(img_path,padding=0.5)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(str(img_path),1)
        img = cv2.resize(img,(self.img_size,self.img_size))
        img = self.transform(img).astype(np.float32)
        return torch.from_numpy(np.transpose(img,(2,0,1)))  , label

class HatDataset(Dataset):
    def __init__(self,data_dir,data_type,img_size=224,augment=False):
        assert(data_type in ("train","test","valid"))
        self.img_size = img_size
        self.augment = augment
        if augment:
            self.transform = ImgAugTransform()
        else:
            self.transform = lambda i:i
        self.x = []
        self.y = []   #帽子标签，戴帽子为1，不戴帽子为0
        img_dir = Path(data_dir).joinpath(data_type)
        hat_dir = Path(img_dir).joinpath('wear_hat')
        noHat_dir = Path(img_dir).joinpath('no_hat')
        for _,_,file_name_list in os.walk(hat_dir):
            for file_name in file_name_list:
                img_path = hat_dir.joinpath(file_name)
                assert(img_path.is_file())
                self.x.append(str(img_path))
                self.y.append(1)
        for _,_,file_name_list in os.walk(noHat_dir):
            for file_name in file_name_list:
                img_path = noHat_dir.joinpath(file_name)
                assert(img_path.is_file())
                self.x.append(str(img_path))
                self.y.append(0)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        img_path = self.x[idx]
        label = self.y[idx]
        img = cv2.imread(str(img_path),1)
        img = cv2.resize(img,(self.img_size,self.img_size))
        img = self.transform(img).astype(np.float32)
        return torch.from_numpy(np.transpose(img,(2,0,1))),label

class GlassDataset(Dataset):
    def __init__(self,data_dir,data_type,img_size=224,augment=False):
        assert(data_type in ('train','test','valid'))
        img_dir = Path(data_dir).joinpath(data_type)
        self.img_size = img_size
        self.augment = augment
        if FACE_ALIGNMENT:
            self.detector = dlib.get_frontal_face_detector()
            self.sp = dlib.shape_predictor(DLIB_PREDICTOR)
        if augment:
            self.transform = ImgAugTransform()
        else:
            self.transform = lambda i:i
        self.x = []
        self.y = []  #戴眼镜标签，1为戴眼镜，0为不戴眼镜
        glasses_dir = img_dir.joinpath('wear_glasses')
        noGlasses_dir = img_dir.joinpath('no_glasses')
        for _,_,file_name_list in os.walk(glasses_dir):
            for file_name in file_name_list:
                img_path = glasses_dir.joinpath(file_name)
                assert(img_path.is_file())
                self.x.append(img_path)
                self.y.append(1)
        for _,_,file_name_list in os.walk(noGlasses_dir):
            for file_name in file_name_list:
                img_path = noGlasses_dir.joinpath(file_name)
                assert(img_path.is_file())
                self.x.append(img_path)
                self.y.append(0)
    def face_aligment(self,img_path,padding):
        img = cv2.imread(str(img_path), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = self.detector(img,1)
        if len(dets)==0:
            return img
        face = self.sp(img,dets[0])
        return dlib.get_face_chip(img,face,224,padding=padding)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        img_path = self.x[idx]
        label = self.y[idx]
        if FACE_ALIGNMENT:
            img = self.face_aligment(img_path,padding=0.25)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(str(img_path),1)
        img = cv2.resize(img,(self.img_size,self.img_size))
        img = self.transform(img).astype(np.float32)
        return torch.from_numpy(np.transpose(img,(2,0,1))) , label

def run():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir',type=str,required=True)
    args = parser.parse_args()
    dataset = AgeDataset(args.data_dir,'train')
    print('train dataset len: {}'.format(len(dataset)))

if __name__ == '__main__':
    run()