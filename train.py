import argparse
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter
import pretrainedmodels
import pretrainedmodels.utils
from sklearn.metrics import confusion_matrix
from model import get_model
import dataset
from default import _C as cfg

import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

WINDOWS_TEST_CONFUSION_MATRIX = 0
def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--age_dir",type=str,required=True,help="Age data root directory")

    parser.add_argument("--resume",type=str,default=None,help="Resume from checkpoint if any")
    parser.add_argument("--checkpoint",type=str,default="checkpoint",help="Checkpoint directory")
    parser.add_argument("--tensorboard",type=str,default=None,help="Tensorboard log directory")

    parser.add_argument("opts",default=[],nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self,val,n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def train(x,y,model,criterion,optimizer,device,train_type):
    assert(train_type in ('gender', 'glasses', 'hat', 'age'))
    model.train()
    x = x.to(device)
    y = y.to(device)
    outputs = model(x)
    gender = outputs[:,0:2]
    glasses = outputs[:,2:4]
    hat = outputs[:,4:6]
    age = outputs[:,6:16]
    loss = None
    if train_type == 'gender':
        loss = criterion(gender,y)
    elif train_type == 'glasses':
        loss = criterion(glasses,y)
    elif train_type == 'hat':
        loss = criterion(hat, y)
    else:
        y//=10
        loss = criterion(age, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def validate(validate_loader,model,criterion,epoch,device,valid_type):
    assert(valid_type in ('gender','glasses','hat','age'))
    model.eval()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()
    preds = []
    gt = []

    with torch.no_grad():  #表示该计算图下的计算不参与梯度,不进行反向传播
        with tqdm(validate_loader) as _tqdm:
            for x,y in _tqdm:
                x = x.to(device)
                y = y.to(device)

                outputs = model(x)
                gender = outputs[:, 0:2]
                glasses = outputs[:, 2:4]
                hat = outputs[:, 4:6]
                age = outputs[:, 6:16]
                preds.append(F.softmax(age,dim=-1).cpu().numpy())
                gt.append(y.cpu().numpy())
                if criterion is not None:
                    loss, predicted = None, None
                    if valid_type == 'gender':
                        loss = criterion(gender, y)
                        predicted = torch.max(gender, 1)[1]
                    elif valid_type == 'glasses':
                        loss = criterion(glasses, y)
                        predicted = torch.max(glasses, 1)[1]
                    elif valid_type == 'hat':
                        loss = criterion(hat, y)
                        predicted = torch.max(hat, 1)[1]
                    else:
                        y//=10
                        loss = criterion(age, y)
                        predicted = torch.max(age, 1)[1]
                    cur_loss = loss.item()
                    correct_num = predicted.eq(y).sum().item()
                    sample_num = x.size(0)
                    loss_monitor.update(cur_loss, sample_num)
                    accuracy_monitor.update(correct_num, sample_num)
                    _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg),
                                      acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

    if valid_type != 'age':
        return loss_monitor.avg , accuracy_monitor.avg
    else:
        preds = np.concatenate(preds,axis=0)
        gt = np.concatenate(gt,axis=0)
        ages = np.arange(5,100,10)
        ave_preds = (preds * ages).sum(axis=1)
        if WINDOWS_TEST_CONFUSION_MATRIX:
            ave_preds = ave_preds.astype(np.int)

            import pandas as pd
            df = pd.crosstab(gt, ave_preds)
            true_label = df.index.values
            pred_label = df.columns.values
            cm = df.values
            cm = cm/cm.sum(axis=1)[:,np.newaxis]
            for i in range(len(true_label)):
                xLabel = [true_label[i]]
                yLabel = pred_label
                data = cm[i,:].reshape(len(pred_label),1)
                fig,ax = plt.subplots(figsize=(50,20))
                ax.set_yticks(range(yLabel.shape[0]))
                ax.set_yticklabels(yLabel)
                ax.set_xticks(range(len(xLabel)))
                ax.set_xticklabels(xLabel)
                im = ax.imshow(data,cmap=plt.cm.hot_r,aspect=0.2)
                plt.colorbar(im)
                plt.title('age prediction of '+str(xLabel[0]))
                plt.savefig('./cm/pred_of_'+str(xLabel[0]))
        diff = ave_preds - gt
        mae = np.abs(diff).mean()
        return loss_monitor.avg , accuracy_monitor.avg , mae

def main():
    args = get_args()
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()         #使cfgNode和其所有子节点不可变
    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True,exist_ok=True)

    #create model
    print("=> creating model ")
    model = get_model()
    if cfg.TRAIN.OPT == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),lr=cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr=cfg.TRAIN.LR)
    #我在135服务器上一般选择3、4、5三张卡进行训练
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    #从checkpoint继续训练
    resume_path = args.resume

    if resume_path:
        if Path(resume_path).is_file():
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path,map_location='cpu')  #先将checkpoint的模型读取到cpu上
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model,device_ids=[3,4,5])

    if 'cuda' in device:
        cudnn.benchmark = True   #自动寻找当前配置的高效算法

    criterion = nn.CrossEntropyLoss().to(device)

    #年龄训练集和验证集
    age_train_dataset = dataset.AgeDataset(args.age_dir,"train",img_size=cfg.MODEL.IMG_SIZE,augment=True,
                                           age_stddev=cfg.TRAIN.AGE_STDDEV)
    age_train_loader = DataLoader(dataset=age_train_dataset,batch_size=cfg.TRAIN.BATCH_SIZE,shuffle=True,
                                  num_workers=cfg.TRAIN.WORKERS,drop_last=False)
    age_val_dataset = dataset.AgeDataset(args.age_dir,"valid",img_size=cfg.MODEL.IMG_SIZE,augment=False)
    age_val_loader = DataLoader(dataset=age_val_dataset,batch_size=cfg.TEST.BATCH_SIZE,shuffle=False,
                                num_workers=cfg.TRAIN.WORKERS,drop_last=False)


    #定义学习率衰减器
    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.LR_DECAY_RATE, last_epoch=start_epoch-1)
    best_val_mae = 10000.0
    train_writer = None

    train_loader_list = [age_train_loader]
    valid_loader_list = [age_val_loader]
    iters_list = [iter(age_train_loader)]
    valid_type = ['age','gender','hat','glasses']
    max_acc = [0,0.7,0.7,0.7]
    for epoch in range(start_epoch,cfg.TRAIN.EPOCHS):
        for step in range(cfg.TRAIN.STEPS):
            try:
                batch_x,batch_y = next(iters_list[step%4])
                if step%4 == 0:
                    train(batch_x, batch_y, model, criterion, optimizer, device, 'age')
                elif step%4==1:
                    train(batch_x, batch_y, model, criterion, optimizer, device, 'gender')
                elif step%4==2:
                    train(batch_x, batch_y, model, criterion, optimizer, device, 'hat')
                else:
                    train(batch_x, batch_y, model, criterion, optimizer, device, 'glasses')
            except StopIteration:
                iters_list[step % 4] = iter(train_loader_list[step % 4]) #重新从头开始迭代
            if step % 100 == 0:
                for i in range(4):
                    if i==0:
                        val_loss, val_acc, val_mae = validate(valid_loader_list[i], model, criterion, epoch, device, valid_type[i])
                        if val_mae < best_val_mae:
                            print(f"=> [epoch {epoch:03d},step {step:03d}] best val mae was improved from {best_val_mae:.3f} to {val_mae:.3f}")
                            model_state_dict = model.module.state_dict()
                            torch.save(
                                {
                                    'epoch': epoch + 1,
                                    'state_dict': model_state_dict,
                                    'optimizer_state_dict': optimizer.state_dict()
                                },
                                str(checkpoint_dir.joinpath(
                                    "epoch{:03d}step{:03d}_{:.5f}_{:.4f}.pth".format(epoch,step,val_loss, val_mae)))
                            )
                            best_val_mae = val_mae
                        else:
                            print(f"=> [epoch {epoch:03d},step {step:03d}] best val mae was not improved from {best_val_mae:.3f} ({val_mae:.3f})")
                    else:
                        val_loss,val_acc = validate(valid_loader_list[i], model, criterion, epoch, device, valid_type[i])
                        if val_acc > max_acc[i]:
                            print(f"=> [epoch {epoch:03d},step {step:03d}] best val acc of {valid_type[i]} was improved from {max_acc[i]:.3f} to {val_acc:.3f}")
                            max_acc[i] = val_acc
                            model_state_dict = model.module.state_dict()
                            torch.save(
                                {
                                    'epoch': epoch + 1,
                                    'state_dict': model_state_dict,
                                    'optimizer_state_dict': optimizer.state_dict()
                                },
                                str(checkpoint_dir.joinpath("epoch{:03d}step{:03d}_{:.5f}_{:.4f}{}.pth".format(epoch, step, val_loss, val_acc, max_acc[i])))
                            )
                        else:
                            print(f"=> [epoch {epoch:03d},step {step:03d}] best val acc of {valid_type[i]} was not improved from {max_acc[i]:.3f} ({val_acc:.3f}")
        scheduler.step()  #调整学习率

    print('=> training finished')
    print(f"additional opts: {args.opts}")
    print(f"best val mae: {best_val_mae:.3f}")
    print("max val acc (gender,hat,glasses) is {max_acc[1]:.3f},{max_acc[2]:.3f},{max_acc[3]:.3f")

if __name__ == '__main__':
    main()