import argparse
from pathlib import Path
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import pretrainedmodels
import pretrainedmodels.utils
from model import get_model
import dataset
from default import _C as cfg
from train import validate
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--age_dir", type=str, required=True, help="Age data root directory")
    # parser.add_argument("--gender_dir", type=str, required=True, help="Gender data root directory")
    # parser.add_argument("--hat_dir", type=str, required=True, help="Hat data root directory")
    # parser.add_argument("--glasses_dir", type=str, required=True, help="Glasses data root directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint if any")
    # parser.add_argument("--checkpoint", type=str, default="checkpoint", help="Checkpoint directory")
    # parser.add_argument("--tensorboard", type=str, default=None, help="Tensorboard log directory")
    # parser.add_argument("--multi_gpu",action="store_true",help="Use multi GPUs (data parallel")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()

    # creat model
    print("=> creating model ")
    model = get_model()
    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # load checkpoint
    resume_path = args.resume

    if Path(resume_path).is_file():
        print("=> loading checkpoint '{}".format(resume_path))
        checkpoint = torch.load(resume_path,map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(resume_path))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))

    if 'cuda' in device:
        cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model,device_ids=[3,4,5])

    age_test_dataset = dataset.AgeDataset(args.age_dir,'test',cfg.MODEL.IMG_SIZE,augment=False)
    age_test_loader = DataLoader(dataset=age_test_dataset,batch_size=cfg.TEST.BATCH_SIZE,shuffle=True)

    # gender_test_dataset = dataset.GenderDataset(args.gender_dir,'test',cfg.MODEL.IMG_SIZE,augment=False)
    # gender_test_loader = DataLoader(dataset=gender_test_dataset,batch_size=cfg.TEST.BATCH_SIZE,shuffle=True,
    #                                 num_workers=cfg.TEST.WORKERS)
    #
    # hat_test_dataset = dataset.HatDataset(args.hat_dir,'test',cfg.MODEL.IMG_SIZE,augment=False)
    # hat_test_loader = DataLoader(dataset=hat_test_dataset,batch_size=cfg.TEST.BATCH_SIZE,shuffle=True,
    #                              num_workers=cfg.TEST.WORKERS)
    #
    # glasses_test_dataset = dataset.GlassDataset(args.glasses_dir,'test',cfg.MODEL.IMG_SIZE,augment=False)
    # glasses_test_loader = DataLoader(dataset=glasses_test_dataset,batch_size=cfg.TEST.BATCH_SIZE,shuffle=True,
    #                                  num_workers=cfg.TEST.WORKERS)

    print("=> start testing")

    _,_,test_mae = validate(age_test_loader,model,None,0,device,'age')
    #gender_loss,gender_acc = validate(gender_test_loader,model,None,0,device,'gender')
    #hat_loss,hat_acc = validate(hat_test_loader,model,None,0,device,'hat')
    #glasses_loss,glasses_acc = validate(glasses_test_loader,model,None,0,device,'glasses')

    print(f"age test mae: {test_mae:.3f}")
    #print(f"gender test accuracy: {gender_acc:.3f}")
    #print(f"hat test accuracy: {hat_acc:.3f}")
    #print(f"glasses test accuracy: {glasses_acc:.3f}")


if __name__ == '__main__':
    main()