import argparse
from datetime import datetime
import os
from pathlib import Path
import time
import torch
import cv2
from PIL import Image
import PIL.ImageOps
import torch.backends.cudnn as cudnn
import numpy as np
import models_maeroad
import timm.optim.optim_factory as optim_factory
from dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.score import get_MSE

def get_args_parser():
    parser = argparse.ArgumentParser('TMAE', add_help=False)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch4', type=str, metavar='MODEL',
                        help='Name of model to train')


    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Dataset parameters
    parser.add_argument('--data_path', default='P1', type=str,
                        help='dataset path')
    parser.add_argument('--channel', default='1', type=int,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=2017, type=int)
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=2e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    
    parser.add_argument('--patch_size', type=int, default=4,
                        help='the size of the patch')
    parser.add_argument('--img_size', type=int, default=128,
                        help='the size of the flow')
    
    return  parser
    
def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cuda = True if torch.cuda.is_available() else False
    cudnn.benchmark = True

    datapath=os.path.join('D:\\项目\\TMAE\\tmae\\data',args.data_path)
    dataset_train=Dataset(datapath,channel=args.channel)
    dataset_valid=Dataset(datapath,'valid',channel=args.channel)
    dataloader_train=DataLoader(dataset_train, batch_size=16, shuffle=True,drop_last=True)
    dataloader_valid=DataLoader(dataset_valid, batch_size=16, shuffle=False,drop_last=True)

    bei_road_path = 'D:\\项目\\TMAE\\tmae\\road_map/beij1.png'
    road_map = torch.from_numpy(np.expand_dims(cv2.resize(np.array(PIL.ImageOps.invert(Image.open(bei_road_path).convert('L'))), (128,128), interpolation=cv2.INTER_LINEAR), 0)).float().to(device)
    road_map=road_map.reshape(1,road_map.shape[0],road_map.shape[1],road_map.shape[2])

    model = models_maeroad.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)
    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optim = torch.optim.AdamW(param_groups, lr=args.blr, betas=(0.9, 0.95))
    def patchify(imgs,patch_size):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], args.channel, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w,-1))
        return x
    def calc_entropy_2d(img):
        N,C,W,H=img.shape
        img=img.reshape(N,-1)
        norm_p=torch.nn.functional.normalize(img, dim=1)
        norm_p=norm_p.reshape(N,C,W,H)
        patches=patchify(norm_p,args.patch_size)     
        entropy=-(patches*torch.log2(patches))
        entropy[entropy.isnan()]=0
        entropy=entropy.sum(dim=2)
        return entropy
    entropy_all=torch.zeros(1024).to('cuda')
    with tqdm(range(len(dataloader_train))) as pbar:
        for(i,(_,flow_f,_)) in zip(pbar,dataloader_train):
                entropy=calc_entropy_2d(flow_f)
                entropy=entropy.sum(dim=0)
                entropy_all+=entropy
    entropy=entropy_all

    rmses = [np.inf]
    maes = [np.inf]
    out_path=os.path.join(args.output_dir,args.data_path)
    for e in range(args.epochs):
        model.train()
        losses = []
        with tqdm(range(len(dataloader_train))) as pbar:
            for(i,(_,flow_f,_)) in zip(pbar,dataloader_train):
                loss = model(flow_f,road_map,entropy,i)
                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(loss.item())
        avg_loss = np.sqrt(np.mean(losses))
        print(f'In epoch {e},  traning loss is {avg_loss}.')
        if (e % 5 == 0 and e!=0):
            print("validating...")
            model.eval()
            valid_time = datetime.now()
            total_mse, total_mae, total_mape = 0, 0, 0
            for j, (_,v_flow_f,v_ext) in enumerate(dataloader_valid):
                mse,mae,mape= model.forward_test(v_flow_f,entropy,road_map)
                total_mse += mse.cpu().detach().numpy() * len(v_flow_f)
                total_mae += mae * len(v_flow_f)
                total_mape += mape * len(v_flow_f)
            rmse = np.sqrt(total_mse / len(dataloader_valid.dataset))
            mae = total_mae / len(dataloader_valid.dataset)
            mape = total_mape / len(dataloader_valid.dataset)
            print("epoch\t{}\tRMSE\t{:.6f}\tMAE\t{:.6f}\tMAPE\t{:.6f}\ttime\t{}".format(e, rmse,mae,mape,datetime.now()-valid_time))
            if rmse < np.min(rmses):
                torch.save(model.state_dict(),
                           f'{out_path}/final_model.pt')
                f = open(f'{out_path}/results.txt', 'a')
                f.write("epoch\t{}\tRMSE\t{:.6f}\tMAE\t{:.6f}\tMAPE\t{:.6f}\n".format(e, rmse,mae,mape))
                f.close()
            rmses.append(rmse)
        pbar.set_postfix(loss=np.round(loss.cpu().detach().numpy().item(), 5))
        pbar.update(0)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        out_path=os.path.join(args.output_dir,args.data_path)
        dir_path = Path(out_path)
        if not dir_path.exists():  
            dir_path.mkdir(parents=True) 

    main(args)