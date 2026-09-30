import argparse
import datetime
import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import load_datasets
# Add Baselines_1 to path so we can import its modules from project root
ROOT = Path(__file__).parent
BASELINES_DIR = ROOT / "Models"
if str(BASELINES_DIR) not in sys.path:
    sys.path.append(str(BASELINES_DIR))
from utils import (  # type: ignore
    AverageMeter,
    initialize_logger,
    save_checkpoint,
    time2file_name,
    Loss_MAE,
    Loss_RMSE,
    Loss_PSNR,
)
from AWAN import AWAN  # type: ignore
from Restormer import Restormer  # type: ignore
from baseneuralop import ushapedno, fourierno
from SSRAN import SSRAN
def build_model(method: str, pretrained: bool=False):
    method = method.lower()
    if method == "restormer":
        model = Restormer(inp_channels=12, out_channels=229)
    elif method == "awan":
        model = AWAN(inplanes=12, planes=229)
    elif method =="uno":
        model = ushapedno(in_channels=12, out_channels =229, hidden_channels = 512)
    elif method =="fno":
        model = fourierno(in_channels=1, out_channels =1, hidden_channels = 16)
    elif method =="ssran":
        model = SSRAN(in_channels=12, out_channels =229)
    else:
        raise ValueError(f"Unknown method {method}")

    if pretrained:
        if not os.path.isfile(pretrained):
            raise FileNotFoundError(f"Pretrained model not found: {pretrained}")
        state = torch.load(pretrained, map_location="cpu")
        if "state_dict" in state:
            model.load_state_dict(state["state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained weights from {pretrained}")
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        default="awan",
        choices=["restormer", "awan","uno","fno", "ssran"],
        help="Model to train",
    )
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--init_lr", type=float, default=5e-5, help="initial learning rate")
    parser.add_argument("--outf", type=str, default="./exp/", help="path log files")
    parser.add_argument("--data_root", type=str, default="../dataset/")
    parser.add_argument("--patch_size", type=int, default=16, help="patch size")
    parser.add_argument("--stride", type=int, default=8, help="stride")
    parser.add_argument("--gpu_id", type=str, default="0", help="CUDA visible devices")
    return parser.parse_args()


def main():
    opt = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    # load dataset
    print("\nloading dataset ...")
    train_data, val_data, _ = load_datasets()
    print(f"Iteration per epoch: {len(train_data)}")
    print("Validation set samples: ", len(val_data))

    # iterations
    total_iteration = opt.epochs

    # loss function
    criterion_mae = Loss_MAE()
    criterion_rmse = Loss_RMSE()
    criterion_psnr = Loss_PSNR()
    srf = np.load('/workspace/SRNO/srf_weights_clipped.npy')
    # model
    model = build_model(opt.method, opt.pretrained_model_path).cuda()
    print("Parameters number is ", sum(param.numel() for param in model.parameters()))
    model = model.float()
    # output path
    date_time = time2file_name(str(datetime.datetime.now()))
    opt.outf = os.path.join(opt.outf, opt.method, date_time)
    os.makedirs(opt.outf, exist_ok=True)

    if torch.cuda.is_available():
        print('cuda!')
        model.cuda()
        criterion_mae.cuda()
        criterion_rmse.cuda()
        criterion_psnr.cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=opt.init_lr)

    # logging
    log_dir = os.path.join(opt.outf, "train.log")
    logger = initialize_logger(log_dir)

    # Resume
    resume_file = opt.pretrained_model_path
    if resume_file is not None and os.path.isfile(resume_file):
        print(f"=> loading checkpoint '{resume_file}'")
        checkpoint = torch.load(resume_file)
        iteration = checkpoint["iter"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        iteration = 0

    cudnn.benchmark = True
    record_mse_loss = 1000
    train_loader = DataLoader(
            dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True
        )
    val_loader = DataLoader(dataset=val_data, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    total_iteration = opt.epochs*len(train_loader)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)
    while iteration < total_iteration:
        model.train()
        losses = AverageMeter()
        epoch_loss=0
        for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader),bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",):
            labels = labels.cuda().float()
            if opt.method == 'fno' :
                basis=images.cuda().float()
                B, C, H, W = images.shape
                idx = torch.round(
                    torch.linspace(5, 204, steps=C, device=images.device)
                ).long()

                out = torch.zeros(B, 229, H, W, device=images.device, dtype=images.dtype)
                images = out.index_copy_(1, idx, images)

                B,C, H,W = images.shape
                images=images.reshape(B,1,H,W,C)
            images = images.cuda().float()
            optimizer.zero_grad()
            output = model(images)
            if opt.method=='fno':
                output= output.squeeze(1) #B,H,W,C
                output= output.permute(0, 3, 1, 2).contiguous()
            if torch.isnan(output).any() or torch.isinf(output).any():
                print("NaN/Inf output detected — stopping")
                iteration = total_iteration
                break
            loss = criterion_mae(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            if not torch.isfinite(loss):
                print("NaN/Inf loss detected — stopping")
                iteration = total_iteration
                break
            losses.update(loss.item())
            iteration+=1    
        
        print(f"[iter:{iteration}/{total_iteration}],lr={lr:.9f},train_losses.avg={losses.avg:.9f}")
        mse_loss, rmse_loss, psnr_loss = validate(val_loader, model, criterion_mae, criterion_rmse, criterion_psnr, opt.method)
        print(f"MAE: {mse_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}")
        # Save model
        if ( mse_loss < record_mse_loss
            ):
            print(f"Saving to {opt.outf}")
            save_checkpoint(opt.outf, (iteration // len(train_loader)), iteration, model, optimizer)
            record_mse_loss = mse_loss
        # print loss
        print(
            " Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MAE: %.9f, Test MAE: %.9f, Test RMSE: %.9f, Test PSNR: %.9f "
            % (iteration, iteration // len(train_loader), lr, losses.avg, mse_loss, rmse_loss, psnr_loss)
        )
        logger.info(
            " Iter[%06d], Epoch[%06d], learning rate : %.9f, Train Loss: %.9f, Test MAE: %.9f, Test RMSE: %.9f, Test PSNR: %.9f "
            % (iteration, iteration // len(train_loader), lr, losses.avg, mse_loss, rmse_loss, psnr_loss)
        )

    print(torch.__version__)


def validate(val_loader, model, criterion_mae, criterion_rmse, criterion_psnr, method):
    model.eval()
    losses_mse = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        
        if method == 'fno':
            B, C, H, W = input.shape
            idx = torch.round(
                torch.linspace(5, 204, steps=C, device=input.device)
            ).long()

            out = torch.zeros(B, 229, H, W, device=input.device, dtype=input.dtype)
            input = out.index_copy_(1, idx, input)
            B,C, H,W = input.shape
            input=input.reshape(B,1,H,W,C)
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            if method=='fno':
                output= output.squeeze(1)
                output= output.permute(0, 3, 1, 2).contiguous()
            loss_mse = criterion_mae(output, target)
            loss_rmse = criterion_rmse(output, target)
            loss_psnr = criterion_psnr(output, target)
        # record loss
        losses_mse.update(loss_mse.item())
        losses_rmse.update(loss_rmse.item())
        losses_psnr.update(loss_psnr.item())
    return losses_mse.avg, losses_rmse.avg, losses_psnr.avg


if __name__ == "__main__":
    main()

