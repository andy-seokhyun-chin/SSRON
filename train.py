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
from torch.utils.data import DataLoader
from dataset import load_ssron_datasets
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
from SSRON import SSRON

def build_coord_grid(H, W, lambda_idx, C_hsi, device):
    """
    Create coordinates for all pixels at a fixed wavelength.

    Returns:
        coords: (H*W, 3) tensor of (x, y, lambda)
    """
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )

    x_norm = xs.reshape(-1) / (W - 1)
    y_norm = ys.reshape(-1) / (H - 1)
    l_norm = torch.full_like(x_norm, lambda_idx / (C_hsi - 1))

    coords = torch.stack([x_norm, y_norm, l_norm], dim=1)
    return coords
def reconstruct_hsi_cube(
    model,
    msi,
    C_hsi,
    lambda_batch_size=8,
):
    """
    Reconstruct full HSI cube from MSI using DeepONet model.

    Args:
        model: trained HyperspectralDeepONet
        msi:   (B, C_msi, H, W)
        C_hsi: number of hyperspectral bands
        lambda_batch_size: how many wavelengths to evaluate at once

    Returns:
        hsi_pred: (B, C_hsi, H, W)
    """
    device = msi.device
    B, _, H, W = msi.shape

    hsi_pred = torch.zeros(B, C_hsi, H, W, device=device)

    for l_start in range(0, C_hsi, lambda_batch_size):
        l_end = min(l_start + lambda_batch_size, C_hsi)

        coords_list = []
        for l in range(l_start, l_end):
            coords = build_coord_grid(H, W, l, C_hsi, device)
            coords_list.append(coords)

        # (N, 3) where N = H*W*(l_end - l_start)
        coords = torch.cat(coords_list, dim=0)
        # Expand to (B, N, 3)
        coords = coords.unsqueeze(0).expand(B, -1, -1).contiguous()

        # Predict
        pred = model(msi, coords)  # (B, N)

        # Reshape and store
        offset = 0
        for l in range(l_start, l_end):
            band = pred[:, offset : offset + H * W]
            band = band.view(B, H, W)
            hsi_pred[:, l] = band
            offset += H * W

    return hsi_pred
def build_model(pretrained: bool=False):
    model = SSRON(msi_channels=12, latent_dim=128, n_freq=12) 
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
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--init_lr", type=float, default=5e-5, help="initial learning rate")
    parser.add_argument("--outf", type=str, default="./exp/", help="path log files")
    parser.add_argument("--data_root", type=str, default="../dataset/")
    parser.add_argument("--patch_size", type=int, default=16, help="patch size")
    parser.add_argument("--stride", type=int, default=8, help="stride")
    parser.add_argument("--gpu_id", type=str, default="0", help="CUDA visible devices")
    parser.add_argument("--downsampling", type=int, default=100, help="How low resolution of training data is")
    return parser.parse_args()


def main():
    opt = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    # load dataset
    print("\nloading dataset ...")
    train_data, val_data = load_ssron_datasets(wavelength_downsample_factor=opt.downsampling)
    print(f"Iteration per epoch: {len(train_data)}")
    print("Validation set samples: ", len(val_data))

    # iterations
    total_iteration = opt.epochs

    # loss function
    criterion_mse = Loss_MAE()
    criterion_rmse = Loss_RMSE()
    criterion_psnr = Loss_PSNR()
    srf = np.load('/workspace/SRNO/srf_weights_clipped.npy')
    # model
    model = build_model(opt.pretrained_model_path).cuda()
    print("Parameters number is ", sum(param.numel() for param in model.parameters()))
    model = model.float()
    # output path
    date_time = time2file_name(str(datetime.datetime.now()))
    opt.outf = os.path.join(opt.outf, 'DeepONet_universal', date_time)
    os.makedirs(opt.outf, exist_ok=True)

    if torch.cuda.is_available():
        print('cuda!')
        model.cuda()
        criterion_mse.cuda()
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
    val_loader = DataLoader(dataset=val_data, batch_size=opt.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    total_iteration = opt.epochs*len(train_loader)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)
    while iteration < total_iteration:
        model.train()
        losses = AverageMeter()
        for i, (images, coords, vals) in tqdm(enumerate(train_loader), total=len(train_loader),bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",):
            images = images.cuda().float()
            coords = coords.cuda().float()
            vals = vals.cuda().float()
            optimizer.zero_grad()
            output = model(images, coords)
            if torch.isnan(output).any() or torch.isinf(output).any():
                print("NaN/Inf output detected — stopping")
                iteration = total_iteration
                break
            loss = criterion_mse(output, vals)
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
        if iteration % len(train_loader) == 0:
            mse_loss, rmse_loss, psnr_loss = validate(val_loader, model, criterion_mse, criterion_rmse, criterion_psnr)
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
@torch.no_grad()
def validate(val_loader, model, criterion_mse, criterion_rmse, criterion_psnr):
    model.eval()
    losses_mse = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    for msi, hsi in val_loader:
        msi = msi.cuda()
        hsi_pred = reconstruct_hsi_cube(
            model,
            msi,
            C_hsi=229,
            lambda_batch_size=8,
        )
        hsi=hsi.cuda()
        loss_mse = criterion_mse(hsi_pred, hsi)
        loss_rmse = criterion_rmse(hsi_pred, hsi)
        loss_psnr = criterion_psnr(hsi_pred, hsi)
        # record loss
        losses_mse.update(loss_mse.item())
        losses_rmse.update(loss_rmse.item())
        losses_psnr.update(loss_psnr.item())
    return losses_mse.avg, losses_rmse.avg, losses_psnr.avg


if __name__ == "__main__":
    main()

