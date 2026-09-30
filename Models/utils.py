from __future__ import division

import torch
import torch.nn as nn
import logging
import os
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))
class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error) + 1e-8)
        return rmse

class Loss_MAE(nn.Module):
    def __init__(self):
        super(Loss_MAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label)
        mae = torch.mean(error)
        return mae

class Loss_MSE(nn.Module):
    def __init__(self):
        super(Loss_MSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs - label
        mse = torch.mean(error ** 2)
        return mse
class Loss_SSIM(nn.Module):
    def __init__(self):
        super(Loss_SSIM, self).__init__()
        self.eps_x=1e-5
        self.eps_y=1e-5

    def forward(self, x, y):
        mu_x = x.mean(dim=(2, 3), keepdim=True)
        mu_y = y.mean(dim=(2, 3), keepdim=True)

        # Variance and covariance over spatial dimensions
        sigma_x = ((x - mu_x) ** 2).mean(dim=(2, 3), keepdim=True)
        sigma_y = ((y - mu_y) ** 2).mean(dim=(2, 3), keepdim=True)
        sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(2, 3), keepdim=True)

        # SSIM per channel
        ssim_c = ((2 * mu_x * mu_y + self.eps_x) * (2 * sigma_xy + self.eps_y)) / \
                ((mu_x ** 2 + mu_y ** 2 + self.eps_x) * (sigma_x + sigma_y + self.eps_y))

        # Average over channels → per image
        return ssim_c.mean(dim=1).squeeze()


class Loss_PSNR(nn.Module):
    def __init__(self, data_range=29.7, eps=1e-8):
        super(Loss_PSNR, self).__init__()
        self.data_range = data_range
        self.eps = eps
    def forward(self, im_true, im_pred):
        mse = torch.mean((im_true - im_pred) ** 2, dim=(1,2,3))
        psnr = 10.0 * torch.log10((self.data_range ** 2) / (mse + self.eps))
        return psnr.mean()

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()
    loss_csv.close
