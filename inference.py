import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from dataset import PatchDataset
# Add Baselines_1 to path so we can import its modules from project root
ROOT = Path(__file__).parent
BASELINES_DIR = ROOT / "Models"
if str(BASELINES_DIR) not in sys.path:
    sys.path.append(str(BASELINES_DIR))
from AWAN import AWAN  # type: ignore
from Restormer import Restormer  # type: ignore
from baseneuralop import ushapedno, fourierno
from SSRON import SSRON
from SSRAN import SSRAN
import matplotlib.pyplot as plt
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
    elif method =="ssron":
        model = SSRON(msi_channels=12, latent_dim=128, n_freq=12) 
    else:
        raise ValueError(f"Unknown method {method}")

    if pretrained:
        if not os.path.isfile(pretrained):
            raise FileNotFoundError(f"Pretrained model not found: {pretrained}")
        state = torch.load(pretrained, map_location="cpu", weights_only=False)
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
        default="ssron",
        choices=["restormer", "awan","uno","fno", "ssran", "ssron"],
        help="Model to train",
    )
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--dataPath", type=str, default=r'./test_data.npz', help="Path to inference data")
    return parser.parse_args()

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
    Reconstruct full HSI cube from MSI using ssron.

    Args:
        model: trained ssron
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


def main():
    opt = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    def load_one(path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path, mmap_mode="r")
        return data["X"], data["y"]
    # load dataset
    print("\nloading dataset ...")
    X_test, y_test = load_one(opt.dataPath)
    test_ds  = PatchDataset(X_test, y_test)
    # iterations
    model = build_model(opt.method, opt.pretrained_model_path).cuda()
    print("Parameters number is ", sum(param.numel() for param in model.parameters()))
    model = model.float()
    model.cuda()

    cudnn.benchmark = True
    model.eval()
# ---- Accumulate error over entire test set ----
    sum_error_map = None
    num_samples = len(test_ds)

    with torch.no_grad():
        for i in range(num_samples):
            input, target = test_ds[i]
            input = input.unsqueeze(0).cuda()
            target = target.unsqueeze(0).cuda()

            # ---- Handle FNO / LocalNO input formatting ----
            if opt.method in ['fno', 'localno']:
                B, C, H, W = input.shape
                idx = torch.round(
                    torch.linspace(5, 204, steps=C, device=input.device)
                ).long()

                out = torch.zeros(B, 229, H, W, device=input.device, dtype=input.dtype)
                input = out.index_copy_(1, idx, input)
                input = input.reshape(B, 1, H, W, 229)

            # ---- Forward pass ----
            if opt.method == 'deeponet':
                output = reconstruct_hsi_cube(
                    model,
                    input,
                    C_hsi=229,
                    lambda_batch_size=8,
                )
            else:
                output = model(input)

            if opt.method == 'fno':
                output = output.squeeze(1)
                output = output.permute(0, 3, 1, 2).contiguous()

            if opt.method == 'localno':
                output = output.squeeze(1)

            # ---- Error map for this sample ----
            error_map = torch.abs(output - target)  # (1, 229, H, W)
            error_map = error_map.squeeze(0)        # (229, H, W)

            if sum_error_map is None:
                sum_error_map = error_map
            else:
                sum_error_map += error_map

    # ---- Average over dataset ----
    avg_error_map = sum_error_map / num_samples   # (229, H, W)

    # ---- ERROR MAP ----
    error_map_2d = avg_error_map.mean(dim=0).cpu().numpy()
    plt.figure(figsize=(6, 6))
    im = plt.imshow(error_map_2d, cmap='hot')
    plt.axis('off')

    # ---- Colorbar on top ----
    cbar = plt.colorbar(
        im,
        orientation='horizontal',
        pad=0.05,
        fraction=0.05
    )
    plt.colorbar(label='Mean Absolute Error')

    save_path = f"error_map_2d_avg_{opt.method}2.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Dataset-averaged 2D error map saved to {save_path}")


if __name__ == "__main__":
    main()

