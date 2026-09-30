import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
class PatchDataset(Dataset):

    def __init__(
        self,
        msi: np.ndarray,
        hsi: np.ndarray,
        ids: np.ndarray,
        indices: np.ndarray,
        msi_mean: np.ndarray = None,
        msi_std: np.ndarray = None,
        hsi_mean: np.ndarray = None,
        hsi_std: np.ndarray = None,
    ):
        self.msi_all = msi
        self.hsi_all = hsi
        self.ids_all = ids
        self.indices = np.asarray(indices)

        # Normalization parameters (computed from training set)
        self.msi_mean = (
            msi_mean if msi_mean is not None else np.zeros(msi.shape[-1])
        )
        self.msi_std = (
            msi_std if msi_std is not None else np.ones(msi.shape[-1])
        )
        self.hsi_mean = (
            hsi_mean if hsi_mean is not None else np.zeros(hsi.shape[-1])
        )
        self.hsi_std = (
            hsi_std if hsi_std is not None else np.ones(hsi.shape[-1])
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        x = self.msi_all[real_idx]  # (H, W, C_msi)
        y = self.hsi_all[real_idx]  # (H, W, C_hsi)

        # Normalize
        x = (x - self.msi_mean) / (self.msi_std + 1e-8)
        y = (y - self.hsi_mean) / (self.hsi_std + 1e-8)

        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite MSI at index {real_idx}")

        if not np.isfinite(y).all():
            raise ValueError(f"Non-finite HSI at index {real_idx}")

        x = torch.from_numpy(x).permute(2, 0, 1).float()
        y = torch.from_numpy(y).permute(2, 0, 1).float()

        return x, y
class ssronPatchDataset(Dataset):
    """
    Dataset for DeepOnet-style MSI -> HSI learning.

    Returns:
        msi:    (C_msi, H, W)
        coords: (N, 3)  -> (x, y, lambda) normalized to [0, 1]
        values: (N,)    -> HSI values at those coords
    """

    def __init__(
        self,
        msi: np.ndarray,
        hsi: np.ndarray,
        ids: np.ndarray,
        indices: np.ndarray,
        num_samples: int = 2048,
        wavelength_downsample_factor: int = 100,
        msi_mean: np.ndarray = None,
        msi_std: np.ndarray = None,
        hsi_mean: np.ndarray = None,
        hsi_std: np.ndarray = None,
    ):
        self.msi_all = msi
        self.hsi_all = hsi
        self.ids_all = ids
        self.indices = np.asarray(indices)
        self.num_samples = num_samples
        self.wavelength_downsample_factor = wavelength_downsample_factor

        self.H = msi.shape[1]
        self.W = msi.shape[2]
        self.C_hsi = hsi.shape[-1]
        
        # Compute effective number of wavelengths to sample from
        self.C_hsi_sampled = max(1, self.C_hsi * wavelength_downsample_factor//100)
        
        # Create mapping from sampled indices to actual HSI channel indices
        # Evenly space the wavelengths across the full range
        if wavelength_downsample_factor == 100:
            self.wavelength_indices = np.arange(self.C_hsi)
        else:
            # Select evenly spaced wavelengths
            bands = np.linspace(0, self.C_hsi-1, self.C_hsi_sampled)
            bands = np.unique(np.round(bands).astype(int))

            # If rounding caused fewer than 200 bands, fix by interpolation
            if len(bands) < self.C_hsi_sampled:
                bands = np.unique(np.floor(np.linspace(0, self.C_hsi - 1, self.C_hsi_sampled)).astype(int))
            self.wavelength_indices=bands

        # Normalization
        self.msi_mean = msi_mean if msi_mean is not None else np.zeros(msi.shape[-1])
        self.msi_std = msi_std if msi_std is not None else np.ones(msi.shape[-1])
        self.hsi_mean = hsi_mean if hsi_mean is not None else np.zeros(hsi.shape[-1])
        self.hsi_std = hsi_std if hsi_std is not None else np.ones(hsi.shape[-1])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        # Load patches
        msi = self.msi_all[real_idx]  # (H, W, C_msi)
        hsi = self.hsi_all[real_idx]  # (H, W, C_hsi)

        # Normalize
        msi = (msi - self.msi_mean) / (self.msi_std + 1e-8)
        hsi = (hsi - self.hsi_mean) / (self.hsi_std + 1e-8)

        if not np.isfinite(msi).all():
            raise ValueError(f"Non-finite MSI at index {real_idx}")
        if not np.isfinite(hsi).all():
            raise ValueError(f"Non-finite HSI at index {real_idx}")

        # Convert MSI to tensor
        msi = torch.from_numpy(msi).permute(2, 0, 1).float()  # (C_msi, H, W)

        # -----------------------------
        # Sample spatio-spectral points
        # -----------------------------
        N = self.num_samples

        ys = np.random.randint(0, self.H, size=N)
        xs = np.random.randint(0, self.W, size=N)
        # Sample from reduced wavelength set
        ls_sampled = np.random.randint(0, self.C_hsi_sampled, size=N)
        # Map to actual HSI channel indices
        ls = self.wavelength_indices[ls_sampled]

        # Normalize coordinates to [0, 1]
        x_norm = xs / (self.W - 1)
        y_norm = ys / (self.H - 1)
        # Normalize using the actual wavelength indices (not sampled indices)
        l_norm = ls / (self.C_hsi - 1)

        coords = np.stack([x_norm, y_norm, l_norm], axis=1)  # (N, 3)

        # Gather HSI values using actual channel indices
        values = hsi[ys, xs, ls]  # (N,)

        coords = torch.from_numpy(coords).float()
        values = torch.from_numpy(values).float()

        return msi, coords, values
def load_ssron_datasets(
    npz_path: str = r"",
    num_samples: int = 2048,
    wavelength_downsample_factor: int = 100,
):
    npz_path = Path(npz_path)
    print("data loading...")
    data = np.load(npz_path, mmap_mode="r")
    msi = data["msi"]
    hsi = data["hsi"]
    ids = data["ids"]
    print("dataset loaded!")

    # Select MSI bands
    msi = msi[:, :, :, [0,1,2,3,4,5,6,7,8,10,11,12]]

    train_idx = np.load('train_idx.npy')
    val_idx = np.load('val_idx.npy')
    # Load normalization stats
    hsi_mean = np.load("hsi_mean.npy")
    hsi_std  = np.clip(np.load("hsi_std.npy"), 1e-3, None)

    msi_mean = np.load("msi_mean.npy")[[0,1,2,3,4,5,6,7,8,10,11,12]]
    msi_std  = np.clip(
        np.load("msi_std.npy")[[0,1,2,3,4,5,6,7,8,10,11,12]],
        1e-3,
        None,
    )
    train_ds = ssronPatchDataset(
        msi, hsi, ids, train_idx,
        num_samples=num_samples,
        wavelength_downsample_factor=wavelength_downsample_factor,
        msi_mean=msi_mean,
        msi_std=msi_std,
        hsi_mean=hsi_mean,
        hsi_std=hsi_std,
    )
    val_ds = PatchDataset(
        msi, hsi, ids, val_idx,
        msi_mean, msi_std, hsi_mean, hsi_std
    )
    return train_ds, val_ds

def load_datasets(
    npz_path: str = "dataset.npz",
):
    """
    Load combined NPZ and return (train_ds, val_ds)
    test_ds can be computed in inference.py
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    print("data loading...")
    data = np.load(npz_path, mmap_mode="r")
    msi = data["msi"]
    hsi = data["hsi"]
    ids = data["ids"]
    print("dataset loaded!")

    # Select MSI bands
    msi = msi[:, :, :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]]

    train_idx = np.load('train_idx.npy')
    val_idx = np.load('val_idx.npy')

    # Load normalization stats (computed on training set)
    hsi_mean = np.load("hsi_mean.npy")
    hsi_std = np.clip(np.load("hsi_std.npy"), 1e-3, None)

    msi_mean = np.load("msi_mean.npy")[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]]
    msi_std = np.clip(
        np.load("msi_std.npy")[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]],
        1e-3,
        None,
    )

    # Create datasets (all normalized using training stats)
    
    train_ds = PatchDataset(
        msi, hsi, ids, train_idx,
        msi_mean, msi_std, hsi_mean, hsi_std
    )
    
    val_ds = PatchDataset(
        msi, hsi, ids, val_idx,
        msi_mean, msi_std, hsi_mean, hsi_std
    )

    return train_ds, val_ds
