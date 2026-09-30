#!/usr/bin/env python3
"""
End-to-end preprocessing:
- Scan for EMIT .nc radiance files
- Convert to hyperspectral cube (HSI) and pseudo MSI (via srf_weights)
- Save ONE combined HDF5 (.h5) with all data (msi, hsi, ids)

"""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import h5py
from netCDF4 import Dataset


DATA_ROOT = Path(r".\SRNO_data")
PATCH_SIZE = 16
SRF_PATH = Path(__file__).parent / "srf_weights.npy"


def find_nc_files(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    return sorted(p for p in root_dir.rglob("*.nc") if p.is_file())


def load_radiance_cube(nc_path: Path) -> np.ndarray:
    with Dataset(nc_path) as ds:
        data = ds.variables["radiance"][:, :, :]
        cube = np.asarray(data, dtype=np.float32)

    # Ensure bands-last (H, W, B)
    if cube.shape[0] == 285:
        cube = np.moveaxis(cube, 0, -1)
    elif cube.shape[-1] == 285:
        pass
    else:
        raise ValueError(f"Unexpected spectral dimension for {nc_path}: {cube.shape}")

    return cube


def to_pseudo_msi(hsi_cube: np.ndarray, srf_weights: np.ndarray) -> np.ndarray:
    # hsi_cube: (H, W, 285); srf_weights: (12, 285)
    return np.tensordot(hsi_cube, srf_weights.T, axes=([2], [0])).astype(np.float32)


def main():
    srf_weights = np.load(SRF_PATH)
    nc_files = find_nc_files(DATA_ROOT)

    if not nc_files:
        print(f"No .nc files found under {DATA_ROOT}")
        return

    patches_dir = DATA_ROOT / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    h5_path = patches_dir / "all_data.h5"

    with h5py.File(h5_path, "w") as f:
        msi_ds = None
        hsi_ds = None
        id_ds = None

        str_dt = h5py.string_dtype(encoding="utf-8")

        offset = 0

        for nc_path in nc_files:
            pair_id = nc_path.stem
            print(f"Processing {pair_id}")

            hsi_cube = load_radiance_cube(nc_path)
            msi_cube = to_pseudo_msi(hsi_cube, srf_weights)

            # Expand dims so we can concatenate along axis 0
            hsi_cube = hsi_cube[None, ...]  # (1, H, W, 285)
            msi_cube = msi_cube[None, ...]  # (1, H, W, 12)
            ids = np.array([pair_id], dtype=object)

            if msi_ds is None:
                # Create datasets
                msi_ds = f.create_dataset(
                    "msi",
                    data=msi_cube,
                    maxshape=(None, *msi_cube.shape[1:]),
                    chunks=True,
                    compression="gzip",
                )
                hsi_ds = f.create_dataset(
                    "hsi",
                    data=hsi_cube,
                    maxshape=(None, *hsi_cube.shape[1:]),
                    chunks=True,
                    compression="gzip",
                )
                id_ds = f.create_dataset(
                    "ids",
                    data=ids,
                    maxshape=(None,),
                    dtype=str_dt,
                )
            else:
                # Append
                new_size = offset + 1

                msi_ds.resize(new_size, axis=0)
                hsi_ds.resize(new_size, axis=0)
                id_ds.resize(new_size, axis=0)

                msi_ds[offset] = msi_cube
                hsi_ds[offset] = hsi_cube
                id_ds[offset] = pair_id

            offset += 1

    print(f"Saved combined dataset to {h5_path}")


if __name__ == "__main__":
    main()
