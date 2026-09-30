#!/usr/bin/env python3
"""Build a .npz dataset and train/val/test index files from preprocessing output.
python build_dataset_split.py --h5 "path/to/all_data.h5" 
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np



def split_indices(n_samples: int, train_ratio: float, val_ratio: float, seed: int = 42):
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0 and 0.0 < 1.0 - train_ratio - val_ratio):
        raise ValueError("train_ratio and val_ratio must produce a valid test split.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    n_train = int(np.floor(train_ratio * n_samples))
    n_val = int(np.floor(val_ratio * n_samples))
    n_test = n_samples - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


def save_split_files(npz_path: Path, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray):
    np.save(npz_path.with_name("train_idx.npy"), train_idx)
    np.save(npz_path.with_name("val_idx.npy"), val_idx)
    np.save(npz_path.with_name("test_idx.npy"), test_idx)


def save_test_array(npz_path: Path, msi: np.ndarray, hsi: np.ndarray, ids: np.ndarray, test_idx: np.ndarray):
    np.save(npz_path.with_name("test.npz"), {
        "msi": msi[test_idx],
        "hsi": hsi[test_idx],
        "ids": ids[test_idx],
    }, allow_pickle=True)


def save_normalization_stats(npz_path: Path, msi: np.ndarray, hsi: np.ndarray, train_idx: np.ndarray):
    msi_train = msi[train_idx]
    hsi_train = hsi[train_idx]

    msi_mean = msi_train.reshape(-1, msi_train.shape[-1]).mean(axis=0)
    msi_std = np.clip(msi_train.reshape(-1, msi_train.shape[-1]).std(axis=0), 1e-3, None)
    hsi_mean = hsi_train.reshape(-1, hsi_train.shape[-1]).mean(axis=0)
    hsi_std = np.clip(hsi_train.reshape(-1, hsi_train.shape[-1]).std(axis=0), 1e-3, None)

    np.save(npz_path.with_name("msi_mean.npy"), msi_mean.astype(np.float32))
    np.save(npz_path.with_name("msi_std.npy"), msi_std.astype(np.float32))
    np.save(npz_path.with_name("hsi_mean.npy"), hsi_mean.astype(np.float32))
    np.save(npz_path.with_name("hsi_std.npy"), hsi_std.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description="Convert preprocessing HDF5 output into dataset.npz + split index files.")
    parser.add_argument("--h5", type=Path, required=True, help="Path to the .h5 file created by preprocessing.py")
    parser.add_argument(
        "--npz",
        type=Path,
        default="dataset.npz",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    h5_path = args.h5.resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
    npz_path = args.npz.resolve()

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, 'r') as f:
        msi = np.asarray(f["msi"])
        hsi = np.asarray(f["hsi"])
        ids = np.asarray(f["ids"])
    ids = np.asarray(ids, dtype=str)
    np.savez_compressed(npz_path, msi=msi, hsi=hsi, ids=ids)

    train_idx, val_idx, test_idx = split_indices(
        n_samples=msi.shape[0],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    save_split_files(npz_path, train_idx, val_idx, test_idx)
    save_test_array(npz_path, msi, hsi, ids, test_idx)
    save_normalization_stats(npz_path, msi, hsi, train_idx)

    print(f"Saved full dataset NPZ: {npz_path}")
    print(f"Saved split files and other files next to it:")
    print(f"  - {npz_path.with_name('train_idx.npy')}")
    print(f"  - {npz_path.with_name('val_idx.npy')}")
    print(f"  - {npz_path.with_name('test_idx.npy')}")
    print(f"  - {npz_path.with_name('test.npz')}")
    print(f"  - {npz_path.with_name('msi_mean.npy')}")
    print(f"  - {npz_path.with_name('msi_std.npy')}")
    print(f"  - {npz_path.with_name('hsi_mean.npy')}")
    print(f"  - {npz_path.with_name('hsi_std.npy')}")
    print(f"Counts: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")


if __name__ == "__main__":
    main()
