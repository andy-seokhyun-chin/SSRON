# Spectral Super-Resolution using Spatial-Spectral Residual Operator Networks (IGARSS 2026)
## Introduction
This is the research code of the IEEE International Geoscience and Remote Sensing Symposium 2026 paper (to appear in proceedings).
[S. Chin, "Spectral Super-Resolution using Spatial-Spectral Residual Operator Networks," IEEE International Geoscience and Remote Sensing Symposium 2026](https://arxiv.org/abs/2609.35410v1)

This repository contains the preprocessing pipeline, dataset builder, model training scripts, and inference code. The intended execution order is:

1. Build the raw dataset from the EMIT .nc files
2. Convert the HDF5 dataset into the .npz + split/index files that the PyTorch loaders expect
3. Train the model
4. Run inference with a saved checkpoint
## Requirements:
torch>=2.0 \
neuraloperator>=1.0 \
einops \
numpy \
tqdm \
h5py \
netCDF4 \
*Note: the original environment was not preserved; these versions are verified to import and run a forward pass, but the full training pipeline has not been re-run with them.*
## Recommended end-to-end order

For the standard SSRON workflow, use this order:

```bash
python preprocessing.py
python build_dataset_split.py --h5 "SRNO_data/patches/all_data.h5" --npz "dataset.npz"
python train.py
python inference.py --method ssron --pretrained_model_path "path/to/checkpoint.pth"
```

This is the intended sequence for this repository.


## Notes

- `dataset.py` is the central loader code; it expects the split and normalization files to already exist.
- `preprocessing.py` creates the raw dataset container.
- `build_dataset_split.py` creates the training/validation/test split and statistics used by the loaders.
- `train.py` and `baselinestrain.py` are training entry points, not preprocessing tools.
- `inference.py` is for evaluation and prediction using a trained model.

If you find the code useful to your work, please consider citing our paper: 

```bibtex
@misc{Chin2026,
  doi = {10.48550/ARXIV.2609.35410},
  url = {https://arxiv.org/abs/2609.35410},
  author = {Chin,  Seokhyun},
  keywords = {Computer Vision and Pattern Recognition (cs.CV),  Artificial Intelligence (cs.AI),  FOS: Computer and information sciences},
  title = {Spectral Super-Resolution using Spatial-Spectral Residual Operator Networks},
  publisher = {arXiv},
  year = {2026},
  copyright = {Creative Commons Attribution 4.0 International}
}