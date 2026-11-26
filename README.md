# TrajBridge

TrajBridge is a multi-agent trajectory reconstruction project for the INTERACTION dataset. The repository keeps a single, reproducible training and validation path for the model, with the package layout and command-line workflow organized in the spirit of [QCNet](https://github.com/ZikangZhou/QCNet).

> **Current scope**  The public code in this directory covers INTERACTION training and validation only. HkIntersection data, ablation runners, baseline scripts, test-only entry points, and generated experiment artifacts are not included.

## Overview

<p align="center">
  <img src="./overview.png" alt="TrajBridge overview" width="100%">
</p>

The main task is reconstruction. `train.py` builds the `TrajBridge` Lightning module, trains it on the INTERACTION training split, evaluates the validation split during training, and saves checkpoints. `val.py` restores a checkpoint and evaluates the INTERACTION validation split with the same model configuration.

The current repository title follows the project description, **Future-Constrained Conditional Flow Bridge for Multi-Agent Trajectory Reconstruction in Dense Traffic**. The command-line training path is intentionally kept to the reconstruction task so that the released training and validation loop is easy to run and inspect.

## Repository layout

```text
Trajbridge/
├── train.py                    # training entry point
├── val.py                      # checkpoint validation entry point
├── model/TrajBridge.py         # TrajBridge LightningModule
├── datamodules/                # INTERACTION LightningDataModule
├── datasets/                   # INTERACTION reader and reconstruction masks
├── modules/                    # backbone, map encoder, and bridge modules
├── layers/                     # graph attention and MLP layers
├── losses/                     # training losses
├── metrics/                    # validation metrics
├── transforms/                 # training augmentations
└── utils/                      # geometry and preprocessing utilities
```

## Environment

The code was checked with Python 3.8 and the following versions.

```text
PyTorch              2.1.0
PyTorch Geometric    2.3.1
PyTorch Lightning    2.0.3
TorchMetrics         1.4.0.post0
```

An example environment setup is shown below. Install a PyTorch build that matches the CUDA version on your machine.

```bash
conda create -n trajbridge python=3.8
conda activate trajbridge
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric pytorch-lightning torchmetrics numpy pandas tqdm
```

## Dataset

Set `--root` to the INTERACTION data root. The reader accepts raw files and prepared tensor caches.

```text
INTERACTION_ROOT/
├── train/                     # raw training CSV files
├── val/                       # raw validation CSV files
├── maps/                      # Lanelet2 OSM maps
├── train_processed/           # prepared .pt files
└── val_processed/             # prepared .pt files
```

Use `--processed_only true` when the processed directories are already available. This avoids reading and preprocessing the raw CSV and map files during a run.

## Training

```bash
python train.py \
  --root /path/to/INTERACTION_ROOT \
  --train_batch_size 4 \
  --val_batch_size 4 \
  --devices 1
```

By default, the run is created under `runs/<timestamp>/`. The trainer saves `last.ckpt` and the checkpoint with the best `val_recon_minJointFDE` under the run's `checkpoints/` directory.

For a quick end-to-end smoke run, use one batch from each split.

```bash
python train.py \
  --root /path/to/INTERACTION_ROOT \
  --processed_only true \
  --train_batch_size 1 \
  --val_batch_size 1 \
  --num_workers 0 \
  --accelerator cpu \
  --devices 1 \
  --max_epochs 1 \
  --limit_train_batches 1 \
  --limit_val_batches 1 \
  --default_root_dir runs/smoke
```

## Validation

Pass a checkpoint produced by `train.py` to `val.py`.

```bash
python val.py \
  --root /path/to/INTERACTION_ROOT \
  --ckpt_path runs/<timestamp>/checkpoints/last.ckpt \
  --val_batch_size 4 \
  --devices 1
```

For a fast validation check, add `--processed_only true --num_workers 0 --limit_val_batches 1`. The validation entry point restores the model hyperparameters from the checkpoint and reads only the `val` split.

## Closed-loop verification

The following two commands exercise the complete public path from data loading to checkpoint restoration and validation.

```bash
python train.py --root /path/to/INTERACTION_ROOT \
  --processed_only true --train_batch_size 1 --val_batch_size 1 \
  --num_workers 0 --accelerator cpu --devices 1 --max_epochs 1 \
  --limit_train_batches 1 --limit_val_batches 1 --default_root_dir runs/smoke

python val.py --root /path/to/INTERACTION_ROOT \
  --processed_only true --num_workers 0 --accelerator cpu --devices 1 \
  --limit_val_batches 1 --ckpt_path runs/smoke/checkpoints/last.ckpt
```

The smoke path was executed successfully on a prepared INTERACTION cache. It is intended as an integration check, not as a reported benchmark result.

## Reproducibility notes

- Training and validation use the same historical and future horizon defaults, reconstruction mask configuration, and checkpoint-defined model hyperparameters.
- Use `--seed` to control the Lightning and data-loader seed.
- Keep the dataset cache, command-line arguments, checkpoint, and reported metrics together when recording an experiment.
- The repository does not include raw INTERACTION data, processed caches, checkpoints, or TensorBoard output.

## Reference

The directory split and the separation between training and validation entry points follow the public organization of [QCNet](https://github.com/ZikangZhou/QCNet).

## Citation and contact

The paper citation will be added after publication. For questions or reproducibility issues, please open an issue in the [TrajBridge repository](https://github.com/Nova-chen151/TrajBridge).
