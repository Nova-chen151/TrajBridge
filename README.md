# TrajBridge

### Future-Constrained Conditional Flow Bridge for Multi-Agent Trajectory Reconstruction in Dense Traffic

Yuhang Chen, Ruyi Fu, Liyong Zheng, Zuduo Zheng, Jian Sun, and Jie Sun

TrajBridge is a multi-agent trajectory reconstruction project for the INTERACTION dataset. The repository keeps a single, reproducible training and validation path for the model, with the package layout and command-line workflow organized in the spirit of [QCNet](https://github.com/ZikangZhou/QCNet).

**INTERACTION trajectory reconstruction** · [Getting Started](#getting-started) · [Training](#training) · [Validation](#validation) · [Citation](#citation-and-contact)

## Overview

<p align="center">
  <img src="./overview.png" alt="TrajBridge overview" width="100%">
</p>

The main task is reconstruction. `train.py` builds the `TrajBridge` Lightning module, trains it on the INTERACTION training split, evaluates the validation split during training, and saves checkpoints. `val.py` restores a checkpoint and evaluates the INTERACTION validation split with the same model configuration.

## Method in the paper

TrajBridge reconstructs an internal trajectory gap using observed motion before the gap, measured motion after it, neighboring agents, and optional map context. The paper describes two stages.

1. **Interaction-aware priors.** Factorized agent, temporal, and mode attention generates multiple joint trajectory hypotheses from the observed history and map.
2. **Future-constrained refinement.** A future-tail encoder conditions a trajectory-space vector field trained with conditional flow matching. The flow bridge progressively refines the priors toward trajectories compatible with the observations around the gap.

The paper evaluates reconstruction on INTERACTION, zero-shot transfer to 500 Waymo Open Motion Dataset scenes, and downstream kinematic and traffic-conflict analysis. Its reported INTERACTION GapFDE and boundary error are 0.050 m and 0.029 m, respectively. These are manuscript results for the full method. They have not been reproduced by the public training command described below.

**Released training scope.** The current `train.py` uses the history/map backbone for reconstruction and disables the future-tail and flow bridge branches. It does not train the full paper model or reproduce its reported metrics. The bridge modules remain in the source tree, but the public CLI does not configure their training.

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

## Getting Started

### Environment

The code was checked with Python 3.8 and the following versions.

```text
PyTorch              2.1.0
PyTorch Geometric    2.3.1
PyTorch Lightning    2.0.3
TorchMetrics         1.4.0.post0
```

Create the CUDA 12.1 environment using the supplied configuration.

```bash
git clone https://github.com/Nova-chen151/TrajBridge.git
cd TrajBridge
conda env create -f environment.yml
conda activate trajbridge
python train.py --help
python val.py --help
```

The configuration follows the PyTorch, PyG, and Lightning versions used by HPNet and the existing TrajBridge smoke run. The environment file targets Linux with an NVIDIA GPU and CUDA 12.1. For raw map preprocessing, install the Lanelet2 Python bindings separately, as in HPNet.

```bash
pip install lanelet2==1.2.1
```

### Dataset

Set `--root` to the INTERACTION data root. The reader accepts raw files and prepared tensor caches.

```text
INTERACTION_ROOT/
├── train/                     # raw training CSV files
├── val/                       # raw validation CSV files
├── maps/                      # Lanelet2 OSM maps
├── train_processed/           # prepared .pt files
└── val_processed/             # prepared .pt files
```

Raw preprocessing requires the Lanelet2 Python bindings and corresponding OSM maps. Use `--processed_only true` when compatible processed directories are already available. This reads the prepared tensors directly.

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

For a fast validation check, add `--processed_only true --num_workers 0 --limit_val_batches 1`. The validation entry point restores the model hyperparameters from the checkpoint and reads only the `val` split. If training used `--window_protocol fixed_gap --gap_len 20`, pass those same arguments to `val.py` to preserve the evaluation window.

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

- Training and validation use checkpoint-defined model horizons. Pass matching `--window_protocol` and `--gap_len` arguments to validation when changing them from their defaults.
- Use `--seed` to control the Lightning and data-loader seed.
- Keep the dataset cache, command-line arguments, checkpoint, and reported metrics together when recording an experiment.
- The repository does not include raw INTERACTION data, processed caches, checkpoints, or TensorBoard output.

## Reference

The training and validation organization follows [QCNet](https://github.com/ZikangZhou/QCNet). Environment setup and INTERACTION data preparation also follow [HPNet](https://github.com/XiaolongTang23/HPNet).

## Citation and contact

The paper citation will be added after publication. For questions or reproducibility issues, please open an issue in the [TrajBridge repository](https://github.com/Nova-chen151/TrajBridge).
