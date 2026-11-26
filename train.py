"""Train TrajBridge on the INTERACTION dataset."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from datamodules import INTERACTIONDataModule
from model import TrajBridge


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def limit_batches(value):
    if value is None:
        return None
    value = str(value)
    if value.lower() in {"none", "null"}:
        return None
    return float(value) if "." in value else int(value)


def add_model_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("TrajBridge")
    group.add_argument("--hidden_dim", type=int, default=128)
    group.add_argument("--num_historical_steps", type=int, default=10)
    group.add_argument("--num_future_steps", type=int, default=30)
    group.add_argument("--duration", type=int, default=10)
    group.add_argument("--a2a_radius", type=float, default=80.0)
    group.add_argument("--l2a_radius", type=float, default=80.0)
    group.add_argument("--num_visible_steps", type=int, default=3)
    group.add_argument("--num_modes", type=int, default=6)
    group.add_argument("--num_attn_layers", type=int, default=3)
    group.add_argument("--num_hops", type=int, default=4)
    group.add_argument("--num_heads", type=int, default=8)
    group.add_argument("--dropout", type=float, default=0.1)
    group.add_argument("--lr", type=float, default=5e-4)
    group.add_argument("--weight_decay", type=float, default=1e-4)
    group.add_argument("--warmup_epochs", type=int, default=4)
    group.add_argument("--T_max", type=int, default=64)
    group.add_argument("--reconstruction_tail_lens", type=int, nargs="+", default=[0, 5, 10, 15])
    group.add_argument("--tail_len", type=int, default=None)
    group.add_argument("--mixed_tail_len", type=str2bool, default=True)
    group.add_argument("--train_mixed_tail_len", type=str2bool, default=True)
    group.add_argument("--eval_tail_len", type=int, default=5)
    group.add_argument("--use_map_encoder", type=str2bool, default=True)
    group.add_argument("--target_agent_policy", choices=["category", "all_valid"], default="category")
    group.add_argument("--target_categories", type=int, nargs="+", default=[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TrajBridge on INTERACTION.")
    parser.add_argument("--root", type=str, required=True, help="INTERACTION dataset root.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--shuffle", type=str2bool, default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--persistent_workers", type=str2bool, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--flip_p", type=float, default=0.5)
    parser.add_argument("--agent_occlusion_ratio", type=float, default=0.0)
    parser.add_argument("--lane_occlusion_ratio", type=float, default=0.2)
    parser.add_argument("--gap_len", type=int, default=None)
    parser.add_argument("--window_protocol", choices=["fixed_gap", "fixed_total"], default="fixed_total")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--max_epochs", type=int, default=64)
    parser.add_argument("--limit_train_batches", type=limit_batches, default=None)
    parser.add_argument("--limit_val_batches", type=limit_batches, default=None)
    parser.add_argument(
        "--default_root_dir",
        type=str,
        default=None,
        help="Run directory. Defaults to runs/<timestamp>.",
    )
    parser.add_argument("--save_top_k", type=int, default=1)
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--allow_processed_only", type=str2bool, default=True)
    parser.add_argument("--processed_only", type=str2bool, default=False)
    parser.add_argument("--max_raw_files", type=int, default=None)
    parser.add_argument("--max_cases_per_file", type=int, default=None)
    parser.add_argument("--scenario_filter", type=str, nargs="*", default=None)
    parser.add_argument("--case_id_filter", type=int, nargs="*", default=None)
    parser.add_argument("--processed_dir_suffix", type=str, default="")
    add_model_args(parser)
    return parser.parse_args()


def make_run_dir(args: argparse.Namespace) -> Path:
    if args.default_root_dir:
        return Path(args.default_root_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "runs" / timestamp


def main() -> None:
    args = parse_args()
    args.task_mode = "reconstruction"
    args.map_encoder_mode = "true"
    args.use_posterior_bridge = False
    args.use_progressive_bridge = False
    args.use_flow_bridge = False

    pl.seed_everything(args.seed, workers=True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(args.matmul_precision)

    run_dir = make_run_dir(args)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = TrajBridge(**vars(args))
    datamodule = INTERACTIONDataModule(**vars(args))

    monitor = "val_recon_minJointFDE"
    checkpoint = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="{epoch:02d}-{" + monitor + ":.3f}",
        monitor=monitor,
        mode="min",
        save_top_k=args.save_top_k,
        save_last=True,
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=str(run_dir),
        logger=TensorBoardLogger(save_dir=str(run_dir), name="TrajBridge"),
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches is not None else 1.0,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches is not None else 1.0,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    trainer.fit(model, datamodule=datamodule)
    print(f"best_checkpoint={checkpoint.best_model_path}")


if __name__ == "__main__":
    main()
