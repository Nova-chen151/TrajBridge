"""Validate a TrajBridge checkpoint on the INTERACTION validation split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader

from datasets import INTERACTIONDataset
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a TrajBridge checkpoint on INTERACTION.")
    parser.add_argument("--root", type=str, required=True, help="INTERACTION dataset root.")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--persistent_workers", type=str2bool, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--limit_val_batches", type=limit_batches, default=None)
    parser.add_argument("--default_root_dir", type=str, default=None)
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="high")
    parser.add_argument("--gap_len", type=int, default=None)
    parser.add_argument("--window_protocol", choices=["fixed_gap", "fixed_total"], default="fixed_total")
    parser.add_argument("--allow_processed_only", type=str2bool, default=True)
    parser.add_argument("--processed_only", type=str2bool, default=False)
    parser.add_argument("--max_raw_files", type=int, default=None)
    parser.add_argument("--max_cases_per_file", type=int, default=None)
    parser.add_argument("--scenario_filter", type=str, nargs="*", default=None)
    parser.add_argument("--case_id_filter", type=int, nargs="*", default=None)
    parser.add_argument("--processed_dir_suffix", type=str, default="")
    return parser.parse_args()


def build_validation_dataset(args: argparse.Namespace, model: TrajBridge) -> INTERACTIONDataset:
    tail_lens = getattr(model, "reconstruction_tail_lens", None) or [0, 5, 10, 15]
    return INTERACTIONDataset(
        root=args.root,
        split="val",
        transform=None,
        num_historical_steps=int(model.num_historical_steps),
        num_future_steps=int(model.num_future_steps),
        task_mode=getattr(model, "task_mode", "reconstruction"),
        reconstruction_tail_lens=list(tail_lens),
        tail_len=getattr(model, "eval_tail_len", 5),
        mixed_tail_len=False,
        gap_len=args.gap_len,
        window_protocol=args.window_protocol,
        allow_processed_only=args.allow_processed_only,
        processed_only=args.processed_only,
        max_raw_files=args.max_raw_files,
        max_cases_per_file=args.max_cases_per_file,
        scenario_filter=args.scenario_filter,
        case_id_filter=args.case_id_filter,
        processed_dir_suffix=args.processed_dir_suffix,
    )


def main() -> None:
    args = parse_args()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(args.matmul_precision)

    model = TrajBridge.load_from_checkpoint(args.ckpt_path, map_location="cpu")
    dataset = build_validation_dataset(args, model)
    loader_kwargs = {
        "batch_size": args.val_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers and args.num_workers > 0,
    }
    if args.num_workers > 0 and args.prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    dataloader = DataLoader(dataset, **loader_kwargs)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=args.default_root_dir,
        precision=args.precision,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches is not None else 1.0,
        logger=False,
        enable_checkpointing=False,
    )
    results = trainer.validate(model=model, dataloaders=dataloader, verbose=True)
    print(results)


if __name__ == "__main__":
    main()
