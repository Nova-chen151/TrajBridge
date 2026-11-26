from typing import Optional

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from datasets import INTERACTIONDataset
from transforms import AgentRandomOcclusion
from transforms import HorizontalFlip
from transforms import LaneRandomOcclusion


class INTERACTIONDataModule(pl.LightningDataModule):

    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 prefetch_factor: Optional[int] = None,
                 flip_p: float = 0.5,
                 agent_occlusion_ratio: float = 0.0,
                 lane_occlusion_ratio: float = 0.2,
                 num_historical_steps: int = 10,
                 num_future_steps: int = 30,
                 task_mode: str = 'reconstruction',
                 reconstruction_tail_lens=None,
                 tail_len: Optional[int] = None,
                 mixed_tail_len: bool = True,
                 gap_len: Optional[int] = None,
                 window_protocol: str = 'fixed_total',
                 train_mixed_tail_len: bool = True,
                 eval_tail_len: int = 5,
                 allow_processed_only: bool = True,
                 processed_only: bool = False,
                 max_raw_files: Optional[int] = None,
                 max_cases_per_file: Optional[int] = None,
                 scenario_filter=None,
                 case_id_filter=None,
                 processed_dir_suffix: str = '',
                 **kwargs) -> None:
        super(INTERACTIONDataModule, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor
        train_transforms = [
            HorizontalFlip(flip_p),
            AgentRandomOcclusion(agent_occlusion_ratio, num_historical_steps),
            LaneRandomOcclusion(lane_occlusion_ratio),
        ]
        self.train_transform = Compose(train_transforms)
        self.val_transform = LaneRandomOcclusion(0.0)
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.task_mode = task_mode
        self.reconstruction_tail_lens = reconstruction_tail_lens or [0, 5, 10, 15]
        self.tail_len = tail_len
        self.mixed_tail_len = mixed_tail_len
        self.gap_len = gap_len
        self.window_protocol = window_protocol
        self.train_mixed_tail_len = train_mixed_tail_len
        self.eval_tail_len = eval_tail_len
        self.allow_processed_only = allow_processed_only
        self.processed_only = processed_only
        self.max_raw_files = max_raw_files
        self.max_cases_per_file = max_cases_per_file
        self.scenario_filter = scenario_filter
        self.case_id_filter = case_id_filter
        self.processed_dir_suffix = processed_dir_suffix

    def _dataset(self, split: str, transform):
        is_train = split == 'train'
        return INTERACTIONDataset(
            root=self.root,
            split=split,
            transform=transform,
            num_historical_steps=self.num_historical_steps,
            num_future_steps=self.num_future_steps,
            task_mode=self.task_mode,
            reconstruction_tail_lens=self.reconstruction_tail_lens,
            tail_len=None if is_train and self.train_mixed_tail_len else self.eval_tail_len,
            mixed_tail_len=self.train_mixed_tail_len if is_train else False,
            gap_len=self.gap_len,
            window_protocol=self.window_protocol,
            allow_processed_only=self.allow_processed_only,
            processed_only=self.processed_only,
            max_raw_files=self.max_raw_files,
            max_cases_per_file=self.max_cases_per_file,
            scenario_filter=self.scenario_filter,
            case_id_filter=self.case_id_filter,
            processed_dir_suffix=self.processed_dir_suffix,
        )

    def prepare_data(self) -> None:
        self._dataset('train', self.train_transform)
        self._dataset('val', self.val_transform)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = self._dataset('train', self.train_transform)
        self.val_dataset = self._dataset('val', self.val_transform)

    def train_dataloader(self):
        loader_kwargs = self._loader_kwargs()
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          **loader_kwargs)

    def val_dataloader(self):
        loader_kwargs = self._loader_kwargs()
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          **loader_kwargs)

    def _loader_kwargs(self):
        kwargs = dict(
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs['prefetch_factor'] = self.prefetch_factor
        return kwargs
