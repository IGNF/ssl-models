"""SSL data module."""

from pathlib import Path
from typing import Literal

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from ssl_models.conf.dataset.utils import DatasetConfig
from ssl_models.conf.opt import OptFinetuneConfig, OptPretrainConfig, OptProbeConfig


class SSLDataModule(LightningDataModule):
    """SSL data module."""

    def __init__(
        self,
        dataset: DatasetConfig,
        root_dir: str,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        opt: OptPretrainConfig | OptProbeConfig | OptFinetuneConfig,
        num_workers: int,
        random_dates: bool,
        random_crop: bool,
    ) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = opt.batch_size

        dataset_kwargs = {
            "dataset": dataset,
            "root_dir": Path(root_dir) / dataset.rel_dir,
            "ssl_phase": ssl_phase,
            # read-time augmentations (other augmentations are on GPU)
            "random_dates": random_dates,
            "random_crop": random_crop,
        }
        self.train_dataset = dataset.dataset_class(stage="train", **dataset_kwargs)
        self.val_dataset = dataset.dataset_class(stage="val", **dataset_kwargs)
        self.test_dataset = dataset.dataset_class(stage="test", **dataset_kwargs)

    def is_distributed(self) -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def _dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        sampler = (
            DistributedSampler(dataset, shuffle=shuffle, drop_last=True)
            if self.is_distributed()
            else None
        )
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle and sampler is None,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        """Return train data loader."""
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return val data loader."""
        return self._dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return test data loader."""
        return self._dataloader(self.test_dataset, shuffle=False)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch of the train dataset (for epoch-dependent sampling)."""
        self.train_dataset.set_epoch(epoch)
