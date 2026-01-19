"""SSL data module."""

from pathlib import Path
from typing import Literal

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from ssl_models.conf.datasets import DatasetsConfig
from ssl_models.conf.opt import OptFinetuneConfig, OptPretrainConfig, OptProbeConfig
from ssl_models.train.sampler import DistributedWeightedSampler


class SSLDataModule(LightningDataModule):
    """SSL data module."""

    def __init__(
        self,
        datasets: DatasetsConfig,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        opt: OptPretrainConfig | OptProbeConfig | OptFinetuneConfig,
        num_workers: int,
        use_transform: bool,
        random_dates: bool,
        random_crop: bool,
    ) -> None:
        super().__init__()
        self.batch_size = opt.batch_size
        self.num_workers = num_workers
        self.balance_weights = None

        root_dir = Path(datasets.root_dir) / datasets.dataset.rel_dir
        self.full_train_dataset = datasets.dataset_class(
            dataset=datasets.dataset,
            root_dir=root_dir,
            stage="train",
            ssl_phase=ssl_phase,
            use_transform=use_transform,
            random_dates=random_dates,
            random_crop=random_crop,
        )
        self.train_dataset = self.full_train_dataset

        self.val_dataset = datasets.dataset_class(
            dataset=datasets.dataset,
            root_dir=root_dir,
            stage="val",
            ssl_phase=ssl_phase,
            use_transform=use_transform,
            random_dates=random_dates,
            random_crop=random_crop,
        )

        self.test_dataset = datasets.dataset_class(
            dataset=datasets.dataset,
            root_dir=root_dir,
            stage="test",
            ssl_phase=ssl_phase,
            use_transform=use_transform,
            random_dates=random_dates,
            random_crop=random_crop,
        )

    def train_dataloader(self) -> DataLoader:
        """Return train data loader."""
        if self.balance_weights is not None:
            sampler = DistributedWeightedSampler(
                weights=self.balance_weights,              # numpy or tensor
                num_samples=len(self.train_dataset),
                replacement=True,
            )
        else:
            sampler = None

        return DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=sampler is None,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Return val data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False,
        )
