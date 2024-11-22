"""SSL data module."""

from pathlib import Path
from typing import Literal

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from conf.datasets import DatasetsConfig


class SSLDataModule(LightningDataModule):
    """SSL data module."""

    def __init__(
        self,
        datasets: DatasetsConfig,
        ssl_phase: Literal["pretrain", "probe", "finetune"],
        batch_size: int,
        num_workers: int,
        use_transform: bool,
    ) -> None:
        super().__init__()

        root_dir = Path(datasets.root_dir) / datasets.dataset.rel_dir
        train_dataset = datasets.dataset_class(
            root_dir=root_dir,
            csv_path=root_dir / "train.csv",
            ssl_phase=ssl_phase,
            use_transform=use_transform,
            dataset=datasets.dataset,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
        )

        val_dataset = datasets.dataset_class(
            root_dir=root_dir,
            csv_path=root_dir / "val.csv",
            ssl_phase=ssl_phase,
            use_transform=use_transform,
            dataset=datasets.dataset,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
        )

        test_dataset = datasets.dataset_class(
            root_dir=root_dir,
            csv_path=root_dir / "test.csv",
            ssl_phase=ssl_phase,
            use_transform=use_transform,
            dataset=datasets.dataset,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
        )

    def train_dataloader(self) -> DataLoader:
        """Return train data loader."""
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        """Return val data loader."""
        return self.val_loader

    def test_dataloader(self) -> DataLoader:
        """Return test data loader."""
        return self.test_loader
