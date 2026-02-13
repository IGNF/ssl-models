"""SSL data module."""

from functools import partial
from pathlib import Path
from typing import Literal

from lightning.pytorch.utilities.combined_loader import CombinedLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ssl_models.conf.datasets import DatasetsConfig
from ssl_models.conf.opt import OptFinetuneConfig, OptPretrainConfig, OptProbeConfig


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
        self.num_workers = num_workers

        self.batch_size = {}
        self.train_dataset, self.val_dataset, self.test_dataset = {}, {}, {}
        for name_dataset, dataset in datasets.datasets.items():
            root_dir = Path(datasets.root_dir) / dataset.rel_dir
            dataset_class = partial(
                dataset.dataset_class,
                dataset=dataset,
                root_dir=root_dir,
                ssl_phase=ssl_phase,
                use_transform=use_transform,
                random_dates=random_dates,
                random_crop=random_crop,
            )
            self.train_dataset[name_dataset] = dataset_class(stage="train")
            self.val_dataset[name_dataset] = dataset_class(stage="val")
            self.test_dataset[name_dataset] = dataset_class(stage="test")

            self.batch_size[name_dataset] = (
                opt.batch_size
                * datasets.sample[name_dataset]
                / sum(datasets.sample.values())
            )
            self.batch_size[name_dataset] = max(round(self.batch_size[name_dataset]), 1)

    def train_dataloader(self) -> DataLoader:
        """Return train data loader."""
        train_loader = {}
        for name_dataset in self.train_dataset:
            train_loader[name_dataset] = DataLoader(
                self.train_dataset[name_dataset],
                sampler=DistributedSampler(
                    self.train_dataset[name_dataset], shuffle=True,
                ),
                batch_size=self.batch_size[name_dataset],
                num_workers=self.num_workers,
                shuffle=False,
                drop_last=True,
            )
        return CombinedLoader(train_loader, "min_size")

    def val_dataloader(self) -> DataLoader:
        """Return val data loader."""
        val_loader = {}
        for name_dataset in self.val_dataset:
            val_loader[name_dataset] = DataLoader(
                self.val_dataset[name_dataset],
                sampler=DistributedSampler(
                    self.val_dataset[name_dataset], shuffle=True,
                ),
                batch_size=self.batch_size[name_dataset],
                num_workers=self.num_workers,
                shuffle=False,
                drop_last=True,
            )
        return CombinedLoader(val_loader, "max_size")

    def test_dataloader(self) -> DataLoader:
        """Return test data loader."""
        test_loader = {}
        for name_dataset in self.test_dataset:
            test_loader[name_dataset] = DataLoader(
                self.test_dataset[name_dataset],
                sampler=DistributedSampler(
                    self.test_dataset[name_dataset], shuffle=True,
                ),
                batch_size=self.batch_size[name_dataset],
                num_workers=self.num_workers,
                shuffle=False,
                drop_last=True,
            )
        return CombinedLoader(test_loader, "max_size")
