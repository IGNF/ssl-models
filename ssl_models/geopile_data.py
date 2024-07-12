"""GeoPile data module."""

from functools import partial
from pathlib import Path

import albumentations as A  # noqa: N812
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision.transforms import RandAugment, v2
from torchvision.transforms.functional import InterpolationMode


class GeoPileData(LightningDataModule):
    """GeoPile data module."""

    def __init__(
        self,
        ssl_phase: str,
        root_dir: str,
        batch_size: int = 32,
        num_workers: int = 1,
        use_transform: bool = True,
        use_mixup: bool = False,
        rand_augment: bool = False,
    ) -> None:
        super().__init__()
        if use_transform:
            transform = A.Compose(
                [
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ],
            )
        else:
            transform = None

        match ssl_phase:
            case "pretrain":
                use_mixup = False
                rand_augment = False
            case "probe" | "finetune":
                pass

        train_path = Path(root_dir) / "train.csv"
        train_dataset = GeoPileDataset(
            train_path,
            root_dir,
            ssl_phase,
            transform,
            rand_augment,
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=partial(self.collate_fn, use_mixup=use_mixup),
        )

        val_path = Path(root_dir) / "val.csv"
        val_dataset = GeoPileDataset(
            val_path,
            root_dir,
            ssl_phase,
            transform,
            rand_augment,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=partial(self.collate_fn, use_mixup=False),
        )

    def train_dataloader(self) -> DataLoader:
        """Return train data loader."""
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        """Return val data loader."""
        return self.val_loader

    def collate_fn(self, batch: list, use_mixup: bool) -> dict[str, Tensor]:
        """Collate function."""
        batch = default_collate(batch)
        for dataset, num_classes in GeoPileDataset.NUM_CLASSES.items():
            inds = (
                (batch[f"target_{dataset}"] != GeoPileDataset.MISSING_VAL)
                .nonzero()
                .squeeze(dim=1)
            )
            image_dataset = torch.index_select(batch["image"], dim=0, index=inds)
            target_dataset = torch.index_select(
                batch[f"target_{dataset}"],
                dim=0,
                index=inds,
            )
            if use_mixup:
                image_dataset, one_hot_dataset = GeoPileDataset.MIXUP[dataset](
                    image_dataset,
                    target_dataset,
                )
            else:
                one_hot_dataset = F.one_hot(
                    target_dataset,
                    num_classes=num_classes,
                ).float()

            B, C, H, W = batch["image"].shape  # noqa: N806
            batch[f"one_hot_{dataset}"] = torch.zeros(
                B,
                num_classes,
                dtype=torch.float,
            )
            batch[f"one_hot_{dataset}"].scatter_(
                0,
                inds[:, None].repeat(1, num_classes),
                one_hot_dataset,
            )
            batch["image"].scatter_(
                0,
                inds[:, None, None, None].repeat(1, C, H, W),
                image_dataset,
            )
        return batch


class GeoPileDataset(Dataset):
    """GeoPile dataset."""

    NUM_CLASSES: dict[str, int] = {  # noqa: RUF012
        "mlrsnet": 46,
        "patternnet": 38,
        "rsd46-whu": 46,
        "resisc45": 45,
    }
    NORM_FAC: int = 255.0
    MISSING_VAL: int = -1
    MIXUP: dict[str, int] = {  # noqa: RUF012
        dataset: v2.MixUp(alpha=0.2, num_classes=num_classes)
        for dataset, num_classes in NUM_CLASSES.items()
    }

    def __init__(
        self,
        csv_path: str | Path,
        root_dir: str | Path,
        ssl_phase: str,
        transform: A.Compose | None,
        rand_augment: bool,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform

        csv_data = pd.read_csv(csv_path)
        match ssl_phase:
            case "pretrain":
                pass
            case "probe" | "finetune":
                class_cols = [f"class_index_{dataset}" for dataset in self.NUM_CLASSES]
                csv_data = csv_data[
                    (csv_data[class_cols] != self.MISSING_VAL).any(axis=1)
                ]

        self.file_name = csv_data["image"].tolist()
        self.targets = {
            dataset: csv_data[f"class_index_{dataset}"].tolist()
            for dataset in self.NUM_CLASSES
        }

        if rand_augment and ssl_phase != "pretrain":
            self.rand_augment = RandAugment(
                num_ops=2,
                magnitude=10,
                interpolation=InterpolationMode.BILINEAR,
            )
        else:
            self.rand_augment = None

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get dataset item."""
        file_path = self.root_dir / self.file_name[idx]
        image = Image.open(file_path)

        if self.rand_augment:
            image = self.rand_augment(image)

        image = np.array(image, dtype=np.float32) / self.NORM_FAC
        if self.transform is not None:
            image = self.transform(image=image)["image"]
            image = image.transpose([2, 0, 1])

        targets = {
            f"target_{dataset}": target[idx] for dataset, target in self.targets.items()
        }
        return {"image": image, **targets}

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.file_name)
