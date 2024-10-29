"""FLAIR dataset module."""

from torch.utils.data import Dataset


class FLAIRDataset(Dataset):
    """FLAIR dataset."""

    def __init__(
        self,
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> None:
        raise NotImplementedError
