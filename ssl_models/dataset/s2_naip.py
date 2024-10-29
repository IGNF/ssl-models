"""S2-NAIP dataset module."""

from torch.utils.data import Dataset


class S2NAIPDataset(Dataset):
    """S2-NAIP dataset."""

    def __init__(
        self,
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> None:
        raise NotImplementedError
