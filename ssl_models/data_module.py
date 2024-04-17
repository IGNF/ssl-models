"""
LightningDataModule for SSL.
"""
from odeon.data import Input

DEFAULT_PARAMS = {
    "input_fields": {
        "image": {
            "name": "image",
            "type": "raster",
            "dtype": "uint8",
            "mean": 0,
            "std": 1,
        },
    }
}


class SSLDataModule(Input):
    def __init__(
        self,
        root_dir: str,
        train_dataset: str,
        val_dataset: str,
    ):
        fit_params = {
            **DEFAULT_PARAMS,
            "input_file": train_dataset,
            "root_dir": root_dir,
        }
        validate_params = {
            **DEFAULT_PARAMS,
            "input_file": val_dataset,
            "root_dir": root_dir,
        }
        super().__init__(fit_params=fit_params, validate_params=validate_params)
