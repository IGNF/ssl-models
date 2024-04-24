import pytest

from ssl_models.ssl_module import SSLModule


@pytest.mark.parametrize("stage", ["train", "val", "test", "predict"])
def test_mae(stage):
    ssl_module = SSLModule()
    ssl_module.setup(stage=stage)
