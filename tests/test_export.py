import json

from omegaconf import ListConfig

from ssl_models.conf.dataset.flair import FLAIRConfig
from ssl_models.conf.datasets import DatasetsConfig
from ssl_models.conf.utils import datasets_from_dict, datasets_to_dict, json_default


def test_export_listconfig_roundtrip():
    filter_inputs = ["aerial", "dem", "s2"]
    datasets = DatasetsConfig(
        root_dir="data",
        name_datasets=ListConfig(["flair"]),
        flair=FLAIRConfig(filter_inputs=ListConfig(filter_inputs)),
    )

    cfg = json.loads(json.dumps(datasets_to_dict(datasets), default=json_default))
    assert cfg["name_datasets"] == ["flair"]
    assert cfg["flair"]["filter_inputs"] == filter_inputs

    reloaded = datasets_from_dict(cfg)
    assert reloaded.flair.filter_inputs == filter_inputs
