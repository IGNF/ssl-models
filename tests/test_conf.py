import pytest

from ssl_models.conf.dataset.flair import FLAIRConfig
from ssl_models.conf.dataset.hi_ucd import HiUCDConfig
from ssl_models.conf.dataset.pastis_hd import PASTISHDConfig
from ssl_models.conf.dataset.s2_naip import S2NAIPConfig
from ssl_models.conf.dataset.tempetes import (
    REMAPPED_CLASS_NAMES,
    TempetesConfig,
)
from ssl_models.conf.dataset.utils import TargetConfig, TargetRasterConfig


def test_class_names_default_to_none():
    assert TargetConfig(type_target="segment", num_classes=2).class_names is None


def test_class_names_length_is_validated():
    TargetConfig(type_target="segment", num_classes=2, class_names=["a", "b"])
    with pytest.raises(ValueError, match="class names"):
        TargetConfig(type_target="segment", num_classes=3, class_names=["a", "b"])


def test_tempetes_targets():
    config = TempetesConfig()
    assert len(config.tempetes.class_names) == config.tempetes.num_classes == 22
    remap = config.tempetes_remap
    assert len(remap.class_names) == remap.num_classes == 9
    assert list(config.targets) == ["tempetes_remap"]

    both = TempetesConfig(filter_targets=["tempetes", "tempetes_remap"])
    assert list(both.targets) == ["tempetes", "tempetes_remap"]


def test_tempetes_custom_remap_needs_matching_class_names():
    custom_remap = {**TempetesConfig().class_remap, 21: 0}  # leaves 8 classes
    with pytest.raises(ValueError, match="class names"):
        TempetesConfig(class_remap=custom_remap)

    config = TempetesConfig(
        class_remap=custom_remap,
        tempetes_remap=TargetRasterConfig(
            type_target="segment",
            num_classes=8,
            class_names=REMAPPED_CLASS_NAMES[:8],
        ),
    )
    assert config.tempetes_remap.num_classes == 8


def test_segmentation_datasets_have_class_names():
    targets = [
        FLAIRConfig().cosia,
        HiUCDConfig().hi_ucd_seg_1,
        PASTISHDConfig().pastis_seg,
        S2NAIPConfig().osm_seg,
    ]
    for target in targets:
        assert len(target.class_names) == target.num_classes
