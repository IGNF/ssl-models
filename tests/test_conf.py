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


def test_tempetes_class_names():
    remapped = TempetesConfig().tempetes
    assert len(remapped.class_names) == remapped.num_classes == 9

    original = TempetesConfig(class_remap=None).tempetes
    assert len(original.class_names) == original.num_classes == 22

    custom = TempetesConfig(class_remap={1: 0}).tempetes
    assert custom.class_names is None


def test_tempetes_recomputes_generated_names_after_remap():
    generated_names = TargetRasterConfig(
        type_target="segment",
        num_classes=9,
        class_names=list(REMAPPED_CLASS_NAMES),
    )
    remapped = TempetesConfig(
        class_remap={1: 0},
        tempetes=generated_names,
    ).tempetes
    assert remapped.num_classes == 22
    assert remapped.class_names is None

    custom_nine_classes = TempetesConfig(
        class_remap={**TempetesConfig().class_remap, 8: 0},
        tempetes=TargetRasterConfig(
            type_target="segment",
            num_classes=9,
            class_names=list(REMAPPED_CLASS_NAMES),
        ),
    ).tempetes
    assert custom_nine_classes.num_classes == 9
    assert custom_nine_classes.class_names is None


def test_tempetes_preserves_explicit_class_names():
    class_names = [f"class_{i}" for i in range(9)]
    config = TempetesConfig(
        class_remap={**TempetesConfig().class_remap, 8: 0},
        tempetes=TargetRasterConfig(
            type_target="segment",
            num_classes=9,
            class_names=class_names,
        ),
    )
    assert config.tempetes.class_names == class_names


def test_segmentation_datasets_have_class_names():
    targets = [
        FLAIRConfig().cosia,
        HiUCDConfig().hi_ucd_seg_1,
        PASTISHDConfig().pastis_seg,
        S2NAIPConfig().osm_seg,
    ]
    for target in targets:
        assert len(target.class_names) == target.num_classes
