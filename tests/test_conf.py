import pytest

from ssl_models.conf.dataset.cosia_histo import NOMENCLATURES, COSIAHistoConfig
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


def test_cosia_histo_targets():
    config = COSIAHistoConfig()
    assert list(config.targets) == ["cosia"]
    assert len(config.cosia.class_names) == config.cosia.num_classes == 19

    names = list(NOMENCLATURES)
    config = COSIAHistoConfig(filter_targets=["cosia", *names])
    assert list(config.targets) == ["cosia", *names]
    for name, num_classes in zip(names, [8, 7, 5]):
        target = config.targets[name]
        assert len(target.class_names) == target.num_classes == num_classes
        assert len(config.class_remaps[name]) == 19
        assert max(config.class_remaps[name]) == num_classes - 1


def test_cosia_histo_remap_to_others():
    config = COSIAHistoConfig(remap_to_others_after_class=2)
    assert config.cosia.num_classes == 3
    assert config.cosia.class_names == ["building", "greenhouse", "other"]
    assert config.class_remaps["cosia"] == [0, 1, *[2] * 17]


@pytest.mark.parametrize("n_kept", [0, 19])
def test_cosia_histo_remap_to_others_out_of_range(n_kept: int):
    with pytest.raises(ValueError, match="remap_to_others_after_class"):
        COSIAHistoConfig(remap_to_others_after_class=n_kept)


def test_segmentation_datasets_have_class_names():
    targets = [
        FLAIRConfig().cosia,
        HiUCDConfig().hi_ucd_seg_1,
        PASTISHDConfig().pastis_seg,
        S2NAIPConfig().osm_seg,
    ]
    for target in targets:
        assert len(target.class_names) == target.num_classes
