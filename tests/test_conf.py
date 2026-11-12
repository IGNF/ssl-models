import pytest

from ssl_models.conf.dataset.utils import TargetConfig


def test_class_names_default_to_none():
    assert TargetConfig(type_target="segment", num_classes=2).class_names is None


def test_class_names_length_is_validated():
    TargetConfig(type_target="segment", num_classes=2, class_names=["a", "b"])
    with pytest.raises(ValueError, match="class names"):
        TargetConfig(type_target="segment", num_classes=3, class_names=["a", "b"])
