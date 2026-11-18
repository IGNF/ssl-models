from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ssl_models.train.augment import Augment


def fake_dataset(sizes, targets=(), rescale_elev=(), local_sizes=None):
    """Minimal dataset config: inputs {name: image size}, raster targets."""
    local_sizes = local_sizes or sizes
    inputs = {
        name: SimpleNamespace(
            image_size=size,
            local_size=local_sizes[name],
            rescale_elev=name in rescale_elev,
        )
        for name, size in sizes.items()
    }
    targets = {name: SimpleNamespace() for name in targets}
    return SimpleNamespace(
        inputs=inputs, targets=targets, rasters={**inputs, **targets}
    )


def coordinate_batch(sizes, batch_size=3, num_dates=2):
    """Each modality holds its normalized x / y coordinates as two bands."""
    batch = {}
    for name_mod, size in sizes.items():
        c = (torch.arange(size, dtype=torch.float32) + 0.5) / size * 2 - 1
        yy, xx = torch.meshgrid(c, c, indexing="ij")
        batch[name_mod] = torch.stack([xx, yy])[None, None].repeat(
            batch_size, num_dates, 1, 1, 1
        )
    return batch


def test_crops_aligned_across_modalities():
    """The same box is applied to all modalities, whatever their size."""
    torch.manual_seed(0)
    augment = Augment(
        fake_dataset({"high": 150, "low": 6}, local_sizes={"high": 60, "low": 4}),
        n_global=2,
        global_scale=[0.32, 1.0],
        n_local=2,
        local_scale=[0.05, 0.32],
    )
    out = augment(coordinate_batch({"high": 300, "low": 30}))
    for suffix in ("", "_local"):
        high, low = out[f"high{suffix}"], out[f"low{suffix}"]
        # 2 views of 3 samples, stacked along the batch dimension
        assert high.shape[0] == low.shape[0] == 2 * 3
        high_pooled = F.adaptive_avg_pool2d(high.flatten(1, 2), low.shape[-1])
        # flips / transposition consistent, coordinates match up to resampling
        assert torch.allclose(high_pooled, low.flatten(1, 2), atol=0.1)
        # dates are augmented consistently
        assert torch.equal(high[:, 0], high[:, 1])


def test_views_stacked_sample_major():
    """Views are stacked sample-major, other entries repeated per view."""
    torch.manual_seed(0)
    augment = Augment(fake_dataset({"a": 16}), n_global=3, global_scale=[0.3, 1])
    batch = coordinate_batch({"a": 32}, batch_size=2)
    batch["a_dates"] = torch.arange(2 * 2 * 3).reshape(2, 2, 3)
    out = augment(batch)

    assert out["a"].shape == (2 * 3, 2, 2, 16, 16)
    dates = out["a_dates"].unflatten(0, (2, 3))
    for idx in range(3):
        assert torch.equal(dates[:, idx], batch["a_dates"])
    # different random boxes for the views of a same sample
    views = out["a"].unflatten(0, (2, 3))
    assert not torch.equal(views[0, 0], views[0, 1])


def test_crop_scale():
    torch.manual_seed(0)
    augment = Augment(fake_dataset({}))
    theta = augment.sample_boxes(1000, (0.05, 0.32), "cpu")
    area = theta[:, 0, 0] * theta[:, 1, 1]
    assert area.min() >= 0.05 - 1e-6 and area.max() <= 0.32 + 1e-6
    # boxes stay inside the image
    assert (theta[:, 0, 2].abs() + theta[:, 0, 0] <= 1 + 1e-6).all()
    assert (theta[:, 1, 2].abs() + theta[:, 1, 1] <= 1 + 1e-6).all()


def test_jitter_blur():
    torch.manual_seed(0)
    augment = Augment(fake_dataset({}), jitter_prob=1.0, brightness=0.2, contrast=0.2)
    x = torch.rand(4, 2, 3, 64, 64)
    assert augment.jitter(x).shape == x.shape
    blurred = augment.blur(x, prob=1.0)
    assert blurred.shape == x.shape
    assert blurred.std() < x.std()
    assert torch.equal(augment.blur(x, prob=0.0), x)


@pytest.mark.parametrize("interpolate", ("nearest", "bilinear"))
@pytest.mark.parametrize("random", (True, False))
def test_default_view_is_resize(interpolate, random):
    """Without flips, the default (single global) view is a plain resize."""
    augment = Augment(fake_dataset({"a": 10}), interpolate=interpolate, flip=False)
    batch = {"a": torch.rand(2, 3, 4, 20, 20), "a_dates": torch.zeros(2, 3, 3)}
    out = augment(batch, random=random)
    expected = F.interpolate(batch["a"].flatten(0, 1), size=10, mode=interpolate)
    assert torch.equal(out["a"], expected.unflatten(0, (2, 3)))
    assert out["a_dates"] is batch["a_dates"]  # other keys kept


def test_view_not_random():
    """random=False: no crop, jitter or blur, even if configured."""
    augment = Augment(
        fake_dataset({"a": 64}),
        flip=False,
        jitter_prob=1.0,
        brightness=0.2,
        contrast=0.2,
        global_scale=[0.1, 0.5],
        global_blur_probs=[1.0],
    )
    batch = {"a": torch.rand(2, 1, 3, 64, 64)}
    out = augment(batch, random=False)
    assert torch.equal(out["a"], batch["a"])


def test_view_targets_aligned():
    """Raster targets get the same crop and flips as the inputs."""
    torch.manual_seed(0)
    augment = Augment(
        fake_dataset({"img": 16}, targets=("seg",)), global_scale=[0.3, 1.0]
    )
    batch = coordinate_batch({"img": 64, "seg": 32}, num_dates=1)
    out = augment(batch, random=True)

    assert out["seg"].shape == batch["seg"].shape  # targets keep their size
    seg = F.adaptive_avg_pool2d(out["seg"].flatten(1, 2), 16)
    assert torch.allclose(seg, out["img"].flatten(1, 2), atol=0.15)
    assert not torch.equal(out["seg"], batch["seg"])  # actually transformed


def test_rescale_elev():
    augment = Augment(
        fake_dataset({"dem": 8, "rgb": 8}, rescale_elev=("dem",)), flip=False
    )
    batch = {"dem": torch.rand(2, 1, 2, 8, 8), "rgb": torch.rand(2, 1, 3, 8, 8)}
    out = augment(batch, random=False)
    dem = batch["dem"]
    assert torch.equal(out["dem"][:, :, :1], dem[:, :, :1])
    assert torch.allclose(out["dem"][:, :, 1:], 30 * (dem[:, :, :1] - dem[:, :, 1:]))
    assert torch.equal(out["rgb"], batch["rgb"])


def test_local_views_keys():
    """Local views at `{key}_local` (dates and raster targets included)."""
    augment = Augment(
        fake_dataset({"img": 16}, targets=("seg",), local_sizes={"img": 8}),
        n_global=2,
        n_local=3,
    )
    batch = coordinate_batch({"img": 32, "seg": 32}, batch_size=2, num_dates=1)
    batch["img_dates"] = torch.zeros(2, 1, 3)
    out = augment(batch)

    assert set(out) == {
        "img",
        "seg",
        "img_dates",
        "img_local",
        "seg_local",
        "img_dates_local",
    }
    assert out["img"].shape == (2 * 2, 1, 2, 16, 16)
    assert out["seg"].shape == (2 * 2, 1, 2, 32, 32)
    assert out["img_local"].shape == (3 * 2, 1, 2, 8, 8)
    assert out["img_dates_local"].shape == (3 * 2, 1, 3)
    assert out["seg_local"].shape == (3 * 2, 1, 2, 32, 32)  # native size
