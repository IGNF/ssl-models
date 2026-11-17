import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from ssl_models.conf.augment import DinoAugmentConfig
from ssl_models.conf.loss import DINOv3LossConfig
from ssl_models.conf.mask import DINOv3MaskConfig
from ssl_models.dino import dino as dino_module
from ssl_models.dino.augment import MultiCropAugment
from ssl_models.dino.dino import (
    DINOv3,
    dinov3_base,
    dinov3_huge_plus,
    dinov3_large,
    dinov3_small,
    dinov3_tiny,
    local_view_size,
)
from ssl_models.dino.head import DINOHead
from ssl_models.dino.loss import (
    dino_loss,
    ibot_loss,
    koleo_loss,
    sinkhorn_knopp,
    teacher_stats,
)
from ssl_models.dino.masking import ibot_masks, view_keep
from ssl_models.dino.schedule import cosine, linear_warmup, lr_factor
from ssl_models.train.lit_dino import LitDINOv3

FUSION_CONFIGS = [
    ("shared", 0),
    ("monotemp", 0),
    ("mod", 0),
    ("mod", 2),
    ("group", 0),
    ("group", 2),
]


def make_batch(dataset, batch_size=2, image_size=None):
    """Random batch following the dataset config (dates as int16)."""
    dataset = dataset
    batch = {}
    for name_mod, mod in dataset.inputs.items():
        C = mod.bands if isinstance(mod.bands, int) else sum(map(len, mod.bands))
        size = image_size or mod.image_size
        batch[name_mod] = torch.rand(batch_size, mod.num_dates, C, size, size)
        dates = torch.tensor([2019, 100, 12], dtype=torch.int16)
        dates = dates.repeat(batch_size, mod.num_dates, 1)
        dates[:, :, 1] += torch.arange(mod.num_dates, dtype=torch.int16) * 10
        batch[f"{name_mod}_dates"] = dates
    batch["ref_date"] = torch.tensor([2019, 180, 0], dtype=torch.int16).repeat(
        batch_size,
        1,
        1,
    )
    return batch


def view_inputs(batch, view):
    inputs = dict(view)
    inputs.update({k: v for k, v in batch.items() if "date" in k})
    return inputs


def small_backbone(dataset, fusion_mode, inter_depth, **kwargs):
    return DINOv3(
        dataset=dataset,
        fusion_mode=fusion_mode,
        inter_depth=inter_depth,
        embed_dim=64,
        depth=3,
        heads=2,
        **kwargs,
    )


# ---------------------------------------------------------------- backbone
def test_local_view_size():
    assert local_view_size(300, 20, 112 / 256) == 140
    assert local_view_size(6, 2, 112 / 256) == 4  # at least 2 patches
    assert local_view_size(2, 2, 112 / 256) == 2  # at most image size


@pytest.mark.parametrize(
    "size_fn",
    (dinov3_tiny, dinov3_small, dinov3_base, dinov3_large, dinov3_huge_plus),
)
def test_backbone_sizes(monkeypatch, size_fn):
    # capture the size arguments without building the backbone
    monkeypatch.setattr(dino_module, "DINOv3", lambda **kw: kw)
    args = size_fn()
    assert args["embed_dim"] % args["heads"] == 0
    assert (args["embed_dim"] // args["heads"]) % 4 == 0  # RoPE head dim
    assert size_fn(ffn="swiglu", mlp_ratio=3.0)["ffn"] == "swiglu"


@pytest.mark.parametrize("fusion_mode, inter_depth", FUSION_CONFIGS)
@pytest.mark.parametrize("view", ("global", "local"))
def test_backbone(dataset_treesat, fusion_mode, inter_depth, view):
    torch.manual_seed(0)
    backbone = small_backbone(dataset_treesat, fusion_mode, inter_depth)
    batch = make_batch(dataset_treesat)
    views = MultiCropAugment(n_local=1)(batch, backbone.view_size)
    inputs = view_inputs(batch, views[view][0])

    shapes = backbone.token_shapes(view)
    mask = ibot_masks(shapes, 2, 1.0, (0.3, 0.5), 4, 1.0, "cpu")
    keep = view_keep(
        {"aerial": 1, "s2": 16, "s1_asc": 4, "s1_des": 4}, 2, 0.3, 0.3, "cpu"
    )
    out = backbone(inputs, view, ibot_mask=mask, keep=keep)

    num_tokens = sum(s * h * w for s, h, w in shapes.values())
    assert out["cls"].shape == (2, 64)
    assert out["patch_flat"].shape == (2, num_tokens, 64)
    assert out["keep_flat"].shape == (2, num_tokens)
    for name_mod, (s, h, w) in shapes.items():
        assert out["patch"][name_mod].shape == (2, s, h * w, 64)
    assert torch.isfinite(out["cls"]).all()
    assert torch.isfinite(out["patch_flat"]).all()

    out["cls"].sum().backward()
    assert backbone.mask_token.grad is not None


@pytest.mark.parametrize("fusion_mode, inter_depth", FUSION_CONFIGS)
def test_backbone_dropped_tokens_ignored(dataset_treesat, fusion_mode, inter_depth):
    """Dropped (modality, date) have no influence on the CLS embedding."""
    torch.manual_seed(0)
    backbone = small_backbone(dataset_treesat, fusion_mode, inter_depth).eval()
    batch = make_batch(dataset_treesat, batch_size=1)
    keep = {
        "aerial": torch.ones(1, 1, dtype=torch.bool),
        "s2": torch.ones(1, 16, dtype=torch.bool),
        "s1_asc": torch.zeros(1, 4, dtype=torch.bool),
        "s1_des": torch.ones(1, 4, dtype=torch.bool),
    }
    keep["s2"][0, :8] = False
    out = backbone(batch, "global", keep=keep)

    batch_mod = {k: v.clone() for k, v in batch.items()}
    batch_mod["s1_asc"] = torch.rand_like(batch["s1_asc"])
    batch_mod["s2"][:, :8] = torch.rand_like(batch["s2"][:, :8])
    out_mod = backbone(batch_mod, "global", keep=keep)
    assert torch.allclose(out["cls"], out_mod["cls"], atol=1e-5)


@pytest.mark.parametrize(
    "ffn, norm",
    itertools.product(("mlp", "swiglu"), ("layernorm", "rmsnorm")),
)
def test_backbone_arch(dataset_treesat, ffn, norm):
    # ffn is fixed per model size in the Lightning module, so test it here
    backbone = small_backbone(
        dataset_treesat, "mod", 0, ffn=ffn, norm=norm, n_registers=0
    )
    batch = make_batch(dataset_treesat)
    views = MultiCropAugment(n_local=1)(batch, backbone.view_size)
    out = backbone(view_inputs(batch, views["global"][0]), "global")
    assert torch.isfinite(out["cls"]).all()


def test_backbone_invalid(dataset_treesat):
    with pytest.raises(ValueError):
        small_backbone(dataset_treesat, "msgfm", 0)
    with pytest.raises(NotImplementedError):
        small_backbone(dataset_treesat, "shared", 2)
    with pytest.raises(ValueError):
        small_backbone(dataset_treesat, "group", 4)


def test_backbone_wrong_grid(dataset_treesat):
    backbone = small_backbone(dataset_treesat, "group", 0)
    batch = make_batch(dataset_treesat)
    with pytest.raises(ValueError):
        backbone(batch, "local")


# ------------------------------------------------------------ augmentation
def coordinate_batch(sizes, batch_size=3):
    """Each modality holds its normalized x / y coordinates as two bands."""
    batch = {}
    for name_mod, size in sizes.items():
        c = (torch.arange(size, dtype=torch.float32) + 0.5) / size * 2 - 1
        yy, xx = torch.meshgrid(c, c, indexing="ij")
        batch[name_mod] = torch.stack([xx, yy])[None, None].repeat(
            batch_size, 2, 1, 1, 1
        )
    return batch


def test_crops_aligned_across_modalities():
    """The same box is applied to all modalities, whatever their size."""
    torch.manual_seed(0)
    augment = MultiCropAugment(n_local=2, jitter_prob=0.0, blur_probs=(0, 0, 0))
    in_sizes = {"high": 300, "low": 30}
    sizes = {"global": {"high": 150, "low": 6}, "local": {"high": 60, "low": 4}}
    views = augment(coordinate_batch(in_sizes), sizes)

    for view in views["global"] + views["local"]:
        high = F.adaptive_avg_pool2d(view["high"].flatten(1, 2), view["low"].shape[-1])
        low = view["low"].flatten(1, 2)
        # flips / transposition consistent, coordinates match up to resampling
        assert torch.allclose(high, low, atol=0.1)
        # dates are augmented consistently
        assert torch.equal(view["high"][:, 0], view["high"][:, 1])


def test_crop_scale():
    torch.manual_seed(0)
    augment = MultiCropAugment(flip=False, jitter_prob=0.0, blur_probs=(0, 0, 0))
    theta = augment.sample_boxes(1000, (0.05, 0.32), "cpu")
    area = theta[:, 0, 0] * theta[:, 1, 1]
    assert area.min() >= 0.05 - 1e-6 and area.max() <= 0.32 + 1e-6
    # boxes stay inside the image
    assert (theta[:, 0, 2].abs() + theta[:, 0, 0] <= 1 + 1e-6).all()
    assert (theta[:, 1, 2].abs() + theta[:, 1, 1] <= 1 + 1e-6).all()


def test_jitter_blur():
    torch.manual_seed(0)
    augment = MultiCropAugment(jitter_prob=1.0)
    x = torch.rand(4, 2, 3, 64, 64)
    assert augment.jitter(x).shape == x.shape
    blurred = augment.blur(x, prob=1.0)
    assert blurred.shape == x.shape
    assert blurred.std() < x.std()
    assert torch.equal(augment.blur(x, prob=0.0), x)


# ----------------------------------------------------------------- masking
def test_ibot_masks():
    torch.manual_seed(0)
    shapes = {"a": (1, 15, 15), "b": (16, 3, 3)}
    masks = ibot_masks(shapes, 64, 0.5, (0.1, 0.5), 4, 1.0, "cpu")
    assert masks["a"].shape == (64, 1, 225)
    assert masks["b"].shape == (64, 16, 9)
    ratio = masks["a"].float().mean(dim=(1, 2))
    assert ratio.max() <= 0.5 + 1e-6
    assert 0.2 < (ratio > 0).float().mean() < 0.8  # about half of the samples
    # fully shared noise: all dates of a modality get the same mask
    assert (masks["b"] == masks["b"][:, :1]).float().mean() > 0.95


def test_ibot_masks_aligned():
    """Shared masks are aligned in space across modalities."""
    torch.manual_seed(0)
    masks = ibot_masks(
        {"a": (1, 12, 12), "b": (1, 3, 3)}, 32, 1.0, (0.5, 0.5), 2, 1.0, "cpu"
    )
    a = masks["a"].float().reshape(32, 1, 12, 12)
    b = masks["b"].float().reshape(32, 1, 3, 3)
    a_low = F.adaptive_avg_pool2d(a, 3)
    agreement = ((a_low > 0.5).float() == b).float().mean()
    assert agreement > 0.8


def test_view_keep():
    torch.manual_seed(0)
    keep = view_keep({"a": 1, "b": 16}, 1000, 0.5, 0.5, "cpu")
    assert keep["a"].shape == (1000, 1) and keep["b"].shape == (1000, 16)
    any_kept = keep["a"].any(dim=1) | keep["b"].any(dim=1)
    assert any_kept.all()
    # 0.5 kept + 0.25 restored (both modalities dropped -> all kept)
    assert 0.7 < keep["a"].float().mean() < 0.8
    assert view_keep({"a": 1}, 4, None, None, "cpu") is None
    # single modality, single date: never dropped
    assert view_keep({"a": 1}, 100, 0.9, 0.9, "cpu")["a"].all()


# ---------------------------------------------------------- heads / losses
def test_dino_head():
    head = DINOHead(in_dim=32, n_prototypes=100, hidden_dim=64, bottleneck_dim=16)
    logits = head(torch.randn(8, 32))
    assert logits.shape == (8, 100)
    assert logits.abs().max() <= 1 + 1e-5


def test_sinkhorn():
    torch.manual_seed(0)
    logits = (
        F.normalize(torch.randn(256, 32), dim=-1)
        @ F.normalize(
            torch.randn(64, 32),
            dim=-1,
        ).t()
    )
    probs = sinkhorn_knopp(logits, temp=0.1, n_iter=20)
    assert torch.allclose(probs.sum(dim=1), torch.ones(256), atol=1e-4)
    # equipartition over prototypes
    assert torch.allclose(probs.mean(dim=0), torch.full((64,), 1 / 64), atol=1e-3)
    assert sinkhorn_knopp(logits[:0], temp=0.1).shape == (0, 64)


def test_dino_loss():
    torch.manual_seed(0)
    teacher = torch.softmax(torch.randn(2, 8, 16), dim=-1)
    student_global = torch.randn(2, 8, 16)
    student_local = torch.randn(3, 8, 16)
    losses = dino_loss(student_global, student_local, teacher, student_temp=1.0)
    n_global_pairs, n_local_pairs = 2, 6
    expected = (n_global_pairs * losses["global"] + n_local_pairs * losses["local"]) / (
        n_global_pairs + n_local_pairs
    )
    assert torch.allclose(losses["total"], expected)

    # minimum = entropy of targets, reached when student matches the other view
    teacher = torch.softmax(torch.randn(1, 8, 16), dim=-1).repeat(2, 1, 1)
    losses = dino_loss(teacher.log(), None, teacher, student_temp=1.0)
    entropy = -(teacher[0] * teacher[0].log()).sum(-1).mean()
    assert torch.allclose(losses["global"], entropy, atol=1e-5)
    assert losses["local"] == 0


def test_ibot_loss():
    torch.manual_seed(0)
    teacher = torch.softmax(torch.randn(5, 16), dim=-1)
    student = torch.randn(5, 16, requires_grad=True)
    num_masked = torch.tensor([2, 0, 3])
    loss = ibot_loss(student, teacher, num_masked, student_temp=1.0)
    ce = -(teacher * torch.log_softmax(student, -1)).sum(-1)
    expected = (ce[:2].mean() + ce[2:].mean()) / 2
    assert torch.allclose(loss, expected)
    loss.backward()
    empty = ibot_loss(student[:0], teacher[:0], torch.tensor([0, 0]), 1.0)
    assert empty == 0


def test_koleo_loss():
    spread = torch.eye(8)
    collapsed = torch.ones(8, 8) + 1e-3 * torch.randn(8, 8)
    assert koleo_loss(spread) < koleo_loss(collapsed)
    assert koleo_loss(torch.randn(1, 8)) == 0


def test_teacher_stats():
    uniform = torch.full((4, 10), 0.1)
    stats = teacher_stats(uniform)
    assert torch.allclose(stats["entropy"], torch.tensor(math.log(10)))
    assert torch.allclose(stats["kl_uniform"], torch.tensor(0.0), atol=1e-6)
    onehot = F.one_hot(torch.zeros(4, dtype=torch.long), 10).float()
    assert teacher_stats(onehot)["kl_uniform"] > 2


def test_schedules():
    assert cosine(1.0, 0.0, 0, 100) == 1.0
    assert cosine(1.0, 0.0, 100, 100) == pytest.approx(0.0)
    assert cosine(1.0, 0.0, 50, 100) == pytest.approx(0.5)
    assert linear_warmup(0.04, 0.07, 0, 10) == 0.04
    assert linear_warmup(0.04, 0.07, 20, 10) == 0.07
    assert lr_factor(0, 100, 10, "cosine", 0.0) == pytest.approx(0.1)
    assert lr_factor(9, 100, 10, "cosine", 0.0) == pytest.approx(1.0)
    assert lr_factor(100, 100, 10, "cosine", 0.0) == pytest.approx(0.0)
    assert lr_factor(60, 100, 10, "constant", 0.0) == 1.0


# ------------------------------------------------------------------ module
def tiny_module(dataset, fusion_mode="group", inter_depth=3, **kwargs):
    args = {
        "dataset": dataset,
        "mask": DINOv3MaskConfig(),
        "loss": DINOv3LossConfig(
            dino_n_prototypes=128,
            ibot_n_prototypes=128,
            head_hidden_dim=64,
            head_bottleneck_dim=32,
        ),
        "fusion_mode": fusion_mode,
        "inter_depth": inter_depth,
        "augment": DinoAugmentConfig(n_local_crops=2),
        "model_size": "tiny",
    }
    args.update(kwargs)
    return LitDINOv3(**args)


@pytest.mark.parametrize(
    "fusion_mode, inter_depth",
    [("shared", 0), ("monotemp", 0), ("mod", 3), ("group", 0), ("group", 3)],
)
def test_dino_module_losses(dataset_treesat, fusion_mode, inter_depth):
    torch.manual_seed(0)
    module = tiny_module(dataset_treesat, fusion_mode, inter_depth)
    module.train()
    assert not module.teacher.training
    assert all(not p.requires_grad for p in module.teacher.parameters())

    batch = make_batch(dataset_treesat, batch_size=4, image_size=32)
    out = module.forward_batch(batch, teacher_temp=0.04)
    assert torch.isfinite(out["loss"])
    for name, value in out["components"].items():
        assert torch.isfinite(value), name
    out["loss"].backward()

    student = module.student
    assert student["dino_head"].prototypes.grad is not None
    assert student["ibot_head"].prototypes.grad is not None
    assert student["backbone"].mask_token.grad is not None
    assert all(p.grad is None for p in module.teacher.parameters())


def test_dino_module_teacher_update(dataset_treesat):
    module = tiny_module(dataset_treesat, "shared", 0)
    with torch.no_grad():
        for param in module.student.parameters():
            param.add_(1.0)
    student = [p.clone() for p in module.student.parameters()]
    teacher = [p.clone() for p in module.teacher.parameters()]
    module.update_teacher(0.9)
    for p_s, p_t, p_new in zip(student, teacher, module.teacher.parameters()):
        assert torch.allclose(p_new, 0.9 * p_t + 0.1 * p_s)


def test_dino_module_param_groups(dataset_treesat):
    module = tiny_module(dataset_treesat, "group", 3)
    names_no_wd = [
        name
        for name, p in module.student.named_parameters()
        if p.ndim <= 1 or any(k in name for k in ("cls_token", "register_tokens"))
    ]
    assert any("cls_token" in name for name in names_no_wd)
    assert any("ls1.gamma" in name for name in names_no_wd)


@pytest.mark.parametrize("norm", ("layernorm", "rmsnorm"))
def test_dino_module_arch(dataset_treesat, norm):
    module = tiny_module(dataset_treesat, "mod", 0, norm=norm, n_registers=0)
    batch = make_batch(dataset_treesat, batch_size=2, image_size=32)
    out = module.forward_batch(batch, teacher_temp=0.04)
    assert torch.isfinite(out["loss"])


def _sinkhorn_worker(rank, world_size, port, logits, splits, queue):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    local = logits.split(splits)[rank]
    queue.put((rank, sinkhorn_knopp(local, temp=0.1).numpy()))
    dist.destroy_process_group()


def test_sinkhorn_distributed():
    """Sinkhorn over 2 processes with unequal sizes == single process."""
    import socket

    import torch.multiprocessing as mp

    torch.manual_seed(0)
    logits = (
        F.normalize(torch.randn(50, 16), dim=-1)
        @ F.normalize(
            torch.randn(32, 16),
            dim=-1,
        ).t()
    )
    splits = [30, 20]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_sinkhorn_worker,
            args=(rank, 2, port, logits, splits, queue),
        )
        for rank in range(2)
    ]
    for proc in procs:
        proc.start()
    results = dict(queue.get(timeout=60) for _ in procs)
    for proc in procs:
        proc.join(timeout=60)

    expected = sinkhorn_knopp(logits, temp=0.1)
    result = torch.cat([torch.from_numpy(results[0]), torch.from_numpy(results[1])])
    assert torch.allclose(result, expected, atol=1e-6)
