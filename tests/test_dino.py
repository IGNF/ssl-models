import itertools
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ssl_models.conf.augment import DINOv3AugmentConfig
from ssl_models.conf.dataset.treesatai_ts import TreeSatAITSConfig
from ssl_models.conf.loss import DINOv3LossConfig
from ssl_models.conf.mask import DINOv3MaskConfig
from ssl_models.layers.head import DINOHead
from ssl_models.layers.loss import dino_loss, ibot_loss, koleo_loss, sinkhorn_knopp
from ssl_models.ssl import dino as dino_module
from ssl_models.ssl.dino import (
    DINOv3,
    dinov3_base,
    dinov3_huge_plus,
    dinov3_large,
    dinov3_small,
    dinov3_tiny,
)
from ssl_models.train.augment import Augment, local_view_size
from ssl_models.train.lit_dino import LitDINOv3, teacher_stats
from ssl_models.train.schedule import cosine, linear_warmup, lr_factor

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


def make_views(batch, dataset):
    """One random global view and one random local view per sample."""
    config = DINOv3AugmentConfig(num_global=1, num_local=1)
    return Augment(dataset, **vars(config))(batch)


def make_tokens(model, batch_size):
    """Zero tokens [B, D, L, 1] of a global view, grouped."""
    return model.group(
        {
            name_mod: torch.zeros(
                batch_size, model.num_dates[name_mod], grid_size**2, 1
            )
            for name_mod, grid_size in model.grid_size.items()
        },
    )


def small_model(dataset, fusion_mode, inter_depth, mask=None, **kwargs):
    return DINOv3(
        dataset=dataset,
        mask=mask or DINOv3MaskConfig(),
        fusion_mode=fusion_mode,
        inter_depth=inter_depth,
        embed_dim=64,
        depth=3,
        heads=2,
        dino_prototypes=32,
        ibot_prototypes=32,
        head_hidden_dim=64,
        head_bottleneck_dim=32,
        **kwargs,
    )


# ------------------------------------------------------------------- model
def test_local_view_size():
    assert local_view_size(300, 20, 112 / 256) == 140
    assert local_view_size(6, 2, 112 / 256) == 4  # at least 2 patches
    assert local_view_size(2, 2, 112 / 256) == 2  # at most image size


@pytest.mark.parametrize(
    "size_fn",
    (dinov3_tiny, dinov3_small, dinov3_base, dinov3_large, dinov3_huge_plus),
)
def test_model_sizes(monkeypatch, size_fn):
    # capture the size arguments without building the model
    monkeypatch.setattr(dino_module, "DINOv3", lambda **kw: kw)
    args = size_fn()
    assert args["embed_dim"] % args["heads"] == 0
    assert (args["embed_dim"] // args["heads"]) % 4 == 0  # RoPE head dim
    assert size_fn(ffn="swiglu", mlp_ratio=3.0)["ffn"] == "swiglu"


@pytest.mark.parametrize("fusion_mode, inter_depth", FUSION_CONFIGS)
@pytest.mark.parametrize("view", ("global", "local"))
def test_model(dataset_treesat, fusion_mode, inter_depth, view):
    torch.manual_seed(0)
    mask = DINOv3MaskConfig(ibot_mask_prob=1.0)
    model = small_model(dataset_treesat, fusion_mode, inter_depth, mask=mask)
    batch = make_views(make_batch(dataset_treesat), dataset_treesat)
    out = model(batch, ssl_phase="pretrain", view=view)

    assert out["cls"].shape == (2, 64)
    assert out["dino"].shape == (2, 32)
    assert torch.isfinite(out["cls"]).all()
    assert torch.isfinite(out["dino"]).all()
    if view == "global":
        num_masked = sum(int(m.sum()) for m in out["mask_rec"].values())
        assert out["ibot"].shape == (num_masked, 32)
        assert out["ibot_weights"].shape == (num_masked,)
        assert torch.isfinite(out["ibot"]).all()
    else:
        assert out["ibot"] is None and out["mask_rec"] is None

    out["dino"].sum().backward()
    assert (model.enc_mask_token.grad is not None) == (view == "global")


@pytest.mark.parametrize("fusion_mode, inter_depth", FUSION_CONFIGS)
def test_model_dropped_tokens_ignored(
    monkeypatch, dataset_treesat, fusion_mode, inter_depth
):
    """Dropped (modality, date) have no influence on the image-level embedding."""
    torch.manual_seed(0)
    mask = DINOv3MaskConfig(ibot_mask_prob=0.0)
    model = small_model(dataset_treesat, fusion_mode, inter_depth, mask=mask).eval()
    keep = {
        name_mod: torch.ones(1, mod.num_dates, dtype=torch.bool)
        for name_mod, mod in dataset_treesat.inputs.items()
    }
    keep["s1_asc"][:] = False
    keep["s2"][:, :8] = False
    keep = model.group(
        {
            name_mod: keep[name_mod][:, :, None, None].expand(-1, -1, grid_size**2, -1)
            for name_mod, grid_size in model.grid_size.items()
        },
    )
    monkeypatch.setattr(model, "keep_view", lambda x, **kwargs: keep)

    batch = make_batch(dataset_treesat, batch_size=1)
    out = model(batch, ssl_phase="pretrain")

    batch_mod = {k: v.clone() for k, v in batch.items()}
    batch_mod["s1_asc"] = torch.rand_like(batch["s1_asc"])
    batch_mod["s2"][:, :8] = torch.rand_like(batch["s2"][:, :8])
    out_mod = model(batch_mod, ssl_phase="pretrain")
    assert torch.allclose(out["cls"], out_mod["cls"], atol=1e-5)


@pytest.mark.parametrize(
    "ffn, norm",
    itertools.product(("mlp", "swiglu"), ("layernorm", "rmsnorm")),
)
def test_model_arch(dataset_treesat, ffn, norm):
    # ffn is fixed per model size in the Lightning module, so test it here
    model = small_model(
        dataset_treesat, "mod", 0, ffn=ffn, norm=norm, num_registers=0
    )
    batch = make_views(make_batch(dataset_treesat), dataset_treesat)
    out = model(batch, ssl_phase="pretrain")
    assert torch.isfinite(out["cls"]).all()


def test_model_invalid(dataset_treesat):
    with pytest.raises(ValueError):
        small_model(dataset_treesat, "msgfm", 0)
    with pytest.raises(NotImplementedError):
        small_model(dataset_treesat, "shared", 2)
    with pytest.raises(ValueError):
        small_model(dataset_treesat, "group", 4)


def test_model_wrong_size(dataset_treesat):
    """Images that have neither the global nor the local view sizes."""
    model = small_model(dataset_treesat, "group", 0)
    batch = make_batch(dataset_treesat, image_size=60)
    with pytest.raises(ValueError):
        model(batch, ssl_phase="pretrain")


# ----------------------------------------------------------------- masking
def test_ibot_masks(dataset_treesat):
    torch.manual_seed(0)
    mask = DINOv3MaskConfig(ibot_mask_prob=0.5, ibot_mask_ratio=[0.1, 0.5])
    model = small_model(dataset_treesat, "group", 0, mask=mask)
    masks = model.ungroup(model.ibot_masks(make_tokens(model, 64)))
    assert masks["aerial"].shape == (64, 1, 225, 1)
    assert masks["s2"].shape == (64, 16, 9, 1)
    ratio = masks["aerial"].float().mean(dim=(1, 2, 3))
    assert ratio.max() <= 0.5 + 1e-6
    assert 0.2 < (ratio > 0).float().mean() < 0.8  # about half of the samples
    # fully shared noise: all dates of a modality get the same mask
    assert (masks["s2"] == masks["s2"][:, :1]).float().mean() > 0.95


def test_ibot_masks_aligned(dataset_treesat):
    """Shared masks are aligned in space across modalities."""
    torch.manual_seed(0)
    mask = DINOv3MaskConfig(
        ibot_mask_prob=1.0, ibot_mask_ratio=[0.5, 0.5], ibot_blob_grid=2
    )
    model = small_model(dataset_treesat, "group", 0, mask=mask)
    masks = model.ungroup(model.ibot_masks(make_tokens(model, 32)))
    aerial = masks["aerial"].float().reshape(32, 1, 15, 15)
    s2 = masks["s2"][:, :1].float().reshape(32, 1, 3, 3)
    aerial_low = F.adaptive_avg_pool2d(aerial, 3)
    agreement = ((aerial_low > 0.5).float() == s2).float().mean()
    assert agreement > 0.8


def test_keep_view(dataset_treesat):
    torch.manual_seed(0)
    mask = DINOv3MaskConfig(view_drop_mod=0.5, view_drop_dates=0.5)
    model = small_model(dataset_treesat, "group", 0, mask=mask)
    x = make_tokens(model, 1000)
    keep = model.ungroup(model.keep_view(x, ssl_phase="pretrain"))
    assert keep["aerial"].shape == (1000, 1, 225, 1)
    assert keep["s2"].shape == (1000, 16, 9, 1)
    # whole (modality, date) are kept or dropped, and at least one is kept
    assert (keep["s2"] == keep["s2"][:, :, :1]).all()
    any_kept = torch.stack([k.flatten(1).any(dim=1) for k in keep.values()])
    assert any_kept.any(dim=0).all()
    # 0.5 kept + about 0.07 restored (all modalities dropped -> all kept)
    assert 0.5 < keep["aerial"].float().mean() < 0.65

    # only for the student in pretraining
    assert model.keep_view(x, ssl_phase="pretrain", teacher=True) is None
    assert model.keep_view(x, ssl_phase="probe") is None
    no_drop = DINOv3MaskConfig(view_drop_mod=None, view_drop_dates=None)
    model = small_model(dataset_treesat, "group", 0, mask=no_drop)
    assert model.keep_view(x, ssl_phase="pretrain") is None
    # single modality, single date: never dropped
    dataset = TreeSatAITSConfig(rel_dir="", filter_inputs=["aerial"])
    mask = DINOv3MaskConfig(view_drop_mod=0.9, view_drop_dates=0.9)
    model = small_model(dataset, "group", 0, mask=mask)
    keep = model.keep_view(make_tokens(model, 100), ssl_phase="pretrain")
    assert keep["aerial"].all()


# ---------------------------------------------------------- heads / losses
def test_dino_head():
    head = DINOHead(in_dim=32, num_prototypes=100, hidden_dim=64, bottleneck_dim=16)
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
    probs = sinkhorn_knopp(logits, temp=0.1, num_iter=20)
    assert torch.allclose(probs.sum(dim=1), torch.ones(256), atol=1e-4)
    # equipartition over prototypes
    assert torch.allclose(probs.mean(dim=0), torch.full((64,), 1 / 64), atol=1e-3)
    assert sinkhorn_knopp(logits[:0], temp=0.1).shape == (0, 64)


def test_dino_loss():
    torch.manual_seed(0)
    teacher = torch.softmax(torch.randn(8 * 2, 16), dim=-1)
    student_global = torch.randn(8 * 2, 16)
    student_local = torch.randn(8 * 3, 16)
    loss, loss_global, loss_local = dino_loss(
        student_global,
        student_local,
        teacher,
        student_temp=1.0,
        num_global=2,
        num_local=3,
    )
    num_global_pairs, num_local_pairs = 2, 6
    expected = (num_global_pairs * loss_global + num_local_pairs * loss_local) / (
        num_global_pairs + num_local_pairs
    )
    assert torch.allclose(loss, expected)

    # minimum = entropy of targets, reached when student matches the other view
    teacher = torch.softmax(torch.randn(8, 16), dim=-1).repeat_interleave(2, dim=0)
    _, loss_global, loss_local = dino_loss(
        teacher.log(), None, teacher, student_temp=1.0, num_global=2, num_local=0
    )
    entropy = -(teacher * teacher.log()).sum(-1).mean()
    assert torch.allclose(loss_global, entropy, atol=1e-5)
    assert loss_local == 0


def test_ibot_loss():
    torch.manual_seed(0)
    teacher = torch.softmax(torch.randn(5, 16), dim=-1)
    student = torch.randn(5, 16, requires_grad=True)
    weights = torch.tensor([1 / 2] * 2 + [1 / 3] * 3)  # 2 and 3 masked tokens
    loss = ibot_loss(student, weights, teacher, student_temp=1.0)
    ce = -(teacher * torch.log_softmax(student, -1)).sum(-1)
    expected = (ce[:2].mean() + ce[2:].mean()) / 2
    assert torch.allclose(loss, expected)
    loss.backward()
    empty = ibot_loss(student[:0], weights[:0], teacher[:0], student_temp=1.0)
    assert empty == 0


def test_koleo_loss():
    spread = torch.eye(8)
    collapsed = torch.ones(8, 8) + 1e-3 * torch.randn(8, 8)
    loss = koleo_loss(spread, num_global=1)
    assert loss < koleo_loss(collapsed, num_global=1)
    assert koleo_loss(torch.randn(1, 8), num_global=1) == 0
    # the neighbors are within each global crop
    views = spread.repeat_interleave(2, dim=0)
    assert torch.allclose(koleo_loss(views, num_global=2), loss)


def test_teacher_stats():
    uniform = torch.full((4, 10), 0.1)
    stats = teacher_stats(uniform)
    assert torch.allclose(stats["teacher_entropy"], torch.tensor(math.log(10)))
    assert torch.allclose(stats["teacher_kl_uniform"], torch.tensor(0.0), atol=1e-6)
    onehot = F.one_hot(torch.zeros(4, dtype=torch.long), 10).float()
    assert teacher_stats(onehot)["teacher_kl_uniform"] > 2


def test_schedules():
    assert cosine(1.0, 0.0, 0, 100) == 1.0
    assert cosine(1.0, 0.0, 100, 100) == pytest.approx(0.0)
    assert cosine(1.0, 0.0, 50, 100) == pytest.approx(0.5)
    assert linear_warmup(0.04, 0.07, 0, 10) == 0.04
    assert linear_warmup(0.04, 0.07, 20, 10) == 0.07
    assert lr_factor(0, 100, 0.1, "cosine", 0.0) == pytest.approx(0.1)
    assert lr_factor(9, 100, 0.1, "cosine", 0.0) == pytest.approx(1.0)
    assert lr_factor(100, 100, 0.1, "cosine", 0.0) == pytest.approx(0.0)
    assert lr_factor(60, 100, 0.1, "constant", 0.0) == 1.0


# ------------------------------------------------------------------ module
def tiny_module(dataset, fusion_mode="group", inter_depth=3, **kwargs):
    args = {
        "dataset": dataset,
        "mask": DINOv3MaskConfig(),
        "loss": DINOv3LossConfig(
            dino_prototypes=128,
            ibot_prototypes=128,
            head_hidden_dim=64,
            head_bottleneck_dim=32,
        ),
        "fusion_mode": fusion_mode,
        "inter_depth": inter_depth,
        "augment": DINOv3AugmentConfig(num_local=2),
        "model_size": "tiny",
    }
    args.update(kwargs)
    module = LitDINOv3(**args)
    module._trainer = SimpleNamespace(ssl_phase="pretrain")  # no actual trainer
    return module


@pytest.mark.parametrize(
    "fusion_mode, inter_depth",
    [("shared", 0), ("monotemp", 0), ("mod", 3), ("group", 0), ("group", 3)],
)
def test_dino_module_losses(dataset_treesat, fusion_mode, inter_depth):
    torch.manual_seed(0)
    module = tiny_module(dataset_treesat, fusion_mode, inter_depth)
    module.train()
    assert not module.ema_model.training
    assert all(not p.requires_grad for p in module.ema_model.parameters())

    batch = make_batch(dataset_treesat, batch_size=4, image_size=32)
    out = module.pretrain_step(module.augment(batch), stage="train")
    for name, value in out.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value), name
    out["loss"].backward()

    student = module.model
    assert student.dino_head.prototypes.grad is not None
    assert student.ibot_head.prototypes.grad is not None
    assert student.enc_mask_token.grad is not None
    assert all(p.grad is None for p in module.ema_model.parameters())


def test_dino_module_ema_update(monkeypatch, dataset_treesat):
    module = tiny_module(dataset_treesat, "shared", 0)
    monkeypatch.setattr(module, "log", lambda *args: None)
    monkeypatch.setattr(module, "momentum", lambda: 0.9)
    with torch.no_grad():
        for param in module.model.parameters():
            param.add_(1.0)
    student = [p.clone() for p in module.model.parameters()]
    teacher = [p.clone() for p in module.ema_model.parameters()]
    module.update_ema()
    for p_s, p_t, p_new in zip(student, teacher, module.ema_model.parameters()):
        assert torch.allclose(p_new, 0.9 * p_t + 0.1 * p_s)


def test_dino_module_param_groups(dataset_treesat):
    module = tiny_module(dataset_treesat, "group", 3)
    names_no_wd = [
        name
        for name, p in module.model.named_parameters()
        if p.ndim <= 1 or any(k in name for k in ("cls_token", "register_tokens"))
    ]
    assert any("cls_token" in name for name in names_no_wd)
    assert any("ls1.gamma" in name for name in names_no_wd)


@pytest.mark.parametrize("norm", ("layernorm", "rmsnorm"))
def test_dino_module_arch(dataset_treesat, norm):
    module = tiny_module(dataset_treesat, "mod", 0, norm=norm, num_registers=0)
    batch = make_batch(dataset_treesat, batch_size=2, image_size=32)
    out = module.pretrain_step(module.augment(batch), stage="train")
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
