import itertools

import pytest
import torch

from ssl_models.layers.vit import (
    Attention,
    DropPath,
    RopePositionEmbedding,
    SwiGLUFFN,
    Transformer,
    apply_rope,
    grid_coords,
    pad_rope,
)


def test_grid_coords():
    coords = grid_coords(4, 2)
    assert coords.shape == (8, 2)
    assert coords.min() > -1 and coords.max() < 1
    # row-major: second token is same row, next column
    assert coords[0, 0] == coords[1, 0]
    assert coords[0, 1] < coords[1, 1]
    # grids of different resolution covering the same extent share their center
    assert torch.allclose(grid_coords(3, 3)[4], grid_coords(1, 1)[0])


def test_rope_shapes():
    rope = RopePositionEmbedding(head_dim=64).eval()
    sin, cos = rope(grid_coords(5, 5))
    assert sin.shape == cos.shape == (25, 64)
    sin, cos = rope(grid_coords(5, 5).expand(3, -1, -1))
    assert sin.shape == (3, 25, 64)


def test_rope_relative():
    """q.k after RoPE only depends on the coordinate difference."""
    torch.manual_seed(0)
    rope = RopePositionEmbedding(head_dim=32).eval()
    q, k = torch.randn(1, 1, 1, 32), torch.randn(1, 1, 1, 32)
    p, r = torch.tensor([[0.3, -0.2]]), torch.tensor([[-0.5, 0.7]])
    delta = torch.tensor([[0.4, 0.1]])

    def dot(p, r):
        return (apply_rope(q, rope(p)) * apply_rope(k, rope(r))).sum()

    assert torch.allclose(dot(p, r), dot(p + delta, r + delta), atol=1e-5)
    assert not torch.allclose(dot(p, r), dot(p, r + delta), atol=1e-3)


def test_rope_augment_train_only():
    rope = RopePositionEmbedding(
        head_dim=32,
        shift_coords=0.5,
        jitter_coords=2.0,
        rescale_coords=2.0,
    )
    coords = grid_coords(4, 4)
    rope.eval()
    assert torch.equal(rope(coords)[0], rope(coords)[0])
    rope.train()
    assert not torch.equal(rope(coords)[0], rope(coords)[0])


def test_rope_invalid_head_dim():
    with pytest.raises(ValueError):
        RopePositionEmbedding(head_dim=30)


def test_pad_rope_identity():
    rope = RopePositionEmbedding(head_dim=16).eval()
    padded = pad_rope(rope(grid_coords(2, 2)), num_prefix=3)
    assert padded[0].shape == (7, 16)
    x = torch.randn(2, 1, 7, 16)
    out = apply_rope(x, padded)
    assert torch.allclose(out[:, :, :3], x[:, :, :3])
    assert not torch.allclose(out[:, :, 3:], x[:, :, 3:])


def test_attention_mask():
    """Masked tokens have no influence on kept tokens."""
    torch.manual_seed(0)
    attn = Attention(dim=32, heads=4).eval()
    x = torch.randn(2, 6, 32)
    mask = torch.tensor([[True] * 4 + [False] * 2, [True] * 6])
    out = attn(x, attn_mask=mask)
    x_mod = x.clone()
    x_mod[0, 4:] = torch.randn(2, 32)
    out_mod = attn(x_mod, attn_mask=mask)
    assert torch.allclose(out[0, :4], out_mod[0, :4], atol=1e-6)
    assert torch.allclose(out[1], out_mod[1], atol=1e-6)


def test_swiglu_hidden_dim():
    ffn = SwiGLUFFN(dim=64, hidden_dim=256)
    assert ffn.w3.in_features % 8 == 0
    assert ffn(torch.randn(2, 3, 64)).shape == (2, 3, 64)


def test_drop_path():
    drop = DropPath(0.5)
    x = torch.ones(1000, 4)
    drop.eval()
    assert torch.equal(drop(x), x)
    drop.train()
    out = drop(x)
    assert set(out.unique().tolist()) <= {0.0, 2.0}
    assert 0.3 < (out == 0).float().mean() < 0.7


@pytest.mark.parametrize(
    "ffn, norm, layerscale_init",
    itertools.product(("mlp", "swiglu"), ("layernorm", "rmsnorm"), (1e-5, None)),
)
def test_transformer(ffn, norm, layerscale_init):
    torch.manual_seed(0)
    dim, heads, num_prefix = 64, 4, 5
    model = Transformer(
        dim=dim,
        depth=2,
        heads=heads,
        ffn=ffn,
        norm=norm,
        layerscale_init=layerscale_init,
        drop_path=0.1,
    )
    rope_embed = RopePositionEmbedding(head_dim=dim // heads)
    rope = pad_rope(rope_embed(grid_coords(3, 3)), num_prefix)
    x = torch.randn(2, num_prefix + 9, dim, requires_grad=True)
    mask = torch.ones(2, num_prefix + 9, dtype=torch.bool)
    mask[0, -3:] = False
    out = model(x, rope, mask)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
