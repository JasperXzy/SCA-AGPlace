import torch
from types import SimpleNamespace

from network_mm.fuse_block_toshallow import FuseBlockToShallow
from network_mm.ode_cq import DeltaQ
from network_mm.vlaq import VLAQ


def test_delta_q_alpha_zero_outputs_zero_but_keeps_alpha_gradient():
    torch.manual_seed(0)
    delta_q = DeltaQ(C=8, S=4, D=6, r=3, alpha_init=0.0, alpha_learn=True)
    e_fuse = torch.randn(2, 8)
    q_bias = delta_q(e_fuse)

    assert q_bias.shape == (2, 4, 6)
    assert torch.allclose(q_bias, torch.zeros_like(q_bias))

    target = torch.randn_like(q_bias)
    loss = (q_bias * target).sum()
    loss.backward()
    assert delta_q.alpha.grad is not None
    assert delta_q.alpha.grad.abs() > 0


def test_vlaq_zero_q_bias_matches_static_queries():
    torch.manual_seed(0)
    vlaq = VLAQ(
        n_queries=4,
        query_dim=8,
        token_dim=8,
        out_dim=16,
        dropout=0.0,
        q_init="orthogonal",
    )
    tokens = torch.randn(2, 5, 8)
    q_bias = torch.zeros(2, 4, 8)

    static_out = vlaq(tokens, q_bias=None)
    biased_out = vlaq(tokens, q_bias=q_bias)

    assert torch.allclose(static_out, biased_out, atol=1e-6)


def test_fuse_summary_accepts_charted_2d_tokens():
    block = object.__new__(FuseBlockToShallow)
    block.args = SimpleNamespace(fuse_summary_mode="mean")
    tokens = torch.arange(24, dtype=torch.float32).view(2, 3, 4)

    summary = FuseBlockToShallow.per_scale_summary(block, tokens, "2d", 0)

    assert summary.shape == (2, 4)
    assert torch.allclose(summary, tokens.mean(dim=1))
