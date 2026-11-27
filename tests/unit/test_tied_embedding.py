"""tied word embedding（Qwen3-4B 等）：lm_head 与 embed_tokens 共享权重"""

import torch

from qwen3 import Qwen3Config, Qwen3ForCausalLM


def tiny_cfg(tie: bool) -> Qwen3Config:
    return Qwen3Config(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, num_hidden_layers=2, vocab_size=128,
        intermediate_size=256, rms_norm_eps=1e-6, rope_theta=1e6,
        tie_word_embeddings=tie,
    )


def test_tied_lm_head_shares_parameter():
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(tiny_cfg(tie=True))
    assert model.lm_head.weight.data_ptr() == \
        model.model.embed_tokens.weight.data_ptr()
    assert model.lm_head.weight is model.model.embed_tokens.weight

    # 前向可用，且 logits = hidden @ embed^T
    ids = torch.arange(6).unsqueeze(0)
    logits = model(ids, mode="prefill")
    assert logits.shape == (1, 6, 128)
    assert torch.isfinite(logits).all()


def test_untied_lm_head_is_independent():
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(tiny_cfg(tie=False))
    assert model.lm_head.weight.data_ptr() != \
        model.model.embed_tokens.weight.data_ptr()


def test_tied_forward_matches_manual_matmul(monkeypatch=None):
    """logits 的最后一维必须等于 hidden @ embed_tokens.weight.T"""
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(tiny_cfg(tie=True)).eval()
    ids = torch.arange(5).unsqueeze(0)
    with torch.no_grad():
        logits = model(ids, mode="prefill")
        # 用 embed 权重手工算 lm_head
        hidden = model.model(ids, mode="prefill")
        manual = torch.nn.functional.linear(
            hidden, model.model.embed_tokens.weight)
    assert torch.allclose(logits, manual, atol=1e-6)
