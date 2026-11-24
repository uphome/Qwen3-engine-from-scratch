"""Qwen3Config：字段推导与本地加载"""

import json

from qwen3 import Qwen3Config


CONFIG_JSON = {
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_hidden_layers": 28,
    "vocab_size": 151936,
    "intermediate_size": 3072,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "max_position_embeddings": 40960,
    "tie_word_embeddings": False,
}


def test_from_pretrained_local(tmp_path):
    """本地目录读取 config.json"""
    (tmp_path / "config.json").write_text(json.dumps(CONFIG_JSON))
    cfg = Qwen3Config.from_pretrained(str(tmp_path))

    assert cfg.hidden_size == 1024
    assert cfg.num_attention_heads == 16
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 128
    assert cfg.num_hidden_layers == 28
    assert cfg.vocab_size == 151936
    assert cfg.rope_theta == 1_000_000.0
    assert cfg.max_position_embeddings == 40960


def test_head_dim_fallback(tmp_path):
    """config.json 无 head_dim 时用 hidden_size / num_attention_heads 推导"""
    data = dict(CONFIG_JSON)
    data.pop("head_dim")
    (tmp_path / "config.json").write_text(json.dumps(data))

    cfg = Qwen3Config.from_pretrained(str(tmp_path))
    assert cfg.head_dim == 1024 // 16


def test_max_position_default(tmp_path):
    """max_position_embeddings 缺省为 4096"""
    data = dict(CONFIG_JSON)
    data.pop("max_position_embeddings")
    (tmp_path / "config.json").write_text(json.dumps(data))

    cfg = Qwen3Config.from_pretrained(str(tmp_path))
    assert cfg.max_position_embeddings == 4096


def test_tie_word_embeddings_default(tmp_path):
    data = dict(CONFIG_JSON)
    data.pop("tie_word_embeddings")
    (tmp_path / "config.json").write_text(json.dumps(data))

    cfg = Qwen3Config.from_pretrained(str(tmp_path))
    assert cfg.tie_word_embeddings is False
