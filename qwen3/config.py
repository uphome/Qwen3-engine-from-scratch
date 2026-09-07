"""
Qwen3 模型配置 — 纯数据类，不负责 I/O

用法:
    # 从本地目录或 HF repo 读取
    config = Qwen3Config.from_pretrained("Qwen/Qwen3-0.6B")

    # 手动构建（方便测试小模型）
    config = Qwen3Config(
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_hidden_layers=4,
        vocab_size=32000,
        intermediate_size=1024,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
    )
"""

import json
import os
from dataclasses import dataclass


@dataclass
class Qwen3Config:
    """Qwen3 模型超参数（纯数据，不绑定 I/O）"""

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int = 4096
    tie_word_embeddings: bool = False

    @classmethod
    def from_pretrained(cls, model_path: str) -> "Qwen3Config":
        """
        从本地目录或 HuggingFace repo 读取 config.json

        Args:
            model_path: 本地路径或 HF repo id（如 "Qwen/Qwen3-0.6B"）
        """
        if os.path.isdir(model_path):
            config_file = os.path.join(model_path, "config.json")
        else:
            from huggingface_hub import hf_hub_download

            config_file = hf_hub_download(repo_id=model_path, filename="config.json")

        with open(config_file) as f:
            data = json.load(f)

        return cls(
            hidden_size=data["hidden_size"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_key_value_heads"],
            head_dim=data.get("head_dim", data["hidden_size"] // data["num_attention_heads"]),
            num_hidden_layers=data["num_hidden_layers"],
            vocab_size=data["vocab_size"],
            intermediate_size=data["intermediate_size"],
            rms_norm_eps=data["rms_norm_eps"],
            rope_theta=data["rope_theta"],
            max_position_embeddings=data.get("max_position_embeddings", 4096),
            tie_word_embeddings=data.get("tie_word_embeddings", False),
        )
