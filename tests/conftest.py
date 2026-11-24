"""pytest 公共配置与 fixture

分层约定：
  - 默认（无 marker）: CPU 可跑，进 CI
  - @pytest.mark.gpu:  需要 CUDA GPU（无 GPU 时自动 skip）
  - @pytest.mark.slow: 模型加载 / 长耗时（大模型、捕获 CUDA Graph 等）
"""

import os
import sys

import pytest
import torch

# 保证从仓库根目录导入 qwen3（pytest rootdir 不自动进 sys.path）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen3 import Qwen3Config, Qwen3ForCausalLM, KVCachePool


def pytest_collection_modifyitems(config, items):
    """无 GPU 环境自动跳过 gpu 标记的用例（CI 友好）"""
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="需要 CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def gpu_device() -> str:
    """选择显存最空闲的 GPU（多卡共享环境），并设为当前设备"""
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA GPU")
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    best = free.index(max(free))
    torch.cuda.set_device(best)
    return f"cuda:{best}"


@pytest.fixture
def tiny_config() -> Qwen3Config:
    """微型 Qwen3 配置（CPU 上秒级前向，用于逻辑/数值测试）"""
    return Qwen3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        vocab_size=512,
        intermediate_size=256,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        max_position_embeddings=128,
    )


@pytest.fixture
def tiny_model(tiny_config) -> Qwen3ForCausalLM:
    """随机初始化的小模型（eval，CPU）"""
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(tiny_config)
    model.eval()
    return model


@pytest.fixture
def kv_pool_factory(tiny_config):
    """KV 池工厂：默认 CPU / fp32；GPU 测试传 device='cuda' 得到 bf16"""

    def _make(device: str = "cpu", num_blocks: int = 32,
              max_seq_len: int = 128, block_size: int = 16) -> KVCachePool:
        torch.manual_seed(0)
        return KVCachePool(
            num_blocks=num_blocks,
            num_layers=tiny_config.num_hidden_layers,
            block_size=block_size,
            num_kv_heads=tiny_config.num_key_value_heads,
            head_dim=tiny_config.head_dim,
            device=device,
            dtype=torch.float32 if device == "cpu" else torch.bfloat16,
            max_seq_len=max_seq_len,
        )

    return _make
