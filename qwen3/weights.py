"""
权重加载 — 从 HuggingFace 预训练模型或本地 safetensors 文件获取权重

两种加载方式:
  1. load_weights_from_hf: 通过 transformers.AutoModelForCausalLM 加载
     (利用 HF 成熟的权重管理, 处理下载、分片、dtype 等)
  2. load_weights_from_safetensors: 直接从本地 safetensors 文件加载
     (不依赖 transformers 库, 更轻量)
"""

import os
import glob

import torch
import torch.nn as nn


def load_weights_from_hf(model: nn.Module, model_name: str, device: torch.device, dtype: torch.dtype):
    """从 HuggingFace 预训练模型加载权重到我们手写的模型"""
    from transformers import AutoModelForCausalLM

    print(f"Loading HuggingFace model: {model_name} ...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="cpu",  # 先加载到 CPU
    )

    # 提取 HF 模型的权重 (只取 parameters, 不取 buffers 如 inv_freq)
    hf_state_dict = hf_model.state_dict()
    print(f"  HF model has {len(hf_state_dict)} tensors")

    # 加载到我们的模型 (strict=False 忽略 inv_freq 等非持久化 buffer)
    result = model.load_state_dict(hf_state_dict, strict=False)
    loaded = len(hf_state_dict) - len(result.unexpected_keys)
    print(f"  Loaded {loaded} tensors, skipped {len(result.unexpected_keys)} unexpected keys")
    if result.missing_keys:
        print(f"  Missing (non-persistent buffers, will be re-created): {result.missing_keys}")

    # 释放 HF 模型, 节省内存
    del hf_model, hf_state_dict
    torch.cuda.empty_cache()

    # 移动到目标设备
    model.to(device=device, dtype=dtype)


def load_weights_from_safetensors(model: nn.Module, model_path: str, device: torch.device, dtype: torch.dtype):
    """从本地目录的 safetensors 文件直接加载权重（不依赖 transformers）"""
    import safetensors

    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    # 读取所有权重文件到一个大的 state_dict 中
    state_dict = {}
    for file in files:
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                state_dict[name] = f.get_tensor(name)

    # 数据类型转换，device 转换
    state_dict = {k: v.to(device=device, dtype=dtype) for k, v in state_dict.items()}

    # 加载权重到模型
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f" Warning: Missing keys when loading weights: {result.missing_keys}")
    if result.unexpected_keys:
        print(f" Warning: Unexpected keys when loading weights: {result.unexpected_keys}")
    # 加载后显式转 dtype + 设备（load_state_dict 的 copy_ 保持参数原 dtype，不会自动转）
    model.to(device=device, dtype=dtype)
