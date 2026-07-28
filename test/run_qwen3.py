"""
Lesson 2: 从零手写 Qwen3 模型 — 端到端推理

仅依赖 PyTorch，从零实现 Qwen3 的完整推理流程：
  1. 模型配置 (AutoConfig)
  2. 基础组件 (RMSNorm, RoPE, SwiGLU MLP)
  3. 注意力机制 (GQA + QK-Norm)
  4. Transformer Decoder
  5. 权重加载 (通过 transformers from_pretrained 获取权重)
  6. 自回归生成

用法:
    python run_qwen3.py --model Qwen/Qwen3-0.6B
    python run_qwen3.py --model Qwen/Qwen3-8B --prompt "Tell me about AI" --temperature 0
"""

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 2. RMSNorm — 比 LayerNorm 更高效的归一化
# ============================================================================
#
# 标准 LayerNorm: y = (x - mean) / std * gamma + beta    (需要 mean 和 std)
# RMSNorm:        y = x / RMS(x) * gamma                 (只需要 RMS, 无 beta)
#
# RMS(x) = sqrt(mean(x^2) + eps)
#
# 优势：省去均值计算和偏置参数，速度更快，效果相当

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先转 float32 做归一化（避免 bfloat16 精度问题），再转回原 dtype
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_float * rms).to(x.dtype) * self.weight


# ============================================================================
# 3. Rotary Position Embedding (RoPE)
# ============================================================================
#
# RoPE 核心思想：用旋转矩阵编码位置信息
#   - 对 Q, K 的每对相邻维度 (x0, x1)，应用 2D 旋转：
#     [x0', x1'] = [x0*cos(mθ) - x1*sin(mθ), x0*sin(mθ) + x1*cos(mθ)]
#   - m 是位置索引，θ_i = base^(-2i/d) 是频率
#   - 低维频率高（变化快），高维频率低（变化慢）
#
# 优势：
#   - Q·K 的点积自然包含 (m-n) 的相对位置信息
#   - 可以外推到训练时没见过的序列长度

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 1000000.0):
        super().__init__()
        # 频率: theta_i = 1 / (base^(2i/d)), i = 0, 1, ..., d/2-1
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (batch, seq_len) — 每个 token 的位置
        # 计算 angle = position * freq, 形状 (seq_len, head_dim/2)
        freqs = torch.outer(position_ids[0].float(), self.inv_freq)
        # 复制拼接成完整 head_dim: (seq_len, head_dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        # 返回 cos, sin, 形状 (1, seq_len, head_dim) — batch 维度广播
        return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将后半部分取负并与前半部分交换: [x1, x2] -> [-x2, x1]"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对 Q 和 K 应用旋转位置编码

    q, k: (batch, heads, seq_len, head_dim)
    cos, sin: (1, seq_len, head_dim)  -> unsqueeze 成 (1, 1, seq_len, head_dim)
    """
    cos = cos.unsqueeze(1).to(q.dtype)
    sin = sin.unsqueeze(1).to(q.dtype)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


# ============================================================================
# KV Cache — 最朴素的实现：每层一个 list，新 token 的 KV 拼到后面
# ============================================================================

class NaiveKVCache:
    """最简 KV Cache：每层存 (B, num_kv_heads, seq_len, head_dim) 的 K/V tensor"""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.k_cache = [None] * num_layers   # 每层一个 tensor 或 None
        self.v_cache = [None] * num_layers
        self.seq_len = 0                      # 当前缓存了多少个 token

    def get_kv(self, layer_id: int):
        """返回该层已缓存的 K, V（可能为 None）"""
        return self.k_cache[layer_id], self.v_cache[layer_id]

    def update(self, layer_id: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """将新 token 的 K, V 拼到缓存末尾"""
        # k_new, v_new: (B, num_kv_heads, S_new, head_dim)，S_new 通常为 1
        if self.k_cache[layer_id] is None:
            self.k_cache[layer_id] = k_new
            self.v_cache[layer_id] = v_new
        else:
            self.k_cache[layer_id] = torch.cat([self.k_cache[layer_id], k_new], dim=2)
            self.v_cache[layer_id] = torch.cat([self.v_cache[layer_id], v_new], dim=2)

    def advance_seq_len(self, n: int = 1):
        self.seq_len += n


# ============================================================================
# 4. Grouped Query Attention (GQA) + QK-Norm
# ============================================================================
#
# GQA 是 MHA 和 MQA 的折中：
#   - MHA (Multi-Head): 每个 Q 头有独立的 K, V 头 → 参数多，KV cache 大
#   - MQA (Multi-Query): 所有 Q 头共享 1 组 K, V    → 参数少，但质量可能下降
#   - GQA (Grouped):     每 G 个 Q 头共享 1 组 K, V  → 折中方案
#
# Qwen3-8B: 32 个 Q heads, 8 个 KV heads → 每 4 个 Q 头共享一组 KV
#
# QK-Norm (Qwen3 特有):
#   在 RoPE 之前，对 Q 和 K 的每个头做 RMSNorm。稳定训练、提升质量。

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将 KV 头复制 n_rep 次以匹配 Q 头数量 (用于 GQA)"""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_attention_heads #注意力头的数量 Q被分成了16份
        self.num_kv_heads = config.num_key_value_heads# KV只有8个 原本是16个的，但是使用GQA技术
        self.num_kv_groups = self.num_heads // self.num_kv_heads # 16 // 8 = 2   每 2 个 Q 头拼 1 个 KV 头
        self.head_dim = config.head_dim #每个头的宽度  128
        self.scaling = self.head_dim ** -0.5

        # Q, K, V, O 线性投影
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # QK-Norm: 对每个头的 Q, K 做 RMSNorm (在 head_dim 上归一化)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,            # (batch, seq_len, hidden_size)
        position_embeddings: tuple[torch.Tensor, torch.Tensor],  # (cos, sin)
        attention_mask: torch.Tensor,            # (1, 1, seq_len, kv_len)
        kv_cache: "NaiveKVCache | None" = None,  # ← 新增
        layer_idx: int = 0,                      # ← 新增：当前是第几层
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        # 投影 Q, K, V 并 reshape 成多头格式
        # (B, S, hidden) -> (B, S, heads, head_dim) -> (B, heads, S, head_dim)
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK-Norm (Qwen3 特有 — 在 RoPE 之前归一化 Q, K)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 应用 RoPE
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV cache: 先存新的（只存新 token 的 K,V），再拼旧的做 attention
        if kv_cache is not None:
            k_old, v_old = kv_cache.get_kv(layer_idx)
            kv_cache.update(layer_idx, k, v)    # 只存 k_new, v_new（S_new 个 token）
            if k_old is not None:
                # Decode 模式：拼上旧缓存，Q 才能看到所有历史
                k = torch.cat([k_old, k], dim=2)
                v = torch.cat([v_old, v], dim=2)
                # K=K
                # V=V

        # GQA: 将 KV 头复制以匹配 Q 头数量
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # 注意力计算: softmax(Q @ K^T / sqrt(d)) @ V
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # reshape 回 (B, S, hidden) 并投影输出
        attn_output = attn_output.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(attn_output)


# ============================================================================
# 5. SwiGLU MLP
# ============================================================================
#
# 标准 Transformer MLP:  y = W2 @ relu(W1 @ x)
# SwiGLU MLP (Qwen3):    y = down_proj(silu(gate_proj(x)) * up_proj(x))
#
# 其中 silu(x) = x * sigmoid(x) (也叫 Swish)
#
# gate 和 up 是两个独立的投影, gate 通过 silu 激活后作为 "门控" 来调节 up 的输出
# 这种设计比单层 relu 有更好的表达能力

class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# 6. Transformer Decoder Layer
# ============================================================================
#
# Pre-Norm 架构 (和 Post-Norm 的区别):
#
#   Pre-Norm (Qwen3 使用):
#     x → RMSNorm → Attention → + → RMSNorm → MLP → +
#     |______________________|    |__________________|
#           残差连接                    残差连接
#
#   Post-Norm (原始 Transformer):
#     x → Attention → + → LayerNorm → MLP → + → LayerNorm
#
# Pre-Norm 训练更稳定，收敛更快，是现代 LLM 的标配

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        kv_cache: "NaiveKVCache | None" = None,  # ← 新增
        layer_idx: int = 0,                      # ← 新增
    ) -> torch.Tensor:
        # 自注意力 + 残差
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask,
                                       kv_cache=kv_cache, layer_idx=layer_idx)
        hidden_states = residual + hidden_states

        # MLP + 残差
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ============================================================================
# 7. Qwen3Model — Transformer backbone
# ============================================================================

class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(self, input_ids: torch.Tensor,
                kv_cache: "NaiveKVCache | None" = None) -> torch.Tensor:
        B, S = input_ids.shape

        # Token embedding: 每个 token ID -> 向量
        hidden_states = self.embed_tokens(input_ids)

        # 位置 ID
        if kv_cache is not None and kv_cache.seq_len > 0:
            # Decode 模式：新 token 的位置 = 已缓存的长度 ~ 已缓存长度+当前序列长度-1
            start_pos = kv_cache.seq_len
            position_ids = torch.arange(start_pos, start_pos + 1, device=input_ids.device).unsqueeze(0)  #torch.Size([1, 1]) 就是当前这个token的总的位置ID
        else:
            # Prefill 模式：位置从 0 开始
            position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1) #torch.Size([1, 32])


        # 提前计算 RoPE cos/sin
        position_embeddings = self.rotary_emb(position_ids)

        # 注意力掩码
        if kv_cache is not None and kv_cache.seq_len > 0:
            # Decode 模式：Q 只有 1 个位置，可以看到所有缓存的 K + 自己的 K
            # mask 形状 (1, 1, 1, kv_len)，全部为 0（不遮蔽） 
            kv_len = kv_cache.seq_len + S # 这个加的S就是自己的K
            causal_mask = torch.zeros(1, 1, S, kv_len, device=input_ids.device, dtype=hidden_states.dtype)
        else:
            # Prefill 模式：标准因果掩码
            causal_mask = torch.full((S, S), float("-inf"), device=input_ids.device, dtype=hidden_states.dtype)
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

        # 逐层前向传播
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, causal_mask, position_embeddings,
                                  kv_cache=kv_cache, layer_idx=i)

        # 更新缓存计数（prefill 加 S，decode 加 1）
        if kv_cache is not None:
            kv_cache.advance_seq_len(S)

        # 最终归一化
        return self.norm(hidden_states)


# ============================================================================
# 8. Qwen3ForCausalLM — 完整语言模型
# ============================================================================

class SimpleConfig:
    """绕过 transformers.AutoConfig，直接读 config.json（兼容老版本 transformers）"""
    def __init__(self, model_path: str):
        import json, os
        # 如果是本地路径直接读，否则从 HF 缓存找
        if os.path.isdir(model_path):
            config_file = os.path.join(model_path, "config.json")
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(repo_id=model_path, filename="config.json")
        with open(config_file) as f:
            data = json.load(f)
        self.hidden_size = data["hidden_size"]
        self.num_attention_heads = data["num_attention_heads"]
        self.num_key_value_heads = data["num_key_value_heads"]
        self.head_dim = data.get("head_dim", data["hidden_size"] // data["num_attention_heads"])
        self.num_hidden_layers = data["num_hidden_layers"]
        self.vocab_size = data["vocab_size"]
        self.intermediate_size = data["intermediate_size"]
        self.rms_norm_eps = data["rms_norm_eps"]
        self.rope_theta = data["rope_theta"]
        self.tie_word_embeddings = data.get("tie_word_embeddings", False)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config) #这个也是Qwen3的核心块
        # lm_head: 将 hidden_states 映射到 vocab_size 维度, 输出每个 token 的概率
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor,
                kv_cache: "NaiveKVCache | None" = None) -> torch.Tensor:
        """
        input_ids: (batch, seq_len)
        kv_cache: 传入则使用 KV cache（prefill 存，decode 读+存）
        returns: logits (batch, seq_len, vocab_size)
        """
        hidden_states = self.model(input_ids, kv_cache=kv_cache)
        return self.lm_head(hidden_states)
    #输入是 (batch, seq_len)   输出是有  (batch, seq_len, vocab_size)  即每个位置都有一个 vocab_size 的向量  这个向量就是每个token的概率分布


# ============================================================================
# 9. 权重加载 — 从 HuggingFace 预训练模型获取权重
# ============================================================================
#
# 思路:
#   1. 用 transformers 的 from_pretrained 加载 HF 模型 (它处理下载、分片、dtype 等)
#   2. 从 HF 模型中提取 state_dict
#   3. 将 state_dict 加载到我们手写的模型中
#
# 这样可以利用 HF 成熟的权重管理, 同时用自己的模型做推理
# 后续课程会教如何直接从 safetensors 文件加载 (绕过 transformers)

def load_weights_from_hf(model: Qwen3ForCausalLM, model_name: str, device: torch.device, dtype: torch.dtype):
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


# ============================================================================
# 10. 自回归生成
# ============================================================================
#
# LLM 生成文本的核心循环:
#   1. 将整个序列送入模型, 得到 logits
#   2. 取最后一个位置的 logits, 采样出下一个 token
#   3. 将新 token 拼接到序列末尾
#   4. 重复直到生成结束 token 或达到最大长度
#
# 注意: 没有 KV cache 时, 每一步都要重新计算整个序列的注意力
#   Step 0: forward(prompt)             — 处理 N 个 token
#   Step 1: forward(prompt + tok1)      — 处理 N+1 个 token (重新计算 prompt!)
#   Step 2: forward(prompt + tok1,2)    — 处理 N+2 个 token
#   总计算量 O(N^2), KV cache (Lesson 4) 可将每步降为 O(1)

@torch.no_grad()  #关闭梯度推导
def generate(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor, #输入的token的ID 就是在分词表中的ID
    max_new_tokens: int = 128, #新生成的token长度
    temperature: float = 0.7,  #温度，  0 是贪心算法 >0是对logits进行缩放 
    top_k: int = 50,  #取前50个作为后续
    top_p: float = 0.9,  #从大到小 概率相加 加到0.9为止， 后面的全部舍去
    eos_token_id: int = 151645, #终止token 遇到就停止自回归生成
) -> tuple[torch.Tensor, dict]:
    """自回归生成 (有 KV cache)"""
    generated = input_ids.clone()
    stats = {"step_times": [], "input_len": input_ids.shape[1]}

    # 创建 KV cache
    kv_cache = NaiveKVCache(model.config.num_hidden_layers)

    for step in range(max_new_tokens):
        t0 = time.perf_counter()

        if step == 0:
            # Prefill: 第一次送入完整 prompt，所有 K,V 写入 cache
            logits = model(generated, kv_cache=kv_cache)
        else:
            # Decode: 只送入最后一个 token，其余 K,V 从 cache 读
            logits = model(generated[:, -1:], kv_cache=kv_cache)

        dt = time.perf_counter() - t0
        stats["step_times"].append(dt)  

        # 取最后一个位置的 logits
        next_logits = logits[:, -1, :] # shape: (batch, T，vocab_size) T是当前输入的长度    

        # 采样策略
        if temperature == 0:
            # 贪心解码: 直接选概率最高的 token
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature #除于温度

            # Top-K: 只保留概率最高的 K 个 token
            if top_k > 0:
                topk_vals = torch.topk(next_logits, min(top_k, next_logits.size(-1))).values #取前K个最大地值
                #这也有个防御性编程  top_k 一定小于分词表大小 
                next_logits = next_logits.masked_fill(next_logits < topk_vals[..., -1:], float("-inf"))
                #将前K个中最小地值作为阈值，比起小的变为 负无穷。

            # Top-P (Nucleus): 保留累积概率不超过 P 的 token    核采样
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)  
                #把当前所有 token 的分数（logits）从高到低排序。
                #sorted_logits 是排好序的分数，sorted_idx 是这些分数对应的原始 token ID（下标）。
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)  #里面有个softmax 将分数改为概率 然后进行累加
                #假设 top_p = 0.7，概率分布是 [0.5, 0.3, 0.1, 0.1]。累积概率 [0.5, 0.8, 0.9, 1.0]
                mask = cumprobs > top_p  
                mask[..., 1:] = mask[..., :-1].clone()  #整体右移动
                # 原始：[False, True, True, True] 变为 结果：[False, False, True, True]  第一位的False不变  移动的本质是为了保留临界点
                mask[..., 0] = False
                # 强制把第一个位置（概率最高的 token）设为 False
                remove = mask.scatter(-1, sorted_idx, mask)
                next_logits = next_logits.masked_fill(remove, float("-inf"))
                # 这个词汇表对应问题有点不懂

            probs = F.softmax(next_logits, dim=-1)
            # 将剩下的词元再次转换为 logits
            next_token = torch.multinomial(probs, num_samples=1) 
            # sample one token  这是按照概率分布加权采样

        generated = torch.cat([generated, next_token], dim=-1) #将新生成的ID 与 之前的输入拼接在一起 

        if next_token.item() == eos_token_id:
            break

    return generated, stats

def load_weights_from_safetensors(model: nn.Module, model_path: str, device: torch.device, dtype: torch.dtype):
    """从本地目录的 safetensors 文件直接加载权重（不依赖 transformers）"""
    import glob
    import safetensors

    # model_path 是本地目录，直接从中加载
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
    model.to(device=device)
# ============================================================================
# 11. 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Lesson 2: 从零手写 Qwen3 推理")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace 模型名或本地路径, e.g. Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", type=str, default="写一篇300字的文章。")
    parser.add_argument("--max-tokens", type=int, default=1280)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default=None,
                        help="指定设备, e.g. 'cuda:1' (默认: 自动选择最空闲的 GPU)")
    args = parser.parse_args()

    print("=" * 60)
    print("Lesson 2: 从零手写 Qwen3 — 端到端推理")
    print("=" * 60)

    # 不依赖 transformers：用 tokenizers 库直读 + SimpleConfig 直读 config.json

    # --- 选择设备 ---
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        best = free.index(max(free))
        device = torch.device(f"cuda:{best}")
    else:
        device = torch.device("cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"\nDevice: {device}, dtype: {dtype}")

    # --- 读取配置 ---
    print(f"\n[1/4] 读取模型配置...")
    config = SimpleConfig(args.model)  # 直读 config.json，不依赖新版 transformers
    print(f"  Model: {args.model}")
    print(f"  Layers: {config.num_hidden_layers}, Hidden: {config.hidden_size}, "
          f"Heads: {config.num_attention_heads}, KV Heads: {config.num_key_value_heads}")

    # --- 创建模型 ---
    print(f"\n[2/4] 创建模型结构...")
    model = Qwen3ForCausalLM(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {param_count:,} ({param_count / 1e9:.2f}B)")

    # --- 加载权重 ---
    print(f"\n[3/4] 从 safetensors 加载权重...")
    load_weights_from_safetensors(model, args.model, device, dtype)
    model.eval()

    # --- 生成 ---
    print(f"\n[4/4] 生成文本...")
    # 不依赖 transformers：用 tokenizers 库直读 tokenizer.json
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    eos_token_id = 151645  # Qwen3 的 EOS token ID

    # Qwen3 Chat Template（手动实现 Jinja 模板逻辑）
    def format_chat(messages):
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        text += "<|im_start|>assistant\n"
        return text

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Respond directly without thinking."},
        {"role": "user", "content": args.prompt},
    ]
    text = format_chat(messages)
    input_ids = torch.tensor([tokenizer.encode(text).ids], device=device)

    sampling = "greedy" if args.temperature == 0 else f"T={args.temperature}"
    print(f"  Prompt: {args.prompt}")
    print(f"  Input tokens: {input_ids.shape[1]}")
    print(f"  Sampling: {sampling}, max_tokens: {args.max_tokens}")

    t0 = time.perf_counter()
    output_ids, stats = generate(
        model, input_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
    )
    t_total = time.perf_counter() - t0

    # --- 输出 ---
    new_tokens = output_ids[0][input_ids.shape[1]:]
    output_text = tokenizer.decode(new_tokens.tolist())

    print(f"\n{'=' * 60}")
    print("Generated:")
    print("=" * 60)
    print(output_text)
    print("=" * 60)

    # --- 统计 ---
    n = len(new_tokens)
    step_times = stats["step_times"]
    print(f"\n统计:")
    print(f"  生成 tokens: {n}")
    print(f"  总耗时: {t_total:.2f}s")
    if n > 0:
        print(f"  平均每步: {sum(step_times) / len(step_times) * 1000:.1f} ms")
        print(f"  首步 (prefill): {step_times[0] * 1000:.1f} ms")
        if len(step_times) > 1:
            print(f"  末步: {step_times[-1] * 1000:.1f} ms")

    print(f"\nKV cache 已启用:")
    print(f"  Prefill: 处理 {stats['input_len']} tokens（存入 cache）")
    print(f"  Decode:  每次只处理 1 token（其余从 cache 读）")
    print(f"  每步时间应保持稳定，不随序列长度增长")


if __name__ == "__main__":
    main()
