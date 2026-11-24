"""Prefix Cache（哈希 KV 复用）— 块哈希工具

设计（阶段 1：完全命中）：
  - prompt 按 block_size 切块，块哈希是**链式**的：
      h_i = blake2b(h_{i-1}, tokens[i*B:(i+1)*B])
    这样相同的 token 前缀一定得到相同哈希序列，不同前缀（哪怕只差
    一个 token）会从差异块开始全部不同 —— 天然支持前缀匹配。
  - 只对**完整块**（block_size 个 token）计算哈希：不完整块的内容会
    在 decode 时被后续 token 写满，不能作为前缀缓存（内容与哈希不对应）。
  - 阶段 1 只使用"全部完整块命中"的快速路径；部分命中的后缀 prefill
    （需要 prefill attention 读历史 KV）留到阶段 2。

与 vLLM 的对应关系：
  - vLLM 用 block hash + ref_count + LRU 做 prefix caching；
  - 这里同样以"块"为粒度（块对齐），块内容只读共享，
    decode 新 token 写入新分配的块，不覆盖共享块。
"""

import hashlib


def hash_block(parent_hash: int, tokens: list[int]) -> int:
    """链式块哈希：h = blake2b(parent_hash || token_ids)"""
    h = hashlib.blake2b(digest_size=16)
    h.update(parent_hash.to_bytes(16, "little"))
    for t in tokens:
        h.update(int(t).to_bytes(4, "little"))
    return int.from_bytes(h.digest(), "little")


def compute_prefix_hashes(token_ids: list[int], block_size: int) -> list[int]:
    """计算 prompt 全部**完整块**的哈希链

    返回长度 = len(token_ids) // block_size（不足一块的尾部不缓存）。
    """
    if block_size <= 0:
        raise ValueError("block_size 必须为正")
    ids = list(token_ids)
    n_full = len(ids) // block_size
    hashes: list[int] = []
    parent = 0
    for i in range(n_full):
        chunk = ids[i * block_size:(i + 1) * block_size]
        parent = hash_block(parent, chunk)
        hashes.append(parent)
    return hashes
