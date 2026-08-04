"""
Request — 单个推理请求的状态容器

Continuous batching 的"被动数据"部分：
  Request 只维护自己的数据与状态机，不做任何调度决策
  （调度决策全部在 Scheduler 里）。

状态机:
    waiting → prefilling → decoding → finished

  waiting    排队等待 prefill（已分配 kv_cache 句柄，未占物理块）
  prefilling 本步参与 prefill 批（只持续一个调度步）
  decoding   逐 token 自回归推进
  finished   EOS / 达到 max_new_tokens → 归还 KV 物理块

每个请求独占一个 PagedKVCache 句柄（block_table + seq_len），
finish 时 kv_cache.free() 归还物理块到共享池。

类比:
  Request 是"旅客的行李"——只装自己的东西（prompt、已生成 token、
  KV cache 句柄、剩余步数），自己不动。
  什么时候上车（被调度）、坐哪趟车（进哪个 batch）、下车后行李
  放哪（归还物理块）——全部由 Scheduler 这个"调度员"决定。
"""

from __future__ import annotations

import torch

from .PagedKVcache import PagedKVCache

# ============================================================
# 状态常量
# ============================================================
# 用字符串常量而非硬编码，避免魔法值；状态名同时也是
# _VALID_TRANSITIONS 字典的键，迁移表与状态常量一一对应。
WAITING = "waiting"        # 已入队，等调度器选中做 prefill
PREFILLING = "prefilling"  # 本步被选中，正在参与 prefill 前向（只持续一个调度步）
DECODING = "decoding"      # prefill 完成，每步 decode 一个 token
FINISHED = "finished"      # EOS / 达上限，KV 物理块已归还

# ============================================================
# 合法迁移表（状态机校验用）
# ============================================================
# 每个状态列出它"允许迁往"的下一个状态，其余迁移一律报错。
# 相当于给状态机上了一道保险：调度器写错顺序（比如没 prefill
# 就进入 decoding）会在 assert 处立刻暴露，而不是带病跑出
# 错误结果再排查。
#
#   waiting   → prefilling         （被调度器选中）
#   prefilling → decoding          （prefill 前向完成）
#   decoding  → decoding, finished （继续自回归 / 结束）
#   finished  → （终点，不可再迁移）
_VALID_TRANSITIONS = {
    WAITING: (PREFILLING,),
    PREFILLING: (DECODING,),
    DECODING: (DECODING, FINISHED),
    FINISHED: (),
}


class Request:
    """单个请求：prompt + 生成状态 + KV cache 句柄

    字段分三类：
      1. 输入（构造时固定）: request_id / input_ids / max_new_tokens / eos_token_id
      2. 运行状态（生成过程中变化）: output_ids / num_generated / state
      3. 资源句柄: kv_cache（指向共享池的请求级块表）

    对外行为只通过方法暴露（start_prefill / append / finish ...），
    不让外部直接改 state，保证状态机迁移总是经过合法性校验。
    """

    def __init__(
        self,
        request_id: int,
        input_ids: torch.Tensor,     # (1, S) prompt tokens（与 generate() 接口形状一致）
        max_new_tokens: int,         # 最多生成多少个新 token
        eos_token_id: int = 151645,  # 遇到该 token 提前结束（151645 = Qwen3 的 <|im_end|>）
        kv_cache: PagedKVCache | None = None,  # 由 Scheduler.add() 注入，用户一般不直接传
        arrival_step: int = 0,       # 到达时调度器的步数（后续做公平调度/超时用）
    ):
        self.request_id = request_id
        self.input_ids = input_ids
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id
        self.kv_cache = kv_cache
        self.arrival_step = arrival_step

        # 生成过程中的可变状态
        self.output_ids: list[int] = []   # 已生成的 token 序列（逐步 append，尾部即最新）
        self.num_generated: int = 0       # 已生成 token 数（与 len(output_ids) 同步维护）
        self.state: str = WAITING         # 初始状态：刚创建，等在队列里

    # ============================================================
    # 状态查询（只读，不修改任何字段）
    # ============================================================

    @property
    def input_len(self) -> int:
        """prompt 长度（S）。decode 步判断"取 logits 的哪个位置"也要用它"""
        return self.input_ids.shape[1]

    @property
    def total_len(self) -> int:
        """已消耗的 token 数（prompt + 已生成）。

        它直接决定 KV cache 需要占多少个物理块：
        num_blocks_needed = ceil(total_len / block_size)。
        调度器/update 用它判断是否需要申请新页。
        """
        return self.input_len + self.num_generated

    @property
    def cur_token(self) -> torch.Tensor:
        """当前 decode 步要送入模型的 token: (1, 1)。

        自回归第 n 步的输入 = 上一步采样出的 token（output_ids 的最后一个）。
        独立成 property 是因为 decode 时模型只吃这 1 个 token，
        K/V 历史全部从 KV cache 里读，不再重复过模型。
        """
        return torch.tensor([[self.output_ids[-1]]], device=self.input_ids.device)

    def is_finished(self) -> bool:
        """生成是否结束（三种情况任一即结束）：
          1. 状态已是 finished（防重复判定）
          2. 达到 max_new_tokens 上限
          3. 采样出 eos_token_id

        由 Scheduler 在每步收尾时调用；为 true 则请求出批、
        finish() 归还 KV 物理块。注意 eos 判定只看"最后一个 token"，
        所以 output_ids 非空才检查。
        """
        return (
            self.state == FINISHED
            or self.num_generated >= self.max_new_tokens
            or (self.output_ids and self.output_ids[-1] == self.eos_token_id)
        )

    # ============================================================
    # 状态迁移（唯一的改状态入口，全部经过合法性校验）
    # ============================================================

    def _transition(self, new_state: str):
        """统一的迁移校验函数：合法才更新，非法立刻 assert 报错。

        报错信息带上 request_id 和当前状态，便于在多请求场景下
        一眼定位是哪个请求、从哪到哪的迁移非法。
        """
        assert new_state in _VALID_TRANSITIONS[self.state], \
            f"[Request {self.request_id}] 非法状态迁移: {self.state} -> {new_state}"
        self.state = new_state

    def start_prefill(self):
        """被调度器选中本步做 prefill（waiting → prefilling）。

        只持续一个调度步：prefill 前向完成并写入 KV 后，
        调度器立即调 enter_decoding() 转入自回归。
        """
        self._transition(PREFILLING)

    def enter_decoding(self):
        """prefill 前向完成 → 进入自回归（prefilling → decoding）"""
        self._transition(DECODING)

    def finish(self):
        """生成结束：标记 finished 并归还 KV 物理块。

        归还 = kv_cache.free()（PagedKVCache 把块表里的物理块
        放回共享池的 free_block_ids，供其他请求复用）+ 置空句柄
        （防止后续误用已释放的句柄）。
        """
        if self.state != FINISHED:
            self._transition(FINISHED)
            if self.kv_cache is not None:
                self.kv_cache.free()
                self.kv_cache = None

    def append(self, token: int):
        """记录一个新生成的 token（采样步的收尾动作）。

        只追加 token 与计数，不判定是否结束——结束判定统一
        在 is_finished() 里由调度器做，保持单一职责。
        """
        self.output_ids.append(int(token))
        self.num_generated += 1

    # ============================================================
    # 输出
    # ============================================================

    def to_output_ids(self) -> torch.Tensor:
        """完整输出序列: (1, input_len + num_generated)。

        prompt + 生成的 token 拼成一条完整序列，供收集结果用
        （对应单请求 generate() 的返回值形状）。
        """
        if self.output_ids:
            gen = torch.tensor([self.output_ids], device=self.input_ids.device)
            return torch.cat([self.input_ids, gen], dim=-1)
        return self.input_ids

    def __repr__(self):
        """调试打印：一眼看清请求 ID、状态、进度（已生成/上限）"""
        return (f"Request(id={self.request_id}, state={self.state}, "
                f"in={self.input_len}, gen={self.num_generated}/{self.max_new_tokens})")
