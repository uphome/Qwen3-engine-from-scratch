"""
GraphRunner — CUDA Graph 图池：decode 步捕获成多张图（每 bucket 一张）

原理（vLLM MultiGraphRunner 同款）：
  - decode 步形状固定（B 恒定、S=1），可捕获成 CUDA Graph
  - 但 batch 大小动态（请求进出 running）→ 按 bucket 捕获多张图，
    运行时选">= 当前请求数的最小 bucket"，空槽用哑行占位
  - 图 replay 不传参数：capture 时所有指针被烧进图。运行期动态内容
    （input_ids / position / row_ids）通过 copy_ 写进固定张量

关键认知：
  - attention 里 caches 列表只用于取 pool + 校验长度，kernel 全靠
    row_ids（GPU 常驻 2D 块表行号）寻址 → 运行期请求进出只需更新
    s_rows 内容（copy_ 换 row_id），capture 时的 caches 列表可复用
  - 空槽填"哑行"row_id（pool 预留一行、seq_len=0、无页），kernel 对
    seq_len=0 请求循环 0 页、输出未定义但不采样

接口:
  runner = GraphRunner(model, pool, caches, buckets=[1,2,4,8], reserve_pages=64)
  logits = runner.replay(input_ids, positions, rows)   # 自动选 bucket
"""

import torch


class GraphRunner:
    """
    CUDA Graph 图池管理器。

    核心思想：
      1. decode 步形状固定，因此可以预先捕获成 CUDA Graph。
      2. 请求数动态变化，因此按 2 的幂准备多个 bucket。
      3. 每次推理只 copy_ 少量动态输入，然后 replay 整张图，
         避免每步几十次 kernel 启动的 CPU 开销。
    """

    def __init__(self, model, pool, caches, buckets=(1, 2, 4, 8),
                 reserve_pages=64):
        """初始化图池。

        Args:
            model: Qwen3ForCausalLM，必须提供 capture 友好的 forward_decode()。
            pool: KVCachePool，所有请求共享的物理 KV 池。
            caches: list[PagedKVCache]，长度 >= max(buckets)。
                    这些是“占位句柄”，只用于 capture 时提供稳定的
                    row_id 和 KV 池状态；运行期实际请求通过 copy_ 替换 row_id。
            buckets: tuple[int]，要捕获的 batch 大小集合，例如 (1,2,4,8)。
            reserve_pages: 每个占位句柄在 capture 前预留的物理页数。
        """
        self.model = model
        self.pool = pool

        # bucket 去重并排序，保证后面能用 min(x >= k) 快速选择。
        self.buckets = tuple(sorted(set(buckets)))
        self.max_batch = max(self.buckets)

        # 占位句柄数量必须覆盖最大 bucket，否则 capture 时不够用。
        assert len(caches) >= self.max_batch, \
            f"caches 数量 {len(caches)} 需 >= max_bucket {self.max_batch}"

        # ------------------------------------------------------------
        # 1. 预留页：capture 前把占位句柄页预分配够
        # ------------------------------------------------------------
        # CUDA Graph capture 期间禁止执行 Python 页分配（_allocate_block），
        # 所以必须在 capture 之前把所有需要的物理页提前分配好。
        for c in caches:
            # 每个占位句柄预留 reserve_pages 个物理页。
            while c.num_pages < reserve_pages:
                c._allocate_block()

            # 占位句柄必须处于 decode 状态。
            # attention.py 以 caches[0].seq_len > 0 判断走 decode 分支；
            # 如果 seq_len == 0，会走 prefill flash 分支，其中有 .cpu()，
            # 这是 CUDA Graph capture 禁区。
            c.seq_len = 1
            # 同步 GPU 端 seq_lens，因为 update kernel 会读取它作为 write_pos。
            self.pool.seq_lens[c.row_id] = 1

        # ------------------------------------------------------------
        # 2. 哑行：空槽占位用
        # ------------------------------------------------------------
        # 当实际请求数 k < bucket b 时，空出的槽位不能留无效 row_id，
        # 否则 kernel 可能读到 block_table 里的 -1 导致越界。
        # 哑行专门用于填充这些空槽。
        self.dummy_row = pool.alloc_row()          # 分配一个常驻块表行号
        dummy_phys = pool.alloc(1)[0]              # 给哑行分配 1 个物理页
        # 把哑行的第 0 个逻辑页指向这个物理页。
        # 这样空槽即使被 update_kv_batch 写入，也不会访问 -1 物理块。
        self.pool.block_table_2d[self.dummy_row, 0] = dummy_phys
        self._dummy_phys = dummy_phys

        # ------------------------------------------------------------
        # 3. 共享 CUDA Graph 内存池
        # ------------------------------------------------------------
        # 多个 bucket 的图共用同一个 graph memory pool，
        # 避免每张图各自预留内存导致显存翻倍。
        self._pool_handle = torch.cuda.graph_pool_handle()

        # ------------------------------------------------------------
        # 4. 为每个 bucket 捕获一张图
        # ------------------------------------------------------------
        # bucket -> (CUDAGraph, fixed_tensors)
        self.graphs = {}
        self._capture_all(caches)

    # ------------------------------------------------------------------

    def _capture_all(self, caches):
        """对每个 bucket 捕获一张 CUDA Graph。"""
        device = self.pool.k_buffer.device

        for b in self.buckets:
            # ------------------------------------------------
            # 每个 bucket 的固定输入张量。
            # 这些张量的“地址”在 capture 时会被烧进图里，
            # 之后运行期只能 copy_ 内容，不能重新赋值。
            # ------------------------------------------------
            # 当前 batch 的 token id，形状 (b, 1)
            s_input = torch.zeros((b, 1), dtype=torch.long, device=device)
            # 当前 batch 的绝对位置，形状 (b, 1)，用于 RoPE 查表
            s_pos = torch.zeros((b, 1), dtype=torch.long, device=device)
            # 当前 batch 的 row_id，对应 GPU 常驻 2D 块表行号。
            # 注意：必须用真实 row_id，不能用 zeros。
            # 如果 row_id 指向未分配行，kernel 读 block_table 可能拿到 -1。
            s_rows = torch.tensor([c.row_id for c in caches[:b]],
                                  dtype=torch.int32, device=device)

            # ------------------------------------------------
            # 预热
            # ------------------------------------------------
            # 目的：
            #   1. 触发 Triton kernel 编译
            #   2. 让 PyTorch 内存池稳定
            #   3. 避免 capture 过程中出现动态内存分配
            # RoPE 查表已经在 forward_decode 内部，因此这里只需要传 positions。
            pos = torch.full((b, 1), 0, device=device, dtype=torch.long)
            for _ in range(3):
                self.model.forward_decode(
                    s_input, pos, s_rows, caches[:b])
            torch.cuda.synchronize()

            # ------------------------------------------------
            # 正式捕获
            # ------------------------------------------------
            # 在 torch.cuda.graph 上下文内执行的所有 kernel 和指针关系
            # 都会被记录成一张静态 CUDA Graph。
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool_handle):
                logits = self.model.forward_decode(
                    s_input, s_pos, s_rows, caches[:b])
            torch.cuda.synchronize()

            # 保存图和固定张量，replay 时直接使用。
            self.graphs[b] = (g, {
                "input": s_input,   # (b, 1) token id
                "pos": s_pos,       # (b, 1) position id
                "rows": s_rows,     # (b,) row_id
                "logits": logits,   # 图输出，replay 后从这里读结果
            })
            print(f"[GraphRunner] 捕获 bucket={b}")

    # ------------------------------------------------------------------

    def replay(self, input_ids, positions, rows):
        """执行一步 decode。

        Args:
            input_ids: (k, 1) 当前 k 个请求的新 token。
            positions: (k, 1) 各请求当前绝对位置，用于 RoPE 查表。
            rows: (k,) 各请求在 GPU 常驻块表中的 row_id。

        Returns:
            logits: (b, 1, vocab)。有效行在前 k 行，空槽行应被调用方丢弃。
        """
        # 当前实际请求数
        k = input_ids.shape[0]

        # 选择 >= k 的最小 bucket。
        # 例如 buckets=(1,2,4,8)，k=3 时选 4。
        b = min(x for x in self.buckets if x >= k)
        g, t = self.graphs[b]

        # ------------------------------------------------
        # 用 copy_ 更新动态内容。
        # copy_ 是原地拷贝，不会改变张量地址，
        # 因此已经捕获的 CUDA Graph 仍然有效。
        # ------------------------------------------------
        t["input"][:k].copy_(input_ids)
        t["pos"][:k].copy_(positions)
        t["rows"][:k].copy_(rows)

        # 如果实际请求数小于 bucket 大小，空槽填哑行。
        # 哑行 seq_len=0，kernel 不会真正读取它的 KV，只用于占位防越界。
        if k < b:
            t["rows"][k:].fill_(self.dummy_row)

        # 一次图启动，代替 eager 模式下每层几十次 kernel 启动。
        g.replay()
        return t["logits"]

    def free(self):
        """释放哑行资源。

        只归还哑行占用的物理页和 row_id。
        CUDA Graph 本身通常跟随服务生命周期，不在这里释放。
        """
        self.pool.free(self._dummy_phys)
        self.pool.free_row(self.dummy_row)
