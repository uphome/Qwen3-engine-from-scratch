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
    def __init__(self, model, pool, caches, buckets=(1, 2, 4, 8),
                 reserve_pages=64):
        """捕获图池

        model:   Qwen3ForCausalLM（用其 forward_decode）
        pool:    KVCachePool（共享池，caches 从它 alloc_row）
        caches:  list[PagedKVCache]，长度 == max(buckets)。capture 时
                 用作"占位句柄"（只贡献 pool + 长度校验）；运行期请求
                 的 row_id 通过 s_rows copy_ 更新，无需更换 caches。
        buckets: tuple[int]，捕获的 batch 大小集合（自动去重 + 升序）
        reserve_pages: 每个"占位句柄"预留的页数（捕获内零分配）
        """
        self.model = model
        self.pool = pool
        self.buckets = tuple(sorted(set(buckets)))
        self.max_batch = max(self.buckets)
        assert len(caches) >= self.max_batch, \
            f"caches 数量 {len(caches)} 需 >= max_bucket {self.max_batch}"

        # ---- 1. 预留页：capture 前把占位句柄页预分配够 ----
        #     捕获期间禁止页分配（Python _allocate_block），全部提前做完。
        for c in caches:
            while c.num_pages < reserve_pages:
                c._allocate_block()
            # 占位句柄必须处于 decode 状态：attention.py 以 caches[0].seq_len>0
            # 判断走 decode 分支（否则走 prefill flash，其中 cu_seqlens.cpu()
            # 是捕获禁区 → "operation not permitted when stream is capturing"）。
            # 同步 GPU seq_lens 保持一致性（update kernel 读它定 write_pos）。
            c.seq_len = 1
            self.pool.seq_lens[c.row_id] = 1

        # ---- 2. 哑行：空槽占位用（seq_len=0、无页、永不写入）----
        #     注意：空槽在图中也会被 update_kv_batch 写（write_pos=seq_len=0
        #     → block_idx=0 → 读 block_table_2d[row,0]）。若该行全 -1，phys=-1
        #     越界崩溃。所以哑行必须预分配 1 个物理页，让 update 写到该页
        #     （页永不被读，无害）。kernel 的 attention 部分对 seq_len=0 请求
        #     循环 0 页，不读块表 → 安全。
        self.dummy_row = pool.alloc_row()          # block_table_2d 一行，seq_len=0
        dummy_phys = pool.alloc(1)[0]              # 给哑行一个物理页（防 update 越界）
        self.pool.block_table_2d[self.dummy_row, 0] = dummy_phys
        self._dummy_phys = dummy_phys

        # ---- 3. 共享内存池：多张图捕获共用，避免内存翻倍 ----
        self._pool_handle = torch.cuda.graph_pool_handle()

        # ---- 4. 每 bucket 捕获一张图 ----
        self.graphs = {}       # bucket -> (graph, tensors)
        self._capture_all(caches)

    # ------------------------------------------------------------------

    def _capture_all(self, caches):
        """对每个 bucket 捕获一张图"""
        device = self.pool.k_buffer.device
        head_dim = self.pool.head_dim

        for b in self.buckets:
            # 每 bucket 的固定张量（形状锁定，运行期只 copy_ 内容）
            s_input = torch.zeros((b, 1), dtype=torch.long, device=device)
            s_pos = torch.zeros((b, 1), dtype=torch.long, device=device)
            s_cos = torch.zeros((b, 1, head_dim), device=device,
                                dtype=torch.bfloat16)
            s_sin = torch.zeros_like(s_cos)
            # 必须用真实 row_id（不能 zeros）：row 0 可能 seq_len=0 且无页，
            # kernel 对无效行寻址会崩。空槽的哑行在 replay 时 fill。
            s_rows = torch.tensor([c.row_id for c in caches[:b]],
                                  dtype=torch.int32, device=device)

            # 预热：kernel 编译 + 内存池稳定（捕获期间禁止分配）
            pos = torch.full((b, 1), 0, device=device, dtype=torch.long)
            cos, sin = self.model.model.rotary_emb(pos)
            for _ in range(3):
                self.model.forward_decode(
                    s_input, (cos.to(torch.bfloat16), sin.to(torch.bfloat16)),
                    s_rows, caches[:b])
            torch.cuda.synchronize()

            # 捕获
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool_handle):
                logits = self.model.forward_decode(
                    s_input, (s_cos, s_sin), s_rows, caches[:b])
            torch.cuda.synchronize()

            self.graphs[b] = (g, {
                "input": s_input, "pos": s_pos,
                "cos": s_cos, "sin": s_sin, "rows": s_rows,
                "logits": logits,
            })
            print(f"[GraphRunner] 捕获 bucket={b}")

    # ------------------------------------------------------------------

    def replay(self, input_ids, positions, cos, sin, rows):
        """执行一步 decode：选 bucket → copy_ 动态内容 → replay → logits

        input_ids: (k, 1) 当前 k 个请求的新 token
        positions: (k, 1) 各请求当前位置（start_pos）
        cos, sin:  (k, 1, head_dim) 调用方按 positions 算好的 RoPE 角度
        rows:      (k,)   各请求的 row_id（常驻块表行号）
        Returns:   logits (b, 1, vocab)，有效行在前 k 行，空槽行丢弃

        cos/sin 每步内容变（位置+1），必须由调用方先 rotary_emb 算出再 copy_，
        否则图内 cos/sin 停留在捕获时的位置——RoPE 全错。
        """
        k = input_ids.shape[0]
        b = min(x for x in self.buckets if x >= k)
        g, t = self.graphs[b]

        # 动态内容写进固定张量（copy_ 不换指针）
        t["input"][:k].copy_(input_ids)
        t["pos"][:k].copy_(positions)
        t["cos"][:k].copy_(cos)
        t["sin"][:k].copy_(sin)
        t["rows"][:k].copy_(rows)
        # 空槽填哑行（seq_len=0，kernel 循环 0 页，输出不采样）
        if k < b:
            t["rows"][k:].fill_(self.dummy_row)

        g.replay()
        return t["logits"]

    def free(self):
        """释放哑行：归还行号 + 物理页（图本身随 pool 生命周期）"""
        self.pool.free(self._dummy_phys)
        self.pool.free_row(self.dummy_row)
