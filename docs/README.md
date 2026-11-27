# 学习文档（docs）

结合代码讲清楚六个核心机制。建议按顺序读，每篇都可以独立阅读。

| 序号 | 文档 | 讲什么 | 对应代码 |
|---|---|---|---|
| 01 | [LEARN_01_STREAMING.md](LEARN_01_STREAMING.md) | SSE 协议、跨线程 token 队列、增量解码、断开取消 | `server.py` |
| 02 | [LEARN_02_PREFIX_CACHE.md](LEARN_02_PREFIX_CACHE.md) | 块哈希链、引用计数/链感知 LRU、完全/部分命中、COW | `qwen3/prefix_cache.py`、`PagedKVcache.py`、`scheduler.py` |
| 03 | [LEARN_03_SCHEDULING.md](LEARN_03_SCHEDULING.md) | 状态机、组批、chunked prefill、容量记账 | `qwen3/scheduler.py`、`request.py`、`batch.py` |
| 04 | [LEARN_04_TESTING_BENCH.md](LEARN_04_TESTING_BENCH.md) | 四层测试、CI、TTFT/TPOT 口径、压测与长稳 | `tests/`、`bench_*.py`、`scripts/` |
| 05 | [LEARN_05_TRAINING_MONITORING.md](LEARN_05_TRAINING_MONITORING.md) | 训推同机监控：快照接口、JSONL、GRPO 集成 | `qwen3/monitoring.py`、`integrations/` |
| 06 | [LEARN_06_BLOCK_POOL_STATE_MACHINE.md](LEARN_06_BLOCK_POOL_STATE_MACHINE.md) | 块池三态、ref 计数、COW、淘汰、不变量 | `qwen3/PagedKVcache.py`、`scheduler.py` |

每篇的结构：**概念 → 代码逐段 → 为什么这样设计 → 踩过的坑 → 动手实验 → 自测问题**。

相关文档：

- [PREFIX_CACHE.md](../PREFIX_CACHE.md)：前缀缓存的设计与实测数据
- [TESTING.md](../TESTING.md)：测试矩阵与复现命令
- [BENCHMARK.md](../BENCHMARK.md)：压测报告
- [ENGINE_COMPARISON.md](../ENGINE_COMPARISON.md)：与工业级引擎的差距与路线图
- [PERFORMANCE.md](../PERFORMANCE.md)：版本性能演进（v0.1 → v3.3）
