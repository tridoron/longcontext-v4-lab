# Memory Profile

本文件记录实际显存、tokens/s、KV cache 估算和压缩比例。当前未运行正式 profile。

生成命令：

```bash
uv run python -m longcontext.eval.eval_kv_cache --config configs/main_350m_hybrid.yaml
uv run python -m longcontext.eval.eval_speed --config configs/main_350m_hybrid.yaml --lengths 4096 8192 16384 32768 --steps 3
```
