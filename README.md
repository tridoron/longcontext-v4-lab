# LongContext-V4-Lab

DeepSeek-V4 Inspired Long-Context LM：基于 CSA/HCA 混合压缩注意力的长上下文语言模型训练与评测项目。

## 项目简介

本仓库实现一个 Decoder-only 长上下文语言模型训练框架，包含 Full Attention、Sliding Window Attention、CSA-lite、HCA-lite、Hybrid Attention、Static-mHC 和 Muon + AdamW 参数分组优化器。代码目标是支撑 120M Debug Model 和 350M Main Model 的训练、继续训练、评测与消融。

## 硬件资源

任务书目标硬件为单卡 RTX 5090 / 32GB。当前仓库提供工程代码、配置、脚本和测试，不包含已经完成训练的模型权重。

## 论文参考点

项目参考 DeepSeek-V4 中的 CSA、HCA、mHC、Muon 和长文档数据构造思想，并将其裁剪成单卡可执行版本。

## 项目边界

本项目不是完整 DeepSeek-V4 复现，不实现 MoE、FP4 QAT、TileLang kernel、多机多卡 Expert Parallelism、GRPO/OPD、MTP Loss、百万 token 上下文训练和完整动态 mHC。当前代码不会自动启动正式训练，训练入口要求显式传入 `--max-steps`。

## 模型结构

模型位于 `longcontext/model/`：

- `transformer.py`：Decoder-only LM、loss、greedy generation。
- `block.py`：Pre-Norm Transformer Block 与 mHC 集成。
- `attention_full.py` / `attention_swa.py` / `attention_csa.py` / `attention_hca.py` / `attention_hybrid.py`：五类 attention。
- `mhc.py`：Static-mHC 与最终 residual stream 混合。
- `kv_cache.py`：理论 KV cache 估算。

## CSA-lite 原理

CSA-lite 将连续 token 的 K/V 压缩成 block entry，使用独立 indexer 选择 top-k 压缩块，再与 sliding local window 拼接成统一 memory 做一次 attention。`beta_raw` 作为可学习 gate，把 selected indexer score 加到 compressed logits。

## HCA-lite 原理

HCA-lite 使用更大的压缩块，对所有合法历史 compressed blocks 做 dense compressed attention，并拼接 local window 保留局部细节。

## Hybrid Attention 设计

Hybrid 采用固定层间调度：

```python
if layer_id < 2:
    attention_type = "swa"
elif layer_id % 3 == 0:
    attention_type = "hca"
else:
    attention_type = "csa"
```

## mHC 实现

`StaticMHCUpdate` 实现 `X_{l+1}=B_l X_l + C_l F_l(A_l X_l)`，其中 `B_l` 使用 Sinkhorn 归一化，最终通过 `A_final_raw` 将多条 residual stream 混合回 `[B,T,D]`。

## Muon 实现

`longcontext/optim/muon.py` 实现 5-step Hybrid Newton-Schulz。`build_optimizer.py` 将二维矩阵参数放入 Muon，将 embedding、lm_head、RMSNorm、bias、mHC raw 参数、`A_final_raw`、CSA `beta_raw` 和一维参数放入 AdamW。

## 数据构造

`data/` 包含流式清洗、BPE tokenizer 训练、tokenize、document-aware pack 和 mmap dataset。数据脚本默认读取 `configs/data_sources.yaml`，不会完整下载公开数据集。

典型流程：

```bash
uv run python data/prepare_corpus.py --sources configs/data_sources.yaml
uv run python data/split_validation.py --input data/metadata/accepted_samples.jsonl
uv run python data/train_tokenizer.py --input data/metadata/train_docs.jsonl
uv run python data/tokenize_corpus.py --input data/metadata/train_docs.jsonl --prefix train
uv run python data/pack_dataset.py --inputs data/shards/train_web_00000.bin --output data/shards/train_main_seq4096.bin --seq-len 4096
```

## 训练命令

只构建模型和优化器，不训练：

```bash
uv run python -m longcontext.train.pretrain --config configs/debug_120m_full.yaml --dry-run
```

启动有限步训练必须显式指定步数：

```bash
uv run python -m longcontext.train.pretrain --config configs/debug_120m_full.yaml --max-steps 10
```

## 评测命令

```bash
uv run python -m longcontext.eval.eval_kv_cache --config configs/main_350m_hybrid.yaml
uv run python -m longcontext.eval.eval_speed --config configs/main_350m_hybrid.yaml --lengths 1024 4096 --steps 1
uv run python -m longcontext.eval.eval_ppl --config configs/debug_120m_full.yaml --weights outputs/artifacts/debug_120m_full/weights_final.safetensors --seq-len 1024
```

## 实验结果

仓库当前未包含正式训练产物。完成训练后，结果应写入 `outputs/artifacts/` 和 `reports/`。

## 消融结论

`reports/ablation.md` 提供消融报告模板，正式结论需要由实际训练和评测 CSV 填充。

## 面试问答

核心解释路径：为什么长上下文需要压缩注意力；CSA 与 HCA 分别解决什么问题；Hybrid 为什么按层混合；mHC 如何影响 residual stream；Muon 如何优化二维矩阵参数；单卡复现与原论文规模的边界在哪里。
