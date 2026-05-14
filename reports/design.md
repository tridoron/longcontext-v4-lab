# Design

## 1. 为什么选择长上下文效率作为项目主线

长上下文模型的主要瓶颈来自注意力计算和 KV cache 随序列长度线性或平方增长。CSA/HCA 类结构可以把远程上下文压缩到更短 memory，同时保留局部窗口。

## 2. 为什么不能完整复现 DeepSeek-V4

原论文系统涉及超大规模模型、私有数据、复杂并行和底层 kernel。本项目面向单卡工程复现，只实现可训练的简化机制。

## 3. 单卡 RTX 5090 下的模型规模设计

Debug Model 约 120M 参数，用于验证训练链路。Main Model 约 350M 参数，用于结构消融和最终长上下文继续训练。

## 4. CSA-lite 的实现细节

CSA-lite 使用独立 indexer 投影，按 block 压缩 K/V，top-k 选择 compressed memory，并与 gather local window 拼接后统一 attention。

## 5. HCA-lite 的实现细节

HCA-lite 使用更大 block size，不使用 indexer，对合法历史压缩块做 dense compressed attention。

## 6. Hybrid Attention 的层间调度

前两层使用 SWA；之后每 3 层中 1 层 HCA、2 层 CSA。

## 7. mHC 的数学形式和工程实现

Static-mHC 将 residual stream 扩展为 `n_hc` 条，使用 `A/B/C` 参数控制输入混合、状态传递和更新注入。

## 8. Muon 的参数分组策略

二维投影矩阵进入 Muon；embedding、lm_head、norm、bias、mHC raw、`A_final_raw`、`beta_raw` 和一维参数进入 AdamW。

## 9. KV cache 估算方法

按每层实际保留 token 数估算压缩结构 KV cache，并与 dense full attention KV cache 对比。

## 10. 训练稳定性处理

代码支持 gradient checkpointing、梯度裁剪、bf16/fp16 autocast、checkpoint 保存和显式步数训练。

## 11. 数据过滤与 validation set 构建

数据脚本实现文本标准化、doc hash、拒绝原因记录、代码语言白名单和固定长度 pack。

## 12. Checkpoint 与磁盘空间控制

权重使用 safetensors；完整训练状态默认仅保留 `outputs/active/state_latest.pt` 的规则由训练脚本支持。
