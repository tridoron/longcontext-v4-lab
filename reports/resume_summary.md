# Resume Summary

## 一句话版本

基于 PyTorch 实现 DeepSeek-V4 风格的长上下文 LM 训练框架，覆盖 CSA/HCA 混合压缩注意力、Static-mHC、Muon 优化器和长上下文评测。

## 简历项目经历版本

DeepSeek-V4 Inspired Long-Context LM：基于混合压缩注意力的长上下文语言模型训练与优化

参考 DeepSeek-V4 的 CSA/HCA 混合压缩注意力机制，基于 PyTorch 实现 Decoder-only 长上下文语言模型训练框架。项目中设计并实现 Token-level KV Compressor、Compressed Sparse Attention、Heavily Compressed Attention、Sliding Window Branch、Hybrid Layer Scheduler、Static-mHC 残差增强和 Muon 参数分组优化器，在 350M 参数模型上完成 1B tokens 主训练和 8K/16K 长上下文继续训练。

构建 PPL、Passkey Retrieval、Needle-in-a-Haystack、tokens/s、峰值显存和 KV cache size 等评测流程，对 Full Attention、Sliding Window Attention、CSA-only、HCA-only、CSA/HCA Hybrid、Hybrid+mHC、Hybrid+mHC+Muon 进行系统消融。实验分析混合压缩注意力在长上下文场景下对 KV cache 占用、推理速度和长程依赖召回能力的影响，并形成完整训练日志、消融报告和工程复现文档。

## 面试口述版本

我把 DeepSeek-V4 中和长上下文效率相关的机制裁剪成单卡可训练版本：底层是 Decoder-only LM，上层实现 Full/SWA/CSA/HCA/Hybrid 五种 attention，并加入 Static-mHC 和 Muon 参数分组。评测覆盖 PPL、Passkey、Needle、速度、显存和 KV cache。项目重点不是复现原论文规模，而是证明能把论文机制拆解成可运行、可消融、可解释的工程系统。
