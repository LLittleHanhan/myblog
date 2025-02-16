## 大模型推理框架
- vllm
- sglang
- tensorrt-llm
- mooncake
- lmdeploy

## 开源算子库
fast transformer
flash attention
fast infer
marlin

## 大模型推理方法
### 指标 
- [x] TTFT TPOT

### 模型结构
- [x] mla
- [x] Speculative Decoding

### 算子
- [x] flash attention
- [x] flash decoding

### 分布式并行
- [x] dp & ddp
- [x] tp
- [x] pp
- [x] sp ring attention

### 内存管理kv cache
page attentionn
Prefix Cache - RadixAttention

### 系统调度
- Continuous Batching
- Chunked Prefills
- prefill decode 分离

### 量化