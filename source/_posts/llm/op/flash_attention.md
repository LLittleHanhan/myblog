# flash attention

# 性能分析
4090
- 512 tensorcores 
- 128 flop per cycle
- 2.52 cycle/ns
- tensor fp16 算力 165.2 * 10^12 flops
input
- bs = 256
- head = 12
- head_dim = 64
- seq = 1024
- time = 5.89ms
```
flop = 1024 * 1024 * 64 * 2 + 1024 * 64 * 1024 * 2 = 8.24 * 10^11
max_flop = 165.2 * 10^12 * 5.89 * 10^-3 = 9.73 * 10^11
~84%
```
