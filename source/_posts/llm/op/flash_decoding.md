# flash decoding
![alt text](flash_decoding-1.png)
背景：在decode阶段，bs和head的维度小，并行性不足，因此考虑在kv的方向（N）做并行
方案：思路很简单，kv分块，最后再做reduce
# flash decoding++
提前预估一个max，避免了放缩
# reference 
[flash decoding](https://zhuanlan.zhihu.com/p/696075602)