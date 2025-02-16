# Parallel Decoding
## Blockwise Parallel Decoding for Deep Autoregressive Models
![alt text](image.png)
![](image-1.png)
假设当前序列seq 1 - j
pi i范围1-k
- predict: 使用pi模型输出第j+i个token的值
- verify: 使用p1模型使用seq(1 - j+i)验证第j+i个token的值
- accept：接受第一个不一致的token之前的token
  
性能分析，假设p的个数为k
原来需要解码m次
现在需要解码m/k * 2(predict和verify各一次)

## Speculative Decoding
![alt text](image-3.png)
![alt text](image-4.png)
- 使用小模型正常decode r个token
- 使用大模型验证概率，所有token验证通过，则生成一个新的token作为下一次迭代的开始，否则修复未通过token作为下一次迭代开始

## specinfer

## Medusa

## lookahead

## reference
[reference](https://www.53ai.com/news/finetuning/2024071109285.html)