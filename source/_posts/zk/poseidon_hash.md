---
title: poseidon hash
date: 2024-01-30 
tags: zk
---

<!--more-->
[poseidon的原始论文]()
[hedes的原始论文](pdf/hades.pdf)
## Merkel-Damgard & sponge function
- MD
md5和sha256就是这个结构
![](pic/poseidon-MD.png)
- Sponge
这个是sha3算法（keccak算法）提出的新型结构
![](pic/poseidon-Sponge.png)
## Substitution-Permutation Network & Hades
- 代换置换网络，可以回顾aes
SPN网络通常由以下几个关键部分组成：
- 代换层（Substitution Layer）：这一部分通常使用S盒（Substitution Box）来进行代换操作。S盒是一个非线性的函数，将输入的比特序列映射到输出比特序列.
- 置换层（Permutation Layer）：这一部分涉及对数据进行比特置换操作，通常包括比特的置换、轮换或排列等操作。
- 轮函数（Round Function）：SPN网络通常包括多个轮次，每一轮都会对数据进行代换和置换操作。轮函数是每个轮次中所使用的代换和置换操作的组合。

## Poseidon hash
![](pic/poseidon.webp)
