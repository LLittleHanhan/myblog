---
title: zksnarks
date: 2024-01-30 
tags: zk
---

>这篇笔记关注协议大致过程。
>ZK-SNARKS协议，是`Zero-Knowl­edge Suc­cinct Non-In­ter­ac­tive Ar­gu­ment of Knowl­edge` 的缩写，意思是简洁零知识非交互式论证。
>可以把zk-SNARKS协议看成一系列组件拼装。
>之后的工作是电路系统，其他简单看看就好。
<!--more-->
## 概述

主要参考文献：
### groth16
- 这篇文章**前半部分**多项式证明比较易懂，从R1CS->QAP开始不大行，难懂！[链接](./reference/Why%20and%20How%20zk-SNARK%20Works:%20Definitive%20Explanation.pdf),这是中文翻译[链接](https://secbit.io/blog/tags/zero-knowledge-proof/)
- 这篇文章介绍详细作为重点[链接](./reference/zkSNARK_intro.pdf)
- Vitalik Buterin的文章，R1CS->QAP解释很舒服[链接](https://medium.com/@VitalikButerin/quadratic-arithmetic-programs-from-zero-to-hero-f6d558cea649)这是中文翻译[链接](https://snowolf0620.xyz/index.php/zkp/435.html)




## Groth16
1. f->r1cs
2. r1cs->qap
3. 之后用椭圆曲线证明多项式
### groth16的算子


## plonk
## plonk的算子