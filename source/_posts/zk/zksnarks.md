---
title: zksnarks
date: 2024-01-30 
tags: zk
---

<!--more-->
## 概述
>这篇笔记关注协议大致过程。
>ZK-SNARKS协议，是`Zero-Knowl­edge Suc­cinct Non-In­ter­ac­tive Ar­gu­ment of Knowl­edge` 的缩写，意思是简洁零知识非交互式论证。
>可以把zk-SNARKS协议看成一系列组件拼装。
>之后的工作是f->计算电路->R1CS，其他简单看看就好。

主要参考文献：
- 这篇文章**前半部分**多项式证明比较易懂，从R1CS->QAP开始不大行，难懂！[链接](./reference/Why%20and%20How%20zk-SNARK%20Works:%20Definitive%20Explanation.pdf),这是中文翻译[链接](https://secbit.io/blog/tags/zero-knowledge-proof/)
- 这篇文章介绍详细作为重点[链接](./reference/zkSNARK_intro.pdf)
- Vitalik Buterin的文章，R1CS->QAP解释很舒服[链接](https://medium.com/@VitalikButerin/quadratic-arithmetic-programs-from-zero-to-hero-f6d558cea649)这是中文翻译[链接](https://snowolf0620.xyz/index.php/zkp/435.html)
- zcash官方科普文，这个简单看看作为补充[链接](https://electriccoin.co/blog/snark-explain/)这是中文翻译[链接](https://blog.csdn.net/u010088996/article/details/96499169?spm=1001.2014.3001.5502)


## 组件
### 多项式证明

### 同态加密-多项式盲估

### 椭圆曲线

### 系数知识测试（KCA）

### 计算电路->R1CS->QAP

## Pinocchio

## Groth16
