---
title: cs260（一）cost model and scan 
date: 2024-05-30
tags: parallel
---
cs260并行算法

<!--more-->

## cost model
`cost = W/P + D`
- W是该并行算法的计算量
- P是并行处理器的个数
- D是深度，即假设处理器无限多，该问题最少需要的轮数
公式的意义：假设P的数量无限多，那算法所需要的时间主要受D影响，如果P的数量为1,那么算法的时间和问题W的规模相关
实际问题中，若在cpu上进行计算，则P的个数较小，则应该主要关注算法的W，若在gpu上进行计算，则主要关注D

## scan
这里描述scan算法应用了两个思想
1. 分治策略（递归，自顶向下）
2. 缩小问题规模（循环，自底向上）
## fork-join模型