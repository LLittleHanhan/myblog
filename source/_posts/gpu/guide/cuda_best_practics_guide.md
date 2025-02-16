---
title: cuda c best practics guide
date: 2024-01-30 
tags: gpu
---
cuda官方文档
<!--more-->

## memory optimizations

### data transfer between host and device

### global memory and l2 cache
> High Priority: Ensure global memory accesses are coalesced whenever possible.


1. For devices of compute capability 6.0 or higher, the requirements can be summarized quite easily: the concurrent accesses of the threads of a warp will coalesce into a number of transactions equal to **the number of 32-byte transactions necessary** to service all of the threads of the warp.
2. Memory allocated through the CUDA Runtime API, such as via cudaMalloc(), is guaranteed to be aligned to at least **256 bytes**.
[链接](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
3. 
对于cc6以上，一个内存事务就是32字节


## Execution Configuration Optimizations

## Instruction Optimization

## Control Flow