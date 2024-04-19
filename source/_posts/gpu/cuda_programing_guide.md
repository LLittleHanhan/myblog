---
title: CUDA C++ Programming Guide
date: 2024-01-30 
tags: gpu
---
本篇记录CUDA C++ Programming Guide,以新的东西为主
大致看下来重要的是2，4，5，
<!--more-->
## 2.Programming Modle
> 两个新的东西，thread block cluster和Asynchronous SIMT Programming Model，这是新的架构提出的
### thread hierarchy
- thread blcok clusters
![](pic/cuda_programing_guide.png)
> With the introduction of **NVIDIA Compute Capability 9.0**, the CUDA programming model introduces an optional level of hierarchy called Thread Block Clusters that are made up of thread blocks. Similar to how threads in a thread block are guaranteed to be co-scheduled on a streaming multiprocessor, thread blocks in a cluster are also guaranteed to be co-scheduled on a GPU Processing Cluster (GPC) in the GPU.

暂时没接触到，啥是GPC，TPC？
[以H100，讲述了GPC的作用，但TPC呢？](https://loop.houmin.site/context/gpu-arch/)
### memory hierarchy
这个没啥东西
### Heterogeneous Programming
- Unified Memory
统一内存
### Asynchronous SIMT Programming Model
Starting with devices based on the NVIDIA Ampere GPU architecture, the CUDA programming model provides acceleration to memory operations via the asynchronous programming model
这个也不是很理解，用到再说mark
### Compute Capability
简单记录一下
H100 Hopper 9.0
A100 Ampere 8.0
T4   Turing 7.5
V100 Volta  7.0
P100 Pascal 6.0
Maxwell,Kepler,Fermi


## 3.Programming Interface
### nvcc
c/c++和ptx通过nvcc编译成二进制代码sass
[详细过程](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
### cuda runtime
#### device memory
- linear memory :Linear memory is allocated in a single unified address space
- cuda arrays :CUDA arrays are opaque memory layouts optimized for texture fetching
- cudamallocpitch cudamalloc3D
#### l2 cache
> Starting with CUDA 11.0, devices of compute capability 8.0 and above have the capability to influence persistence of data in the L2 cache, potentially providing higher bandwidth and lower latency accesses to global memory