---
title: cuda c编程权威指南
date: 2024-01-30 
tags: gpu
---
这篇是cuda c编程权威指南的学习笔记，书比较老，后续计划看官方文档和项目
<!--more-->
## 编程和执行模型
- 人的角度：一个kernel函数对应一个grid，一个grid中有多个block，一个block中分多个thread
- 硬件角度：一个block分配在一个sm上，在block划分多个warp，warp之间抢占计算核心，warp是调度的基本单位
- 同步：warp天然同步，block内的warp可以使用`__syncthreads()`同步，block间没有同步机制
- 分支，wrap分化
    > **SIMD**：单指令多数据的执行属于向量机，这种机制的问题就是过于死板，不允许每个分支有不同的操作，所有分支必须同时执行相同的指令。
    > **SIMT**：给所有thread广播指令，但不要求必须执行。
    > **分支**：wrap遇到分支时所有thread都把if和else全部执行（近似）![Alt text](pic/gpgpubranch.png)
    > **优化思路**手动分配thread到不同的wrap
- sm会给block分配一套寄存器，因此上下文切换开销很少
- 多个warp抢占sm的core以充分利用计算资源：延迟隐藏
- tips
  - 首先要考虑到线程块这一级别，确定一个线程块处理那些数据块，因为线程块内的线程可以同步
  - 其次考虑线程束分化问题
  - 主机和设备是并行的，`cudaDeviceReset()`同步主机和设备

## 内存
这部分需要明确什么是我可以手动设置的，什么是透明的
先不管常量和纹理

|   种类   | 存储位置 | 声明位置 |    标识符    |  RW   |    作用域     |   生命周期    |
| :------: | :------: | :------: | :----------: | :---: | :-----------: | :-----------: |
|  寄存器  |   片上   |  核函数  |      ——      |  RW   |     线程      |     线程      |
| 本地内存 |   片外   |  核函数  |      ——      |  RW   |     线程      |     线程      |
| 共享内存 |   片上   |    ？    | `__shared__` |  RW   |    线程块     |    线程块     |
| 全局内存 |   片外   |   全局   | `__device__` |  RW   | 所有线程+主机 | 所有线程+主机 |

- 缓存
  - 一级：和共享内存处于同一位置，可以手动划分两者大小`cudaFuncSetCacheConfig()`
  - 二级

### 全局内存
> 主机代码不能直接访问设备变量，必须调用cuda API**获取真正的内存地址**
#### 编程方法
 ```c++
//全局变量
//静态
int a = 9;
__device__ int ga = 0;
__device__ int gb;//要么直接初始化，要么调用cudaMemcpytoSymbol
//cudaMemcpytoSymbol(const void* symbol,const void* src,size_t count)
cudaMemcpytoSymbol(gb,&a,sizeof(int))//对于主机来说，gb只是个标识符，而不是一个内存地址
gb = a;//error
cudaMemcpy(&gb,&a,sizeof(int),cudaMemcpyHostToDevice); // error
//一定要用cudaMemcpy的话，可以用下面的方法
int *temp = NULL;
cudaGetSymbolAddress(&temp,gb)
cudaMemcpy(&temp,&a,sizeof(int),cudaMemcpyHostToDevice);

//动态
int* gc = NULL;
cudaMalloc(&gc,sizeof(int));//这里传递的是指针的地址，即在gpu全局内存处申请一块空间，将该空间地址赋值给gc，也就是说gc指向gpu中的一块内存
cudaMemcpy();
cudaFree(gc);
```
#### 固定内存`cudaMallocHost()`
主机中操作系统的内存管理机制，虚拟内存+内存分页，进程的内存页可能会换出
> CUDA的驱动程序检查内存范围是否被锁定，然后它将使用不同的代码路径。锁定的内存存储在物理内存 (RAM) 中，因此设备可以在没有CPU帮助的情况下获取它（DMA，也称为异步副本；设备只需要物理页面列表）。非锁定内存在访问时会产生页面错误，并且它不仅存储在内存中（例如它可以在交换中），因此驱动程序需要访问非锁定内存的每一页，将其**复制到固定缓冲区**并传递到 DMA（同步，逐页复制）

#### 零拷贝内存`cudaHostAlloc()`
```c++
cudaError_t cudaHostAlloc(void ** pHost,size_t count,unsigned int flags)
//flag
//cudaHostAllocDefalt 同cudaMallocHost()
//cudaHostAllocPortable
//cudaHostAllocWriteCombined
//cudaHostAllocMapped 零拷贝

// 设备不能直接使用pHost，需要转化为pDevice
cudaError_t cudaHostGetDevicePointer(void ** pDevice,void * pHost,unsigned flags);//UVA出现后就没用了
```
- 零拷贝的是cpu固定内存
- 通过PCI-e远程访问，不再需要使用memcopy

#### 统一虚拟寻址UVA cuda2.0
UVA提出了零拷贝内存，即设备可以直接访问主机内存（单向），是否一定要是零拷贝？
> UVA启用了zero-copy技术，在CPU端分配内存，将CUDA VA映射上去，通过PCI-E进行每个操作。而且注意，UVA永远不会为你进行内存迁移。
#### 统一内存寻址 cuda6.0
```c++
cudaError_t cudaMallocManaged(void ** devPtr,size_t size,unsigned int flags=0)
```
Unified memory在程序员的视角中，维护了一个统一的内存池，在CPU与GPU中共享。使用了单一指针进行托管内存，由系统来自动地进行内存迁移。


以上参考
[文章一](https://zhuanlan.zhihu.com/p/82651065)
[文章二](https://www.cnblogs.com/maomaozi/p/16175725.html)
[文章三](https://forums.developer.nvidia.com/t/unified-virtual-addressing-uva-vs-unified-memory-perceived-difference/72399)

## 共享内存

