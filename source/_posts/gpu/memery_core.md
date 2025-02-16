---
title: GPU memery core
date: 2024-01-30 
tags: gpu
---
本篇是《通用图形处理器设计GPGPI编程模型与架构原理》的存储架构
![](pic/gpgpumemery.png)
<!--more-->
## 寄存器文件
- 寄存器文件将32个线程的标量寄存器打包成一个线程束寄存器
### 并行多板块
- 例如一条乘加指令需要读三个线程束寄存器，写一个线程束寄存器，读写操作最好在一个周期完成
- gpgpu寄存器文件会采用多个板块（bank）的单端口模拟多端口访问

无板块冲突的过程：
1. 译码指令存入流水线寄存器，以及线程束WID
2. 仲裁器根据WID和寄存器编号，打开相应位置的板块
3. 读取后经过交叉开关将数据传输到下一流水线寄存器
4. 所有数据准备好后将指令送入执行单元
5. 执行完成后写回

有冲突时，延迟到下一个周期
> 同一条指令的读操作可能产生板块冲突，不同指令之间的读写也可能冲突
### 操作数收集器
![](pic/gpgpucollector.png)
这就相当于在流水线中间放了一个缓存箱
将解码的指令放入空闲的收集单元，同时指令的读写请求会加入到仲裁器中的某个板块的请求队列中。仲裁器最多对4个请求（不同板块）处理，将数据发送给收集器
当一个收集器单元所有操作数准备好后，就可以发送给执行单元
> 按上述，不同的线程束可以同时访问寄存器文件只要板块不冲突即可
> 具体的时钟周期没讲明白，不过目前不是很重要？                                         
### 板块交错分布
原
| bank1 | bank2 | bank3 | bank4 |
| :---: | :---: | :---: | :---: |
|  w0   |  w1   |  w2   |  w3   |
|  v0   |  v1   |  v2   |  v3   |

交错
| bank1 | bank2 | bank3 | bank4 |
| :---: | :---: | :---: | :---: |
|  w0   |  w1   |  w2   |  w3   |
|  v1   |  v2   |  v3   |  v0   |
### 数据相关性
采用数据收集器可能会产生数据冲突，发射有顺序，但是到收集器后具体谁先取完数谁后取完数就无法控制


## L1缓存和共享存储器
![](pic/gpgpureg.png)
L1缓存和共享内存共用一套结构，编程人员可以手动调整两者的大小
### 共享存储器
共享存储器分32板块，一个板块一个单元32bit（4B）
无冲突：一个线程束32和线程的访问地址落在32个板块上或者指向同一板块的同一位置
有冲突：不同线程访问地址在同一板块的不同地址

#### 无冲突的访问过程
1. 共享内存取数指令发射到load/store单元
2. load/store单元会识别请求和判断地址信息，识别出是共享内存请求，地址无板块冲突，之后交给仲裁器，绕过tag unit
3. 如果是读请求，load/store会同时给寄存器文件调度一个写操作
4. 如果是写请求，会将待写入数据写入写缓冲，之后再写入SRAM阵列
#### 有冲突的访问
把请求分为不冲突的多部分，第一部分正常请求，其他部分采用重播
1. 可以退回到指令缓存中重播，缺点占用了指令缓存，需要重新计算地址
2. 可以在load/store单元单独设置缓存空间重播
#### 共享存储器加载的数据通路
共享存储器最初的数据怎么加载？直接看图

### L1缓存
#### cpu中的cache
- 映射：全相连，直接映像，组相连
- 替换算法：FIFO，LRU
- 写策略：写直达，写回法
- 写不命中：写分配

#### 读操作
1. 全局内存访问指令发射到load/store单元
2. load/store单元会识别请求和判断地址信息，拆分合并地址请求，之后交给仲裁器，同时给寄存器文件调度一个写操作
3. 仲裁器检测tag是否命中，如果命中直接从板块中读取
4. 如果不命中，并将请求写入mshr单元
> MSHR
> On a cache hit, a request will be served by sending data to the register ﬁle immediately. On a cache miss, the miss handling logic will ﬁrst check the miss status holding register (MSHR) to see if the same request is currently pending from prior ones. If so, this request will be merged into the same entry and no new data request needs to be issued. Otherwise, a new MSHR entry and cache line will be reserved for this data request. A cache status handler may fail on resource unavailability events such as when there are no free MSHR entries, all cache blocks in that set have been reserved but still haven’t been ﬁlled, the miss queue is full, etc.
> cahce不命中会把请求写入mshr单元，mshr单元会进行地址合并减少下一层次访存
1. mshr单元处理后发送到mmu单元，mmu单元进行虚实地址转换读取数据并返回，1.通过fill unit填入最后写入cache（可能有涉及cache替换），并锁定这一行在被读取前不能被替换，2.告知load/store重播该指令
#### 写操作
详细过程不写了，有几个点
1. 缓存行要能够部分写
2. 写命中的策略：
   - 对于局部存储器（寄存器溢出部分），因为是线程私有，一般不会产生一致性问题，因此可以采用写回法
   - 对于全局存储器可以采用写逐出，写入L2缓存，同时L1缓存置为无效

## 全局存储器
全局存储器的地址合并要求在DRAM中连续
- 地址对齐
- 合并访问

按照cpu缓存理解，访存指令先访问cache，cache未命中，先取数，之后指令重放，若启用L1,粒度128B，若不启用L1,粒度32B