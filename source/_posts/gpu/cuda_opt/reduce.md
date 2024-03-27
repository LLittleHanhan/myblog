---
title: cuda reduce
date: 2024-03-26 
tags: gpu
---
reduce本质是一组向量运算，满足交换律和结合律，比如一组向量的和、一组向量的最大值
![](pic/reduce-1.png)
<!--more-->
具体思路如上图所示，每个线程块处理一组数据，之后将每个线程块的结果拿出来，再由一个线程块处理。
- op5之前是nv的官方教程，有些过时，看看思路
- op6使用shuffle指令，这个是重点

参考文章：
[官方教程](./reduction.pdf)
[有了琦琦的棍子](https://zhuanlan.zhihu.com/p/426978026)
[BBuf](https://zhuanlan.zhihu.com/p/596012674)
[zzk](https://mp.weixin.qq.com/s?src=11&timestamp=1711529201&ver=5163&signature=4keLvSGMDuzYkg8qdEcEHcPOvaSSg4sX*BTfecKXO8rzwlkdm3KcMJuxNmC*F1Wxu4OzaMwQOnrXoCdNOqfaFmoISMp0kpPYNiYflu4HLice2Iu*nqPV3wnvPSZt3V7F&new=1)
[用一个函数来得到最终的归约值](https://zhuanlan.zhihu.com/p/635456406)
## baseline
![](pic/reduce-2.png)
基本思想：
- 加速存取过程，将向量存储到共享内存中去,这样整个reduce过程以线程块为单位

## op1：减少线程束分化
在baseline的图中，
第一次迭代，t0、t2、t4、t6...
第二次迭代，t0、t4、t8...
一个线程束内的线程并未同步执行，可以这样
![](pic/reduce-3.png)
这样会造成后边的线程啥都没干，op4解决

## op2：解决板块冲突
在op1的图中，以一个warp内为例
第一次迭代，t0访问共享内存0（bank0），1（bank1）...t16访问共享内存32（bank0），33（bank1）...二路
第二次迭代，t0访问共享内存0（bank0），2（bank2）...t8访问共享内存32（bank0），34（bank2）...四路
会发现，随着迭代次数的增加，冲突越来越多
解决方法，for循环反着写，简单画画就明白了
> cuda c编程权威指南中相邻配对->交错配对
## op3：idle线程
op1中，后半线程啥都没干，直接取消就行
也就是说，假设原来一个块256个线程，处理256个数据，现在只需要128个线程即可
在参考文章中，是让256个线程先把**两个256的数据块相加**，之后用前128个线程处理256数据块
> 按照cuda c编程权威指南中是展开循环的优化

总之，结果还是让后128个线程有活干，虽然只做了一次加法

## op4：展开最后一个线程束
当进行到最后几轮迭代时，此时的block中只有warp0在干活时，线程还在进行同步操作
> 有大问题，计算能力7.0开始，warp中的线程是有可能不同步的了，若步调不一致，则计算结果会出现错误。
> 后边还有把for完全循环展开的优化，没必要！！！

## op5：设置合理的block数量
怎么设置O:O?

## op6：使用shuffle指令
https://forums.developer.nvidia.com/t/what-does-mask-mean-in-warp-shuffle-functions-shfl-sync/67697
https://zhuanlan.zhihu.com/p/572820783
## Q
1. 精度问题：大量的浮点数加法cpu和gpu运算结果差距太大
2. gpu整形除法和取模操作成本很高
3. 架构问题：https://zhengqm.github.io/blog/2018/12/07/cuda-nvcc-tips.html
4. 线程束同步问题:https://blog.csdn.net/weixin_42730667/article/details/109838089
5. shuffle指令
6. 测试问题，大数据量，多次？ 带宽怎么算的？
## 评论区
> 大佬您好，这边再跟您讨论一点比较琐碎的东西，希望您发表一下看法。
在Volta架构使用Independent Thread Scheduling以后，由于每个线程都有自己的PC和stack，所以不能随意的假设warp threads在什么时候收敛，什么时候是没有分支的；不能随意的假设代码是lock-step执行的，假设warp内线程的写操作一定会被其他“看见”，这样不安全。在任何情况下，warp内某一线程或线程组完全可能比别的线程或线程组推进的更快。
所以，在官方给出的关于Volta新特性的例子中，关于reduction的写法，采用了利用寄存器将读写完全分离，并用__syncthread确保同步的方法，可以参考链接：[developer.nvidia.com/bl](http://link.zhihu.com/?target=https%3A//developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
官方说不安全与安全的方法我都实验了，得出的结果都是正确的，不过性能都比使用volatile的写法稍差。不过站在官方的角度，使用volatile的写法也是不安全的，可以参考链接：[forums.developer.nvidia.com](http://link.zhihu.com/?target=https%3A//forums.developer.nvidia.com/t/why-syncwarp-is-necessary-in-undivergent-warp-reduction/209893)
我可以想象这样一种情况：对于语句cache[tid] += cache[tid+8]，包含了对shared memory的读和写。但是不能保证所有读操作完成后才进行写操作。例如，tid=0的线程，cache[0]的更新依赖于对cache[8] 的读操作；但是由于tid=8的线程可能推进的更快，即实际上warp内的线程是不同步，存在分支的，cache[8]的更新(写操作)：cache[8] += cache[16] 可能先进行，导致tid=0的线程读到的是错误的数据。
虽然大多数时候线程和warp的执行确实是按照我们所想的那样在何处收敛或者lock-step，但是尤其是Volta以后，这实际上并不是cuda所保证的。我想请问您，您认为是不是只要结果正确，可以不考虑这种所谓的“安全”呢？