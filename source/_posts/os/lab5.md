---
title: lab5
date: 2024-12-13
tags: os.mit6.828
---
lab5 文件系统
没做实验，先搞清除几个概念
<!--more-->
1. 文件在内存上是如何组织的
   1. 连续分配记录起始块和大小
   2. 链表，fat表
   3. 索引，多级索引
2. 空闲块是如何组织的
   1. 位图
   2. 空闲表
3. inode 和 目录
4. 文件描述符
5. read write系统调用
   操作系统先读取到缓冲区，之后拷贝到用户buffer
6. mmap内存映射文件
7. vma管理虚拟内存
8. 管道

# reference
![fd](https://yushuaige.github.io/2020/08/14/%E5%BD%BB%E5%BA%95%E5%BC%84%E6%87%82%20Linux%20%E4%B8%8B%E7%9A%84%E6%96%87%E4%BB%B6%E6%8F%8F%E8%BF%B0%E7%AC%A6%EF%BC%88fd%EF%BC%89/)
![mmap](https://www.cnblogs.com/binlovetech/p/17712761.html)
![](https://www.cnblogs.com/xiaolincoding/p/13499209.html)