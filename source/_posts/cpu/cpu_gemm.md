---
title: cpu gemm
date: 2024-01-30 
tags: cpu
---
<!--more-->

## 技巧
### 调换顺序
`C(i,j)+=A(i,l)*B(l,j)`
把j循环放最后！
### 循环展开
```c++
for (int l = 0; l < k; l += 4)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
            }
        }
```
展开最外层的循环
### 利用寄存器
```c++
for (int l = 0; l < k; l += 4)
        for (int i = 0; i < m; i++) {
            double reg0 = A(i, l);
            double reg1 = A(i, l + 1);
            double reg2 = A(i, l + 2);
            double reg3 = A(i, l + 3);
            for (int j = 0; j < n; j++) {
                C(i, j) += reg0 * B(l, j);
                C(i, j) += reg1 * B(l + 1, j);
                C(i, j) += reg2 * B(l + 2, j);
                C(i, j) += reg3 * B(l + 3, j);
            }
        }
```
c++17 废除了register关键字，这里设置普通double变量，编译器会把其分配到寄存器中
### 减少乘法

## SIMD指令集
1. 内联汇编&编译器intrinsic函数

cmake
benchmark
simd avx
gemm