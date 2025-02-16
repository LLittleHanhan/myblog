---
title: x86的SIMD
date: 2024-08-22
tags: arch
---
## development
[reference](https://www.cnblogs.com/TaigaCon/p/7835340.html)
[reference](https://www.cnblogs.com/moonzzz/p/17806496.html)
1. mmx,使用st浮点寄存器的64bit，只能处理整型
> emms,movd,movq,p*
> st寄存器：x87架构用于浮点计算8个80bit
1. sse，使用xmm寄存器128bit和32-bit的控制寄存器MXCSR
2. 