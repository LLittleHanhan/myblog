---
title: commitment
date: 2024-01-30 
tags: zk
---

>承诺：是对一个既有的确定性的事实（敏感数据）进行陈诉，保证未来的某个时间有验证方可以验证承诺的真假，也就是说承诺当前时间的事实，未来不会发生变化。
<!--more-->
承诺的两个阶段：
1. commit：对敏感数据m计算承诺c并公开
2. reveal：公开m验证

两个特性
1. hiding：reveal前不能verifier不能知道m
2. binding：proofer在commit后不能更改m


## 哈希承诺
- commit: `c=hash(m)`
- reveal: proofer公开m，验证`c==hash(m)`


## Pedersen承诺
- commit: `c=r*G+m*H` G,H为特定椭圆曲线上的生成点，r是盲因子，m是原始信息
- reveal：proofer公开r，m，验证`c=r*G+v*H`

## 多项式承诺
怎么确定一个唯一的多项式
1. 系数
2. n阶，提供n+1个点值对
> FFT和IFFT提供这两种方法的转化

全部打开：
- commit：任取一点r计算`c=f(r)`
- reveal：proofer公开r和c，验证`c=f(r)`

部分打开:
- commit：任取一点r计算`c=f(r)`
- reveal: 
      1. proofer公开r，c
      2. verifer随机选择z
      3. proofer计算`s=f(z) t(x)=(f(x)-s)/(x-z) w=t(r)`发送s，w
      4. verifer验证`f(r)-s==t(r)(r-z)`即`c-s==w(r-z)`
- 这种方法没有暴露最初的多项式
- 但是有问题，原因在于verfier公开的z，因此要在加密空间上运算

## kate承诺
1. setup
   - 选择椭圆曲线生成元$G$，配对函数$e$，随机值$\alpha$
   - 计算{$g,g^\alpha,g^{\alpha^2}...g^{\alpha^t}$}
   - 丢弃$\alpha$
2. commit
   - 令要承诺的多项式为$f(x)=\displaystyle\sum^{j}_{i=0}{a_ix^i}$
   - 计算在$\alpha$点的值$c=\displaystyle\prod^{j}_{i=0}{(g^{\alpha^i})^{a_i}}$
- open reveal
   - 给出原始多项式$f(x)$，verifer验证
- create witness
   - 给定$m$，创造出一个新的多项式$g(x)=\frac{f(x)-f(m)}{x-m}$
> 可以注意到$f(x)-f(m)$具有$x=m$的根，问题转化为proofer知道一个多项式可以被整除


