---
title: rust(一)基本类型
date: 2024-03-01 
tags: rust
---

本篇介绍rust基本变量类型
<!--more-->

## 数值类型

### 整数
- i8 -> i32
- u8 -> u32
- isize usize
- 使用 `wrapping_*` 方法在所有模式下都按照补码循环溢出规则处理，例如 `wrapping_add`
如果使用 `checked_*` 方法时发生溢出，则返回 None 值
使用 `overflowing_*` 方法返回该值和一个指示是否存在溢出的布尔值
使用 `saturating_*` 方法，可以限定计算后的结果不超过目标类型的最大值或低于最小值
### 浮点
- f32 f64
- 浮点数没实现std::cmp::Eq，而且会存在精度问题，不能直接判等
### NaN
`is_nan()`方法
### 字符
unicode编码，4B
### 布尔
1B
### 单元
`()`
### 序列
用于循环
`1..5`
`'a'..'z'`