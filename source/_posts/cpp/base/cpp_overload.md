---
title: cpp overload
date: 2024-01-30 
tags: cpp
---

<!--more-->
- 可以作为函数重载的标志
  - 参数类型和顺序
  - 引用和指针类型的const修饰
  
  ```c++
  void f(int& a){ }
  void f(const int& a){ }

  int a = 0;
  const int b =0;

  f(a);//f(int&)
  f(b);//f(const int&)
  ```
- 无法区分（重定义）
  - 值和值引用
  - 缺省参数   