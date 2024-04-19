---
title: cpp type
date: 2024-01-30 
tags: cpp
---
## 常量表达式
编译期可以得到计算结果，且不会改变
```cpp
const int a =10;//ok
const int b = a+1;//ok
int c = 10//error
const int d = f()//error
```
### constexpr
将变量声明为constexpr类型，可以由编译器验证变量是否是常量表达式
如果你认为它是，就把它声明为常量表达式
- constexpr函数
  - 函数返回值及所有形参类型都得是字面值
  - 隐式内联
  - 只有return语句，也可以包含空语句，类型别名，using等不执行操作的语句
  - 允许constexpr函数返回值为非常量
  ```cpp
  constexpr int f(int a){
    return 8*a;
  }//若传入的a不是常量表达式，则该函数就不是常量表达式
  ```
### 字面值类型
常量表达式需要在编译时计算，因此在表达式里面用到的简单类型叫字面值类型
- 算术类型
- 引用
- 指针
- 字面值常量类

#### 字面值常量类

#### 指针的初始值：
1. 0（nullpr）
2. 某个固定地址，如定义与函数体之外的对象
