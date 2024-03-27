---
title: cpp oopVirtual
date: 2024-01-30 
tags: cpp
---

## 静态多态
1. 模板
2. 函数重载
<!--more-->
## 动态多态
### 编联
- 静态编联，编译期决定调用
- 动态编联
```c++
//静态编联
#include <iostream>
#include <typeinfo>
using namespace std;

class Shape {
   public:
    void Show() const { cout << "面积是 :" << Area() << endl; }
    float Area() const { return 0; }
};

class Rectangle : public Shape {
   public:
    Rectangle(float w, float h) {
        mWidth = w;
        mHeight = h;
    }
    float Area() const { return mWidth * mHeight; }

   private:
    float mWidth, mHeight;
};

int main() {
    Rectangle rect(1, 2);
    rect.Show();//0
}
```
### 继承下的类型转换

### 虚函数
#### 虚函数和析构
#### 虚函数
- 虚函数表
  - 指针数组，指向函数入口地址
  - 一个类只有一个，在首次实例化创建
  - 顺序，前半部分和基类对应（为了方便寻找
- 机制
  - vptr是一个指针，指向虚函数表
![](../../pic/oop2virtual-1.png)     