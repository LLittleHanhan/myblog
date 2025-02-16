---
title: 类型转换
date: 2024-06-27
tags: cpp
---
cpp转换规则
<!--more-->
## 隐式转换
### 发生语境
T1->T2
1. 函数调用或返回时的参数传递
2. 运算符
3. if/switch
### 转换顺序
按照下面顺序进行
1. 零个或一个由标准转换规则组成的标准转换序列，叫做初始标准转换序列
2. 零个或一个由用户自定义的转换规则构成的用户定义转换序列
3. 零个或一个由标准转换规则组成的标准转换序列，叫做第二标准转换序列

标准转换规则包括限定符，数值，数组退化等转换
用户自定义包括自定义构造函数，转换函数
可以看到用户定义的隐式转换只发生一次！！！
```cpp
#include <cstring>
#include <iostream>
using namespace std;

struct B {
    B(string s) {}
};

struct A {
    A(B b) {}
};

int main() {
    string s{"hello"};
    // B b("hello");  // ok
    // B b = "hello";  // error
    // B b = s;  // ok

    A a("hello");//error
}
// 直接初始化相当于调用构造函数，构造函数内部可以做一次自定义的隐式转换
// 复制初始化相当于直接隐式转换
```
## 基本类型的转换
1. 范围
2. 精度

int->float:范围安全，精度损失
> float的有效位是24，在2的24次方范围内精度无损。float a{11111};可以防止窄化转化

float->int:范围不安全
## static_cast
1. 引用和指针
   1. 转换继承关系的类指针(向上和向下都可以，不报错)
   2. void*和其他指针的转换
2. 非引用和指针
   1. 内置类型
   2. 继承关系的类之间，这里的转换相当于调用构造函数。
```cpp
// 使用static_cast对继承类的向下类型转换
#include <iostream>
using namespace std;
class A {
   public:
    A() {
    }
};
class B : public A {
   public:
    int b = 2;
    B() {
    }
    void f() {
        cout << "in B f" << endl;
    }
};
int main() {
    A* aa = new A();
    B* bb = static_cast<B*>(aa);
    cout << bb->b << endl; // 0 访问非法地址，但是不报错
    bb->f();//正常
}
```
```cpp
// 使用dynamic_cast向下类型转换，要求有虚函数
// ps：向上类型转换就算不是虚类也没关系
#include <iostream>
using namespace std;

class A {
   public:
    A() {
    }
    virtual void f() {
        cout << "in A f" << endl;
    }
};
class B : public A {
   public:
    int b = 2;
    B() {
    }
    void f() override {
        cout << "in B f" << endl;
    }
};

int main() {
    // bb->f();

    B* b = new B();
    A* a = dynamic_cast<A*>(b);
    B* from_a = dynamic_cast<B*>(a);
    a->f();
    cout << from_a << endl;

    A* aa = new A();
    B* bb = dynamic_cast<B*>(aa);
    cout << bb << endl;
    cout << bb->b << endl;
    bb->f();
}
```


上述转换在运行期可能出错
## reinterpret_cast
对内存区域的重新解释，不改变内存的bit位
1. 指针到指针
2. 数值到指针

## dynamic_cast
略

## const_cast
去const
## 转换时的问题
1. 类型转换后为右值，尤其注意在隐式类型转换时
2. 数组退化int[100]->int*
## 参考
[彻底理解c++的隐式类型转换](https://www.cnblogs.com/apocelipes/p/14415033.html)