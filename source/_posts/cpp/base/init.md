---
title: init
date: 2024-01-30 
tags: cpp
---
cpp中的默认初始化，值初始化，零初始化
<!--more-->
## 四种初始化形式，三种初始化
```cpp
1. = express //复制初始化
2. = {}      //列表初始化
3. ()        //直接初始化
4. {}        //列表初始化
```
### 复制初始化
1. 复制初始化中的隐式转换必须从初始化器直接生成 T，然而直接初始化可以在T的某个构造函数的实参处发生隐式转换
2. 复制初始化不如直接初始化宽容：explicit修饰
3. c++17之前，复制初始化不开优化`-fno-elide-constructors`，先构造再复制，从c++17起仅有一次构造
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
### 直接初始化
`()`
- 非类对象
- 类对象调用构造函数（注意无参构造函数A a()会产生歧义，见下面默认初始化和值初始化，直接写A a即可，或者用列表初始化）

### 列表初始化
1. 防止窄化转化
2. std::initialier_list

## 以下区分的是初始化给没给值的问题

### 默认初始化
[cpp_reference](https://zh.cppreference.com/w/cpp/language/default_initialization)
```
触发条件
1. T obj
2. new T
3. 构造函数初始化器列表中未提及某个基类或非静态数据成员,且调用该构造函数
```
效果：
1. 非类对象，不进行初始化
2. 类对象，调用构造函数
3. 数组类型，内部每个元素默认初始化
```cpp
#include <string>
struct T2{
    int mem;
    T2() {} // "mem" 不在初始化器列表中
};
 
int n; // 静态非类，进行两阶段初始化：
       // 1) 零初始化将 n 初始化为零
       // 2) 默认初始化不做任何事，令 n 保留为零
 
int main(){
    int n;            // 非类，不初始化，值不确定
    std::string s;    // 类，调用默认构造函数，值是 ""（空字符串）
    std::string a[2]; // 数组，默认初始化其各元素，值是 {"", ""}
    T2 t2;            // 调用用户提供的默认构造函数
                      // t2.mem 被默认初始化（为不确定值）
}
```

### 值初始化
[cpp_reference](https://zh.cppreference.com/w/cpp/language/value_initialization)
```
T{}
new T{}
class T{
    int mem
    T():mem{}{
    }
}
// 以上三项的{}可以替换成（）
T obj{}

// 以上的{}()是空的
```
效果：
1. T没有默认构造函数，或拥有**用户提供**的默认构造函数，则默认初始化
2. T有默认构造函数（不由用户提供），首先零初始化对象,...
3. T是数组，每个元素零初始化
4. 其他（如内置类型）都是零初始化
>用户提供user-provided:如果一个函数由用户声明且没有在它的首个声明被显式default或显式delete，那么它由用户提供

## 零初始化
[cpp_reference](https://zh.cppreference.com/w/cpp/language/zero_initialization)
条件：
1. 在所有初始化之前对静态存储期的，不进行常量初始化的变量
2. 值初始化的一部分情况
3. 不够长初始化数组的剩余部分


## 栗子
```cpp
#include <iostream>

using namespace std;

class Init1 {
   public:
    int i;
};

class Init2 {
   public:
    Init2() = default;

    int i;
};

class Init3 {
   public:
    Init3();
    int i;
};

Init3::Init3() = default;

class Init4 {
   public:
    Init4();
    int i;
};

Init4::Init4() {
    // constructor
}

class Init5 {
   public:
    Init5()
        : i{} {
    }
    int i;
};

int main(int argc, char const* argv[]) {
    Init1 ia1;
    Init1 ia2{};
    cout << "Init1: "
         << "  "
         << "i1.i: " << ia1.i << "\t"//任意值
         << "i2.i: " << ia2.i << "\n";//0

    Init2 ib1;
    Init2 ib2{};
    cout << "Init2: "
         << "  "
         << "i1.i: " << ib1.i << "\t"//任意值
         << "i2.i: " << ib2.i << "\n";//0

    Init3 ic1;
    Init3 ic2{};
    cout << "Init3: "
         << "  "
         << "i1.i: " << ic1.i << "\t"//任意值
         << "i2.i: " << ic2.i << "\n";//任意值

    Init4 id1;
    Init4 id2{};
    cout << "Init4: "
         << "  "
         << "i1.i: " << id1.i << "\t"//指定值
         << "i2.i: " << id2.i << "\n";//指定值

    Init5 ie1;
    Init5 ie2{};
    cout << "Init5: "
         << "  "
         << "i1.i: " << ie1.i << "\t"//0
         << "i2.i: " << ie2.i << "\n";//0

    return 0;
}
```

## 指定初始化
1. c标准，数组指定初始化
2. struct指定初始化