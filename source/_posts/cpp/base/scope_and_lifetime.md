---
title: scope and lifetime
date: 2024-01-30 
tags: cpp
---

<!--more-->
## 分区
## static 和 extern
- 静态无链接：函数内使用static
- 静态内链接：函数外使用static
- 静态外链接：函数外直接声明定义
```c++
//a.cpp
static int a = 1;
int b = 2;
void f(){
    static int c = 3;
}

//b.cpp
extern int a;//error
extern int b;//ok
extern int b = 1;//error,链接时redefine
```
> summer：两个原则，1.任何变量和函数使用前至少要声明。2.单一定义
## const
- 全局const
  - `const int a = 0;`默认为内链接（static）
  - 方便在头文件中使用，若为外链接，多个包含相同头文件的源文件会产生redefine
  - 可以使用`extern const int a = 0;`覆盖
- 指针常量和常量指针
  - `const int* a`
  - `int* const a`
- 从程序员角度引用起别名,内部实现是指针
    ```c++
    int a=1;
    int &b=a;
    int &c=b;
    cout<<a<<b<<c//1,1,1
    ```

## 生存期和命名空间
### 区域
![](pic/cppscope-1.png)
![](pic/cppscope-2.png)
1. 声明域，全局变量->整个文件，函数中的局部变量->函数代码块，循环中局部变量->循环代码块
2. 潜在作用域,从声明点到声明域结束
3. 作用域，潜在作用域可能用命名冲突而覆盖

### 命名空间
只是一个“形式”，在使用using之前，这些变量没有声明和定义
命名空间的“形式”是全局的，不能在代码块中
命名空间里的变量一般具有外部链接性（除非const）
每个文件有一个全局命名空间
```c++
namespace test{
  int a = 1;
  int b;
}
```

### using
- 直接使用作用域解析符
- using声明,与普通声明一样,唯一区别，普通声明可以重复，这个不可以(同级)，保证唯一
```c++
namespace test{
  int a;
}
using test::a
int a = 10 //error conflict
```
```c++
#include <iostream>
using namespace std;
namespace test {
int a = 10;
}
int a = 8;
int main() {
    using test::a;
    cout << a << endl;//10
    {
        int a = 9;
        cout << a << endl;//9
        cout << ::a << endl;//8
        cout << test::a << endl;//10
    }
}
```
- using编译与using声明的区别是，在声明域中，冲突会隐藏
```c++
namespace test {
int a = 10;
}
int a = 8;
int main() {
    cout << a << endl;//8
    using namespace test;
    cout << a << endl;  // error 因为此时a在两个命名空间中，test和全局
    int a = 9;
    cout << a << endl;//9
    cout << test::a << endl;//10
    cout << ::a << endl;//8
}
```
### 其他
1. 嵌套
2. 匿名
乱七八糟的，需要的时候再看

