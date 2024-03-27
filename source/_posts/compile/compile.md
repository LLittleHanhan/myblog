---
title: compile
date: 2024-01-30
tags: compile
---
> g++编译的过程，以及静态链接
<!--more-->

## 程序的编译过程 g++
![](pic/compile-1.png)
![](pic/compile-2.webp)
编译成的目标文件大致包括代码段，数据段，符号表。
- 预处理 将`#include<>和#define`加入
- 编译 源代码->汇编语言
- 汇编 汇编语言->目标文件
- 链接 多个目标文件链接成可执行程序或库
```shell
#gcc中的使用
g++ -o #输出文件名
g++ -E #预处理
g++ -S #汇编文件
g++ -c #目标文件
g++ -I include_path #头文件目录
g++ -L lib_path -l lib #库目录
```
 
**attention**
1. 不同阶段编译器检查什么
   - 源文件->目标文件阶段 编译器只检查变量和函数是否声明
   - 链接阶段 会检查是否定义
   - 具体怎么链接的见后
2. gcc是怎么查找头文件的
   - `-I`指定目录
   - 当前目录`#include"xxx"`
   - 环境变量`C_INLCUDE_PATH`or`CPLUS_INCLUDE_PATH`
   - 默认目录`/usr/include /usr/local/include ...`
3. gcc是怎么找静态库的
   - `-L`指定目录
   - 环境变量`LIBRARY_PATH`，若使用动态库在编译阶段也是这个环境变量，但是动态库运行时还需要指定`LD_LIBRARY_PATH`
   - 默认目录`/usr/lib /usr/local/lib`


## 静态链接
静态链接简单说就是，张三写了一些工具函数，包含了许多cpp，hpp文件。别人使用的时候首先需要include头文件这样可以生成目标文件，但之后要的链接文件，且使用者不知道要链接什么（使用者只能从头文件中了解该库的功能，并不知道具体实现），怎么办？所以张三就可以把这些cpp编译成目标文件然后打包成一个静态库，这样就把链接的活全部交给编译器，完美解决问题。
```shell
ar crs xxx.a a.o b.o ...
```

## 一个例子
```c++
/*tree
project
-main.cpp
-lib
--a.cpp
--a.h
*/

//main.cpp
#include <iostream>
#include "lib/a.hpp"
using namespace std;

int main() {
    cout << test();
}

//a.cpp
#include "a.hpp"

int test() {
    return 1;
}

//a.hpp
#ifndef A_HPP
#define A_HPP
int test();
#endif
```

```shell
gcc ./lib/a.cpp -c -o ./lib/a.o
ar crs ./lib/liba.a ./lib/a.o
gcc main.cpp -L ./lib -la -o test
./test
```
**attention**
  
```c++
#ifndef A_HPP
#define A_HPP
//这个语句只是保证一个cpp文件，include多个头文件时，若重复只包含（展开）一次
//当两个文件include同一个头文件时，链接时就可能出错！！！
//因此头文件中最好不要放定义
```