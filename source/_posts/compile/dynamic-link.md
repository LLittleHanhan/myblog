---
title: dynamic link
date: 2024-01-30 
tags: compile
---
>静态链接会导致，每一个程序要用一个库都要链接库的目标文件，这样在内存中同一个库会有多次拷贝造成浪费，动态链接就是解决这类问题。
<!--more-->
## 动态链接
两种加载方式
- 装入时加载 dynamic linking:常用的是这种
- 运行时加载 dynamic loading:这种一般是显示调用的
### 运行时查找动态库
- 动态库的路径查找有两步，第一步是编译链接时的查找，和静态库一致，检查变量函数是否声明定义，这部分涉及到符号表
- 编译时优先动态库，可以使用`-static`强制指定静态库
- 运行时的查找
  1. `-Wl,-rpath=`指定
  2. `export LD_LIBRARY_PATH=$LD_BRARY_PATH:path`
  3. 默认路径`/usr/lib ...`
  4. 配置`/etc/ld.so.config`文件
  
`-Wl`这里gcc实际上是一个上层工具，编译链接过程中调用了预处理器，编译器，汇编器，链接器，`-Wl,rpath=`就是将参数传递给链接器,链接器对应ld指令
### 一个例子
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
cd lib
g++ a.cpp -fPIC -c -o a.o #-fPIC用于编译，生成位置无关代码
g++ a.o xxx.o ... -shared -o liba.so #链接成动态库
cd ..
g++ main.cpp -L ./lib -la -Wl,-rpath=./lib -o test
```
**重要的事**
1. -fPIC生成位置无关代码，与内存,重定位有关的知识得补补


## 一些工具
> 在实际编译中，各种动态库调用，手写g++编译很麻烦且易出错，linux系统有一些工具帮忙
### ldconfig
ldconfig命令：的用途主要是在默认搜寻目录/lib和/usr/lib以及动态库配置文件/etc/ld.so.conf内所列的目录下，搜索出可共享的动态链接库（格式如lib.so）,进而创建出动态装入程序(ld.so)所需的连接和缓存文件。
缓存文件默认为/etc/ld.so.cache，此文件保存已排好序的动态链接库名字列表，为了让动态链接库为系统所共享，需运行动态链接库的管理命令ldconfig，此执行程序存放在/sbin目录下。
ldconfig通常在系统启动时运行，而当用户安装了一个新的动态链接库时，就需要手工运行这个命令。
### lld
查看可执行程序的共享库

### pkg-config
- pkg-config相当于一个库管理工具
- 需要库的开发者写好.pc文件，文件中会包含头文件和库文件的位置信息，pkgconfig会读取这些文件
- pkgconfig可以在编译链接运行时发挥作用，可以大大简化gcc指令
- 默认路径是`/usr/lib/pkgconfig` 环境变量`PKG_CONFIG_PATH`
```shell
pkg-config --list-all
pkg-config --modversion name
pkg-config --libs name
pkg-config --cflags name
g++ main.cpp -o test `pkg-config --libs --cflags name`
```

### 更高级的工具
make，cmake