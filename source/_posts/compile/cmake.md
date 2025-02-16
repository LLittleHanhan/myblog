---
title: cmake
date: 2024-01-30 
tags: compile
---
> cmake的简单用法 
<!--more-->

## 使用
```shell
cmake -S source_path -B build_path
cmake ---build build_path
```
## 头
```
cmake_minimum_required(VERSION 3.22)
project(name)
set(CMAKE_CXX_STANDARD 11)
```

## 变量
`set(<variable> <value>... CACHE <type> <docstring> [FORCE])`
- cache:缓存变量的标志
- docstring:描述，必须为字符串
- force:强制改变缓存变量
- 使用:`$(MY_VARIABLE)`

### local var
相当于局部变量
### cache var
相当于全局变量，CMake中的缓存变量都会保存在CMakeCache.txt文件中
`cmake -D`可以创建，赋值缓存变量
文件内force是禁止覆盖
### env var
`set(ENV{variable_name} value)`
`$ENV{variable_name}`
### 内置变量
同cache var
`cmake -D`可以创建，赋值缓存变量，文件内用set
```
message(${CMAKE_BINARY_DIR})
message(${CMAKE_SOURCE_DIR})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
message(${EXECUTABLE_OUTPUT_PATH})
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
message(${LIBRARY_OUTPUT_PATH})

message(${CMAKE_CURRENT_SOURCE_DIR})
message(${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER xxx)
set(CMAKE_BUILD_TYPE debug/release)
```

## 编译选项
```
target_compile_options(target PUBLIC|PRIVATE|INTERFACE -XX -XX -XX)
```

## 编译器
`cmake -DCMAKE_CXX_COMPILER=clang++`

## 目标文件构建
> 需要的文件是从当前工作目录查找的，要么写出路径，要么通过add_subdirectories()或findPackage()导入
```
add_executable(target source)
add_library(target [STATIC|SHARED] source)
```
路径问题：
```
# 解决路径问题
target_include_directories(target [PUBLIC|PRIVAE|INTERFACE] src_path)
target_link_directories(target [PUBLIC|PRIVAE|INTERFACE] src_path)
# 解决依赖问题
target_link_libraries(target [PUBLIC|PRIVAE|INTERFACE] src_path)
# 以上不会递归查找
```
以上相当于给target添加属性，并通过`PUBLIC|PRIVATE|INTERFACE`控制传递性
1. PUBLIC:自己使用且传递
2. PRIVATE:自己使用不传递
3. INTERFACE:自己不用且传递
> target_link_libraries:比如libc需要libb，libb需要liba，那么libb链接liba时使用public，那么libc链接libb时就会隐式链接liba?存疑


## file
```
file(GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
# GLOB
# GLOB_RECURSE:递归
```

## 子模块
```
add_subdirectory(src_path)
```
1. 调用子目录的CMakeLists
2. 导入子目录的target


## 函数

## 宏

## 外部调用


## 一个栗子
### 不使用子目录cmakelist，不规范，看看语法即可
```
-project
CMAKELISTS.txt
-main.cpp
-include
--func.h
-src
--func.cpp
```
```
cmake_minimum_required(VERSION 3.22)
project(main)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(main main.cpp)
target_include_directories(main PUBLIC include)

add_library(func ./src/func.cpp)
target_link_libraries(main func)
```
> include的“”和<>，“”是先查找当前目录，没找到去查找默认目录，<>是直接查找默认目录，target_include_directories相当于提供了一个默认目录，查找过程非递归
### 使用子目录cmakelist，以下为规范化的写法
```
```

## 使用共享库

### find_package
[官方文档](https://cmake.org/cmake/help/latest/command/find_package.html#find-package)
`find_package`本质上还是查找文件问题
1. module模式
2. config模式

### 常用的三方库链接
#### CUDA
#### GOOGLE_BENCHMARK