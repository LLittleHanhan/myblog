---
title: cmake
date: 2024-01-30 
tags: compile
---
> cmake的简单用法 
<!--more-->

## 开始
```
cmake_minimum_required()
project()
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
```
<!--more-->
## 编译选项
```
target_compile_options(target PUBLIC|PRIVATE|INTERFACE -XX -XX -XX)
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
### env var
`set(ENV{variable_name} value)`
`$ENV{variable_name}`



## 目标文件构建
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
```
以上相当于给target添加属性，并通过`PUBLIC|PRIVATE|INTERFACE`控制传递性
1. PUBLIC:自己使用且传递
2. PRIVATE:自己使用不传递
3. INTERFACE:自己不用且传递


## 子模块
```
add_subdirectory(src_path)
```
1. 调用子目录的CMakeLists
2. 导入子目录的target,不需要指定路径了,但头文件仍然需要指定路径

## 一个栗子
```
-main
CMAKELISTS.txt
-src
--func.h
--CMAKELISTS.txt
```
### 不使用子目录cmakelist
```
cmake_minimum_required()
project(main)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(main main.cpp)
target_include_directories(main PUBLIC src)
target_link_directories(main PUBLIC src)
target_link_libraries(main func)
```
### 使用子目录cmakelist
```
// main cmakelists
cmake_minimum_required()
project(main)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(src)

add_executable(main main.cpp)
target_include_directories(main PUBLIC src)
target_link_libraries(main func)

// src cmakelists
add_library(func STATIC naive.cpp)
```
可以在src中的cmakelists中这样写,main中的`target_include_directories(main PUBLIC src)`就可以省略
```
add_library(func STATIC naive.cpp)
target_include_directories(func PUBLIC .)
```

## 使用共享库

### find_package
[官方文档](https://cmake.org/cmake/help/latest/command/find_package.html#find-package)
`find_package`本质上还是查找文件问题
1. module模式
2. config模式
