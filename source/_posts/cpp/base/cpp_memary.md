---
title: cpp memory
date: 2024-01-30 
tags: cpp
---

<!--more-->
## 分区
```c++
#include <iostream>
using namespace std;
int a;
static int b;
// 全局变量
int ga = 0;
int gb = 0;
static int s_ga = 0;
static int s_gb = 0;
// 全局常量
const int c_ga = 0;
const int c_gb = 0;

int main() {
    int la = 0;
    int lb = 0;
    static int s_la = 0;
    static int s_lb = 0;
    cout << &a << endl;
    cout << &b << endl;
    cout << &ga << endl;
    cout << &gb << endl;
    cout << &s_ga << endl;
    cout << &s_gb << endl;
    cout << &c_ga << endl;
    cout << &c_gb << endl;
    cout << &la << endl;
    cout << &lb << endl;
    cout << &s_la << endl;
    cout << &s_lb << endl;
    cout << endl;
    char s[] = "hello"; //char* const s
    char* t = "world"; //const char* t
    cout << &s << endl;  // 地址为0x7fffffffd9d2在栈区，内容为hello，即hello存储在栈区
    cout << &t << endl;  // 地址为0x7fffffffd9c8在栈区，内容为0x55555555600c为常量区，即world存储在常量区
    return 0;
}
/*
0x55555555815c
0x555555558168
0x55555555816c
0x555555556004
0x555555556008
0x7fffffffd9c0
0x7fffffffd9c4
0x555555558170
0x555555558174

0x7fffffffd9d2
0x7fffffffd9c8
*/

```
- 堆区
- 栈区
- 静态/全局 变量
- 常量  
## 堆区
### new
```c++
class A{}
A* p = new A;
```
```c++
1. void* mem = operator new()//内部调用malloc
2. p = static_cast<A*>(mem)//指针转换
3. p->A//构造函数
```

### delete
```c++
1. ~A()//调用析构函数
2. operator delete()//内部调用free，删除类
```

### malloc calloc realloc alloca
```c++
int *p = (int*)calloc(0,2*sizeof(int));
//相当于
int *p=(int *)malloc(2*sizeof(int));
memset(p,0,2*sizeof(int));
```