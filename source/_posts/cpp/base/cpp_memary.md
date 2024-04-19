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
    cout << &a << endl;     // 0x576450897154

    cout << &b << endl;     // 0x576450897164

    cout << &ga << endl;    // 0x576450897158
    cout << &gb << endl;    // 0x57645089715c
    
    cout << &s_ga << endl;  // 0x576450897168
    cout << &s_gb << endl;  // 0x57645089716c

    cout << &c_ga << endl;  // 0x576450895004
    cout << &c_gb << endl;  // 0x576450895008

    cout << &la << endl;  // 0x7fff6aafda70
    cout << &lb << endl;  // 0x7fff6aafda74

    cout << &s_la << endl;  // 0x576450897170
    cout << &s_lb << endl;  // 0x576450897174
    cout << endl;
    char s[] = "hello";  // char* const s
    char* t = "world";   // const char* t
    cout << &s << endl;  // 地址为0x7fff6aafda82在栈区，内容为hello，即hello存储在栈区
    cout << &t << endl;  // 地址为0x7fff6aafda78在栈区，内容为0x57645089500c为常量区，即world存储在常量区
    return 0;
}

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