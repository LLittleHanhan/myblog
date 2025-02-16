
```cpp
int main() {
    return 0;
}
[Nr] Name              Type            Address          Off    Size   
[14] .text             PROGBITS        0000000000001040 001040 0000f8 
[16] .rodata           PROGBITS        0000000000002000 002000 000004 
[23] .data             PROGBITS        0000000000004000 003000 000010 
[24] .bss              NOBITS          0000000000004010 003010 000008
```

# const
```cpp
const int c = 1;
int main() {
    return 0;
}
[Nr] Name              Type            Address          Off    Size   
[14] .text             PROGBITS        0000000000001040 001040 0000f8
[16] .rodata           PROGBITS        0000000000002000 002000 000008
[23] .data             PROGBITS        0000000000004000 003000 000010
[24] .bss              NOBITS          0000000000004010 003010 000008
```
- 全局const会放在.rodata段
- 局部const会直接编到.text或在栈段


# global && static
```cpp
int d = 1;
int main() {
    const int a = 1;
    // A a;
    return 0;
}
[Nr] Name              Type            Address          Off    Size  
[14] .text             PROGBITS        0000000000001040 001040 0000ff
[16] .rodata           PROGBITS        0000000000002000 002000 000004
[23] .data             PROGBITS        0000000000004000 003000 000014 
[24] .bss              NOBITS          0000000000004014 003014 000004 
```

```cpp
static int d = 1;
int main() {
    const int a = 1;
    // A a;
    return 0;
}
[Nr] Name              Type            Address          Off    Size  
[14] .text             PROGBITS        0000000000001040 001040 0000ff
[16] .rodata           PROGBITS        0000000000002000 002000 000004
[23] .data             PROGBITS        0000000000004000 003000 000014 
[24] .bss              NOBITS          0000000000004014 003014 000004 
```
有初值在.data
无初值在.bss
> 这里在全局变量依次int a,b,c,d .bss段的增长有点奇怪

## static
```
#include <iostream>
using namespace std;

static int d = 1;

int main() {
    static int d = 2;
    cout << d << endl;// 2
    cout << ::d << endl; // 1
    return 0;
}
13: 0000000000004010     4 OBJECT  LOCAL  DEFAULT   25 _ZL1d
14: 0000000000004014     4 OBJECT  LOCAL  DEFAULT   25 _ZZ4mainE1d
```
static静态变量，生存期都是全局的，作用域是局部的

## strong weak
```cpp
//a.cpp
int x = 10;
__attribute__((weak)) double weak = 1.9;
void f() {
    x = 15212;
}
//b.cpp
#include <stdio.h>
using namespace std;
extern int x;
void f();
// int weak = 1;
// __attribute__((weak)) int weak = 1;
int main() {
    printf("%d", sizeof(weak));
    return 0;
}
```
1. 多个strong重定义
2. 多个weak，选最大的那个

## extern
```cpp
extern int x;
int main() {
    x = 10;
    return 0;
}

Symbol table '.symtab' contains 5 entries:
Num:    Value          Size Type    Bind   Vis      Ndx Name
    0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND 
    1: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS b.cpp
    2: 0000000000000000     0 SECTION LOCAL  DEFAULT    1 .text
    3: 0000000000000000    25 FUNC    GLOBAL DEFAULT    1 main
    4: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND x
```

# class
```cpp
class A {
    static int a;
};

int A::a = 1;
int b = 1;
const int c = 1;
static int d = 1;

int main() {
    A a;
    return 0;
}

[Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
[14] .text             PROGBITS        0000000000001040 001040 0000f8 00  AX  0   0 16
[16] .rodata           PROGBITS        0000000000002000 002000 000008 00   A  0   0  4
[23] .data             PROGBITS        0000000000004000 003000 00001c 00  WA  0   0  8
[24] .bss              NOBITS          000000000000401c 00301c 000004 00  WA  0   0  1
```

## 静态链接，动态链接，重定位