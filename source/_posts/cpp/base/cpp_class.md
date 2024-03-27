---
title: cpp class
date: 2024-01-30 
tags: cpp
---

<!--more-->
## 内联函数
1. 内联不局限与类成员函数
2. `inline`后有实现才有意义
3. 内联是编译器将函数定义（{...}之间的内容）在函数调用处展开，藉此来免去函数调用的开销
4. 同时内联避免了重定义问题，强符号，弱符号？
5. 内联的意义是将本文件中使用该内联函数的地方全部替换，如果内联函数fun()定义在某个编译单元A中，那么其他编译单元中调用fun()的地方将无法解析该符号，因为在编译单元A生成目标文件A.obj后，内联函数fun()已经被替换掉，A.obj中不再有fun这个符号，链接器自然无法解析。因此内联函数要定义在**头文件**中
6. [有空看看](https://blog.csdn.net/qq_35902025/article/details/127912415)
## 访问限制
- public:
- private
- protected
## 成员函数
- const常成员函数
## 友元
```c++
class A{
    friend void f();
    friend class B；
    friend void B::g();
private:
    int a;
    int b;
}
class B{
public:
    g();
}
void f(){
    int fa = A.a;
}
```
- 本质上都是函数，不属于类
- 单向无传递性
## 类静态变量和函数
```c++
class A{
public:
    int a =1;
    static int b;
};
int A::b = 1;
static int A::b = 1;//error,见2
int main(){
    int A::b =1;//error,见1
}
```
1. 不能在类声明中定义，因为类的声明一般放在头文件中，若可以，那么当多个文件包含该头文件，之后这些文件链接时会产生重定义，因此只能在类外定义，且在函数之外，且只能定义一次
2. 此static非彼static,类（cpp）static意义是使对象共享变量，**在该类中是静态的**,而在类外，相当于普通变量（不是静态，外文件可以访问）
先这样理解，这篇文章写的很好
[知乎](https://www.zhihu.com/question/397086631)
## 构造函数
### 声明定义
- 当没有定义构造函数时，编译器会提供一个默认的构造函数。否则需要(最好这样做)自行提供默认构造函数
- 两种提供默认构造函数的方式方式，一是无参，二是缺省
- 构造函数重载
- explicit禁止隐式调用，见类型转换
### 初始化
```c++
class A{
    int a;
    int b;
    A(int _a,int _b):a(_a),b(_b){
    }
}
```
- 常量型，引用型，类成员为没有默认构造函数的类类型，必须使用初始化列表
- 顺序：严格按照定义顺序依次初始化，先在初始化列表中查找是否指定了初始化方式；若找到，按指定初始化；若没找到，按无参构造初始化(内置类型的值不确定)；若没找到无参构造函数，则报错；全部完成后，再进入构造函数的{ }

### 使用
```c++
class A{
A(){}
A(int _a):a(_a){}
};

A a;//ok
A a();//error,这是一个返回值为A类型的函数声明
A a(1);//ok
A a = A();//ok,c++标准允许编译器使用两种方式构造，一是同上，二是先构造一个临时对象，之后拷贝丢弃
A* a = new A()//ok
```
# 一个例子
```c++
class myString {
   private:
    char* s;
   public:
    myString(const char _s[] = 0) {
        if (_s) {
            s = new char[strlen(_s) + 1];
            strcpy(s, _s);
        } else {
            s = new char[1];
            *s = '\0';
        }
    }
    myString(const myString& _string) {
        s = new char[strlen(_string.s) + 1];
        strcpy(s, _string.s);
    }
    myString& operator=(const myString& _string) {
        if(this == &_string)
            return *this;
        delete[] s;
        s = new char[strlen(_string.s) + 1];
        strcpy(s, _string.s);
        return *this;
    }
    ~myString() {
        delete[] s;
    }
    char* get_s() {
        return s;
    }
};
```
## 拷贝构造
- 默认浅拷贝，需要时自行提供深拷贝（自定义）
- private修饰自定义拷贝构造函数，禁止拷贝，或`myString(const myString& s)=delete`
- 应用场景
  1. 值传参
  2. 值返回
  3. `myString s1(s)` or `myString s1=s`  

## 拷贝赋值
1. `return *this` 返回引用类型，是为了连等
2. 检查自我赋值 

## 析构函数
没什么东西

## 运算符重载
