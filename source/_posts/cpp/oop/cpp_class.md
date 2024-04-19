---
title: cpp class
date: 2024-01-30 
tags: cpp
---

<!--more-->
> class和struct定义类的唯一区别是默认访问权限
> 访问权限是class level而不是obj level，类内部可以访问一个实例的私有变量
## 类型成员
```cpp
using
typename
```
## 可变数据成员
mutable修饰
1. const对象的成员也可变
2. 在常成员函数(const修饰)内可也变
## 常成员函数和*this
```cpp
class A{
    A& f() const{
        return *this;
    }
}
```
const函数内部的this指针为`const T* this`
返回**const引用**，就不能接着调用非const函数了
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
## 类静态变量和静态函数
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
- 当没有定义构造函数时，编译器会提供一个**合成的默认的构造函数**，否则需要(最好这样做)自行提供默认构造函数
- 两种提供默认构造函数的方式方式，一是无参，二是缺省
- 构造函数重载
- default
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
A a = A();//ok,c++标准允许编译器使用两种方式构造，一是同上，二是先构造一个临时对象，之后拷贝丢弃，gcc上是一
A* a = new A()//ok
```
---

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
---

## 拷贝构造
- 默认浅拷贝，需要时自行提供深拷贝（自定义）
- private修饰自定义拷贝构造函数，禁止拷贝，或`myString(const myString& s)=delete`
- 应用场景
  1. 值传参
  2. 值返回
  3. `myString s1(s)` or `myString s1=s`  

## 拷贝赋值
1. `return *this` 返回引用类型，是为了使表达式返回原来的变量
   ```
   T a = b = c;
   ```
2. 检查自我赋值 

## 析构函数
没什么东西

## 运算符重载
