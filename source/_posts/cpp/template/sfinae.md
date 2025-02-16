>c++11 enable_if declval
>c++14 enable_if_t void_t
>c++17 if constexpr
>c++20 concept requires

## enable_if
```cpp
template<bool B, class T = void>
struct enable_if {};
 
template<class T> // 类模板偏特化
struct enable_if<true, T> { typedef T type; };     // 只有 B 为 true，才有 type，即 ::type 才合法

template< bool B, class T = void >
using enable_if_t = typename enable_if<B,T>::type; // C++14 引入
```
typename enable_if<B,T>::type = enable_if_t<B,T>
若B为true则返回T，若B为false则什么都不返回
可以想到，可以使用类型萃取技术作为B

## void_t
void_t<T,U,V,W>
返回void类型

## std::declval
将任意类型 T 转换成引用类型，使得在 decltype 说明符的操作数中不必经过构造函数就能使用成员函数。
简单来说通过类型直接获得变量，std::declval只能用于不求值语境，且不要求有定义。


## c++14
```cpp
#include <iostream>
using namespace std;

template <class T, class = void>
class is_true {
   public:
    constexpr static bool flag = false;
};

template <class T>
//class is_true<T, std::void_t<decltype(declvar(T) + declvar(T)), typename T::type, decltype(&T::value), decltype(&T::f(1))>> 
class is_true<T, std::void_t<decltype(T{} + T{}), typename T::type, decltype(&T::value), decltype(&T::f(1))>> {
   public:
    constexpr static bool flag = true;
};

template <class T, enable_if_t<is_true<T>::flag, int> = 0>
void test(T& a, T& b) {
    cout << "int true" << endl;
}
template <class T, enable_if_t<!is_true<T>::flag, int> = 0>
void test(T& a, T& b) {
    cout << "int false" << endl;
}

class A {
   public:
    using type = int;
    int value = 10;
    void f(int a) {
    }
    void operator+(const A& b) {
    }
    // A(int a) {
    // }
};

int main() {
    A a, b;
    test(a, b);
    int x = 1;
    int y = 1;
    test(x, y);
}
// int true
// int false
```
把上面的程序的A的构造函数改为含参,则`test(a, b);`时`decltype(T{} + T{})`出错，`test(a, b);`的is_true返回false

```cpp
template <class T, class = void>
class is_true {
   public:
    constexpr static bool flag = false;
};

template <class T>
class is_true<T, std::void_t<decltype(declval<T>() + declval<T>()), typename T::type, decltype(&T::value), decltype(&T::f)>> {
   public:
    constexpr static bool flag = true;
};

template <class T, enable_if_t<is_true<T>::flag, int> = 0>
void test(T& a, T& b) {
    cout << "int true" << endl;
}
template <class T, enable_if_t<!is_true<T>::flag, int> = 0>
void test(T& a, T& b) {
    cout << "int false" << endl;
}

class A {
   public:
    using type = int;
    int value = 10;
    void f() {
    }
    void operator+(const A& b) {
    }
    A(int a) {
    }
};

int main() {
    A a{1}, b{2};
    test(a, b);
    int x = 1;
    int y = 1;
    test(x, y);
}
```