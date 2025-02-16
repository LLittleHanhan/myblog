---
title: callable
date: 2024-09-05
tags: cpp
---

## lambda
编译器会把lambda表达式转换成一个类，重载operate()
```c++

int main(){
    auto f = []()->int{};
}

// 捕获[]
//
// [a,b]默认按值捕获
// [&a,&b]mutable{ };//&按引用捕获
// [=] [&]捕获所用局部变量！不能捕获静态变量，但是可以直接访问静态变量
int main() {
    static int i{42};

    auto f = [=] { i++; };
    f();

    std::cout << i;  // 43!
}
// [=,&a,&b] [&,a,b]
// [this]类内捕获this指针以访问类成员
```
还有一些特殊的用法参考
[reference](https://quant67.com/post/C/lambda/lambda.html)
1. C++17 可以给 lambda 设定 constexpr
2. auto lambda类型，行参，返回值类型，ps：auto推导规则是模板推导规则，不能推导初始化列表
3. 直接函数调用


## 函数 & 函数指针
```cpp
void test(int a,float b){
    //
}

int main(){
    void (*p)(int,float) = test;
}
```

## 仿函数
就是实现operate()的类

## bind
bind的参数是按值传递
按引用需要ref