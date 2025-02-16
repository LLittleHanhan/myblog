---
title: 可变参数
date: 2024-07-24
tags: cpp
---

## va_list
```cpp
#include <stdarg.h>
#include <stdio.h>

int sum(int count, ...) {
    va_list args;
    int total = 0;
    va_start(args, count);
    for (int i = 0; i < count; i++) {
        total += va_arg(args, int);
    }
    va_end(args);
    return total;
}

int main() {
    printf("Sum of 1, 2, 3, 4: %f\n", sum(4, 1, 2, 3, 4));
    printf("Sum of 5, 10: %f\n", sum(2, 5, 10));
    return 0;
}

```
注意类型提升，类型匹配
1. 整数类型的提升：
   - char、short 和 bool 会被提升为 int。
   - unsigned char 和 unsigned short 会被提升为 unsigned int。
2. 浮点类型的提升：
    - float 会被提升为 double。
## initializer_list
```cpp
template <class T>
class initializer_list {
   public:
    typedef T value_type;
    typedef const T& reference;
    typedef const T& const_reference;
    typedef size_t size_type;
    typedef const T* iterator;
    typedef const T* const_iterator;

   private:
    iterator _M_array;
    size_type _M_len;

    constexpr initializer_list(const_iterator __a, size_type __l)
        : _M_array(__a), _M_len(__l) {};

    constexpr initializer_list()
        : _M_array(0), _M_len(0) {}

    constexpr size_type size() const noexcept { return _M_len; }

    constexpr const_iterator begin() const noexcept { return _M_array; }

    constexpr const_iterator end() const noexcept {
        return begin() + _M_len;
    }
};
```
initializer_list支持列表初始化，而其他容器如果存在以nitializer_list为参数的构造函数也就支持了列表初始化
## template

## reference

[](https://songlee24.github.io/2014/07/22/cpp-changeable-parameter/)