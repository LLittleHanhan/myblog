---
title: rust(三)字符串
date: 2024-03-01 
tags: rust
---

## String
- `push()`&`push_str()`
- `insert()`&`insert_str()`
- `replace("O","N")`&`replacen("O","N",1)`返回新串
- `replace_range(1..2,"N")`返回当前串
- `pop()`
- `remove()`&`truncate()`
- `clear()`
> 有关索引位置的都要注意utf-8编码不定长，按字节索引，可以使用`#String.chars()`变成unicode编码

<br>

`+`&`fn add(self, s: &str) -> String`
1. 返回的是新串
2. 原串发生所有权转移
3. 右值是&str


## 切片
- 对String，数组的部分引用，字符串字面量也是引用
- `[i32]``str`类型是变长的，即类型大小不确定，因此要使用引用表示，这就是切片
- 切片是部分引用，也是引用！遵守引用的规则

## string的索引
- Rust 中的字符是 Unicode 类型，因此每个字符占据 4 个字节内存空间，但是在字符串中不一样，字符串是 UTF-8 编码，也就是字符串中的字符所占的字节数是变化的(1 - 4)。
- Rust 不允许去索引字符串