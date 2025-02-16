---
title: rust(二)所有权和引用
date: 2024-03-01 
tags: rust
---
rust的所有权
<!--more-->
## 所有权原则
rust中每个值都有一个所有者，更进一步说就是，rust中分配的每块内存都有其所有者，所有者负责该内存的释放和读写权限，并且每次每个值只能有唯一的所有者，这就是rust的所有权机制（ownership）

1. Rust 中每一个值都被一个变量所拥有，该变量被称为值的所有者
2. 一个值同时只能被一个变量所拥有，或者说一个值只能拥有一个所有者
3. 当所有者(变量)离开作用域范围时，这个值将被丢弃(drop)

## 四种语义
以下针对内置类型
1. copy：浅拷贝，栈上的数据按位复制
2. move：浅拷贝，栈上的数据按位复制，堆上数据发生所有权转移
3. clone：深拷贝
4. 引用和借用：见下节

## 引用和借用
和cpp相比rust的引用和c一样是指针，在cpp中可以直接看作变量的别名使用，但是在rust中要把它理解为指针使用：
- 解引用*
- 所用权机制，引用没有所用权，只是借用，如果把它当作变量别名，就会在所有权问题上出错，这点在结构体的栗子中可以看到
```rust
fn main() {
    let a = "hello".to_string();
    print!("{:p}\n", &a);

    let mut b = a;
    print!("{:p}\n", &b);

    let c = &b;
    print!("{:p}\n", &c);//c后面没用到，引用释放

    let d = &mut b;
    print!("{:p}\n", &d);//d后面没用到，引用释放

    let e = d;//e是当前引用

    //b.push('a'); //error：push传参为&mut self，与e冲突
    e.push('a');//e释放，之后push就没问题了
    b.push('a');
    print!("{}", b);

}
```
1. 可变引用与不可变引用不能同时存在
2. 可变引用同时只能存在一个
3. 编译器优化：引用作用域的结束位置从花括号变成最后一次使用的位置，**Non-Lexical Lifetimes(NLL)**
## 复合类型的引用
### 结构体，枚举
**即使成员都有copy语义，但是rust也不会默认为其实现copy。需要手工添加 `#[derive(Debug, Copy, Clone)]`**
```rust
#[derive(Debug)]
struct A {
    x: i32,
}

fn main() {
    // let mut a = A {
    //     x: "hello".to_string(),
    // };

    let a = A { x: 1 };
    println!("{:p}", &a);

    let b = a;
    println!("{:p}", &b);

    println!("{}", a.x);//error
}
```
虽然i32有copy语义，但x是move语义
```rust
#[derive(Debug)]
struct A {
    x: String,
}

fn main() {
    let mut a = A {
        x: "hello".to_string(),
    };
    println!("{:p}", &a);

    let b = &a;
    let c = b.x;//error
}
```
b是借用，没有结构体的所有权，所以就不能转移所有权

## 内存
```rust
fn main(){
    let a = vec![1,2,3];
    let b = a;
    let i = &b[1..2];
    let j = &b;
    println!("{}",std::mem::size_of_val(&i));//16 8+8
    println!("{}",std::mem::size_of_val(&j));//8  8
    println!("{}",std::mem::size_of_val(&b));//24 8+8+8
}
```
[rust内存排布](https://www.cnblogs.com/88223100/p/Rust-memory-distribution.html)
## 引用和生命周期
见之后的笔记