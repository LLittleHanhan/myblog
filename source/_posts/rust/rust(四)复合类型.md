---
title: rust(四)复合类型
date: 2024-03-01 
tags: rust
---

## 元组
```rust
fn main() {
    let tup: (i32, f64, u8) = (500, 6.4, 1);

    tup.0//use

    let (x,y,z) = tup //解构

    return (x,y,z)//用作返回值
}
```
## 结构体
```rust
struct User{
    active: bool,
    username: String,
    email: String,
    sign_in_count: u64,
}

let mut user1 = User {
  	email: String::from("someone@example.com"),
  	username: String::from("someusername123"),
  	active: true,
  	sign_in_count: 1,
};
```
- 必须全部初始化，顺序随便。
- 必须要将结构体实例声明为可变的，才能修改其中的字段，Rust 不支持将某个结构体某个字段标记为可变。
```rust
User {
        email,
        username,
        active: true,
        sign_in_count: 1,
    }
```
- 当函数参数和结构体字段同名时，可以直接使用缩略的方式进行初始化
```rust
let user2 = User {
  	email: String::from("another@example.com"),
  	..user1
};
```
- 类似于对象赋值（对象赋值是整体发生所有权转移），注意会发生内部可能会所有权转移
```rust
struct A {
	a: u8,
	b: u32,
	c: u16,
}
fn main() {
	let a = A {
		a: 1,
		b: 2,
		c: 3,
	};
	println!("{:p} {:p} {:p}",&a.a,&a.b,&a.c); 
	//0x7ffca964560e 0x7ffca9645608 0x7ffca964560c
	println!("{}",std::mem::size_of_val(&a));//8
	println!("{}",std::mem::align_of_val(&a));//4
}
```
  - 内存分布：结构体内部数据堆栈同正常单个类型，结构体只是数据打包？
  - 结构体中会存在对齐属性！Rust 会在必要的位置填充空白数据，以保证每一个成员都正确地对齐，同时整个类型的尺寸是对齐属性的整数倍。

## 数组

## hash