---
title: rust
date: 2024-01-30 
tags: rust
---

rust函数的参数和返回值必须是确定大小
对于一个类型的特征函数，时刻注意是`self &self &mut self`

<!--more-->
## Rust的内存管理
- rust中特别的地方是内存管理使用所有权机制，这涉及到赋值，传参，返回时的复制。可以将变量分为两类：
  - 堆上的变量，在栈会存在内容（指针，大小管理之类），没有实现`copy`方法，拷贝时只拷贝栈上内容，可用`clone`完全拷贝
  - 基本类型的变量**整数，浮点，布尔，简单元组，不可变引用**只存在栈上，拷贝整块复制

## 调试方法
```rust
//size
std::mem::size_of_val(&val);
std::mem::size_of(type);
```

## 引用
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
- 引用为**纯指针**
- 可变引用只能单独存在
- 存在引用原变量不能释放。
- 自动解指针？？？
```rust
struct A{
  	x:String,
}
impl A {
  	fn f(&self){
      	let s = self.x;
      	println!("{}",s);
  	}
}
fn main(){
  	let a = A{x:"hello".to_string()};
  	a.f();
}
```
- 代码块会报错`cannot move out of self.x which is behind a shared reference`，目前个人理解，不可变引用改变了原变量

## 结构体
```rust
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
let user2 = User {
  	email: String::from("another@example.com"),
  	..user1
};
```
  - 可缩略初始化
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

## 枚举 匹配 模式

### 枚举

### 匹配
```rust
enum person{
	name(String),
	age(i32),
	location{id:i32,lo:String},
}

fn main(){
	let a = person::location { id: 1, lo: "hello".to_string() };
	let b = person::age(32);
	let c = person::name("world".to_string());

	match a{
		person::age(i) => println!("{}",i),
		person::name(s) => println!("{}",s),
		person::location { id, lo } => println!("{},{}",id,lo),
	}

	if let person::age(s) = b{
		println!("{}",s);
	}

	if matches!(c,person::name(s) if s == "world"){
		println!("yes");
	}
}
```
- match本身同loop为表达式，有返回值
- 三种匹配方法`if let` `match` `matches!`
```rust
fn main() {
	let x = Some(50);
	let y = 10;

	match x {
		Some(50) => println!("Got 50"),//a
		Some(y) => println!("Matched, y = {y}"),//b
		_ => println!("Default case, x = {:?}", x),
	}

	println!("at the end: x = {:?}, y = {y}", x);
}
```
- 匹配顺序——a,b谁在前执行谁，
- 匹配范围 `1..=10`
```rust
match b{
    person::age(i) => println!("{}",i),
    person::name(s) => println!("{}",s),
    person::location { id, lo } => println!("{},{}",id,lo),
}

if let person::age(s) = b{
  	println!("{}",s);
}
```
- 这样会报错？？目前推断是：解构过程中会破坏原来的变量，虽然b为i32没问题,但存在其他类型，编译器为了确保安全仍会报错？？
- 匹配守卫，匹配后加判断
```rust
enum Message {
  	Hello { id: i32 },
}

fn main(){
	let msg = Message::Hello { id: 10 };

	match msg {
		Message::Hello { id: id_variable @ 3..=7 } => {
			println!("Found an id in range: {}", id_variable)
		},
		Message::Hello { id: 10..=12 } => {
			println!("Found an id in another range")
		},
		Message::Hello { id } => {
			println!("Found some other id: {}", id)
		},
		Message::Hello { id } if id>7 && id<20 =>{
			println!("Found some other id: {}", id)
		},
		_ => println!("hello")
	}
}
- `@`绑定
```
### 解构
- 之前的匹配即解构，下面是其他情况
- 枚举的解构即用上面的`match`
```rust
//结构体
struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let p = Point { x: 0, y: 7 };
    let Point { x, y } = p;
    assert_eq!(0, x);
    assert_eq!(7, y);
}
```
```rust
//元组
fn main(){
	let a = (1,"hello".to_string(),3.0);
	let (x,y,z) = a;
	println!("{} {} {}",x,y,z);
}
```
```rust
//数组
fn main(){
	let arr = [1,2,3];
	let [x,y,z] =arr;
	let [a,..,b] = arr;
	println!("{}{}",a,b);
}
```
- `..`省略 `_`忽略（只使用`_`和使用以下划线开头的名称有些微妙的不同：比如`_x`仍会将值绑定到变量，而`_`则完全不会绑定）

## 泛型
- 使用泛型注意提前声明
  ```rust
  struct A<T>{ }
  impl<T> A<T>{ }//这里的T可以换成别的字母，做到对应即可
  fn f<T>(){ }
  ```
- const泛型，在数值上使用泛型
- 
## 特征
- 泛型是一个复合类型或一个函数内部变量的多态，特征是类型的相似行为的多态。个人理解，泛型主要从函数角度看变量，特征是从变量角度看函数
- 特征约束`<T:>`，语法糖`x:impl trait`，语法糖也可以作为函数返回类型，但是缺点是返回值没有多态，可以通过特征对象解决
- 特征对象`Box<dyn trait>`

## 循环和Iterator
- loop是一个表达式，可以break返回值
![](/pic/rust-iter.png)
```rust
//iter = collection.into_iter();
pub trait IntoIterator {
	type Item;
	type IntoIter: Iterator<Item = Self::Item>;
	fn into_iter(self) -> Self::IntoIter;
}

impl<T, A: Allocator> IntoIterator for Vec<T, A> {
	type Item = T;
	type IntoIter = IntoIter<T, A>;
	fn into_iter(self) -> Self::IntoIter{...}
	...
}

pub struct IntoIter<
    T,
    #[unstable(feature = "allocator_api", issue = "32838")] A: Allocator = Global,
> {
    pub(super) buf: NonNull<T>,
    pub(super) phantom: PhantomData<T>,
    pub(super) cap: usize,
    pub(super) alloc: ManuallyDrop<A>,
    pub(super) ptr: *const T,
    pub(super) end: *const T, 
}
```
- `fn into_iter(self) -> Self::IntoIter;`可以看出，`into_iter()`方法输入`self`，输出具有`Iterator`特征的关联类型。即所谓迭代器就是实现`Iterator`特征的类型
- 对于`Vec`来说，`into_iter()`输出的关联类型为`IntoIter<T,A>`
- `IntoIter`类型迭代器会拿走**被迭代值的所有权**
```rust
//iter = collection.iter();
impl<T> [T] {
	pub fn iter(&self) -> Iter<'_, T> {
		Iter::new(self)
	}
}

pub struct Iter<'a, T: 'a> {
	ptr: NonNull<T>,
	end: *const T,
	_marker: PhantomData<&'a T>,
}

```
- 这里先回顾一个问题，集合的切片和集合的自身借用的差别，`&v`为纯指针，`&[]`为指针+长度
- 这里的`iter()`传入切片，返回`Iter`类型，可以看到它只有始末指针
```rust
impl<T> [T] {
  	pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut::new(self)
    }
}

pub struct IterMut<'a, T: 'a> {
	ptr: NonNull<T>,
	end: *mut T,
	_marker: PhantomData<&'a mut T>,
}
```
- 可以很容易看出和`iter()`的区别
```rust
let a = loop{
	if cnt >5{
		break cnt
	}
	cnt+=1;
};
```
- 迭代器上的方法有两大类一类返回新的迭代器，一类返回值
```rust
fn main() {
    let names = ["sunface", "sunfei"];
    let ages = [18, 18];
    
    //fn count(self) -> usize 
    let cnt = names.iter().count();
    //fn last(self) -> Option<Self::Item>
    let last = names.iter().last();
    assert_eq!(last,Some(&"sunfei"));
    //fn nth(&mut self, n: usize) -> Option<Self::Item> 返回第n个元素
    let second = names.iter().nth(1);
    assert_eq!(second,Some(&"sunfei"));
    //fn chain<U>(self, other: U) -> Chain<Self, <U as IntoIterator>::IntoIter>

    //fn zip<U>(self, other: U) -> Zip<Self, <U as IntoIterator>::IntoIter>

    //fn map<B, F>(self, f: F) -> Map<Self, F>

    //fn filter<P>(self, predicate: P) -> Filter<Self, P>

    //fn enumerate(self) -> Enumerate<Self>
}
```

## 生命周期
- 生命周期机制是为了防止悬空指针
- 生命周期机制没有改变原来的生命
- 消除规则
  - 每一个引用参数都会获得独自的生命周期
  - 若只有一个输入生命周期(函数参数中只有一个引用类型)，那么该生命周期会被赋给所有的输出生命周期
  - 若存在多个输入生命周期，且其中一个是 `&self`或`&mut self`，则`&self` 的生命周期被赋给所有的输出生命周期
```rust
struct ImportantExcerpt<'a> {
    part: &'a str,
}

//impl<'a> ImportantExcerpt<'a> {
//    fn announce_and_return_part(&self, announcement: &str) -> &str {
//       println!("Attention please: {}", announcement);
//       self.part
//    }
//}

impl<'a> ImportantExcerpt<'a> {
    fn announce_and_return_part<'b>(&'a self, announcement: &'b str) -> &'a str {
        println!("Attention please: {}", announcement);
        self.part
    }
}
```
- 对于函数来说，参数`'a`表示这个参数的生命标记为`'a`，具体多长不知道，当有多个相同标记的参数时取小值。返回值`'a`表示，返回值的生命周期要小于`'a`标记。因为返回值只能是从参数获得的(`'static` 情况除外）
- 结构体的标记表示，内部的生命要比外部的生命长
```rust
impl<'a> ImportantExcerpt<'a> {
	fn announce_and_return_part<'b>(&'a self, announcement: &'b str) -> &'b str {
		println!("Attention please: {}", announcement);
		self.part
	}
}

impl<'a: 'b, 'b> ImportantExcerpt<'a> {
    fn announce_and_return_part(&'a self, announcement: &'b str) -> &'b str{
        println!("Attention please: {}", announcement);
        self.part
    }
}
```
- 这样会报错，因为返回值为 `self.part` 标记为`'a`，编译器无法判断`'a`和`'b`的关系
- `'a:'b`：`'a`生命要比`'b`长，因为返回值标记`'b`说明返回值生命要比`'b`短，而实际的返回值为`'a`
```rust
struct NoCopyType {}

#[derive(Debug)]
#[allow(dead_code)]
struct Example<'a, 'b> {
    a: &'a u32,
    b: &'b NoCopyType
}

/* 修复函数的签名 */
fn fix_me(foo: &Example) -> &NoCopyType
{ foo.b }

fn main(){
    let no_copy = NoCopyType {};
    let example = Example { a: &1, b: &no_copy };
    fix_me(&example);
    println!("Success!")
}
```

## 闭包
- 闭包是一种匿名函数，它可以赋值给变量也可以作为参数传递给其它函数，不同于函数的是，它允许捕获调用者作用域中的值
```rust
fn fn_once<F>(func: F)
where
    F: FnOnce(usize) -> bool,{
    println!("{}", func(3));
    //println!("{}", func(4));
}

fn main() {
    let x = vec![1, 2, 3];
    let f = |z:usize|{z==x.len()};
    fn_once(f)
}
```
```rust
fn main() {
    let x = vec![1, 2, 3];
    let f = ||{
      	let newx =x;
    };   
}
```
- 可以把闭包看作一个结构体类型，实现`Fn` `FnMut` `FnOnce`特征，编译器会默认顺序选择合适的特征类型
```rust
fn main(){
	let mut num = 10;
	let mut f = |x:i32|{
		num +=x;
		num
	};
	let x = f(5);
	let y = f(5);
	println!("num:{} x:{} y:{}",num,x,y);//20 15 20
}

fn main(){
	let mut num = 10;
	let mut f = move |x:i32|{
		num +=x;
		num
	};
	let x = f(5);
	let y = f(5);
	println!("num:{} x:{} y:{}",num,x,y);//10 15 20
}
```
- `move` 闭包在内存中存储所获得的环境变量（把它理解单独的一个栈帧），对于实现`Copy`特征的类型，不带`move`是复制指针，反之是复制变量到新的栈帧，与原变量无关。
```rust
fn main(){
	let mut s = "hello".to_string();
	let mut f =||{
		s.push_str("world");
		println!("{}",s);
		};
	println!("{}",std::mem::size_of_val(&f));//8
	let x = f();
	let y = f();
	println!("{}",s);
}

fn main(){
	let mut s = "hello".to_string();
	let mut f =move ||{
		s.push_str("world");
		println!("{}",s);
	};
	println!("{}",std::mem::size_of_val(&f));//24
	let x = f();
	let y = f();
	//println!("{}",s); 错误
}
```
- 对于没实现`Copy`特征的类型，`move`涉及所有权转移
```rust
fn factory(x:i32) -> Box<dyn Fn(i32) -> i32> {
	let num = 5;
	if x > 1{
		Box::new(move |x| x + num)
	} else {
		Box::new(move |x| x - num)
	}
}
```
- 闭包作为函数返回值

## 指针
- `Box<T>` 很形象，有个box装着一个存储在堆上的变量，返回一个指针
- `Box::leak`
```rust
use std::ops::Deref;
fn main(){
	let a = Box::new("hello".to_string());
	println!("{:p} {:p}",a,&a);
	println!("{}",std::mem::size_of_val(&a));

	let b = a.deref();
	println!("{:p} {:p}",b, &b);
	println!("{}",std::mem::size_of_val(b));

	let x = *a;
	println!("{:p}",&x);
	println!("{}",std::mem::size_of_val(&x));

	let c = "hello".to_string();
	let d = &c;
	println!("{:p} {:p}",d, &d);
	println!("{}",std::mem::size_of_val(d));
}
```
- `deref`特征
