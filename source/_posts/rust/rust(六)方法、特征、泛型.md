---
title: rust(六)方法、特征、泛型
date: 2024-03-04 
tags: rust
---
> 来看看方法，特征和泛型
> 泛型是一个复合类型或一个函数内部变量的多态，特征是类型的相似行为的多态。
> 泛型主要从函数角度看变量，特征是从变量角度看函数
<!--more-->

## 方法
## 泛型
### 在结构体和方法中使用泛型
```rust
struct Point<T, U> { //这里结构体的类型就是Point<T,U>,T和U是表现形式，可以换成其他的！
    x: T,
    y: U,
}

impl<T, U> Point<T, U> { // 注意impl声明
    fn mixup<V, W>(self, other: Point<V, W>) -> Point<T, W> {
        Point {
            x: self.x,
            y: other.y,
        }
    }
}
```
- 注意声明
- 泛型字母只是标志，对应即可
- turbofish语法
```rust
let mut v = Vec::new();//编译器无法推断类型
let v: Vec<bool> = Vec::new();
let v = Vec::<bool>::new();
```

### 在枚举中使用泛型
### const泛型
数值的泛型

> 泛型具体类型也不是随便给的，有些限制，比如说给泛型T一个具体类型X，之后T要做比大小，但X不能比较，这时需要对泛型作类型限制，见特征

## 特征

### 特征约束
- 做参数
```rust
pub fn notify(item1: &impl Summary, item2: &impl Summary) {} //使用语法糖
pub fn notify<T: Summary>(item1: &T, item2: &T) {} //特征约束，这里item1和item2是同一类型

// 多重约束
pub fn notify(item: &(impl Summary + Display)) {}
pub fn notify<T: Summary + Display>(item: &T) {}

// where约束,约束太多
fn some_function<T, U>(t: &T, u: &U) -> i32
    where T: Display + Clone,
          U: Clone + Debug
{}
```
- 做返回值
这里就一个问题，整个函数的返回类型虽然未知，但是要固定一致
```rust
//error 这里会返回两种类型，不确定
fn returns_summarizable(switch: bool) -> impl Summary {
    if switch {
        Post {}
    } else {
        Weibo {}
    }
}
```

### 特征对象
这个之后补充

### 关联特征
可以理解为特征的泛型
```rust
pub trait Iterator {
    type Item;

    fn next(&mut self) -> Option<Self::Item>;
}
```
### 默认泛型特征
```rust

trait Add<RHS=Self> {
    type Output;
    fn add(self, rhs: RHS) -> Self::Output;
}

struct Point {
    x: i32,
    y: i32,
}

impl Add for Point {
    type Output = Point;

    fn add(self, other: Point) -> Point {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}
```
```rust
struct Millimeters(u32);
struct Meters(u32);

impl Add<Meters> for Millimeters {
    type Output = Millimeters;

    fn add(self, other: Meters) -> Millimeters {
        Millimeters(self.0 + (other.0 * 1000))
    }
}

```

### 同名调用
规则：
1. 调用类型上的方法
2. 调用特征上的方法
```rust
trait Pilot {
    fn fly(&self);
}

trait Wizard {
    fn fly(&self);
}

struct Human;

impl Pilot for Human {
    fn fly(&self) {
        println!("This is your captain speaking.");
    }
}

impl Wizard for Human {
    fn fly(&self) {
        println!("Up!");
    }
}

impl Human {
    fn fly(&self) {
        println!("*waving arms furiously*");
    }
}

fn main() {
    let person = Human;
    Pilot::fly(&person); // 调用Pilot特征上的方法
    Wizard::fly(&person); // 调用Wizard特征上的方法
    person.fly(); // 调用Human类型自身的方法
}
```
以上为正常方法，即含有&self，若为关联函数，使用完全限定语法
```rust
<Type as Trait>::function(receiver_if_method, next_arg, ...);
```
### 特征中使用特征约束
实现A特征之前要实现B
```rust
trait A:B{
}
```

## 一个问题
在看的zk项目中，发现有很多泛型嵌套，乱七八糟的，有新的理解，记录一下。
以下是个人理解：
首先需要明确，泛型只是静态多态，编译期就要**确定类型**，因此代码中手动明确泛型的类型，或者编译器可以推断出类型
结构体，函数，特征中的泛型要什么时间明确，怎样明确呢
```rust
trait test<A, B> {
    type C;
    fn f<G>(&self);
}

struct S<F> {
    sa: F,
}

impl<A, B, F> test<A, B> for S<F> {
    type C = i32;
    fn f<G>(&self) {
        let a: A;
        let b: B;
        let c: F;
    }
}

fn main() {
    let t = S { sa: 1 };
    <S<i32> as test<i32, i32>>::f::<i32>(&t);
}
``` 
这里手写了一个很丑的栗子
- 结构体S的泛型类型是编译器推断出来的
- 关联特征必须在为结构体实现时明确指定，这点是特征泛型和关联特征的区别，特征泛型在实现时可以不用明确给出，但最迟要在使用的时候明确
- 完全限定性语法的另一个用途，指定特征的泛型
- turbofish语法的用途，指定函数的泛型