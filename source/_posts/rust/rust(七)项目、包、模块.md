---
title: rust(七)项目、包、模块
date: 2024-03-04 
tags: rust
---

了解rust的package,crate,module,看完就可以看大点项目了
<!--more-->

1. package > crate > module
2. 一个项目就是一个package，项目中可能会有多个crate，一个crate中会有多个module

## package

```shell
cargo new project # 创建一个package，里面有一个二进制类型的crate
# --project
#   --cargo.toml
#   --src
#     --main.rs

cargo new lib --lib # 创建一个package，里面有一个lib类型的crate
# --lib
#   --cargo.toml
#   --src
#     --lib.rs 
```


## crate
- 二进制类型的crate可以有多个可执行文件
```shell
# --project
#   --cargo.toml
#   --src
#     --main.rs
#     --bin
#       --test.rs
cargo run --bin test.rs
```
- lib类型的crate只能有一个lib.rs文件
- lib类型和二进制类型可以共存

## module
```rust
// --lib
//   --cargo.toml
//   --src
//     --lib.rs

// lib.rs
mod front_of_house {
    mod hosting {
        fn add_to_waitlist() {}
        fn seat_at_table() {}
    }

    mod serving {
        fn take_order() {}
        fn serve_order() {}
        fn take_payment() {}
    }
}
fn eat_at_restaurant() {}
```
```
crate
 └── eat_at_restaurant()
 └── front_of_house
     ├── hosting
     │   ├── add_to_waitlist()
     │   └── seat_at_table()
     └── serving
         ├── take_order()
         ├── serve_order()
         └── take_payment()
```
- 引用方式
  - 绝对路径crate：`crate::front_of_house::hosting::add_to_waitlist()`
  - 相对路径self和super
- 把module视作和函数一样的实体，同级可见，父无法访问子，子可以访问父
- 使用pub访问控制，注意pub只能控制当前的东西，不能连带控制内部

## 文件和module
- 本质上一个crate就一个文件，拆散后用`mod xxx;`连接
- 使用方法和上小节一样
```rust
// --src
//   --main.rs
//   --test.rs

//main.rs
mod test; //相当于c中的include
```

下面是复杂的情况，以目前所看的代码为例
```rust
// --src
//   --lib.rs
//   --poseidon/
//     --mod.rs
//     --matrix.rs
//     --... 

//mod.rs
pub mod matrix;
```

## use
- 引入范围：
  - 本crate中的mod
  - 第三方包，需要在cargo.toml中更改[dependencies]
- 相当于给目标起个别名，方便使用，具体是引用模块还是直接引用函数，需要具体情况具体分析
- 命名冲突问题
    ```rust
    use std::fmt::Result;
    use std::io::Result as IoResult;

    fn function1() -> Result {
        // --snip--
    }
    fn function2() -> IoResult<()> {
        // --snip--
    }
    ```
- `{}`或`*`引用多个目标

## 使用use的可见性问题
这个完全可以让编译器帮忙纠错
```rust
// 一个名为 `my_mod` 的模块
mod my_mod {
    // 模块中的项默认具有私有的可见性
    fn private_function() {
        println!("called `my_mod::private_function()`");
    }

    // 使用 `pub` 修饰语来改变默认可见性。
    pub fn function() {
        println!("called `my_mod::function()`");
    }

    // 在同一模块中，项可以访问其它项，即使它是私有的。
    pub fn indirect_access() {
        print!("called `my_mod::indirect_access()`, that\n> ");
        private_function();
    }

    // 模块也可以嵌套
    pub mod nested {
        pub fn function() {
            println!("called `my_mod::nested::function()`");
        }

        #[allow(dead_code)]
        fn private_function() {
            println!("called `my_mod::nested::private_function()`");
        }

        // 使用 `pub(in path)` 语法定义的函数只在给定的路径中可见。
        // `path` 必须是父模块（parent module）或祖先模块（ancestor module）
        pub(in crate::my_mod) fn public_function_in_my_mod() {
            print!("called `my_mod::nested::public_function_in_my_mod()`, that\n > ");
            public_function_in_nested()
        }

        // 使用 `pub(self)` 语法定义的函数则只在当前模块中可见。
        pub(self) fn public_function_in_nested() {
            println!("called `my_mod::nested::public_function_in_nested");
        }

        // 使用 `pub(super)` 语法定义的函数只在父模块中可见。
        pub(super) fn public_function_in_super_mod() {
            println!("called my_mod::nested::public_function_in_super_mod");
        }
    }

    pub fn call_public_function_in_my_mod() {
        print!("called `my_mod::call_public_funcion_in_my_mod()`, that\n> ");
        nested::public_function_in_my_mod();
        print!("> ");
        nested::public_function_in_super_mod();
    }

    // `pub(crate)` 使得函数只在当前包中可见
    pub(crate) fn public_function_in_crate() {
        println!("called `my_mod::public_function_in_crate()");
    }

    // 嵌套模块的可见性遵循相同的规则
    mod private_nested {
        #[allow(dead_code)]
        pub fn function() {
            println!("called `my_mod::private_nested::function()`");
        }
    }
}

fn function() {
    println!("called `function()`");
}

fn main() {
    // 模块机制消除了相同名字的项之间的歧义。
    function();
    my_mod::function();

    // 公有项，包括嵌套模块内的，都可以在父模块外部访问。
    my_mod::indirect_access();
    my_mod::nested::function();
    my_mod::call_public_function_in_my_mod();

    // pub(crate) 项可以在同一个 crate 中的任何地方访问
    my_mod::public_function_in_crate();

    // pub(in path) 项只能在指定的模块中访问
    // 报错！函数 `public_function_in_my_mod` 是私有的
    //my_mod::nested::public_function_in_my_mod();
    // 试一试 ^ 取消该行的注释

    // 模块的私有项不能直接访问，即便它是嵌套在公有模块内部的

    // 报错！`private_function` 是私有的
    //my_mod::private_function();
    // 试一试 ^ 取消此行注释

    // 报错！`private_function` 是私有的
    //my_mod::nested::private_function();
    // 试一试 ^ 取消此行的注释

    // 报错！ `private_nested` 是私有的
    //my_mod::private_nested::function();
    // 试一试 ^ 取消此行的注释
}
```
限制可见性语法
```rust
pub(crate)
pub(in crate::a)
pub(self)
pub(super)
```
具体可以仔细看`my_mod::nested::public_function_in_my_mod();`