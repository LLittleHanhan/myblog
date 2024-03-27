---
title: circom1
date: 2024-01-30 
tags: zk
---
<!--more-->
## signal
- 三类信号
    ```
    signal input a;
    signal b;
    signal output c;
    ```

- public & private
  - 中间信号总是private
  - main component的输入信号可以指定public`component main {public [in1,in2]} = XXX();`
  - 输出信号是public，否则外部不可见

- var & signal
  - `var a = 0;` 注意：`=`赋值操作没有返回值
  

## template & component
- templates是定义，components是实例化
- templates不能嵌套
<br>
- components实例化时要指定明确的值
- main components的input signal是对外的接口有值，内部templates的input signal需要手动赋值
- 当components的input signal全都赋完值后才会实例化

```
template A(N1,N2){
    signal input in;
    signal output out; 
    out <== N1 * in * N2;
}

template wrong (N) {
    signal input a;
    signal output b;
    a <== 1; //Exception caused by invalid assignment: signal already assigned
    component c = A(a,N); // Every component instantiation must be resolved during the constraint generation phase
    component c = A(1,1);//Component c is created but not all its inputs are initialized
}
component main {public [a]} = wrong(1);
```
- components可以先声明，然后在第二步中初始化。如果有几个初始化指令（在不同的执行路径中），它们都需要是同一模板的实例化（可能具有不同的参数值）
- arrays of components不允许在定义时进行初始化，只能逐个组件进行实例化，所有components都必须是同一模板的实例
<br>
- parallel关键字
- custom模板

## constraint
>总结
>signal 和 var 本质上都是变量，区别在于:At compilation time, the content of a **signal** is always considered unknown
>signal可以赋值var,这时var就是**unknown**
>模板实例化时只能使用确定值的变量
<br>

>R1CS
>这是一个约束系统，If we have an arithmetic circuit with signals s_1,...,s_n, then we define a constraint as an equation of the following form:
>`(a_1*s_1 + ... + a_n*s_n) * (b_1*s_1 + ... + b_n*s_n) + (c_1*s_1 + ... + c_n*s_n) = 0`
>circom allows programmers to define the constraints that define the arithmetic circuit. All constraints must be quadratic of the form A*B + C = 0,
>where A, B and C are linear combinations of signals. circom will apply some minor transformations on the defined constraints in order to meet the format A*B + C = 0:


## function
- 这个就是一个正常的函数，表示一个通用的计算过程


## include
- `include test.circom`同c++
- 编译时同g++ `-l`寻找文件路径


## operator
```circom
//boolean conditional
var z = x>y?x:y;
//boolean
&& || !
//relational operators
> < >= <= == !=
```
```
//arthmetic operators
```
![](pic/circom.png)
```
//bitwise operators
& | ~ ^(异或) >> <<
//arthmetic bitwise 均可+=
```