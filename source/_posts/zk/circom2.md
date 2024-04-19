---
title: circom2
date: 2024-01-30 
tags: zk
---
circom的一个讲座，简单看看
<!--more-->
## Albert Rubio: Circom 2.0:A Scalable Circuit Complier

---

[video](https://www.youtube.com/watch?v=zRngElDdUNE&ab_channel=Delendum)
- two-folded
  1. describe the circuit
  2. compute the witness    
> you have to consider these constraints are polynomial constraints so defining the constraints is not going to give you an effencient way to compute the witness in many cases. so then you have to provide another way to compute the witness.
- three different instructions
  - `out === in1*in2` symbolic level only
  - `out <-- in1*in2` computation level only
  - `out <== in1*in2` both
- simplify
  ![](pic/circom2-1.png)