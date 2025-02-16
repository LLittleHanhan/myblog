---
title: ZENO:A Type-based Optimization Framework for Zero Knowledge Neural Network Inference
date: 2024-01-30 
tags: paper
---
一篇论文
<!--more-->
> We propose **privacy type driven** and **tensor-type driven** optimizations to further optimize the generated zkSNARK circuit

## background
- zk-snarks nn , protect x or w , for inference
  
- funtion -> circuit -> constraint -> proof 
![](pic/zeno-1.png)
    - the first step is Generate, which takes a given arithmetic function F(x) and generates a circuit
    - the second step is Circuit Computation that condenses the circuit into constraints
    - the third step is Security Computation that generates proof
- latency
![](pic/zeno-2.png)

- properties in the constraints
  - privacy plays an important role where multiplying a public value and a private value does not lead to constraints. 
  - the addition is “free" in zkSNARK in terms of not introducing constraints, since a large number of additions can be expressed in a single linear combination by incorporating into the linear combination of private values. 
  - Third, in the circuit computation, children gate(e.g., 𝐺𝑎𝑡𝑒 1 to 𝐺𝑎𝑡𝑒 4 ) need to be computed before parent gates

## Privacy-type Driven Optimization
1. We should introduce privacy only when necessary and exploit as many “free” operations as possible to reduce cost
2. **Privacy-aware Knit Encoding**
   - prove the computation over two dot products 
    ![](pic/zeno-3.png)

## Tensor-type Driven Optimization
1. We present our ZENO circuit as an efficient intermediate representation (IR) from high-level zkSNARK NN arithmetic function to low-level constraints. 
![](pic/zeno-4.png)
   - public w, private x
   - 对向量点乘的优化，扩展加法门
   - 如何从f到电路再到约束？f->电路：树从叶结点生成，电路->约束：从根节点遍历？
   - 哪里慢了？
2. We propose ZENO circuit for fully connected, convolution, and pooling layers as an extension to ZENO circuit for dot product. 
    ![](pic/zeno-5.png)
