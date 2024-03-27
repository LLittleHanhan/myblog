---
title: cryptocurrency
date: 2024-01-30 
tags: zk
---
<!--more-->
## 数字签名和数字证书
### 签名
这个问题之前研究过，简单来说是公私钥的加密解密
- 公钥加密，私钥解密，保证公钥方的消息只有私钥方可以查看，即加密通信
- 私钥加密，公钥解密，用于签名
- - 具体过程：<br>私钥方<br>`digest = hash(message)`<br> `signature = Private(digest)`<br>`send signature message Public`<br>共钥方<br>`verify hash(message)==Public(signature)`
### 证书
证书即信任的起点，目的是保证公钥的正确性防止中间人攻击,简单来书就是在公钥上套上一层ca机构的签名
![](pic/cryptocurrency-ca.jpg)



## 区块链和交易
以下以比特币为背景
### Merkle哈希树
结构很好理解![](pic/cryptocurrency-merkle.png)
用途：
- 快速比较两组数据
- 快速定位修改
- 零知识证明，比如证明者声称知道L1,可以由验证者给Hash0-1,Hash1让证明者计算根结点
### 区块链结构
### UTXO模型
这种模型不直接记录用户资产，而是通过记录未花费的Output来计算资产，任何一笔资产都可以追根溯源![](pic/cryptocurrency-utxo.webp)
实际情况可能是一对多，多对一等
### 区块链中的交易
关于交易有几个问题
- 怎么找到属于属于自己的UTXO，且别人不能用
- 一笔交易，除了交易双方的人的其他人能看到什么，信息隐蔽吗
### zcash
zcash相较于bitcoin使用了zk-snarks协议，来看看用在了哪里，以及和hash的关系


- l1链
- outsourcing computation



