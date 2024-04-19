---
title: java env
date: 2024-01-30 
tags: java
---

java基础概念
<!--more-->
## jdk和jvm
- jdk: 早期叫jdk1.X，之后叫java SE X，最常用的版本jdk1.8（java 8）
- jvm： 执行java程序的虚拟机
## java编译执行过程
1. 首先由.java文件编译成.class文件
2. 之后将.class文件链接在jvm上执行
[javac编译](https://zhuanlan.zhihu.com/p/74229762)
## 安装
```
sudo apt install openjdk-X-jdk
```
## maven
```
mvn clean 
mvn compiler # 编译
mvn package # 制作jar包
mvn install # 下载本的仓库
```