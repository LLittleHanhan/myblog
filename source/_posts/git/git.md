---
title: git
date: 2024-01-30 
tags: git
---
<!--more-->
## config
## git模型
观察.git/object目录
文件名均是哈希
1. git add
    此时，每个文件会生成一个blob类型的object
2. git commit
    创建一个tree类型的object，内容为blob文件名
    创建一个commmit类型的object，内容为**提交信息**，**tree类型的文件名**，**上一个commit类型的object的文件名**
3. 各分支的head存储在refs/head中
   HEAD内容为当前工作分支的头

![](/pic/git-1.png)

## 合并策略
1. git merge
   git merge有两种合并策略
   - 快进合并(faster forward)：当前分支在待合并分支为线性关系，直接合并即可，不会产生merge commit
    ![](/pic/git-2.gif)
   - 递归三路合并（这里简单化，只考虑三路合并）,把待合并分支的所有更改，生成一个新的commit接到当前分支。本次commit object会有两个parent指针，即指向条线的上一个commit object
    ![](/pic/git-3.gif)
2. git rebase
   ```shell
   git checkout feature
   git rebase master
   git checkout master
   git merge feature
   ```
   ![](/pic/git-4.gif)

> 两种合并注意方向

3. git push
   默认仅在 fast-forward 状态下才可以合并，即git push在远端指针不是本地指针的祖先时会拒绝覆盖。而 `–force`，可以让 Git 不进行这个检查，直接覆盖远端对应 master 指针的内容。

## 基本语法
```shell
git init
git add remote
git remote add origin https://github.com/LLittleHanhan/test.git
git remote -v
git config --global https.proxy socks5://127.0.0.1:10800
git config --global http.proxy socks5://127.0.0.1:10800

git push
git push -force
git fetch
git merge
git pull --no-rebase
git pull --rebase

git branch
git branch feature
git checkout branch_name

```
