---
title: rust(九)循环和Iterator
date: 2024-03-07 
tags: rust
---

<!--more-->

## 迭代器
![](/pic/rust-iter.png)
### `iter = collection.into_iter()`
```rust
pub trait IntoIterator {
	type Item;
	type IntoIter: Iterator<Item = Self::Item>;
	fn into_iter(self) -> Self::IntoIter;
}

pub trait Iterator {
    type Item;
}
```
- 为类型实现IntoIterator，可以将类型转换为迭代器
- 何为迭代器——实现Iterator特征的类型
- `IntoIter`类型迭代器会拿走**被迭代值的所有权**
### `iter = collection.iter()`
```rust
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
- Iter结构体实现了Iterator特征，Iter为迭代器
- 以上以slice类型为例，每种可迭代类型有单独的实现，如下
```rust
std::collections::binary_heap::Iter   BinaryHeap 元素上的迭代器。
std::collections::btree_map::Iter     BTreeMap 条目上的迭代器。
std::collections::btree_set::Iter     BTreeSet 项上的迭代器。
std::collections::hash_map::Iter      HashMap 条目上的迭代器。
std::collections::hash_set::Iter      HashSet 项上的迭代器。
std::collections::linked_list::Iter   LinkedList 元素上的迭代器。
std::collections::vec_deque::Iter     VecDeque 元素上的迭代器。
...
```

### `iter = collection.iter_mut()`
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


## 消费者与适配器
- 返回迭代器类型的就是迭代器适配器，返回其他类型的就是消费者适配器
```rust
fn count(self)->usize{}


```
