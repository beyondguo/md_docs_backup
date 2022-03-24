---
title:  Python Tricks
published: 
sidebar: auto
---

# Python Tricks

## `zip`相关操作

解压list of tuple：

```python
pairs = [('A',1),('B',2),('C',3)]
words, nums = list(zip(*pairs))
print(words)
print(nums)
>>>
('A', 'B', 'C')
(1, 2, 3)
```

两个list/tuple变为dictionary：

```python
words = ('A', 'B', 'C')
nums = (1, 2, 3)
dic = dict(zip(words, nums))
print(dic)
>>>
{'A': 1, 'B': 2, 'C': 3}
```

## 当前文件位置

使用`os.path.dirname(__file__)`来获取当前文件的位置，用`os.path.join`来拼接目录层级：

```python
os.path.join(os.path.dirname(__file__), '', '')
```

## `sum()`的骚操作

今天发现sum还可以用来拼接list：

```python
a = [1,2,3]
b = [4,5,6]
c = [7,8,9]
o = [100,200]
```

```python
sum([a,b,c],[])
>>>[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

```python
sum([a,b,c],o)
>>>[100, 200, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

神奇了，主要记住`sum(list of lists, [])`就可以对lists进行拼接。



## `dict`的各种骚操作

**合并**俩字典`d1.update(d2)`

**丢掉某个key**，同时还能获取这个key对应的value：`v = d.pop(k)`

这样这个d就丢掉了k，v对应着原本k的value。

还有，之前总是有这么一个需求：我想构造一个字典，同时每个key对应一个list，而key实现是不知道的，这个时候我就没法直接拿到一个key：value就直接append，还得先判断一下key存不存在，不存在得先添加一个空list，这就挺麻烦的。其实我们可以直接使用下面的方法：

```python
from collections import defaultdict
d = defaultdict(list)
```

然后这个d就默认了每个value都是list类型，我们就可以直接append：

```python
d['a'].append(1)
d
>>>defaultdict(list, {'a': [1]})
```

