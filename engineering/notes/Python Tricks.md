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



## Pandas系列

关于dataframe的各种操作，官方文档：https://pandas.pydata.org/docs/reference/frame.html#

### rename columns

```python
>>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
>>> df.rename(columns={"A": "a", "B": "c"})
   a  c
0  1  4
1  2  5
2  3  6
```



### remove/drop columns

```python
>>> df.drop(columns=['B', 'C'])
   A   D
0  0   3
1  4   7
2  8  11
```

### 统计某column各值的数量

`df.your_column.value_counts()`



### 清理空值

```python
df = df.dropna()
df = df[df.content != ''] 
```



### 筛选

例如有一个column为content，需要过滤过content分词后长度大于10的样本：

`df[df['content'].apply(lambda x: len(x.split(' '))>5)]`





## `random`库

- 随机取一个序号：
  `random.randrange(stop)` or `random.randrange(start,stop)`

- 随机不重复抽样（用sample）：
  `random.sample(population, k, counts)`
  从population中不重复地抽样k个，其中，我们可以通过跟population同等长度的counts来指定每个元素在总体中的个数。For example, `sample(['red', 'blue'], counts=[4, 2], k=5)` is equivalent to `sample(['red', 'red', 'red', 'red', 'blue', 'blue'], k=5)`.

  返回一个list，哪怕只有一个。

- 允许重复的采样（用choices）：
  `random.choices(population,weights,k)`
  返回一个list，哪怕只有一个。

- 随机采样一个（用choice）：
  ``random.choice(population)`，返回一个值。

- 打乱顺序：
  `random.shuffle(x)`

  



## logging

https://stackoverflow.com/questions/6386698/how-to-write-to-a-file-using-the-logging-python-module

https://stackoverflow.com/questions/8455171/how-to-choose-handler-while-logging-in-python

https://docs.python.org/3/howto/logging.html



## string字符串相关骚操作

### 建立一个字符替换表，translate函数

```python
table = str.maketrans({"-":  r"\-", "]":  r"\]", "[":  r"\[", "\\": r"\\", \
                       "^":  r"\^", "$":  r"\$", "*":  r"\*", ".":  r"\.", \
                        "(":  r"\(", ")":  r"\)", \
                       })
s_new = s.translate(table)
```

就不用写一堆`replace`了。

