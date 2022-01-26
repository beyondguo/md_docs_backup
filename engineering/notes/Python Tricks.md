---
title:  Python Tricks
published: 
sidebar: auto
---

# Python Tricks

## zip相关操作

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

