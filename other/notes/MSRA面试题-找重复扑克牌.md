# MSRA 面试题

- 2022.1.19
- 面试老师：Bartuer Zhou
- 整理：郭必扬

### 题目：
一副扑克（54张牌），现在又拿一副，随机抽一张，跟原来那副混在一起（即有55张牌），洗牌。

要求：设计一个算法，把重复的牌找出来。


```python
# 翻译题目：
import random
m = 27  # 随机抽取的一张牌
A = list(range(1,55))
A.append(m)
random.shuffle(A)
```

## 算法1：遍历计数法


```python
def f1(A):
    for a in set(A):
        c = A.count(a)
        if c == 2:
            return a
    return -1

assert f1(A) == m, 'wrong'
```

计算平均耗时：


```python
%timeit f1(A)
```

    12.6 µs ± 52.8 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)


## 算法2：通过哈希表记录


```python
def f2(A):
    dic = {}
    for a in A:
        if a in dic:
            return a
        else:
            dic[a] = 1
    return -1

assert f2(A) == m, 'wrong'
```

计算平均耗时：


```python
%timeit f2(A)
```

    2.1 µs ± 8.73 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)


## 算法3：巧妙利用问题特性
原始的牌1-54的和是固定的，所以重复的那一张牌的数，就一定等于加牌后的求和减去原始的求和。


```python
n = sum(range(1,55))
def f3(A):
    return sum(A) - n

assert f3(A) == m, 'wrong'
```


```python
%timeit f3(A)
```

    358 ns ± 1.76 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)


使用numpy的求和可以更快(理论上，但在Mac M1上测试发现更慢)：


```python
import numpy as np
d = np.array(A, dtype=np.int8)

def f4(d):
    return np.sum(d) - n

assert f4(d) == m, 'wrong'
```


```python
%timeit f4(d)
```

    1.71 µs ± 4.98 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)


### 下附各种求和函数的对比：


```python
%timeit sum(A)
```

    319 ns ± 0.15 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)



```python
%timeit sum(d)
```

    4.58 µs ± 29.2 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)



```python
%timeit np.sum(d)
```

    1.55 µs ± 2.54 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)



```python
%timeit d.sum()
```

    718 ns ± 2.08 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)


## 总结：

Mac M1 测试结果：

| 算法      | 耗时 |
| ----------- | ----------- |
|遍历计数法|12.6 µs ± 52.8 ns|
|通过哈希表记录|2.1 µs ± 8.73 ns|
|求和相减|**358 ns** ± 1.76 ns|
|求和相减(np)|1.71 µs ± 4.98 ns per|



Windows Intel Xeon W-2123 CPU 3.60GHz 测试结果：

| 算法      | 耗时 |
| ----------- | ----------- |
|遍历计数法|26.8 μs ± 752 ns |
|通过哈希表记录| 4.58 μs ± 149 ns|
|求和相减| **663 ns** ± 4.92 ns|
|求和相减(np)|4.43 μs ± 184 ns |

**在Mac M1上和Windows Intel上的测试结果在rank上一致**。


```python

```
