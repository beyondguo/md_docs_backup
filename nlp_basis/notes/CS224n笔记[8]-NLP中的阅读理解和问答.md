---
title: CS224n笔记[8]:NLP中的阅读理解和问答
published: 2021-6-24
sidebar: auto
---

# CS224n笔记[8]:NLP中的阅读理解和问答

## 为什么需要问答：

网络上大量的文本，仅仅获取相关的文本还是不够的，很多时候我们想直接得到答案。这种情况在移动端更加常见。这就是问答系统背后的动机。

问答系统可以分解成两块：
1. 找到那些可能包含答案的文档
    使用传统的信息抽取、web搜索等方式实现
2. 从文档或者段落中找到答案
    这就就是所谓的“阅读理解”问题

所以我们讨论的重点就是“阅读理解”——Reading Comprehension



## 阅读理解（Reading Comprehension）


传统的QA系统，例如LCC QA System，十分复杂，但是对于那些“factoid question”表现还是很好的。所谓的factoid question，一般指那些答案就是一个实体的问题，比如“成龙哪一天出生？”、“湖北的省会是哪里？”这种简单的问题。



数据集：SQuAD（Stanford Question Answering Dataset） 
/丝瓜的/ ——目前QA领域中使用最广泛的数据集

SQuAD 1.1版本：
- 该数据集包含100k从维基百科上构建的问答，包括Question，Passage和Answer。
- Answer必须是passage中的一段文字（a span）。
    因此这种方式，也成为“抽取式问答”（extractive question answering）。
- 数据集中会对每个问题提供三个人的回答作为标准答案（3 gold answers），只要跟其中一个一样就算对。

SQuAD 1.1的评价方式：
- 精确匹配法：简单粗暴，计算对的百分比。
- F1值：使用词袋，计算prec，acc从而计算F1。

SQuAD 2.0版本：
考虑到1.0版本中的问题：所有的问题都有答案，这样就让机器不用去判断是否一段文字是不是真的回答了问题。因此，在2.0版本中，增加了“无答案问题”。
1/3的训练集中没有答案，1/2的验证集和测试集中没有答案。

此时在评价时，如果不回答，则德1分，只要给了答案，就是0分。


从现有的模型，即使是在SQuAD leaderboard上顶尖的模型，我们分析它的一些错误情况，也可以发现，其实这些模型也并没有真正理解文本中的意思，它们做的实际上还是一种“匹配”。


SQuAD数据集的局限性：
1. 回答全都是文中的一段文本，没有“是否”、“数数”、非直接的“为什么”这些问题。
2. 答案全都从段落中找，使得问题本身变得过于简单
3. 不涉及从多个句子中结合答案等等复杂情况

但它依然很不错，因为数据高度结构化、干净、目标明确。工业中也常常作为最开始的训练数据。


---

隔了好久才发现写过这篇笔记，似乎没写完。。。mark一下，日后补上。