---
title: 文本检索、开放域问答与Dense Passage Retrieval (EMNLP-20)
published: 2022-2-15
sidebar: auto
---

# 文本检索、开放域问答与Dense Passage Retrieval (EMNLP-20)

![](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220215131419.png)

- 标题：Dense Passage Retrieval for Open-Domain Question Answering
- 会议：EMNLP-20
- 机构：Facebook AI, University of Washington, Princeton University
- 链接：https://readpaper.com/paper/3099700870



> **一句话总结：**
> 一个很好的文本检索（IR）、问答（QA）的学习材料。开放域问答一般分两步——检索和阅读理解，本文提出的DPR是一个高效的基于语义匹配的检索模型，从而提高整体QA的效果，该思路对后续的对**比学习**的一系列工作都有启发。



## Open-domain question answering (QA)

QA可以分为Close-domain QA和Open-domain QA [1]，前者一般限制在某个特定领域，有一个给定的该领域的知识库，比如医院里的问答机器人，只负责回答医疗相关问题，甚至只负责回答该医院的一些说明性问题，再比如我们在淘宝上的智能客服，甚至只能在它给定的一个问题集合里面问问题；而Open-domain QA则是我们可以问任何事实性问题，一般是给你一个海量文本的语料库，比方Wikipedia/百度百科，让你从这个里面去找回答任意非主观问题的答案，这显然就困难地多。总结一下，Open-domain QA的定义：

> Open-domain QA，是这样一种任务：给定海量文档，来回答一个事实性问题（factoid questions ）。

注意，是回答factoid questions，即一个很客观的问题，不能问一些主观的问题。

这就类似于我们只在在搜索引擎里搜索某个问题的答案，我们希望搜索引擎能直接告诉我们答案，而不单单是找到一篇文章，然后我们需要自己找答案。

举个例子，正好我前几天搜索triplet loss的时候印象深刻：

我在Google里面直接搜triplet loss：

![image-20220216105121926](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220216105121.png)

首先，排在第一的结果就是Wikipedia的link，同时上面显示了一段话，让我不用点开Wikipedia，就能直接知道triplet loss**是什么，有什么用**，另外，仔细看的话，发现它把具体定义的那句话给我加粗了，就是我上面高亮的部分。

如果我继续点击进入Wikipedia，会更清楚的看到搜索引擎帮我把我想要的答案给高亮了（下图中的紫色部分是浏览器自动显示的，不是我选择的）：

![image-20220216104917986](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220216104918.png)

我想这就是个很好的例子，告诉我们open-domain QA想要做的是什么事。

当然，搜索引擎返回答案，还涉及到其他技术，比如有一些事实性问题，比如“姚明比奥尼尔高多少”，就需要借助知识图谱等技术来实现了。

### Open-domain QA的两个步骤

我们这里讲深度学习时代的Open-domain QA，传统的方法往往涉及到十分复杂的组件，而随着基于深度学习的阅读理解（reading comprehension）模型的兴起，我们现在可以把Open-domain QA给简化成两个步骤：文本检索与阅读理解。

① 文本检索：需要一个retriever，从海量文本中，找到跟question最相关的N篇文档，这些文档中包含了该问题的答案；
② 阅读理解：需要一个reader，从上面抽取出来的文档中，找到具体答案。

### 文本检索

对于文本的检索，目前最常用的方案就是基于倒排索引（inverted index）的关键词检索方式，例如最常用的ElasticSearch方案，就是基于倒排索引的，简言之，这是一种关键词搜索，具体的匹配排序规则有TF-IDF和BM25两种方式。这种文本检索的方式，是一种文本的bag-of-words表示，通过词频、逆文档频率等统计指标来计算question和document之间的相关性，可参考BM25的wiki[2]。

这种方式，是比较“硬”的匹配，当你搜索的关键词准确、明确时，搜索结果会非常好，但是当你只知道大概意思，搜索的结果可能就很差，因为bag-of-words的表示，无法认识到词语之间的相似性关系，因此就只能搜索到你输入的关键词，却无法找到词语不同但意思相近的结果。

一般的Open-domain QA都会直接使用这种基于TF-IDF或者BM25的匹配方式来进行检索，本论文则是提出，我们可以使用语义的匹配来达到更好的效果，弥补硬匹配的不足，这也是本论文的主要关注点。具体地，我们可以训练一个语义表示模型，赋予文本一个dense encoding，然后通过向量相似度来对文档进行排序。









## Dense Passage Retriever (DPR)



模型结构

损失函数设计

负样本选择

关键的Trick——In-batch negatives



## 实验设计&结果



参考文献：

[1]Wikipedia: Question Answering https://en.wikipedia.org/wiki/Question_answering
[2]Wikipedia: BM25 https://en.wikipedia.org/wiki/Okapi_BM25

