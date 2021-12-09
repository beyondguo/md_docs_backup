---
title: 利用标签与样本之间的统计信息改善文本分类中的embedding表示(CIKM2020)
published: 2021-6-30
sidebar: auto
---

# 利用标签与样本之间的统计信息改善文本分类中的embedding表示

- 论文标题：Exploiting Class Labels to Boost Performance on Embedding-based Text Classification
- 发表会议：CIKM-2020
- 组织机构：Queen Mary University of London 



> **一句话评价**：
>
> 又是一篇标题听上去挺牛逼，实际简单的不行的论文。不过还是有一点借鉴意义吧。



## 背景

基于文本Embedding表示的文本分类已经非常常见了，基本是文本分类的基本选择之一。然而，传统的embedding方式，都是直接使用预训练好的embedding，比如Word2Vec、Glove等。

这些词向量是通过外部的语料训练的，而没考虑到我们具体分类任务中的不同的词对于各个类别不同的重要性和相关性。**我们希望能得到一个任务相关的文本表示，能让那些跟我们的任务更相关的词语得到更强的表示。**

比方说，我做一个情感分类，实际上我需要关注的就是情感词，其他的很多话对我来说都是废话，甚至是干扰。对于文本向量表示，我们经常是要把文本中所有的词的向量综合起来形成一个统一的表示的，这样的话其他的任务无关的词就会影响我们整体的表示。



## 方法

### Term Frequency-Category Ratio (TF-CR)

作者提出了一个名为**Term Frequency-Category Ratio**（后简称TF-CR）的指标，用于给数据集中的词汇打分。

**某个词针对某个类**的TF-CR的表达式为：
$$
TFCR(w,c) = \frac{|w_c|}{N_c}*\frac{|w_c|}{|w|}
$$

- 其中c是给定的某个类别，w代表某个词，$w_c$则是代表在类别c的预料中的词w
- 第一项$\frac{|w_c|}{N_c}$就是term frequency，是衡量在某个类别的词中，某个词出现的频率。「这个词在这个类中的**重要性**」
- 第二项$\frac{|w_c|}{|w|}$则称为category ratio，衡量某个词出现的总次数中，多大的比例是出现在这个类别中。「这个词跟这个类的分布上的**相关性**」

通过这样的指标，那些在某个类别中既词频高又类别独有的词，会得到很高的得分。而那些虽然类别独有但频率很低，或者高频词但独有程度低的词，得分会较低。

我画了下图来示意TF-CR的计算过程：

![TF-CR计算示意图](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20210704223226tfcr.png)

### 使用TF-CR调整文本表示

首先假设我们的使用场景是文本分类，有k个类别。

1. 每个词都会对每个类别计算一个TF-CR指标作为权重，即一个词有k个权重。
2. 将给定文本中所有词的embedding进行加权求和，得到k个embedding。
3. 将k个embedding拼接起来，得到最终的文本向量表示。

为了方便记忆，上面的过程可以这样表示：

![image-20210704223356029](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20210704223356tfcr%E8%A1%A8%E7%A4%BA%E7%A4%BA%E6%84%8F%E5%9B%BE.png)

这样做有什么意义呢？这k个embedding，各自都是相应类别的重要特征，通过这样的操作，我们「**把原文本混杂在一起的特征，做了一个分离**」，这样对于后面的分类器来说，就可以更好地理解文本的特征。

作者在很多数据集上做了实验，这里贴出其中一部分：

![image-20210704214150509](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20210704214150.png)

实验发现，数据量越大，TF-CR的效果越好，因为对词权重的计算更加准确了。

---

以上就是这个文章的核心内容，是不是贼简单？

而且，为了验证这样对embedding的调整的有效性，作者是直接吧embedding作为特征，输入到LR这种简单的分类器中，没有微调的过程，跟TF-IDF、KLD等权重方法对比了一下，发现效果显著。所以实验也是简单的一批。（CIKM很好发吗？）

提一嘴，baseline使用的TF-IDF和KLD，都做了些调整，为了给TFCR对应，这两种baseline也是针对“词相对于类”计算的指标，很明显，这样的调整，对于TF-IDF的IDF的计算，是大打折扣的，甚至说是畸形的。所以在作者的实验中，这些baseline方法，甚至还不如不加权重。

很明显，我们可以设计出更好的权重指标，来超越TF-CR。不过，这种**将不同类别各自重要的信息进行分离提取，然后喂给模型**的思路，还是值得借鉴和思考的（虽然根据related work，这也不是作者的原创）。





