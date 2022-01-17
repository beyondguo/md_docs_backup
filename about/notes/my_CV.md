---
title:  郭必扬的简历（MSRA）
published: 2022-01-11
sidebar: auto
---

# 郭必扬

- <small>Tel: 185-7275-6723</small>
- <small>E-Mail: guo_biyang(@)163(dot)com</small>

> 目前是**上海财经大学信息管理与工程学院人工智能实验室**（SUFE AI Lab）在读二年级博士生。本科硕士均就读于上海财经大学信息管理与工程学院。


- <small> <bold>微信公众号</bold>：SimpleAI</small>
- <small><bold>个人博客</bold>：<a>https://beyondguo.github.io/</a></small>
- <small><bold>知乎文章</bold>：<a>https://www.zhihu.com/people/guo-bi-yang-78/posts</a></small>


## 研究兴趣：

- NLP中的数据增强（Data Augmentation in NLP）
- 数据为中心的人工智能（Data-centric AI）
- 文本分类与标签表示（Text Classification & Label Embedding）



## 已有科研工作：

<style>
table th:first-of-type {
    width: 5%;
}
table th:nth-of-type(2) {
    width: 95%;
}
</style>

| Year |Publications |
| ---- |----  |
|2021|[**Roles of Words: What Should (n’t) Be Augmented in Text Augmentation on Text Classification Tasks?** (Under Review)](https://openreview.net/pdf?id=_jpxhquKzO9) <p><small>{**Biyang Guo**, Songqiao Han, Hailiang Huang}</small></p>**简介**：传统的文本增强技术很少考虑词的角色对增强的不同影响，本文创造性地从**统计相关度**和**语义相似度**对词语进行了四种角色划分，并基于此提出**针对性文本增强技术**（STA），在9个数据集上验证比传统方法显著有效。<br /><br /><center><img src='https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220109164804.png' width=70%/></center>|
|2020 |[**Label Confusion Learning to Enhance Text Classification Models** (AAAI-2021)](https://arxiv.org/abs/2012.04987) <p><small>{**Biyang Guo**, Songqiao Han, Xiao Han, Hailiang Huang, Ting Lu}</small></p>**简介**：本文提出了**标签混淆模型**（LCM）用来学习标签之间的overlap，从而动态地生成**模拟标签概率分布**来作为分类模型的target。在文本分类和图像分类的实验上都表明，LCM可以取得比one-hot label、smoothed label更好的效果。<br /><br /><center><img src='https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220109164546.png' width=70% align='center'/></center>|


## 科研之外

我是一名技术科普爱好者，<u>**喜欢并追求将艰深复杂的理论知识用通俗易懂的语言描绘出来**</u>。在科研之外的时间，我喜欢撰写技术博客，进行模型、论文解读。代表作品如下：

- <small><a href='https://zhuanlan.zhihu.com/p/147310766'>整理了12小时，只为让你20分钟搞懂Seq2seq「知乎670+赞」</a></small>
- <small><a href='https://zhuanlan.zhihu.com/p/42559190'>从此明白了卷积神经网络（CNN）「知乎1860+赞」</a></small>
- <small><a href='https://zhuanlan.zhihu.com/p/71200936'>何时能懂你的心——图卷积神经网络（GCN）「知乎2700+赞」</a></small>
- <small><a href='https://zhuanlan.zhihu.com/p/74242097'>GraphSAGE：我寻思GCN也没我牛逼「知乎1360+赞」</a></small>


包含上述作品在内，我在知乎上的专栏[DeepLearning学习笔记](https://www.zhihu.com/column/deeplearningnotes)和[NLP学习笔记](https://www.zhihu.com/column/pythontricks)累计被收藏**2.5W次**，获得众多深度学习和自然语言处理领域同学的认可。





---

## ★ 未来工作畅想：

### 1. **特征空间数据增强**

**➤背景：**

在原始数据上进行数据增强，我们需要根据数据本身的特点来设计各种增强方法，这样有两个问题：① 增强方法不通用，比如CV中的旋转、裁剪，就没法用到NLP中 ；②增强样本容易发现巨大含义改变，尤其针对NLP这种离散数据形式。

有研究表明，在特征空间进行增强，相比于原始数据空间，更有可能得到真实的样本。同时，对特征的操作对于所有模态的特征都是通用的。因此，特征空间增强具有非常大的潜力和想象空间。

**➤现有研究的问题：**

特征空间数据增强的现有做法主要包括interpolation，extrapolation，noising。现有的工作存在以下问题：

- 没有统一的task进行对比
- 多数工作缺乏开源代码
- 如果使用全局信息，就难以对特征进行fine tune，而如果想进行fine tune，就只能只用一个batch内的特征。（这导致现在多数的工作，都只能针对静态的特征进行增强，然后对增强后的特征进行下游任务）

以上问题，使得特征空间增强的研究受到很大限制，没有成为数据增强的主流方法。目前特征空间增强最具代表性的还是mixup，但这只是interpolation的一种方法。

**➤未来工作设想：**

① 可以对现有的工作进行梳理，最好能够把它们整理到一个统一的框架中，并进行开源实现，使得特征增强可以作为一个方便易用的算子。这样，可以写一个综述。

②能否把现有的静态特征增强，有效地迁移到fine tune的训练框架中？取得比mixup系列更好的效果？

②相比于原始数据空间的增强，特征增强的方法目前还比较少。因此有很大的潜力可以设计新的特征增强方法/对现有工作进行改进。



### 2. **Data-centric NLP**

**➤背景：**

长期以来，AI的发展都是以模型为中心的，即如何设计出更好的模型、损失函数、正则化手段等等来提高模型在下游任务的表现。然而，随着大规模预训练模型的提出，我们能够在模型层面进行的创新和改进已经越来越少。

近两年，Data-centric AI的概念被提出并受到越来越多人的注意。通过对模型输入的数据——深度学习的原料——的改善，我们可以让模型学得更好、更多、更可信，同时降低训练成本。

**➤未来工作设想：**

目前的Data-centric AI的主要方法包括：

- 数据增强
- 数据治理（例如，错误数据纠正/过滤）
- 标签辅助的数据集改善

我认为，Data-centric AI的基本假设是：“模型学习到的，都是来自我们提供的数据。然而我们提供的数据则不一定好。” 

我曾在针对性文本增强的工作中思考过这个问题：（下图来自针对性文本增强的[preprint](https://arxiv.org/abs/2109.00175)）

![Biyang Guo, 2021, https://arxiv.org/abs/2109.00175](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220111161949.png)

如上图所示意，模型学习到的实际上是由数据集所提供的特征，这往往跟我们下游任务中真实的特征有一定的gap，而data-centric AI能做的事情可以从如下两方面着手：

**① 增大数据集特征，与真实任务特征的overlap；**

**② 减少数据集中带来的bias、noise。**



回顾现有的工作，则可以发现现有工作的一些**问题**：

- 数据增强，只是增大了原数据集的覆盖面，因此在增大overlap的同时，也带来了更多的bias、noise，因此效果有限；
- 错误数据过滤，只是单纯的降低原数据集的noise，所以效果很有限；
- 标签辅助数据集改善，目前常用的是利用标签来生成一些新样本，所以只是稍微增大了上图中的一些overlap，并没有降低bias、noise，因此效果也有限。

因此，未来工作，可以基于上面这些问题来开展。