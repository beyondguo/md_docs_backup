---
title: 「精美手写笔记」Everything about Word2Vec 📒
published: 2021-6-24
sidebar: auto
---

# 「精美手写笔记」Everything about Word2Vec 📒
上一篇文章，讲解了词向量的基本思想，为什么需要词向量，以及如何构建词向量。

然而，仅仅知道思想是不够的，所以这篇笔记详细地展示了word2vec的内部结构（以skip-gram为例）和推导过程。并且介绍了训练过程的优化方法——层级softmax（hierarchical Softmax）和负采样（negative sampling），配有丰富的图示来辅助理解。

本文尝试全新的方法：全部手写(〃'▽'〃)！

希望大家喜欢。


![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624614275547-image.png)


![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624614284629-image.png)



![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624614290690-image.png)


![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624614298374-image.png)

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624614305719-image.png)


![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624614317377-image.png)


![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624614365119-image.png)


![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624614377175-image.png)

---




