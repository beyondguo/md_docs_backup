---
title: 又一些有趣的特征空间增强方法——Delta和GE3
published: 2021-12-16
sidebar: auto
---

# 又一些有趣的特征空间增强方法——Delta和GE3

这里主要涉及到两篇论文：

- A Closer Look At Feature Space Data Augmentation For Few-Shot Intent Classification. (EMNLP-19 workshop)
- Good-Enough Example Extrapolation. (EMNLP-21)

第一篇文章中，对比了6中feature space augmentation方法，其中有一种叫 Linear Delta 的方法最有意思，虽然他实验的效果也不是最好的；第二篇文章中，提出了一种新的基于extrapolation的方法，也挺有意思。所以这里主要记录一下这两种有趣的方法，启发我们开更多的脑洞~

