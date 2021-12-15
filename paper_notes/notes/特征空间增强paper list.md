---
title: 特征空间增强paper list
published: 2021-12-15
sidebar: auto
---

## - Dataset Augmentation in Feature Space. (ICLR-17 workshop)

![](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20211128154757.png)

interpolation和extrapolation

## - A Closer Look At Feature Space Data Augmentation For Few-Shot Intent Classification. (EMNLP-19 workshop)

利用两个样本的difference来应用到第三个样本上，从而产生新样本。

## - Good-Enough Example Extrapolation. (EMNLP-21)

设计了一种跨类别的增强方法：

![image-20211215220023461](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/image-20211215220023461.png)

## - Feature Space Augmentation for Long-Tailed Data. (ECCV-20)

提出使用资源丰富的类别来指导低资源/长尾类别的数据增强，从而复原低资源类别所缺失的信息。

设计了一个class activation map，来对feature的每个维度进行区分，得到class-specific feature和class-generic feature：

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/image-20211215221647548.png)

