---
title: 小样本学习与Triplet Loss，数据增强和课程学习 (NAACL-21)
published: 2022-2-10
sidebar: auto

---

# 小样本学习与Triplet Loss，数据增强和课程学习

![](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220210230114.png)

- 标题：Few-Shot Text Classification with Triplet Networks, Data Augmentation, and Curriculum Learning.
- 会议：NAACL-21 (short paper)
- 链接：https://readpaper.com/paper/3139506831



> **一句话总结：**
> 一个简洁明了的short-paper，没啥技术含量，主要贡献在于设计了使用数据增强的课程学习方式，并验证了在few-shot分类问题上的有效性。



## 课程学习式数据增强（curriculum data augmentation）

这应该就是本文最主要的贡献了。作者**使用文本增强时文本的改动幅度来衡量增强样本的难度，从而设计课程学习策略**。

![curriculum data augmentation](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220210233643.png)
具体分为两种方法：

**① 两阶段法（two-stage）**

先使用原样本进行训练，然后把增强样本混进来训练。这里的增强样本使用都是同样的改动幅度，所以该方法就是分了两个层级的难度。

**②渐进式（gradual）**

设定一个难度范围，从最低难度开始训练，收敛后就增大难度继续训练，直到训练到最大难度。

具体设置就是这段话：

![课程学习的具体设置](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220210234352.png)



然后就可以直接看实验结果了：

<img src="https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220210234611.png" alt="实验结果" style="zoom:67%;" />

![更详细的实验](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220210234752.png)

总之就是，使用课程学习的思路来进行数据增强，确实会比传统的数据增强训练方式有效，这也是为什么two-stage的方式会更好。而那个gradual的方式，相当于设置了更多的课程阶梯，使用了更多的增强样本，所以效果肯定会好一些。



## **小样本学习和Triplet loss**

这里再单独讲一讲few-shot learning和Triplet loss，因为一开始我是从数据增强角度去找到这篇文章的，加上之前对few-shot learning也不太了解，所以搞不懂为什么一定要跟triplet loss扯上关系。

我们回顾上面贴第二个实验结果表，它揭示了Triplet loss在few-shot任务中相比于使用cross-entropy loss的优势。

这个triple loss最开始是用于训练人脸识别模型的，因为人脸识别就是要识别的人一般非常多（类别多），但是我们能够提供给模型拿来训练的人脸样本非常少（few-shot），所以在这种背景下，triple loss就被设计出来，把一个分类问题，转化成相似度问题，使用少量的训练样本，训练出一个相似函数，然后在预测时，就可以计算新样本跟训练集中已有的样本的相似度，从而判断类别。

计算Triplet Loss使用的是一批三元组(A,P,N)，计算公式是这样的：
$$
L = \sum_i max(d(A_i,P_i)-d(A_i,N_i)+\alpha,0)
$$
其中，A代表anchor样本，P代表positive样本，N代表negative样本，$\alpha$则是一个用于缓冲的距离，或者说margin，d则是一个计算距离的函数，可以使用余弦距离或者欧氏距离。

借用B站Shusen Wang老师的教程（链接：https://www.bilibili.com/video/BV1vQ4y1R7dr?t=511.5）中的示意图来帮助理解：

![来源B站Shusen Wang老师的教程，链接：https://www.bilibili.com/video/BV1vQ4y1R7dr?t=511.5](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220211113744.png)

在训练样本很少的情况下，这种基于相似度的方法可以更加细致地刻画不同类别之间的区别，所以可以取得比cross-entropy更好的效果。

### **triplet loss VS. cross-entropy loss**

这里我们不禁要问，那triplet loss和cross-entropy loss各自的适用场景是什么呢？

**triplet loss，一般用于相似度、检索和小样本分类任务**上，而一般的分类任务，则更常使用cross-entropy。

虽然triplet loss我们看起来可以使同类别的样本的表示更近、不同类别的表示更远，在这一点上似乎比cross-entropy loss更优一些，但实际上**由于每次计算triplet loss都只是考虑了两个类别，还涉及到正负样本的采样问题，所以triplet loss计算存在不稳定、收敛慢的问题**，而cross-entropy则是计算时会考虑所有类别，所以在普通分类问题上效果会更好。

更多关于二者的对比，可以参见下面链接：

- 知乎讨论：https://www.zhihu.com/question/402067053/answer/1297490623

- 为什么triplet loss有效？http://bindog.github.io/blog/2019/10/23/why-triplet-loss-works/