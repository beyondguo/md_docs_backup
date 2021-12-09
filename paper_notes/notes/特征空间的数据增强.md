---
title: 在特征空间增强数据集（ICLR workshop 2017）
published: 2021-11-28
sidebar: auto
---

# 在特征空间增强数据集

- 论文标题：DATASET AUGMENTATION IN FEATURE SPACE
- 发表会议：ICLR workshop 2017
- 组织机构：University of Guelph 



> **一句话评价**：
>
> 一篇简单的workshop文章，但是有一些启发性的实验结论。



## 简介

最常用的数据增强方法，无论是CV还是NLP中，都是直接对原始数据进行各种处理。比如对图像的剪切、旋转、变色等，对文本数据的单词替换、删除等等。**对于原始数据进行处理，往往是高度领域/任务相关的，即我们需要针对数据的形式、任务的形式，来设计增强的方法，这样就不具有通用性。比如对于图像的增强方法，就没法用在文本上。**因此，本文提出了一种“领域无关的”数据增强方法——特征空间的增强。具体的话就是对可学习的样本特征进行 1) adding noise, 2) interpolating, 3) extrapolating 来得到新的样本特征。



文本提到的一个想法很有意思：

> When traversing along the manifold it is more likely to encounter realistic samples in feature space than compared to input space.
> 在样本所在的流形上移动，在特征空间上会比在原始输入空间上移动，更容易遇到真实的样本点。

我们知道，对原始的数据进行数据增强，很多时候就根本不是真实可能存在的样本了，比如我们在NLP中常用的对文本进行单词随机删除，这样得到的样本，虽然也能够提高对模型学习的鲁棒性，但这种样本实际上很难在真实样本空间存在。本文提到的这句话则提示我们，如果我们把各种操作，放在高维的特征空间进行，则更可能碰到真实的样本点。文章指出，这种思想，Bengio等人也提过：“higher level representations expand the relative volume of plausible data points within the feature space, conversely shrinking the space allocated for unlikely data points.” 这里我们暂且不讨论这个说法背后的原理，先不妨承认其事实，这样的话就启示我们，在特征空间进行数据增强，我们有更大的探索空间。



## 具体方法

其实这个文章具体方法很简单，它使用的是encoder-decoder的框架，在（预训练好的）encoder之后的样本特征上进行增强，然后进行下游任务。所以是先有了一个表示模型来得到样本的特征，再进行增强，而不是边训练边增强。框架结构如下：

![image-20211128152702006](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20211128152721.png)



下面我们来看作者具体是怎么增强的：

### 1. Adding Noise（加噪音）

直接在encoder得到的特征上添加高斯噪声：
$$
c^{'}_i = c_i + \gamma X, X \sim \mathcal{N}(0,\sigma ^2_i)
$$

### 2. Interpolation（内插值）

在**同类别**点中，寻找K个最近邻，然后任意两个邻居间，进行内插值：
$$
c^{'} = (c_k - c_j)\lambda + c_j
$$

### 3. Extrapolating（外插值）

跟内插的唯一区别在于插值的位置：
$$
c^{'} = (c_j - c_k)\lambda + c_j
$$
下图表示了内插跟外插的区别：

![image-20211128154757628](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20211128154757.png)

在文本中，内插和外插都选择$\lambda = 0.5$.



论文作者为了更加形象地展示这三种增强方式，使用正弦曲线（上的点）作为样本，来进行上述操作，得到新样本：

![image-20211128160256809](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20211128160256.png)

作者还借用一个手写字母识别的数据集进行了可视化，进一步揭示interpolation和extrapolation的区别：

![image-20211128160514332](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20211128160514.png)

作者没有具体说可视化的方法，猜测是通过autoencoder来生成的。可以看到，extrapolation得到的样本，更加多样化，而interpolation则跟原来的样本对更加接近。



## 实验

下面我们来看看使用这三种方式的增强效果。本文的实验部分十分混乱，看得人头大，所以我只挑一些稍微清楚一些的实验来讲解。



### 实验1：一个阿拉伯数字语音识别任务

![实验1](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20211128161250.png)

### 实验2：另一个序列数据集

![image-20211128161607738](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20211128161607.png)

注：interpolation和extrapolation都是在同类别间进行的。

实验结果：

- Adding Noise，效果一般般。
- Interpolation，降低性能！
- Extrapolation，效果最好！



### 实验3：跟再input space进行增强对比

![image-20211128162546341](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20211128162546.png)

实验结果：

- 在特征空间进行extrapolation效果更好
- 特征空间的增强跟input空间的增强可以互补

### 实验4：把增强的特征重构回去，得到的新样本有用吗

![image-20211128163337691](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20211128163337.png)

这个实验还是有一点意思的，一个重要的结论就是，在特征空间这么增强得到的特征，重构回去，屁用没有。可以看到，都比最基础的baseline要差，但是如果把测试集都换成重构图，那么效果就不错。这其实不能怪特征增强的不好，而是重构的不好，因为重构得到的样本，跟特征真实代表的样本，肯定是有差距的，因此效果不好是可以理解的。



## 重要结论&讨论

综上所有的实验，最重要的实验结论就是：**在特征空间中，添加噪音，或是进行同类别的样本的interpolation，是没什么增益的，甚至interpolation还会带来泛化性能的降低。相反，extrapolation往往可以带来较为明显的效果提升。**这个结论还是很有启发意义的。

究其原因，i**nterpolation实际上制造了更加同质化的样本，而extrapolation得到的样本则更加有个性**，却还保留了核心的特征，更大的多样性有利于提高模型的泛化能力。

作者在结尾还提到他们做过一些进一步的探究，发现了一个有意思的现象：**对于类别边界非常简单的任务（例如线性边界、环状边界），interpolation是有帮助的，而extrapolation则可能有害。**这也是可以理解的，因为在这种场景下，extrapolation很容易“越界”，导致增强的样本有错误的标签。但是现实场景中任务多数都有着复杂的类别边界，因此extrapolation一般都会更好一些。作者认为，**interpolation会让模型学习到一个更加“紧密”、“严格”的分类边界，从而让模型表现地过于自信，从而泛化性能不佳。**

总之，虽然这仅仅是一篇workshop的论文，实验也做的比较混乱，可读性较差，但是揭示的一些现象以及背后原理的探讨还是十分有意义的。经典的数据增强方法mixup也部分受到了这篇文章的启发，因此还是值得品味的。









