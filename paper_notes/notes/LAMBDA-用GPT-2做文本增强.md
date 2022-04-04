---
title: LAMBADA——用GPT-2来做文本数据增强
published: 2022-3-27
sidebar: auto
---



# **Do Not Have Enough Data? Deep Learning to the Rescue!**



![image-20220327233812887](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/202203272338878.png)

本文发表在AAAI-20上，作者是IBM AI团队。

> **一句话总结:**
> 思路相当简单，利用GPT-2强大的生成能力来进行文本增强，从而在few-shot场景下达到很好的增强效果。



## 思路一览：

本文提出的方法称为language-model-based data augmentation（LAMBADA）。

分成四个步骤：

![image-20220328223450798](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/202203282234851.png)

### 1. 用已有的有标签数据训练一个classifier $A$.

这个A会被用来当做一个filter，用来筛选生成的样本的质量。

### 2. 在训练集上对 GPT-2 ($G$) 进行 fine-tune，得到 $G_{tuned}$.

这一步就是整个文章的核心了。

我们知道，GPT-2实际上就是一个语言模型，使用的是Next-word-prediction的方式进行训练，这种语言模型称为causal language modeling (CLM) 。

为了生成我们需要的增强语料，这里的方式是使用我们的训练集，来构造一批语料，让GPT-2继续在该语料上进行Next-word-prediction的训练.

语料如何构建呢？假设我们有n个训练样本$\{<x_i,y_i>\}_{i=1}^n$，那么就构造：
$$
U^* = y_1 [SEP] x_1 [EOS] y_2 [SEP] x_2 [EOS]...y_n [SEP] x_n [EOS]
$$
即使用两个特殊的token——[SEP]和[EOS]把训练样本和标签给拼起来。

然后，就使用常规的causal language modeling的损失函数来训练：
$$
L = \sum_i logP(x_i|x_{i-k},...,x_{i-i})
$$


### 3. 使用$G_{tuned}$进行增强样本生成

经过了上面的微调，让模型学习看到`yi [SEP]`就可以生成跟`yi`对应的句子`xi`，这里的`yi [SEP]`实际上就是所谓的prompt。

作者给出了几个生成的例子：

![image-20220328001255002](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/202203280012040.png)

上述例子，比方Flight time这个class，就是直接对GPT-2输入`Flight time [SEP]`，然后GPT-2就输出后面这个句子。



在具体生成的时候，由于存在一些randomness（比方根据概率分布进行采样），所以给定一个prompt之后，模型每次可以生成不同的句子，所以理论上可以无限扩充训练样本。



### 4. 使用 $A$对生成的样本进行筛选

很好理解，因为生成的句子质量是难以保证的，生成一大堆，可能会有很多噪音，所以我们就用前面的一个初级分类器$A$对这批样本进行预测，挑选出那些置信度比较高的来作为我们真正使用的增强样本。



## 实验效果：

使用了三个数据集：

![image-20220328214807792](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/202203282148853.png)

说实话，数据集的选择挺weak的，你做NLP的为啥不用点常见的NLP数据集。

对比的baseline主要包括EDA、CVAE和CBERT：

![image-20220328215650605](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/202203282156644.png)

然后作者设计的低资源场景就是每个类别都只有5/10/20/50/100个样本，反正就是很平衡的情况，这其实也是不太现实的，现实的低资源场景往往是类别不平衡的。

这些槽点咱们也不多说了，看看它的实验效果吧：



![image-20220328215759815](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/202203282157845.png)

上图展示了不同的训练集大小下的效果，可见在每个类别只有5个样本的时候，LAMBADA的效果十分显著。而当每个类别样本量达到100的时候，效果就比较弱了，但是依然是可以提高的。

（有意思的是，当每类别样本量位5的时候，LSTM奇差无比，即使到了100也只比SVM高一点，这说明在小样本的情况下，LSTM这种没有预训练的深度学习模型很垃圾，BERT虽然更是深度学习模型，但是它有强大的预训练，而SVM这种机器学习模型就很适应小样本场景）

然后针对**每个类别只有5个样本**的情况，作者对比了各种baseline：

![image-20220328220414775](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/202203282204804.png)

效果还是比较明显的，我觉得这里主要归功于GPT-2巨大的预训练量。

然而，这个论文没有放出在样本量更多一些的时候的实验结果。。。我猜测是效果不好，不然为啥不放？另外我发现，上面那个表，其实主要是ATIS数据集效果很明显，其他的俩效果都只能说有提高，但是对于few-shot的场景下只有1-2个点的提高，说实话不是什么突破性提高。然后我发现作者的Figure1也是ATIS数据集的，心机啊！



## 总结：

看了实验结果之后，我发现并没有那么惊喜，但是从作者的写作上看，包括取的标题，我就感觉作者仿佛十分激动，仿佛发现了什么不得了的东西，搞得读者一开始也跟着激动一番。

![image-20220328221841193](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/202203282218236.png)

但是呢，虽然实验效果没那么惊喜，数据集的选取、实验的设计也有很多槽点，但这里的方法还是给人一些启发的，告诉了我们NLG模型用于文本数据增强的更多可能。比方我们可以在prompt设计，在GPT-2微调的方式上进行更精细地设计，想办法让GPT-2针对给定的标签可以生成更加diverse的样本，都可以作为进一步改进的方向。

