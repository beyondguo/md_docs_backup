---
title: Google的 Pathways（理想）与 PaLM（现实）
published: 2022-7-13
sidebar: auto
---

# Google的 Pathways（理想）与 PaLM（现实）

## Pathways构想

Google 在2021年提出了Pathways的构想：

当前模型的主要问题：

- 基本都是一个模型做一个任务；
- 在一个通用的模型上继续fine-tune，会遗忘很多其他知识；
- 基本都是单模态；
- 基本都是 **dense** 模型，在完成一个任务时（不管难易程度），网络的所有参数都被激活和使用；

**Pathways** 的愿景 —— 一个跟接近人脑的框架：

- 一个模型，可以做多任务，多模态
- sparse model，在做任务时，只是 **sparsely activated**，只使用一部分的参数

![https://storage.googleapis.com/gweb-uniblog-publish-prod/original_images/pathways_3.gif](https://storage.googleapis.com/gweb-uniblog-publish-prod/original_images/pathways_3.gif)

## Pathways 系统

2022年3月，Google发布了Pathways系统，用于更高效地训练大型模型：

![image-20220713213527311](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202207132135364.png)
这个太工程的东西我也看不懂，所以就不评论了。

## PaLM: Lanugage Modeling with Pathways

2022年4月，Google发布了一个鸿篇巨制——PaLM：

![image-20220713213609786](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202207132136826.png)

说实话，在看完Jeff Dean介绍Pathways愿景博客之后，再看这篇文章前本来是充满期待的，比较牛皮已经吹了一年了，论文一开打一屏幕的作者，文83页（比GPT-3的paper都长），结果浏览一遍，满脑子都是”就这？？？“

Anyway，还是介绍一下：

**一句话介绍：**

> **PaLM** 是第一款基于 Google **Pathways** 系统训练的超大规模的语言模型（但依然是经典结构：a dense, decoder-only, full-attention Transformer model），再次展现了“大力出奇迹”还有很大空间。但是离Google的Pathways愿景还有很大距离。

![image-20220713213633466](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202207132136489.png)

**Key Points：**

- Efficient scaling：在Pathways系统的加持下，PaLM的训练效率比之前的方法有了显著提高；
- Few-shot SOTA：在众多任务上取得了 few-shot 的 SOTA；
- Breakthrough capabilities：在很多推理（reasoning）任务上，PaLM在few-shot的情况就可以超越很多之前需要fine-tune的方法；
- Discontinuous improvements：随着模型规模的提高，边际效益可能会有质的提升（在25%的任务上，观察到了“量变产生质变”的现象）；
- Multilingual understanding：多语言能力大幅提高。

**Model：**

- A **dense**, **decoder**-only, full-attention Transformer model
- 使用 SwiGLU Activation，Parallel Layers，Multi-Query Attention 等提升计算效率的机制
- 完全无损、可逆的vocabulary：空格保留、OOV切分成UTF8 bytes、数值切分成单个token
- 只训练一个epoch——防止overfitting，减轻memorization现象

![image-20220713213654964](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202207132136000.png)

**Training Dataset：**

filtered webpages, books, Wikipedia, news articles, source code, and social media conversations.

![image-20220713213713606](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202207132137642.png)

**Results：**

Few-shot 实验：

![image-20220713213748078](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202207132137104.png)

Finetune实验：

![image-20220713213818286](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202207132138313.png)

比最好的encoder-decoder模型效果要差一点，但是显著高于之前的decoder-only的模型。

Big-Bench：

![image-20220713213838402](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202207132138425.png)

![image-20220713213852176](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202207132138202.png)

**PaLM自己的总结：**

虽然文章没有给人惊喜，但是论文自己的总结也还是挺中肯的：

PaLM只是构建Pathways这个愿景迈出的第一步，PaLM的意义在于进一步扩展了大模型的能力边界（尤其是few-shot），说明了传统的模型架构和训练方法依然有很大的提升空间。另一方面，PaLM验证了Pathways训练系统的有效性，为下一代的模型架构研发做了经验积累。

------

## Other Related Work

其实在这个PaLM之前，Google探索过很多基于**MoE**（Mixture-of-experts）的大型sparse model，包括 GShard，Switch-Transformer，GLaM。这些模型，通过MoE的形式，实现了一个大模型中包含很多子网络，且针对不同的token自动选择不同的子网络（experts）进行推理的能力。其实看了PaLM之后，我觉得MoE这条线才更接近与Jeff Dean所说的Pathways的愿景，但我也猜测可能一个超大的类MoE模型正在Pathways系统上训练呢（我赌一根钟薛高hhh）......

---

参考链接：

- Jeff Dean关于Pathways愿景的介绍：https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/
- PaLM Blog：https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html
- PaLM paper：https://arxiv.org/pdf/2204.02311.pdf
- Pathways ML system paper：https://arxiv.org/pdf/2203.12533.pdf