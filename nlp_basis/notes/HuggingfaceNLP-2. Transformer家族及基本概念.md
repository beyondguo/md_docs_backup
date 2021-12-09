---
title:  Huggingface🤗NLP笔记2：一文看清Transformer大家族的三股势力
published: 2021-9-23
sidebar: auto
---

> **「Huggingface🤗NLP笔记系列-第2集」**
> 最近跟着Huggingface上的NLP tutorial走了一遍，惊叹居然有如此好的讲解Transformers系列的NLP教程，于是决定记录一下学习的过程，分享我的笔记，可以算是官方教程的精简版。但最推荐的，还是直接跟着官方教程来一遍，真是一种享受。

- 官方教程网址：https://huggingface.co/course/chapter1
- 本期内容对应网址：https://huggingface.co/course/chapter1/4?fw=pt
- 本系列笔记的**GitHub**： https://github.com/beyondguo/Learn_PyTorch/tree/master/HuggingfaceNLP

---

# 一文看清Transformer大家族的三股势力

## 1. Transformer结构

Transformer结构最初就是在大2017年名鼎鼎的《Attention Is All You Need》论文中提出的，最开始是用于机器翻译任务。

这里先简单回顾一下Transformer的基本结构：

<img src='https://huggingface.co/course/static/chapter1/transformers_blocks.png' width=200 align="center">

- 左边是encoder，用于对输入的sequence进行表示，得到一个很好特征向量。
- 右边是decoder，利用encoder得到的特征，以及原始的输入，进行新的sequence的生成。

encoder、decoder既可以单独使用，又可以再一起使用，因此，基于Transformer的模型可以分为三大类：

- Encoder-only
- Decoder-only
- Encoder-Decoder


## 2. Transformer家族及三股势力

随后各种基于Transformer结构的模型就如雨后春笋般涌现出来，教程中有一张图展示了一些主要模型的时间轴：

<img src='https://huggingface.co/course/static/chapter1/transformers_chrono.png' width=1000>

虽然模型多到四只jio都数不过来，但总体上可以分为三个阵营，分别有三个组长：

- 组长1：**BERT**。组员都是BERT类似的结构，是一类**自编码模型**。
- 组长2：**GPT**。组员都是类似GPT的结构，是一类**自回归模型**。
- 组长3：**BART/T5**。组员结构都差不多是**encoder-decoder**模型。

### 不同的架构，不同的预训练方式，不同的特长

对于**Encoder-only**的模型，预训练任务通常是“破坏一个句子，然后让模型去预测或填补”。例如BERT中使用的就是两个预训练任务就是**Masked language modeling**和**Next sentence prediction**。
因此，这类模型擅长进行文本表示，适用于做**文本的分类、实体识别、关键信息抽取**等任务。

对于**Decoder-only**的模型，预训练任务通常是**Next word prediction**，这种方式又被称为**Causal language modeling**。这个Causal就是“因果”的意思，对于decoder，它在训练时是无法看到全文的，只能看到前面的信息。
因此这类模型适合做**文本生成**任务。

而**Seq2seq**架构，由于包含了encoder和decoder，所以预训练的目标通常是融合了各自的目标，但通常还会设计一些更加复杂的目标，比如对于T5模型，会把一句话中一片区域的词都mask掉，然后让模型去预测。seq2seq架构的模型，就适合做**翻译、对话**等需要根据给定输入来生成输出的任务，这跟decoder-only的模型还是有很大差别的。

### 总结表如下：

|类型|架构|Transformer组件 |	Examples |	Tasks|
| -------- | -------- | -------- | -------- |-------- |
|**BERT**-like | auto-encoding models|	Encoder  |		ALBERT, BERT, DistilBERT, ELECTRA, RoBERTa | 	Sentence classification, named entity recognition, extractive question answering|
|**GPT**-like |  auto-regressive models |	Decoder |CTRL, GPT, GPT-2, Transformer XL |	 	Text generation|
|**BART/T5**-like |  sequence-to-sequence models|	Encoder-decoder  |		BART, T5, Marian, mBART |	 	Summarization, translation, generative question answering|




---

了解了Transformer一系列模型的来龙去脉，我们就可以更好地玩耍Transformer啦！下一集，我们会慢慢深入查看Huggingface `transformers`库背后的细节，从而更灵活地使用。