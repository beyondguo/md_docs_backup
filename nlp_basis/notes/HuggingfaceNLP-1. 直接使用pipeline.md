---
title:  Huggingface🤗NLP笔记1：直接使用pipeline，是个人就能玩NLP
published: 2021-9-20
sidebar: auto
---

> 「Huggingface🤗NLP笔记系列-第1集」
> 最近跟着Huggingface上的NLP tutorial走了一遍，惊叹居然有如此好的讲解Transformers系列的NLP教程，于是决定记录一下学习的过程，分享我的笔记，可以算是官方教程的精简版。但最推荐的，还是直接跟着官方教程来一遍，真是一种享受。

- 官方教程网址：https://huggingface.co/course/chapter1
- 本期内容对应网址：https://huggingface.co/course/chapter1/3?fw=pt
- 本系列笔记的GitHub： https://github.com/beyondguo/Learn_PyTorch/tree/master/HuggingfaceNLP

---

# 直接使用Pipeline工具做NLP任务

`Pipeline`是Huggingface的一个基本工具，可以理解为一个端到端(end-to-end)的一键调用Transformer模型的工具。它具备了数据预处理、模型处理、模型输出后处理等步骤，可以直接输入原始数据，然后给出预测结果，十分方便。

给定一个任务之后，pipeline会自动调用一个预训练好的模型，然后根据你给的输入执行下面三个步骤：
1. 预处理输入文本，让它可被模型读取
2. 模型处理
3. 模型输出的后处理，让预测结果可读

一个例子如下：


```python
from transformers import pipeline

clf = pipeline('sentiment-analysis')
clf('Haha, today is a nice day!')
```
输出：
```shell
[{'label': 'POSITIVE', 'score': 0.9998709559440613}]
```


还可以**直接接受多个句子**，一起预测：
```python
clf(['good','nice','bad'])
```
输出：
```shell
[{'label': 'POSITIVE', 'score': 0.9998160600662231},
 {'label': 'POSITIVE', 'score': 0.9998552799224854},
 {'label': 'NEGATIVE', 'score': 0.999782383441925}]
```



pipeline支持的**task**包括：

- `"feature-extraction"`: will return a FeatureExtractionPipeline.
- `"text-classification"`: will return a TextClassificationPipeline.
- `"sentiment-analysis"`: (alias of "text-classification") will return a TextClassificationPipeline.
- `"token-classification"`: will return a TokenClassificationPipeline.
- `"ner"` (alias of "token-classification"): will return a TokenClassificationPipeline.
- `"question-answering"`: will return a QuestionAnsweringPipeline.
- `"fill-mask"`: will return a FillMaskPipeline.
- `"summarization"`: will return a SummarizationPipeline.
- `"translation_xx_to_yy"`: will return a TranslationPipeline.
- `"text2text-generation"`: will return a Text2TextGenerationPipeline.
- `"text-generation"`: will return a TextGenerationPipeline.
- `"zero-shot-classification"`: will return a ZeroShotClassificationPipeline.
- `"conversational"`: will return a ConversationalPipeline.

---

下面可以可以来试试用pipeline直接来做一些任务：

## Have a try: Zero-shot-classification
零样本学习，就是训练一个可以预测任何标签的模型，这些标签可以不出现在训练集中。

一种零样本学习的方法，就是通过NLI（文本蕴含）任务，训练一个推理模型，比如这个例子：
```python
premise = 'Who are you voting for in 2020?'
hypothesis = 'This text is about politics.'
```
上面有一个前提(premise)和一个假设(hypothesis)，NLI任务就是去预测，在这个premise下，hypothesis是否成立。

>NLI (natural language inference)任务：it classifies if two sentences are logically linked across three labels (contradiction, neutral, entailment).

通过这样的训练，我们可以直接把hypothesis中的politics换成其他词儿，就可以实现zero-shot-learning了。而Huggingface pipeline中的零样本学习，使用的就是在NLI任务上预训练好的模型。


```python
clf = pipeline('zero-shot-classification')

clf(sequences=["A helicopter is flying in the sky",
               "A bird is flying in the sky"],
    candidate_labels=['animal','machine'])  # labels可以完全自定义
```

输出：
```shell
[{'sequence': 'A helicopter is flying in the sky',
  'labels': ['machine', 'animal'],
  'scores': [0.9938627481460571, 0.006137280724942684]},
 {'sequence': 'A bird is flying in the sky',
  'labels': ['animal', 'machine'],
  'scores': [0.9987970590591431, 0.0012029369827359915]}]
```

参考阅读：
- 官方 Zero-shot-classification Pipeline文档：https://huggingface.co/transformers/main_classes/pipelines.html#transformers.ZeroShotClassificationPipeline
- 零样本学习简介：https://mp.weixin.qq.com/s/6aBzR0O3pwA8-btsuDX82g


## Have a try: Text Generation
Huggingface pipeline默认的模型都是英文的，比如对于text generation默认使用gpt2，但我们也可以指定Huggingface Hub上其他的text generation模型，这里我找到一个中文的：

```python
generator = pipeline('text-generation', model='liam168/chat-DialoGPT-small-zh')  
```
给一个初始词句开始生产：

```python
generator('上午')
```
输出：
```shell
[{'generated_text': '上午上班吧'}]
```



## Have a try: Mask Filling


```python
unmasker = pipeline('fill-mask')

unmasker('What the <mask>?', top_k=3)  # 注意不同的模型，MASK token可能不一样，不一定都是 <mask>
```

输出：


```shell
[{'sequence': 'What the heck?',
  'score': 0.3783760964870453,
  'token': 17835,
  'token_str': ' heck'},
 {'sequence': 'What the hell?',
  'score': 0.32931089401245117,
  'token': 7105,
  'token_str': ' hell'},
 {'sequence': 'What the fuck?',
  'score': 0.14645449817180634,
  'token': 26536,
  'token_str': ' fuck'}]
```



## 其他Tasks

还有很多其他的pipeline，比如NER，比如summarization，这里就不一一尝试了。

想看官方实例的可以参见： https://huggingface.co/course/chapter1/3?fw=pt



---

总之，我们可以看出，Huggingface提供的pipeline接口，就是一个”拿来即用“的端到端的接口，只要Huggingface Hub上有对应的模型，我们几行代码就可以直接拿来做任务了，真是造福大众啊！



下一篇笔记，会回顾一下Transformer模型的发展和基本架构，让我们对这些工具背后的模型更加了解。

