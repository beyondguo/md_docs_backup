---
title:  Huggingface🤗NLP笔记4：Models，Tokenizers，以及如何做Subword tokenization
published: 2021-9-26
sidebar: auto
---

> **「Huggingface🤗NLP笔记系列-第4集」**
> 最近跟着Huggingface上的NLP tutorial走了一遍，惊叹居然有如此好的讲解Transformers系列的NLP教程，于是决定记录一下学习的过程，分享我的笔记，可以算是官方教程的精简+注解版。但最推荐的，还是直接跟着官方教程来一遍，真是一种享受。

- 官方教程网址：https://huggingface.co/course/chapter1
- 本期内容对应网址：https://huggingface.co/course/chapter2/3?fw=pt
- 本系列笔记的**GitHub**： https://github.com/beyondguo/Learn_PyTorch/tree/master/HuggingfaceNLP

---

# Models，Tokenizers，以及如何做Subword tokenization




## Models

前面都是使用的`AutoModel`，这是一个智能的wrapper，可以根据你给定的checkpoint名字，自动去寻找对应的网络结构，故名Auto。

如果明确知道我们需要的是什么网络架构，就可以直接使用具体的`*Model`，比如`BertModel`，就是使用Bert结构。

### 随机初始化一个Transformer模型：通过`config`来加载

`*Config`这个类，用于给出某个模型的网络结构，通过config来加载模型，得到的就是一个模型的架子，没有预训练的权重。


```python
from transformers import BertModel, BertConfig

config = BertConfig()
model = BertModel(config)  # 模型是根据config来构建的，这时构建的模型是参数随机初始化的
```

看看config打印出来是啥：
```python
print(config)
```

```shell
BertConfig {
  "attention_probs_dropout_prob": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.3.3",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
```



更常用的做法则是直接加载预训练模型，然后微调。

### 初始化一个预训练的Transformer模型：通过`from_pretrained`来加载


```python
from transformers import BertModel

model = BertModel.from_pretrained('bert-base-cased')
```


模型的保存：


```python
model.save_pretrained("directory_on_my_computer")
# 会生成两个文件： config.json pytorch_model.bin
```

## Tokenizer
transformer模型使用的分词方法，往往不是直接的word-level分词或者char-level分词。

前者会让词表过大，后者则表示能力很低。

因此主流的方式是进行 **subword-level** 的分词。例如对 "tokenization" 这个词，可能会被分成 "token" 和 "ization" 两部分。

常见的subword tokenization方法有：
- BPE
- WordPiece
- Unigram
- SentencePiece
- ...


这里对BPE做一个简单的介绍，让我们对 sub-word tokenization 的原理有一个基本了解：

## Subword tokenization (☆☆☆)
Subword tokenization的核心思想是：“**频繁出现了词不应该被切分成更小的单位，但不常出现的词应该被切分成更小的单位**”。

比方"annoyingly"这种词，就不是很常见，但是"annoying"和"ly"都很常见，因此细分成这两个sub-word就更合理。中文也是类似的，比如“仓库管理系统”作为一个单位就明显在语料中不会很多，因此分成“仓库”和“管理系统”就会好很多。

这样分词的好处在于，大大节省了词表空间，还能够解决OOV问题。因为我们很多使用的词语，都是由更简单的词语或者词缀构成的，我们不用去保存那些“小词”各种排列组合形成的千变万化的“大词”，而用较少的词汇，去覆盖各种各样的词语表示。同时，相比与直接使用最基础的“字”作为词表，sub-word的语义表示能力也更强。

那么，用什么样的标准得到sub-word呢？一个著名的算法就是 **Byte-Pair Encoding (BPE)** ：

（下面的内容，主要翻译自Huggingface Docs中讲解tokenizer的部分，十分推荐大家直接阅读： https://huggingface.co/transformers/master/tokenizer_summary.html ）

### BPE————Byte-Pair Encoding：

#### **Step1**：首先，我们需要对**语料**进行一个**预分词（pre-tokenization）**：

比方对于英文，我可以直接简单地使用空格加一些标点符号来分词；中文可以使用jieba或者直接字来进行分词。

分词之后，我们就得到了一个**原始词集合**，同时，还会记录每个词在训练语料中出现的**频率**。

假设我们的词集合以及词频是：

```python
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

#### **Step2**：构建**基础词表（base vocab）** 并开始学习 **结合规则（merge rules）**：


对于英语来说，我们选择字母来构成**基础词表**：

`["b", "g", "h", "n", "p", "s", "u"]`

注：这个基础词表，就是我们最终词表的初始状态，我们会不断构建新词，加进去，直到达到我们理想的词表规模。

根据这个基础词表，我们可以对原始的词集合进行细粒度分词，并看到基础词的词频：

```python
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
```

接下来就是BPE的Byte-Pair核心部分————找symbol pair（符号对）并学习结合规则，即，我们从上面这个统计结果中，找出出现次数最多的那个符号对：

统计一下：
```python
h+u   出现了 10+5=15 次
u+g   出现了 10+5+5 = 20 次
p+u   出现了 12 次
...
```
统计完毕，我们发现`u+g`出现了最多次，因此，第一个结合规则就是：**把`u`跟`g`拼起来，得到`ug`这个新词！**

那么，我们就把`ug`加入到我们的基础词表：

`["b", "g", "h", "n", "p", "s", "u", "ug"]`

同时，词频统计表也变成了：
```
("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
```

#### **Step3**：反复地执行上一步，直到达到预设的词表规模。

我们接着统计，发现下一个频率最高的symbol pair是`u+n`，出现了12+4=16次，因此词表中增加`un`这个词；再下一个则是`h+ug`，出现了10+5=15次，因此添加`hug`这个词......

如此进行下去，当达到了预设的`vocab_size`的数目时，就停止，咱们的词表就得到啦！

#### **Step4**：如何分词：

得到了最终词表，在碰到一个词汇表中没有的词的时候，比如`bug`就会把它分成`b`和`ug`。也可以理解成，我首先把`bug`分解成最基本的字母，然后根据前面的结合规律，把`u`跟`g`结合起来，而`b`单独一个。具体在分词时候是如何做的，有时间去读读源码。

---

除了BPE，还有一些其他的sub-word分词法，可以参考 https://huggingface.co/transformers/master/tokenizer_summary.html 。

下面，我们就直接使用Tokenizer来进行分词：


```python
from transformers import BertTokenizer  # 或者 AutoTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
```


```python
s = 'today is a good day to learn transformers'
tokenizer()
```

得到：


```shell
{'input_ids': [101, 2052, 1110, 170, 1363, 1285, 1106, 3858, 11303, 1468, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```



### 了解一下内部的具体步骤：

1. `tokenize()`


```python
s = 'today is a good day to learn transformers'
tokens = tokenizer.tokenize(s)
tokens
```

输出：


```shell
['today', 'is', 'a', 'good', 'day', 'to', 'learn', 'transform', '##ers']
```



注意这里的分词结果，`transformers`被分成了`transform`和`##ers`。这里的##代表这个词应该紧跟在前面的那个词，组成一个完整的词。

这样设计，主要是为了方面我们在还原句子的时候，可以正确得把sub-word组成成原来的词。

2. `convert_tokens_to_ids()`


```python
ids = tokenizer.convert_tokens_to_ids(tokens)
ids
```

输出：


    [2052, 1110, 170, 1363, 1285, 1106, 3858, 11303, 1468]



3. `decode`


```python
print(tokenizer.decode([1468]))
print(tokenizer.decode(ids))  # 注意这里会把subword自动拼起来
```
输出：
```shell
##ers
today is a good day to learn transformers
```



### Special Tokens

观察一下上面的结果，直接call tokenizer得到的ids是：
```shell
[101, 2052, 1110, 170, 1363, 1285, 1106, 3858, 11303, 1468, 102]
```
而通过`convert_tokens_to_ids`得到的ids是：
```shell
[2052, 1110, 170, 1363, 1285, 1106, 3858, 11303, 1468]
```
可以发现，前者在头和尾多了俩token，id分别是 101 和 102。

decode出来瞅瞅：


```python
tokenizer.decode([101, 2052, 1110, 170, 1363, 1285, 1106, 3858, 11303, 1468, 102])
```

输出：


```shell
'[CLS] today is a good day to learn transformers [SEP]'
```



它们分别是 `[CLS]` 和 `[SEP]`。这两个token的出现，是因为我们调用的模型，在pre-train阶段使用了它们，所以tokenizer也会使用。

不同的模型使用的special tokens不一定相同，所以一定要让tokenizer跟model保持一致！


