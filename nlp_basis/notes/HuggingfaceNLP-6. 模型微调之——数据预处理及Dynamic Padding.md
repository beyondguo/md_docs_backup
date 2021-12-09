---
title:  Huggingface🤗NLP笔记6：数据集预处理，使用dynamic padding构造batch
published: 2021-9-27
sidebar: auto
---

> **「Huggingface🤗NLP笔记系列-第6集」**
> 最近跟着Huggingface上的NLP tutorial走了一遍，惊叹居然有如此好的讲解Transformers系列的NLP教程，于是决定记录一下学习的过程，分享我的笔记，可以算是官方教程的**精简+注解版**。但最推荐的，还是直接跟着官方教程来一遍，真是一种享受。

- 官方教程网址：https://huggingface.co/course/chapter1
- 本期内容对应网址：https://huggingface.co/course/chapter3/2?fw=pt
- 本系列笔记的**GitHub**： https://github.com/beyondguo/Learn_PyTorch/tree/master/HuggingfaceNLP

---

# 数据集的预处理，使用dynamic padding构造batch

从这一集，我们就正式开始使用Transformer来训练模型了。今天的部分是关于数据集预处理。

## 试着训练一两条样本


```python
# 先看看cuda是否可用
import torch
torch.cuda.is_available()
```


```shell
>>> True
```

首先，我们加载模型。既然模型要在具体任务上微调了，我们就要加载带有Head的模型，这里做的分类问题，因此加载`ForSequenceClassification`这个Head：


```python
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
```
下面是模型输出的warning：
```shell
>>> 
Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForSequenceClassification: ['cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias']
- This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.weight', 'classifier.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

看到这么一大串的warning出现，不要怕，这个warning正是我们希望看到的。

为啥会出现这个warning呢，因为我们加载的预训练权重是`bert-based-uncased`，而使用的骨架是`AutoModelForSequenceClassification`，前者是没有在下游任务上微调过的，所以用带有下游任务Head的骨架去加载，会随机初始化这个Head。这些在warning中也说的很明白。

接下来，我们试试直接构造一个size=2的batch，丢进模型去。

当输入的batch是带有"labels"属性的时候，模型会自动计算loss，拿着这个loss，我们就可以进行反向传播并更新参数了：

```python
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
batch['labels'] = torch.tensor([1, 1])  # tokenizer出来的结果是一个dictionary，所以可以直接加入新的 key-value

optimizer = AdamW(model.parameters())
loss = model(**batch).loss  #这里的 loss 是直接根据 batch 中提供的 labels 来计算的，回忆：前面章节查看 model 的输出的时候，有loss这一项
loss.backward()
optimizer.step()
```

## 从Huggingface Hub中加载数据集

这里，我们使用MRPC数据集，它的全称是Microsoft Research Paraphrase Corpus，包含了5801个句子对，标签是两个句子是否是同一个意思。

Huggingface有一个`datasets`库，可以让我们轻松地下载常见的数据集：


```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets
```

看看加载的dataset的样子：

```shell
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
```



load_dataset出来的是一个DatasetDict对象，它包含了train，validation，test三个属性。可以通过key来直接查询，得到对应的train、valid和test数据集。

这里的train，valid，test都是Dataset类型，有 features和num_rows两个属性。还可以直接通过下标来查询对应的样本。


```python
raw_train_dataset = raw_datasets['train']
raw_train_dataset[0]
```

看看数据长啥样：


```shell
{'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
 'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .',
 'label': 1,
 'idx': 0}
```

可见，每一条数据，就是一个dictionary。

Dataset的features可以理解为一张表的columns，Dataset甚至可以看做一个pandas的dataframe，二者的使用很类似。

我们可以直接像操作dataframe一样，取出某一列：


```python
type(raw_train_dataset['sentence1'])  # 直接取出所有的sentence1，形成一个list
```


```shell 
>>> list
```



通过Dataset的features属性，可以详细查看数据集特征，包括labels具体都是啥：


```python
raw_train_dataset.features
```


```shell
>>>
{'sentence1': Value(dtype='string', id=None),
 'sentence2': Value(dtype='string', id=None),
 'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),
 'idx': Value(dtype='int32', id=None)}
```



## 数据集的预处理


```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
```

我们可以直接下面这样处理：
```python
tokenized_sentences_1 = tokenizer(raw_train_dataset['sentence1'])
tokenized_sentences_2 = tokenizer(raw_train_dataset['sentence2'])
```
但对于MRPC任务，我们不能把两个句子分开输入到模型中，二者应该组成一个pair输进去。

tokenizer也可以直接处理sequence pair：


```python
from pprint import pprint as print
inputs = tokenizer("first sentence", "second one")
print(inputs)
```

```shell
>>>
{'attention_mask': [1, 1, 1, 1, 1, 1, 1],
 'input_ids': [101, 2034, 6251, 102, 2117, 2028, 102],
 'token_type_ids': [0, 0, 0, 0, 1, 1, 1]}
```

我们把这里的input_ids给decode看一下：

```python
tokenizer.decode(inputs.input_ids)
```


```shell
>>>
'[CLS] first sentence [SEP] second one [SEP]'
```

可以看到这里inputs里，还有一个`token_type_ids`属性，它在这里的作用就很明显了，指示哪些词是属于第一个句子，哪些词是属于第二个句子。tokenizer处理后得到的ids，解码之后，在开头结尾多了`[CLS]`和`[SEP]`，两个句子中间也添加了一个`[SEP]`。另外注意，虽然输入的是一个句子对，但是编码之后是一个整体，通过`[SEP]`符号相连。

**这种神奇的做法，其实是源于bert-base预训练的任务**，即**next sentence prediction**。换成其他模型，比如DistilBert，它在预训练的时候没有这个任务，那它的tokenizer的结果就不会有这个`token_type_ids`属性了。

既然这里的tokenizer可以直接处理pair，我们就可以这么去分词：


```python
tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)
```

但是这样不一定好，因为先是直接把要处理的整个数据集都读进了内存，又返回一个新的dictionary，会占据很多内存。

官方推荐的做法是通过`Dataset.map`方法，来调用一个分词方法，实现批量化的分词：


```python
def tokenize_function(sample):
    # 这里可以添加多种操作，不光是tokenize
    # 这个函数处理的对象，就是Dataset这种数据类型，通过features中的字段来选择要处理的数据
    return tokenizer(sample['sentence1'], sample['sentence2'], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets
```

处理后的dataset的信息：

```shell
DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 408
    })
    test: Dataset({
        features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids'],
        num_rows: 1725
    })
})
```



看看这个map的一些参数：

```shell
raw_datasets.map(
    function,
    with_indices: bool = False,
    input_columns: Union[str, List[str], NoneType] = None,
    batched: bool = False,
    batch_size: Union[int, NoneType] = 1000,
    remove_columns: Union[str, List[str], NoneType] = None,
    keep_in_memory: bool = False,
    load_from_cache_file: bool = True,
    cache_file_names: Union[Dict[str, Union[str, NoneType]], NoneType] = None,
    writer_batch_size: Union[int, NoneType] = 1000,
    features: Union[datasets.features.Features, NoneType] = None,
    disable_nullable: bool = False,
    fn_kwargs: Union[dict, NoneType] = None,
    num_proc: Union[int, NoneType] = None,  # 使用此参数，可以使用多进程处理
    desc: Union[str, NoneType] = None,
) -> 'DatasetDict'
Docstring:
Apply a function to all the elements in the table (individually or in batches)
and update the table (if function does updated examples).
The transformation is applied to all the datasets of the dataset dictionary.
```

关于这个map，在Huggingface的测试题中有讲解，这里搬运并翻译一下，辅助理解：

### Dataset.map方法有啥好处：

- The results of the function are cached, so it won't take any time if we re-execute the code.

    （通过这个map，对数据集的处理会被缓存，所以重新执行代码，也不会再费时间。）
- It can apply multiprocessing to go faster than applying the function on each element of the dataset.

    （它可以使用多进程来处理从而提高处理速度。）
- It does not load the whole dataset into memory, saving the results as soon as one element is processed.

    （它不需要把整个数据集都加载到内存里，同时每个元素一经处理就会马上被保存，因此十分节省内存。）

观察一下，这里通过map之后，得到的Dataset的features变多了：
```python
features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids']
```
多的几个columns就是tokenizer处理后的结果。

注意到，**在这个`tokenize_function`中，我们没有使用`padding`**，因为如果使用了padding之后，就会全局统一对一个maxlen进行padding，这样无论在tokenize还是模型的训练上都不够高效。



## Dynamic Padding 动态padding

实际上，我们是故意先不进行padding的，因为我们想**在划分batch的时候再进行padding**，这样可以避免出现很多有一堆padding的序列，从而可以显著节省我们的训练时间。

这里，我们就需要用到**`DataCollatorWithPadding`**，来进行**动态padding**：


```python
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

注意，我们需要使用tokenizer来初始化这个`DataCollatorWithPadding`，因为需要tokenizer来告知具体的padding token是啥，以及padding的方式是在左边还是右边（不同的预训练模型，使用的padding token以及方式可能不同）。


下面假设我们要搞一个size=5的batch，看看如何使用`DataCollatorWithPadding`来实现：


```python
samples = tokenized_datasets['train'][:5]
samples.keys()
# >>> ['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids']
samples = {k:v for k,v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}  # 把这里多余的几列去掉
samples.keys()
# >>> ['attention_mask', 'input_ids', 'label', 'token_type_ids']

# 打印出每个句子的长度：
[len(x) for x in samples["input_ids"]]
```


```shell
>>>
[50, 59, 47, 67, 59]
```

然后我们使用data_collator来处理：


```python
batch = data_collator(samples)  # samples中必须包含 input_ids 字段，因为这就是collator要处理的对象
batch.keys()
# >>> dict_keys(['attention_mask', 'input_ids', 'token_type_ids', 'labels'])

# 再打印长度：
[len(x) for x in batch['input_ids']]
```


```shell
>>>
[67, 67, 67, 67, 67]
```



可以看到，这个`data_collator`就是一个把给定dataset进行padding的工具，其输入跟输出是完全一样的格式。


```python
{k:v.shape for k,v in batch.items()}
```


```shell
>>>
{'attention_mask': torch.Size([5, 67]),
 'input_ids': torch.Size([5, 67]),
 'token_type_ids': torch.Size([5, 67]),
 'labels': torch.Size([5])}
```



这个batch，可以形成一个tensor了！接下来就可以用于训练了！

---

对了，这里多提一句，`collator`这个单词实际上在平时使用英语的时候并不常见，但却在编程中见到多次。

最开始一直以为是`collector`，意为“收集者”等意思，后来查了查，发现不是的。下面是柯林斯词典中对`collate`这个词的解释：

> **collate**: 
>
> When you collate pieces of information, you **gather** them all together and **examine** them. 

就是归纳并整理的意思。所以在我们这个情景下，就是对这些杂乱无章长短不一的序列数据，进行一个个地分组，然后检查并统一长度。

关于DataCollator更多的信息，可以参见文档：
https://huggingface.co/transformers/master/main_classes/data_collator.html?highlight=datacollatorwithpadding#data-collator