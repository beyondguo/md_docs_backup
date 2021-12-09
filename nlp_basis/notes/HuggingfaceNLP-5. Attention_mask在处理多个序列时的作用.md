---
title:  Huggingface🤗NLP笔记5：attention_mask在处理多个序列时的作用
published: 2021-9-27
sidebar: auto
---

> **「Huggingface🤗NLP笔记系列-第5集」**
> 最近跟着Huggingface上的NLP tutorial走了一遍，惊叹居然有如此好的讲解Transformers系列的NLP教程，于是决定记录一下学习的过程，分享我的笔记，可以算是官方教程的精简+注解版。但最推荐的，还是直接跟着官方教程来一遍，真是一种享受。

- 官方教程网址：https://huggingface.co/course/chapter1
- 本期内容对应网址：https://huggingface.co/course/chapter2/5?fw=pt
- 本系列笔记的**GitHub**： https://github.com/beyondguo/Learn_PyTorch/tree/master/HuggingfaceNLP

---

# `attention_mask`在处理多个序列时的作用

现在我们训练和预测基本都是批量化处理的，而前面展示的例子很多都是单条数据。单条数据跟多条数据有一些需要注意的地方。

## 处理单个序列

我们首先加载一个在情感分类上微调过的模型，来进行我们的实验（注意，这里我们就不能能使用`AutoModel`，而应该使用`AutoModelFor*`这种带Head的model）。


```python
from pprint import pprint as print  # 这个pprint能让打印的格式更好看一点
from transformers import AutoModelForSequenceClassification, AutoTokenizer
checkpoint = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
```

对一个句子，使用tokenizer进行处理：


```python
s = 'Today is a nice day!'
inputs = tokenizer(s, return_tensors='pt')
print(inputs)
```

```shell
{'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]]),
 'input_ids': tensor([[ 101, 2651, 2003, 1037, 3835, 2154,  999,  102]])}
```


可以看到，这里的inputs包含了两个部分：`input_ids`和`attention_mask`.

模型可以直接接受`input_ids`：


```python
model(inputs.input_ids).logits
```

输出：


```shell
tensor([[-4.3232,  4.6906]], grad_fn=<AddmmBackward>)
```



也可以通过`**inputs`同时接受`inputs`所有的属性：


```python
model(**inputs).logits
```

输出：


    tensor([[-4.3232,  4.6906]], grad_fn=<AddmmBackward>)



上面两种方式的**结果是一样的**。

## 但是当我们需要同时处理**多个序列**时，情况就有变了！


```python
ss = ['Today is a nice day!',
      'But what about tomorrow? Im not sure.']
inputs = tokenizer(ss, padding=True, return_tensors='pt')
print(inputs)
```
输出：
```shell
{'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
 'input_ids': tensor([[  101,  2651,  2003,  1037,  3835,  2154,   999,   102,     0,     0,
             0],
        [  101,  2021,  2054,  2055,  4826,  1029, 10047,  2025,  2469,  1012,
           102]])}
```


然后，我们试着直接把这里的`input_ids`喂给模型


```python
model(inputs.input_ids).logits  # 第一个句子原本的logits是 [-4.3232,  4.6906]
```

输出：


```shell
tensor([[-4.1957,  4.5675],
        [ 3.9803, -3.2120]], grad_fn=<AddmmBackward>)
```



发现，**第一个句子的`logits`变了**！

这是**因为在padding之后，第一个句子的encoding变了，多了很多0， 而self-attention会attend到所有的index的值，因此结果就变了**。

这时，就需要我们不仅仅是传入`input_ids`，还需要给出`attention_mask`，这样模型就会在attention的时候，不去attend被mask掉的部分。

因此，**在处理多个序列的时候，正确的做法是直接把tokenizer处理好的结果，整个输入到模型中**，即直接`**inputs`。
通过`**inputs`，我们实际上就把`attention_mask`也传进去了:


```python
model(**inputs).logits
```

输出：


```shell
tensor([[-4.3232,  4.6906],
        [ 3.9803, -3.2120]], grad_fn=<AddmmBackward>)
```

现在第一个句子的结果，就跟前面单条处理时的一样了。

