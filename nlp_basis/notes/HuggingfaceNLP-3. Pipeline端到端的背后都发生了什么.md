---
title:  Huggingface🤗NLP笔记3：Pipeline端到端的背后发生了什么
published: 2021-9-24
sidebar: auto
---

> **「Huggingface🤗NLP笔记系列-第3集」**
> 最近跟着Huggingface上的NLP tutorial走了一遍，惊叹居然有如此好的讲解Transformers系列的NLP教程，于是决定记录一下学习的过程，分享我的笔记，可以算是官方教程的精简版。但最推荐的，还是直接跟着官方教程来一遍，真是一种享受。

- 官方教程网址：https://huggingface.co/course/chapter1
- 本期内容对应网址：https://huggingface.co/course/chapter2/2?fw=pt
- 本系列笔记的**GitHub**： https://github.com/beyondguo/Learn_PyTorch/tree/master/HuggingfaceNLP

---

# Pipeline端到端的背后发生了什么

Pipeline的背后：
<img src='https://huggingface.co/course/static/chapter2/full_nlp_pipeline.png' width=1000>

## 1. Tokenizer

我们使用的tokenizer必须跟对应的模型在预训练时的tokenizer保持一致，也就是词表需要一致。\
Huggingface中可以直接指定模型的checkpoint的名字，然后自动下载对应的词表。\
具体方式是：
- 使用`AutoTokenizer`的`from_pretrained`方法

`tokenizer`这个对象可以直接接受参数并输出结果，即它是callable的。具体参数见：\
https://huggingface.co/transformers/master/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase \
主要参数包括：
- text，可以是单条的string，也可以是一个string的list，还可以是list的list
- padding，用于填白
- truncation，用于截断
- max_length，设置最大句长
- return_tensors，设置返回数据类型


```python
from transformers import AutoTokenizer

checkpoint = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

先看看直接使用tokenizer的结果：


```python
raw_inputs = ['Today is a good day! Woo~~~',
              'How about tomorrow?']
tokenizer(raw_inputs)
```

输出：


```shell
{'input_ids': [[101, 2651, 2003, 1037, 2204, 2154, 999, 15854, 1066, 1066, 1066, 102], [101, 2129, 2055, 4826, 1029, 102]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]}
```



可以加上一个 `padding=Ture` 参数，让得到的序列长度对齐：


```python
tokenizer(raw_inputs, padding=True)
```

输出：


```shell
{'input_ids': [[101, 2651, 2003, 1037, 2204, 2154, 999, 15854, 1066, 1066, 1066, 102], [101, 2129, 2055, 4826, 1029, 102, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]}
```



tokenizer还有`truncation`和`max_length`属性，用于在max_length处截断：


```python
tokenizer(raw_inputs, padding=True, truncation=True, max_length=7) 
```

输出：


```shell
{'input_ids': [[101, 2651, 2003, 1037, 2204, 2154, 102], [101, 2129, 2055, 4826, 1029, 102, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0]]}
```



`return_tensors`属性也很重要，用来指定返回的是什么类型的tensors，`pt`就是pytorch，`tf`就是tensorflow：


```python
tokenizer(raw_inputs, padding=True, truncation=True, return_tensors='pt')
```

输出：


```shell
{'input_ids': tensor([[  101,  2651,  2003,  1037,  2204,  2154,   999, 15854,  1066,  1066,
          1066,   102],
        [  101,  2129,  2055,  4826,  1029,   102,     0,     0,     0,     0,
             0,     0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]])}
```



## 2. Model
也可以通过AutoModel来直接从checkpoint导入模型。

这里导入的模型，是Transformer的基础模型，接受tokenize之后的输入，**输出hidden states，即文本的向量表示**，是一种上下文表示。

这个向量表示，会有三个维度：
1. batch size
2. sequence length
3. hidden size


```python
from transformers import AutoModel
model = AutoModel.from_pretrained(checkpoint)
```

加载了模型之后，就可以把tokenizer得到的输出，直接输入到model中：


```python
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors='pt')
outputs = model(**inputs)  # 这里变量前面的**，代表把inputs这个dictionary给分解成一个个参数单独输进去
vars(outputs).keys()  # 查看一下输出有哪些属性
```

输出：


```shell
dict_keys(['last_hidden_state', 'hidden_states', 'attentions'])
```



>**这里顺便讲一讲这个函数中`**`的用法：**

`**`在函数中的作用就是把后面紧跟着的这个参数，从一个字典的格式，解压成一个个单独的参数。

回顾一下上面tokenizer的输出，我们发现它是一个包含了input_ids和attention_mask两个key的**字典**，因此通过`**`的解压，相当于变成了`intput_ids=..., attention_mask=...`喂给函数。

我们再来查看一下通过AutoModel加载的DistillBertModel模型的输入：
https://huggingface.co/transformers/master/model_doc/distilbert.html#distilbertmodel

可以看到DistillBertModel的直接call的函数是：

`forward(input_ids=None, attention_mask=None, ...)`
正好跟`**inputs`后的格式对应上。


```python
print(outputs.last_hidden_state.shape)
outputs.last_hidden_state
```
输出
```shell
torch.Size([2, 12, 768])

    tensor([[[ 0.4627,  0.3042,  0.5431,  ...,  0.3706,  1.0033, -0.6074],
             [ 0.6100,  0.3093,  0.2038,  ...,  0.3788,  0.9370, -0.6439],
             [ 0.6514,  0.3185,  0.3855,  ...,  0.4152,  1.0199, -0.4450],
             ...,
             [ 0.3674,  0.1380,  1.1619,  ...,  0.4976,  0.4758, -0.5896],
             [ 0.4182,  0.2503,  1.0898,  ...,  0.4745,  0.4042, -0.5444],
             [ 1.1614,  0.2516,  0.9561,  ...,  0.5742,  0.8437, -0.9604]],
    
            [[ 0.7956, -0.2343,  0.3810,  ..., -0.1270,  0.5182, -0.1612],
             [ 0.9337,  0.2074,  0.6202,  ...,  0.1874,  0.6584, -0.1899],
             [ 0.6279, -0.3176,  0.1596,  ..., -0.2956,  0.2960, -0.1447],
             ...,
             [ 0.3050,  0.0396,  0.6345,  ...,  0.4271,  0.3367, -0.3285],
             [ 0.1773,  0.0111,  0.6275,  ...,  0.3831,  0.3543, -0.2919],
             [ 0.2756,  0.0048,  0.9281,  ...,  0.2006,  0.4375, -0.3238]]],
           grad_fn=<NativeLayerNormBackward>)
```


可以看到，输出的shape是`torch.Size([2, 12, 768])`，三个维度分别是 batch，seq_len和hidden size。




## 3. Model Heads
模型头，接在基础模型的后面，用于将hidden states文本表示进一步处理，用于具体的任务。

整体框架图：

<img src='https://huggingface.co/course/static/chapter2/transformer_and_head.png' width=1000>

Head一般是由若干层的线性层来构成的。

Transformers库中的主要模型架构有：
- *Model (retrieve the hidden states)
- *ForCausalLM
- *ForMaskedLM
- *ForMultipleChoice
- *ForQuestionAnswering
- *ForSequenceClassification
- *ForTokenClassification
- ...

单纯的`*Model`，就是不包含 Head 的模型，而有`For*`的则是包含了具体 Head 的模型。

例如，对于前面的那个做在情感分析上pretrain的checkpoint(distilbert-base-uncased-finetuned-sst-2-english)，我们可以使用包含 SequenceClassification 的Head的模型去加载，就可以直接得到对应分类问题的logits，而不仅仅是文本向量表示。


```python
from transformers import AutoModelForSequenceClassification
clf = AutoModelForSequenceClassification.from_pretrained(checkpoint)
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors='pt')
outputs = clf(**inputs)
print(vars(outputs).keys())
outputs.logits
```
输出：
```shell
dict_keys(['loss', 'logits', 'hidden_states', 'attentions'])

tensor([[-4.2098,  4.6444],
        [ 0.6367, -0.3753]], grad_fn=<AddmmBackward>)
```



从outputs的属性就可以看出，带有Head的Model，跟不带Head的Model，输出的东西是不一样的。

**没有Head的Model**，输出的是`'last_hidden_state', 'hidden_states', 'attentions'`这些玩意儿，因为它仅仅是一个表示模型；

**有Head的Model**，输出的是`'loss', 'logits', 'hidden_states', 'attentions'`这些玩意儿，有logits，loss这些东西，因为它是一个完整的预测模型了。

可以顺便看看，加了这个 SequenceClassification Head的DistillBertModel的文档，看看其输入和输出：

https://huggingface.co/transformers/master/model_doc/distilbert.html#distilbertforsequenceclassification

可以看到，输入中，我们还可以提供`labels`，这样就可以直接计算loss了。

## 4. Post-Processing
后处理主要就是两步：
- 把logits转化成概率值 （用softmax）
- 把概率值跟具体的标签对应上 （使用模型的config中的id2label）


```python
import torch
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)  # dim=-1就是沿着最后一维进行操作
predictions
```

输出：


```shell
tensor([[1.4276e-04, 9.9986e-01],
        [7.3341e-01, 2.6659e-01]], grad_fn=<SoftmaxBackward>)
```



得到了概率分布，还得知道具体是啥标签吧。标签跟id的隐射关系，也已经被保存在每个pretrain model的config中了，
我们可以去模型的`config`属性中查看`id2label`字段：


```python
id2label = clf.config.id2label
id2label
```

输出：


```shell
{0: 'NEGATIVE', 1: 'POSITIVE'}
```

综合起来，直接从prediction得到标签：


```python
for i in torch.argmax(predictions, dim=-1):
    print(id2label[i.item()])
```

输出：
```shell
    POSITIVE
    NEGATIVE
```

