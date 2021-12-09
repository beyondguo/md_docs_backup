---
title:  Huggingface🤗NLP笔记8：使用PyTorch来微调模型
published: 2021-10-01
sidebar: auto
---

> **「Huggingface🤗NLP笔记系列-第8集」**
> Huggingface初级教程完结撒花！🌸🌼ヽ(°▽°)ノ🌸🌺
> 最近跟着Huggingface上的NLP tutorial走了一遍，惊叹居然有如此好的讲解Transformers系列的NLP教程，于是决定记录一下学习的过程，分享我的笔记，可以算是官方教程的**精简+注解版**。但最推荐的，还是直接跟着官方教程来一遍，真是一种享受。

- 官方教程网址：https://huggingface.co/course/chapter1
- 本期内容对应网址：https://huggingface.co/course/chapter3/4?fw=pt
- 本系列笔记的**GitHub Notebook(可下载直接运行)**： https://github.com/beyondguo/Learn_PyTorch/tree/master/HuggingfaceNLP

---


# 更加透明的方式——使用PyTorch来微调模型

这里我们不使用Trainer这个高级API，而是用pytorch来实现。


## 1. 数据集预处理
在Huggingface官方教程里提到，在使用pytorch的dataloader之前，我们需要做一些事情：
- 把dataset中一些不需要的列给去掉了，比如‘sentence1’，‘sentence2’等
- 把数据转换成pytorch tensors
- 修改列名 label 为 labels

其他的都好说，但**为啥要修改列名 label 为 labels，好奇怪哦！**
这里探究一下：

首先，Huggingface的这些transformer Model直接call的时候，接受的标签这个参数是叫"labels"。
所以不管你使用Trainer，还是原生pytorch去写，最终模型处理的时候，肯定是使用的名为"labels"的标签参数。


但在Huggingface的datasets中，数据集的标签一般命名为"label"或者"label_ids"，那为什么在前两集中，我们没有对标签名进行处理呢？

这一点在transformer的源码`trainer.py`里找到了端倪：
```python
# 位置在def _remove_unused_columns函数里
# Labels may be named label or label_ids, the default data collator handles that.
signature_columns += ["label", "label_ids"]
```
这里提示了， data collator 会负责处理标签问题。然后我又去查看了`data_collator.py`中发现了一下内容：
```python
class DataCollatorWithPadding:
    ...
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        ...
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch
```
这就真相大白了：不管数据集中提供的标签名叫"label"，还是"label_ids"，
DataCollatorWithPadding 都会帮你转换成"labels"，装进batch里，再返回。

前面使用Trainer的时候，DataCollatorWithPadding已经帮我们自动转换了，因此我们不需要操心这个问题。

但这就是让我疑惑的地方：我们使用pytorch来写，其实也不用管这个，因为在pytorch的data_loader里面，有一个`collate_fn`参数，我们可以把DataCollatorWithPadding对象传进去，也会帮我们自动把"label"转换成"labels"。因此实际上，**这应该是教程中的一个小错误，我们不需要手动设计**（前两天在Huggingface GitHub上提了issue，作者回复我确实不用手动设置）。

---

下面开始正式使用pytorch来训练：

首先是跟之前一样，我们需要加载数据集、tokenizer，然后把数据集通过map的方式进行预处理。我们还需要定义一个`data_collator`方便我们后面进行批量化处理模型：


```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

查看一下处理后的dataset：

```python
print(tokenized_datasets['train'].column_names)
```

```shell
>>>
['attention_mask', 'idx', 'input_ids', 'label', 'sentence1', 'sentence2', 'token_type_ids']
```


huggingface datasets贴心地准备了三个方法：`remove_columns`, `rename_column`, `set_format`

来方便我们为pytorch的Dataloader做准备：


```python
tokenized_datasets = tokenized_datasets.remove_columns(['sentence1', 'sentence2','idx'])
# tokenized_datasets = tokenized_datasets.rename_column('label','labels')  # 实践证明，这一行是不需要的
tokenized_datasets.set_format('torch')

print(tokenized_datasets['train'].column_names)
```

```shell
>>>
['attention_mask', 'input_ids', 'label', 'token_type_ids']
```

查看一下：

```python
tokenized_datasets['train']  # 经过上面的处理，它就可以直接丢进pytorch的Dataloader中了，跟pytorch中的Dataset格式已经一样了
```


```shell
>>>
Dataset({
    features: ['attention_mask', 'input_ids', 'label', 'token_type_ids'],
    num_rows: 3668
})
```



定义我们的**pytorch dataloaders**：

在pytorch的`DataLoader`里，有一个`collate_fn`参数，其定义是："merges a list of samples to form a mini-batch of Tensor(s).  Used when using batched loading from a map-style dataset." 我们可以直接把Huggingface的`DataCollatorWithPadding`对象传进去，用于对数据进行padding等一系列处理：


```python
from torch.utils.data import DataLoader, Dataset
train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=8, collate_fn=data_collator)  # 通过这里的dataloader，每个batch的seq_len可能不同
eval_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=8, collate_fn=data_collator)
```


```python
# 查看一下train_dataloader的元素长啥样
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}
# 可见都是长度为72，size=8的batch
```


```shell
>>>
{'attention_mask': torch.Size([8, 72]),
 'input_ids': torch.Size([8, 72]),
 'token_type_ids': torch.Size([8, 72]),
 'labels': torch.Size([8])}
```

观察一下经过DataLoader处理后的数据，我们发现，标签那一列的列名，已经从`"label"`变为`"labels"`了！

## 2. 模型


```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

前面dataloader出来的batch可以直接丢进模型处理：

```python
model(**batch) 
```


```shell
>>>
SequenceClassifierOutput(loss=tensor(0.7563, grad_fn=<NllLossBackward>), logits=tensor([[-0.2171, -0.4416],
        [-0.2248, -0.4694],
        [-0.2440, -0.4664],
        [-0.2421, -0.4510],
        [-0.2273, -0.4545],
        [-0.2339, -0.4515],
        [-0.2334, -0.4387],
        [-0.2362, -0.4601]], grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)
```



## 定义 optimizer 和 learning rate scheduler

按道理说，Huggingface这边提供Transformer模型就已经够了，具体的训练、优化，应该交给pytorch了吧。但鉴于Transformer训练时，最常用的优化器就是AdamW，这里Huggingface也直接在`transformers`库中加入了`AdamW`这个优化器，还贴心地配备了lr_scheduler，方便我们直接使用。


```python
from transformers import AdamW, get_scheduler

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)  # num of batches * num of epochs
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,  # scheduler是针对optimizer的lr的
    num_warmup_steps=0,
    num_training_steps=num_training_steps)
print(num_training_steps)
```

    1377


## 3. Training

首先，我们设置cuda device，然后把模型给移动到cuda上：


```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```



## 编写pytorch training loops:

这里也很简单，思路就是这样：
1. for每一个epoch
2. 从dataloader里取出一个个batch
3. 把batch喂给model（先把batch都移动到对应的device上）
4. 拿出loss，进行反向传播backward
5. 分别把optimizer和scheduler都更新一个step

最后别忘了每次更新都要清空grad，即对optimizer进行zero_grad()操作。


```python
from tqdm import tqdm

for epoch in range(num_epochs):
    for batch in tqdm(train_dataloader):
        # 要在GPU上训练，需要把数据集都移动到GPU上：
        batch = {k:v.to(device) for k,v in batch.items()}
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
```

```shell
100%|██████████| 459/459 [01:54<00:00,  4.01it/s]
100%|██████████| 459/459 [01:55<00:00,  3.98it/s]
100%|██████████| 459/459 [01:55<00:00,  3.96it/s]
```

## 4. Evaluation

这里跟train loop还是挺类似的，一些细节见注释即可：


```python
from datasets import load_metric

metric= load_metric("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():  # evaluation的时候不需要算梯度
        outputs = model(**batch)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    # 由于dataloader是每次输出一个batch，因此我们要等着把所有batch都添加进来，再进行计算
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
```


```shell
>>>
{'accuracy': 0.8651960784313726, 'f1': 0.9050086355785838}
```

---

至此，Huggingface Transformer初级教程就完结撒花了！

<center>🌸🌼ヽ(°▽°)ノ🌸🌺</center>

更高级的教程，Huggingface也还没出😂，所以咱们敬请期待吧！不过，学完了这个初级教程，我们基本是也可以快乐地操作各种各样Transformer-based模型自由玩耍啦！