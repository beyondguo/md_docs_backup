---
title: 【DL笔记5】用TensorFlow搭建神经网络——手写数字识别
published: 2021-6-24
sidebar: auto
---

# 【DL笔记5】用TensorFlow搭建神经网络——手写数字识别

>之前又有很长一段时间在讲理论，上次实践还是用python实现Logistic regression。那是一个很有意义的尝试，虽然Logistic regression简单，但是真的亲手手动实现并不容易（我指的是在没有任何框架的加成下），但我们也深刻理解了内部的原理，而这么原理是模型再怎么复杂也不变的。
但是想构建更加复杂的网络，用纯python+numpy恐怕就很不容易了，主要是反向传播，涉及到大量的求导，十分麻烦。针对这种痛点，各种深度学习框架出现了，他们基本上都是帮我们自动地进行反向传播的过程，我们只用把正向传播的“图”构建出来即可。
所以，今天，我会介绍如何用TensorFlow这个深度学习最有名的的框架（之一吧，免得被杠），来实现一个3层的神经网络，来对MNIST手写数字进行识别，并且达到95%以上的测试集正确率。

## 一、TensorFlow的运行机制和基本用法

### TensorFlow运行机制：
刚开始接触TensorFlow的同学可能会发现它有点奇怪，跟我们一般的计算过程似乎不同。

首先我们要明确TensorFlow中的几个基本概念：
- Tensor 张量，是向量、矩阵的延伸，是tf中的运算的基本对象
- operation 操作，简称op，即加减乘除等等对张量的操作
- graph 图，由tensor和tensor之间的操作（op）搭建而成
- session 会话，用于启动图，将数据feed到图中，然后运算得到结果

其他的概念先放一边，我们先搞清楚上面这几个玩意儿的关系。

**在TF中构建一个神经网络并训练的过程，是这样的：**
先用tensor和op来搭建我们的graph，也就是要定义神经网络的各个参数、变量，并给出它们之间是什么运算关系，这样我们就搭建好了图（graph），可以想象是我们搭建好了一个管道。

![定义参数、变量，搭建成管道](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624595974113-image.png)

然后我们启动session（想象成一个水泵），给参数、变量初始化，并把我们的训练集数据注入到上面构建好的图（graph）中，让我们的数据按照我们搭建好的管道去流动（flow），并得到最终的结果。


![开启session，数据流动、循环](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624596013660-image.png)

一句话，先搭建数据流图，然后启动会话注入数据。TF自动完成梯度下降及相应的求导过程。

### TensorFlow基本用法：
1. **定义变量**
一般用下面两种方法来定义：
```python
w = tf.Variable(<initial-value>, name=<optional-name>)
```
或者用：
```python
w = tf.get_variable(<name>, <shape>, <initializer>)
```

我更常用后一种方法，因为可以直接指定initializer来赋值，比如我们常用的Xavier-initializer，就可以直接调用tf.contrib.layers.xavier_initializer(),不用我们自己去写函数进行初始化。

2. **placeholder**
我们一般给X、Y定义成一个placeholder，即占位符。也就是在构建图的时候，我们X、Y的壳子去构建，因为这个时候我们还没有数据，但是X、Y是我们图的开端，所以必须找一个什么来代替。这个placeholder就是代替真实的X、Y来进行图的构建的，它拥有X、Y一样的形状。
等session开启之后，我们就把真实的数据注入到这个placeholder中即可。
定义placeholder的方法：
```python
X = tf.placeholder(<dtype>,<shape>,<name>)
```

3. **operation**
op就是上面定义的tensor的运算。比如我们定义了W和b，并给X定义了一个placeholder，那么Z和A怎么计算呢：
```python
Z = tf.matmul(X,W)+b
A = tf.nn.relu(Z)
```
上面两个计算都属于op，op的输入为tensor，输出也为tensor，因此Z、A为两个新的tensor。
同样的，我们可以定义cost，然后可以定义一个optimizer来minimize这个cost（optimizer怎么去minimize cost不用我们操心了，我们不用去设计内部的计算过程，TF会帮我们计算，我们只用指定用什么优化器，去干什么工作即可）。这里具体就留给今天的代码实现了。

4. **session**
我们构建了图之后，就知道了cost是怎么计算的，optimizer是如何工作的。
然后我们需要启动图，并注入数据。
启动session有两种形式，本质上是一样的：
```python
sess = tf.Session()
sess.run(<tensor>,<feed_dic>)
...
sess.close()
```
或者：
```python
with tf.Session() as sess:
    sess.run(<tensor>,<feed_dic>)
    ...
```
后者就是会自动帮我们关闭session来释放资源，不用我们手动sess.close()，因为这个经常被我们忘记。

我们需要计算什么，就把相关的tensor写进`<tensor>`中去，计算图中的placeholder需要什么数据，我们就用feed_dic={X:...,Y:...}的方法来传进去。具体我们下面的代码实现部分见！


>上面就是最基本的TensorFlow的原理和用法了，我们下面开始搭建神经网络！好戏现在开始~


## 二、开始动手，搭建神经网络，识别手写数字

我们要面对的问题是啥呢？以前银行收到支票呀，都要人工去看上面的金额、日期等等手写数字，支票多了，工作量就很大了，而且枯燥乏味。那我们就想，能不能用机器是识别这些数字呢？

深度学习领域的大佬Yann LeCun（CNN的发明者）提供了一个手写数字数据集MNIST，可以说是深度学习的hello world了。数字长这样：

![MNIST手写数字](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624596071692-image.png)


其中每个图片的大小是 **28×28**，我们的 **数据集已经将图片给扁平化了，即由28×28，压扁成了784，也就是输入数据X的维度为784**.

我们今天就设计一个简单的 **3-layer-NN**，让识别率达到95%以上。
假设我们的网络结构是这样的：
第一层 **128**个神经元，第二层 **64**个神经元，第三层是 **Softmax输出层**，有 **10**个神经元，因为我们要识别的数组为0~9，共10个。网络结构如下（数字代表维度）：

![3-layer-NN](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624596094768-image.png)


好了，我们下面一步步地实现：

### （1）加载数据，引入相关的包
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# 下面这一行代码就可以直接从官网下载数据，下载完之后，你应该可以在目录中发现一个新文件夹“MNIST_data”
from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```
下面我们从数据集中，把我们的训练集、测试集都导出：
```python
X_train,Y_train = mnist.train.images,mnist.train.labels
X_test,Y_test = mnist.test.images,mnist.test.labels
# 不妨看看它们的形状：
print(X_train.shape)  # (55000, 784)
print(Y_train.shape)  # (55000, 10)
print(X_test.shape)   # (10000, 784)
print(Y_test.shape)   # (10000, 10)
```
可以看出，我们的训练样本有55000个，测试集有10000个。


### （2）根据网络结构，定义各参数、变量，并搭建图（graph）
```python
tf.reset_default_graph() # 这个可以不用细究，是为了防止重复定义报错

# 给X、Y定义placeholder，要指定数据类型、形状：
X = tf.placeholder(dtype=tf.float32,shape=[None,784],name='X')
Y = tf.placeholder(dtype=tf.float32,shape=[None,10],name='Y')

# 定义各个参数：
W1 = tf.get_variable('W1',[784,128],initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable('b1',[128],initializer=tf.zeros_initializer())
W2 = tf.get_variable('W2',[128,64],initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable('b2',[64],initializer=tf.zeros_initializer())
W3 = tf.get_variable('W3',[64,10],initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable('b3',[10],initializer=tf.zeros_initializer())
```

这里需要说明的有几点呢：

1. 最好给每个tensor **都取个名字**（name属性），这样报错的时候，我们可以方便地知道是哪个
2. **形状的定义要一致**，比如这里的W的形状，我们之前在讲解某些原理的时候，使用的是（当前层维度，上一层维度）,但是 **这里我们采用的是（上一层维度，当前层维度）**,所以分别是(784,128),(128,64),(64,10). 另外，X、Y的维度中的None，是样本数，由于我们同一个模型不同时候传进去的样本数可能不同，所以这里可以写 **None，代表可变的**。
3. **W的初始化**，可以直接用tf自带的initializer，但是注意不能用0给W初始化，这个问题我在之前的“参数初始化”的文章里面讲过。b可以用0初始化。


接着，我们根据上面的变量，来 **计算网络中间的logits（就是我们常用的Z）、激活值**：
```python
A1 = tf.nn.relu(tf.matmul(X,W1)+b1,name='A1')
A2 = tf.nn.relu(tf.matmul(A1,W2)+b2,name='A2')
Z3 = tf.matmul(A2,W3)+b3
```
**为什么我们只用算到Z3就行了呢**，因为TensorFlow中，计算损失有专门的函数，一般都是直接用Z的值和标签Y的值来计算，比如

对于sigmoid函数，我们有：
tf.nn.sigmoid_cross_entropy_with_logits(logits=,labels=)来计算，

对于Softmax，我们有：
tf.nn.softmax_cross_entropy_with_logits(logits=,labels=)来计算。
这个logits，就是未经激活的Z；labels，就是我们的Y标签。

因此我们如何 **定义我们的cos**t呢：
```python
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels=Y))
```
注意，为什么要用 **reduce_mean()**函数呢？因为经过softmax_cross_entropy_with_logits计算出来是，是所有样本的cost拼成的一个向量，有m个样本，它就是m维，因此我们需要去平均值来获得一个整体的cost。

定义好了cost，我们就可以 **定义optimizer来minimize cost**了：
```python
trainer = tf.train.AdamOptimizer().minimize(cost)
```
也是一句话的事儿，贼简单了。这里我们采用Adam优化器，用它来minimize cost。当然，我们可以在AdamOptimizer()中设置一些超参数，比如leaning_rate，但是这里我直接采用它的默认值了，一般效果也不错。


>至此，我们的整个计算图，就搭建好了，从X怎么一步步的加上各种参数，并计算cost，以及optimizer怎么优化cost，都以及明确了。接下来，我们就可以启动session，放水了！


### （3）启动图，注入数据，进行迭代
废话不多说，直接上代码：
```python
with tf.Session() as sess:
    # 首先给所有的变量都初始化（不用管什么意思，反正是一句必须的话）：
    sess.run(tf.global_variables_initializer())

    # 定义一个costs列表，来装迭代过程中的cost，从而好画图分析模型训练进展
    costs = []
    
    # 指定迭代次数：
    for it in range(1000):
        # 这里我们可以使用mnist自带的一个函数train.next_batch，可以方便地取出一个个地小数据集，从而可以加快我们的训练：
        X_batch,Y_batch = mnist.train.next_batch(batch_size=64)

        # 我们最终需要的是trainer跑起来，并获得cost，所以我们run trainer和cost，同时要把X、Y给feed进去：
        _,batch_cost = sess.run([trainer,cost],feed_dict={X:X_batch,Y:Y_batch})
        costs.append(batch_cost)

        # 每100个迭代就打印一次cost：
        if it%100 == 0:
            print('iteration%d ,batch_cost: '%it,batch_cost)

    # 训练完成，我们来分别看看来训练集和测试集上的准确率：
    predictions = tf.equal(tf.argmax(tf.transpose(Z3)),tf.argmax(tf.transpose(Y)))
    accuracy = tf.reduce_mean(tf.cast(predictions,'float'))
    print("Training set accuracy: ",sess.run(accuracy,feed_dict={X:X_train,Y:Y_train}))
    print("Test set accuracy:",sess.run(accuracy,feed_dict={X:X_test,Y:Y_test}))
```
运行，查看输出结果：
```
iteration0 ,batch_cost:  2.3507476
iteration100 ,batch_cost:  0.32707167
iteration200 ,batch_cost:  0.571893
iteration300 ,batch_cost:  0.2989539
iteration400 ,batch_cost:  0.1347334
iteration500 ,batch_cost:  0.24421218
iteration600 ,batch_cost:  0.13563904
iteration700 ,batch_cost:  0.26415896
iteration800 ,batch_cost:  0.1695988
iteration900 ,batch_cost:  0.17325541
Training set accuracy:  0.9624182
Test set accuracy:  0.9571
```
嚯，感觉不错！训练很快，不到5秒，已经达到我们的要求了，而且我们仅仅是迭代了1000次啊。

我们不妨将结果可视化一下，**随机抽查一些图片，然后输出对应的预测：**
**将下列代码放到上面的session中**（不能放在session外部，否则没法取出相应的值），重新运行：
```python
    # 这里改了一点上面的预测集准确率的代码，因为我们需要知道预测结果，所以这里我们单独把Z3的值给取出来，这样通过分析Z3，即可知道预测值是什么了。
    z3,acc = sess.run([Z3,accuracy],feed_dict={X:X_test,Y:Y_test})
    print("Test set accuracy:",acc)
    
    # 随机从测试集中抽一些图片（比如第i*10+j张图片），然后取出对应的预测（即z3[i*10+j]）：
    fig,ax = plt.subplots(4,4,figsize=(15,15))
    fig.subplots_adjust(wspace=0.1, hspace=0.7)
    for i in range(4):
        for j in range(4):
            ax[i,j].imshow(X_test[i*10+j].reshape(28,28))
            # 用argmax函数取出z3中最大的数的序号，即为预测结果：
            predicted_num  = np.argmax(z3[i*10+j])        
            # 这里不能用tf.argmax，因为所有的tf操作都是在图中，没法直接取出来
            ax[i,j].set_title('Predict:'+str(predicted_num))
            ax[i,j].axis('off')
```
得到结果：
  
  
![图片和预测结果](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624596126047-image.png)

可见，我们的模型是真的训练出来了，而且效果不错。这个图中，右下角的那个奇怪的“4”都给识别出来了。唯一有争议的是第三排第三个的那个数字，我感觉是4，不过也确实有点像6，结果模式识别它为6。

**总的来说还是很棒的**，接下来我觉得增大迭代次数，迭代它个10000次！然后看更多的图片(100张图片)。效果如下：
  
![准确率和cost曲线](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624596140633-image.png)

可见，准确率提高到了97%以上！
再展示一下图片：

![手写数字预测](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624596164750-image.png)

至此，我们的实验就完成了。我们成功地利用TensorFlow搭建了一个三层神经网络，并对手写数字进行了出色的识别！


---
对于TensorFlow更丰富更相信的使用，大家可以去TensorFlow中文社区或者TensorFlow官网了解。这里也推荐大家试试TensorFlow的高度封装的api——Keras，也是一个深度学习框架，它可以更加轻松地搭建一个网络。之后的文章我也会介绍keras的使用。





