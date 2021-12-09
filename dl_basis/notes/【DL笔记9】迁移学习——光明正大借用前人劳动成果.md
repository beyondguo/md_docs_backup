---
title: 【DL笔记9】迁移学习——光明正大借用前人劳动成果
published: 2021-6-24
sidebar: auto
---


# 【DL笔记9】迁移学习——光明正大借用前人劳动成果
>好一阵没有继续更新深度学习系列的笔记了，其实文章的技术方面的内容在暑假就已经准备好了，只是开学一个月事情比较多，一直没时间整理下来。今天咱们再续前缘，继续完善深度学习的基础知识框架。今天的主角是——**迁移学习(Transfer learning)**。而且，这次不让大家空手离开，特地准备了一个实践案例——**手势识别，让你的小电脑看得懂手语**！


### 一、什么是迁移学习，有何来历？

在前面学习了CNN的基本原理，并且了解了一些著名的框架之后，我们似乎都有一种冲动————训练一个自己的模型，做出一些有趣的应用。毕竟，学过前面的知识之后，我们发现很多网络结构虽然层数多，但是其实很简单，比如VGG，十分富有规律性。

但是，自己从0到1训练一个模型十分困难，主要在下面两个方面：

1. **没有足够多的数据：**
深度学习模型，最需要的就是大数据。没有大量的数据进行支撑，我们很难训练出一个理想的模型。
2. **没有足够多的计算、时间资源**
即使我们有了足够多的数据，计算量也是一个大问题。那些著名的模型，基本都是用最好的GPU，动辄训练一个星期得到的。对于我们初学者来说，这个条件很难满足。而且，实际操作中，有很多的trick，这需要大量的试验。

因此，从头开始训练一个模型很多时候是不现实的。这个时候，迁移学习就有了它的用武之地。让我们在 **数据量不大、资源也不够的情况下，也可以训练很好的模型**。


<big>**迁移学习**</big>，顾名思义，就是把别人的模型，迁移过来，来学习自己的任务。

对于迁移学习，我可以打一个不是那么恰当的**比方**：别人修建了一间房子，住了一段时候不住了，转手给你住，请问你需要把房子推倒了重建吗？当然不需要，我们顶多把里面的重新装修一下，就行了。

现成的模型就是别人造好的房子，你拿过来，结构、参数都可以不变，因为很多东西是通用的。我们只用把我们属于自己个性化任务的那部分，按照自己喜好改造一下即可。

比方说，我们可以把在ImageNet上用千万张图片训练好的VGG拿过来（可以识别1000个种类），把最后两个FC层（全连接层）给拿掉，换成我们自己定义的FC层和输出层，其他的层则保持结构不变，参数也采用之前的参数。然后拿我们自己的金毛、二哈的照片（也许只有不到一千张）去训练，得到一个分辨金毛二哈的分类器。


### 二、为什么可以这样做？

如果你还记得我之前的文章【DL笔记8】，你应该已经熟悉了卷积神经网络(CNN)各层学习到的特征有什么特点。这里简单回顾一下：
1. 越浅的层，学习的特征越简单基础；越深的层，学习的特征越复杂而具体。所以，我们可以发现，从浅层到深层，识别的特征从边缘、线条、颜色，过渡到纹理、图案，再过渡到复杂图形，甚至到具体的物品。
2. **越前面的层，特征越具有一般性，越深的层，特征越具有场景的特殊性。**

尤其是最后一条，我们知道，**一个训练好的CNN模型，其实已经具备了很多通用的特征**。比如用ImageNet训练出来的VGG，它的各层的特征已经包含了识别各种物体的基本属性，比如棱角、颜色、基本的形状等等，而我们想训练一个金毛二哈分类器，其实也要从这些底层的基本特征来学起，因此，完全可以直接把训练好的VGG和配套的特征(特征即filters的参数)拿过来，我们只用训练一些高级特征即可。

#### ——>运用transfer learning一般有两种方式：

1. 用训练好的模型作为**特征抽取器**(Pre-trained model as a **feature extractor**)

这个是我们最常见的形式。前面说了，其实训练好的很多CNN模型，已经可以从我们的图片中提取出很多有用的特征了，那么这个时候我们可以直接拿过来，把这些CNN当做我们特征提取的工具。特征提取好了之后，我们再在后面接上一个简单的分类器，即可实现我们的学习任务。
这种方法，往往用在我们没有大量的训练样本的时候，也可以取得出色的效果。(今天我们就会用这种方法，来实现一个“手势”的分类器！)


2. 对训练好的模型进行**微调**(**Fine-tuning** a pre-trained model)

Fine-tune这个词大家应该经常听说。它的意思是把训练好的模型，在它原来的基础上，借助我们自己的训练样本，进行微调。
什么意思呢？在上面的直接当做feature extractor中，我们是直接固定好之前训练好的参数，只是把网络最后几层去掉，换成一个小型的分类器进行训练，训练的实际上是我们的小分类器。但是fine-tune不光是把原网络的末端换成我们自己的分类器，还会把整个网络的参数都继续训练，只不过我们不用从头训练，而是在原来的参数的基础上接着训练即可。
这种方法，一般用于我们自己也有大量的数据，这个时候，借用他人成功的网络结构和预训练参数，再充分利用自己的数据，就可以得到十分好的效果。
如果我们的样本量不大，那就没必要去fine-tune了，因为对效果的提升不会有很大的帮助，直接拿来用即可。



### 三、亲手试一试吧！
看了半天猪在跑，不如亲自尝一尝猪肉！
这里，我拿来了吴恩达的一个数据集，是他们团队自己采集的————0~5的手势。长的如下这样：

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624598534927-image.png)



这个任务是不是看起来比之前的MNIST手写数字识别要难多了？手写数字毕竟单纯，而且那个数据集十分地规范，而且图片还都是单通道的！。但是吴恩达的这个数据集就复杂多了，是他们团队亲自拍摄的，每一个数字的手势会有很大的差别，形态各异，还是正经的RGB图片，具体大家可以下载数据集之后自己去查看。这个任务就比较贴近我们的实际生活了。

我们的目标是：达到**90%以上的测试集准确率**！


由于这个任务稍微有点复杂，所以我们需要借助一个更复杂的网络结构来实现。于是自然而然地想到用Transfer learning。这里，我选择的是著名的**VGG-19**，是一个有19层的卷积神经网络。

我先画一张图来表示一下我们要做的事儿：

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624598549463-image.png)

话不多说，上代码吧：

#### 1. 引入相关的包：
```python
import h5py
import numpy as np
import keras
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense,GlobalMaxPooling2D,Input
import matplotlib.pyplot as plt
%matplotlib inline
```

相关说明：
**h5py**是一个python读取h5数据文件的工具，本实验中，我们的数据是存在h5中，这种存储方式十分经济。
**在keras中，已经内置了一些著名的模型的结构**，比如VGG19，因此我们可以直接通过keras.applications调用。

#### 2. 加载数据：
```python
# 导入数据：
train_dataset = h5py.File('datasets/train_signs.h5', "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:]) 
train_set_y_orig = np.array(train_dataset["train_set_y"][:]) 
test_dataset = h5py.File('datasets/test_signs.h5', "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:]) 
test_set_y_orig = np.array(test_dataset["test_set_y"][:]) 

classes = np.array(test_dataset["list_classes"][:]) # the list of classes

train_set_y_orig = train_set_y_orig.reshape((train_set_y_orig.shape[0],1))
test_set_y_orig = test_set_y_orig.reshape((test_set_y_orig.shape[0],1))

print("-----Reshaped data:")
X_train = train_set_x_orig/255
X_test = test_set_x_orig/255
Y_train = keras.utils.to_categorical(train_set_y_orig,6)
Y_test = keras.utils.to_categorical(test_set_y_orig,6)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
```

这部分代码没什么好看的，唯一需要留意的就是后面的“reshape”的过程，首先我们要对训练集进行**归一化处理**，然后要把标签值通过**keras.utils.to_categorical**方法转换成**one-hot**的形式，方便训练。

**为了防止有的读者对one-hot形式不熟悉，我这里解释一下：**
本实验中，我们的手势的标签分别是0,1,2,3,4,5这6个标签。
在原数据中，y就是一维的，就是0~5这些数字，
one-hot表示法就是这样变：
0 --> (1,0,0,0,0,0)
1 --> (0,1,0,0,0,0)
2 --> (0,0,1,0,0,0)
3 --> (0,0,0,1,0,0)
4 --> (0,0,0,0,1,0)
5 --> (0,0,0,0,0,1)

为啥要这么处理呢？
因为我们知道，我们的输出层，无论是sigmoid还是Softmax，都是这样的one-hot形式，所以我们需要把我们的标签改成一致的。那么这个keras.utils.to_categorical方法，大家可得记住了，日后会经常用到。


加载完之后，可以随便挑一个出来看看，我们的训练集长啥样：
```python
plt.imshow(train_set_x_orig[30])
```
得到

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624598565090-image.png)



#### 3.构造模型
我们首先引入训练好的VGG模型，同时把原模型的FC层去掉：
```python
base_model = VGG19(weights='imagenet',include_top=False)
```
我们看看参数：
`weights = 'imagenet'`说明，我们这个VGG19是通过ImageNet的图片训练的，ImageNet上有成千上万张图片，因此已经学得了大量的各种图片的特征，我们完全可以拿过来用。
`include_top = False`表示去掉模型的最后的全连接层（一般是两层，其中一个是Softmax输出层）。

然后，我们重新构造最后两层，我们先把上面的去掉尾巴的模型做一个pooling，简化计算量，然后添加一个128单元的FC层，用relu激活，最后输出层使用Softmax（就是一个6单元的FC层使用了Softmax激活）：
```python
x = base_model.output
x = GlobalMaxPooling2D()(x)
x = Dense(128,activation='relu')(x)
predictions = Dense(6,activation='softmax')(x)
model = Model(inputs=base_model.input,outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False
    
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
```
注意，由于我们的样本数量比较少，所以不用去fine-tune原模型的参数，因此我们直接**把原模型的参数固定**下来，所以上面我们需要设置`layer.trainable = False`.

模型的编译就很简单了，直接选择优化器、损失函数即可。

#### 4.模型的训练和评估
这里就直接上代码了：
```python
model.fit(X_train,Y_train,epochs=20,batch_size=64)
score = model.evaluate(X_test,Y_test)
print("Total loss:",score[0])
print("Test accuracy:",score[1])
```
迭代次数大概20~30次吧，我的电脑只能使用CPU，每次迭代大概70多秒，所以总时间需要约半小时。
最后的输出结果(只展示最后3个epoch的数据，不然太占地儿了)：
```
Epoch 1/3
1080/1080 [==============================] - 72s 67ms/step - loss: 0.2111 - acc: 0.9630
Epoch 2/3
1080/1080 [==============================] - 75s 69ms/step - loss: 0.1942 - acc: 0.9676
Epoch 3/3
1080/1080 [==============================] - 83s 77ms/step - loss: 0.1845 - acc: 0.9685

120/120 [==============================] - 8s 67ms/step
Total loss: 0.2701009293397268
Test accuracy: 0.9166666626930237
```

可以看到，最后的测试集准确率达到了91.67%！目标达成！


准确率不错，为了更直观地看看模型的效果，我们不妨做一个可交互的查看的结果的方法：输入一个图片，计算机返回预测结果：


![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624598578722-image.png)




---
>至此，我们已经知道了迁移学习的原理，并亲自动手实践了一下。
>可以看出，虽然手势识别比我们之前做的数字识别要复杂的多，但是相比ImageNet的图片识别，还是太简单。因此，我们用VGG来做迁移学习，显然是“杀鸡用牛刀”，所以我们简单地训练20来次，就可以达到很高的准确率，如果使用GPU的话，那10分钟的训练，估计准确率就可以接近100%了（我的猜测）。这也说明了迁移学习的强大和方便之处。
>因此，在我们的实际任务中，其实可以多想想，是否有机会**“站在巨人的肩膀上”**，让我们事半功倍。





