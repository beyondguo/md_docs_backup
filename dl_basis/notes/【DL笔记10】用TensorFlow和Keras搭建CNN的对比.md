---
title: 【DL笔记10】用TensorFlow和Keras搭建CNN的对比
published: 2021-6-24
sidebar: auto
---

# 用TensorFlow和Keras搭建CNN的对比

>本篇文章我们会使用**两种框架**（TensorFlow和Keras，虽然Keras从某种意义上是TF的一种高层API）**来实现一个简单的CNN**，来对我们之前的MNIST手写数字进行识别。还记得上一次我们用TF实现了一个简单的三层神经网络，最终测试集准确率达到95%的水平。今天，我们**期望达到99%以上的准确率！**

## 一、使用TensorFlow框架

### 1.引入基本的包和数据集：


```python
import tensorflow as tf
sess = tf.InteractiveSession()
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X_train,Y_train = mnist.train.images,mnist.train.labels
X_test,Y_test = mnist.test.images,mnist.test.labels
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
```


    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
    (55000, 784)
    (55000, 10)
    (10000, 784)
    (10000, 10)



**这里需要多说一句的就是这个InteractiveSession。**

还记得我们上次使用TF的时候，是用`sess = tf.Session()`或者`with tf.Session() as sess:`的方法来启动session。

他们两者有什么区别呢：
`InteractiveSession()`，多用于交互式的环境中，如IPython Notebooks，比`Session()`更加方便灵活。
在前面我们知道，我们所有的计算，都必须在：
```
with tf.Session as sess:
    ... ...
```
中进行，如果出界了，就会报错，说“当前session中没有这个操作/张量”这种话。

但是在`InteractiveSession()`中，我们只要创建了，就会全局默认是这个session，我们可以随时添加各种操作，用`Tensor.eval()`和`Operation.run()`来随机进行求值和计算。


### 2.设置CNN的结构
为了方便，我们不是直接给变量赋值，而是写一些通用的函数：

**对于weights**，我们在前面的文章【】中提到过，需要随机初始化参数，不能直接用0来初始化，否则可能导致无法训练。因此我们这里采用**truncated_normal**方法，normal分布就是我们熟悉的正态分布，truncated_normal就是**将正太分布的两端的过大过小的值去掉了**，使得我们的训练更容易。

而**对于bias**，它怎么初始化就无所谓了，我们简单起见，就直接用0初始化了。


```python
def weights(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias(shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial)
```

然后一些定义层的函数：

定义卷积层、池化层的时候有很多超参数要设置，但是很多参数可能是一样的，所以我们写一个函数会比较方便。

参数的格式有必要在这里说一说，因为这个很容易搞混淆：

**对于卷积层：**
- 输入(inputs/X):[batch, height, width, channels],就是[样本量，高，宽，通道数];
- 权重(filter/W):[filter_height, filter_width, in_channels, out_channels],这个就不解释了吧，英语都懂哈;
- 步长(strides):一开始我很奇怪为什么会是四维的数组，查看文档才只有，原来是对应于input每一维的步长。而我们一般只关注对宽、高的步长，所以假如我们希望步长是2，那么stride就是[1,2,2,1];
- 填白(padding):这个是我们之前讲过的，如果不填白，就设为VALID，如果要填白使得卷积后大小不变，那么就用SAME.

**对于池化层：**
- 输入、步长、填白这三者跟上面一样;
- ksize:指的是kernel-size，就是在pooling的时候的窗口，也是一个四维数组，对应于输入的每一维。跟strides一样，我们一般只关心中间两个维度，比如我们希望一个2×2的窗口，就设为[1,2,2,1].


```python
def conv(X,W):
    return tf.nn.conv2d(X,W,strides=[1,1,1,1],padding='SAME')  

def max_pool(X):
    return tf.nn.max_pool(X,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
```

好了，准备工作都做好了，我们可以开始正式搭建网络结构了！

先为X,Y定义一个占位符，后面运行的时候再注入数据：


```python
X= tf.placeholder(dtype=tf.float32,shape=[None,784])
X_image = tf.reshape(X,[-1,28,28,1])  ## 由于我们的卷积层处理的是思维的输入结构，我们这里需要对输入进行变形
Y = tf.placeholder(dtype=tf.float32,shape=[None,10])
```

我们就来两个卷积层（每一个后面都接一个池化层）,后接一个全连接层吧：


```python
## 第一个卷积层：
W1 = weights([5,5,1,32])
b1 = bias([32])
A1 = tf.nn.relu(conv(X_image,W1)+b1,name='relu1')
A1_pool = max_pool(A1)

## 第二个卷积层：
W2 = weights([5,5,32,64])
b2 = bias([64])
A2 = tf.nn.relu(conv(A1_pool,W2)+b2,name='relu2')
A2_pool = max_pool(A2)

## 全连接层：
W3 = weights([7*7*64,128])
b3 = bias([128])
A2_flatten = tf.reshape(A2_pool,[-1,7*7*64])
A3= tf.nn.relu(tf.matmul(A2_flatten,W3)+b3,name='relu3')

## 输出层（Softmax）：
W4 = weights([128,10])
b4 = bias([10])
Y_pred = tf.nn.softmax(tf.matmul(A3,W4)+b4)
```

定义损失和优化器：


```python
loss = -tf.reduce_sum(Y*tf.log(Y_pred))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
```

### 3.开始训练


```python
sess.run(tf.global_variables_initializer())
```


```python
costs = []
for i in range(1000):
    X_batch,Y_batch = mnist.train.next_batch(batch_size=64)
    _,batch_cost = sess.run([train_step,loss],feed_dict={X:X_batch,Y:Y_batch})
    if i%100 == 0:
        print("Batch%d cost:"%i,batch_cost)
        costs.append(batch_cost)
print("Training finished!")
```

    Batch0 cost: 0.16677594
    Batch100 cost: 0.052068923
    Batch200 cost: 0.5979577
    Batch300 cost: 0.049106397
    Batch400 cost: 0.047060404
    Batch500 cost: 2.0360851
    Batch600 cost: 3.3168547
    Batch700 cost: 0.11393449
    Batch800 cost: 0.06208247
    Batch900 cost: 0.035165284
    Training finished!



```python
correct_prediction = tf.equal(tf.argmax(Y_pred,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
accuracy.eval(feed_dict={X:X_test,Y:Y_test})
```




    0.9927



---
上面，我们已经用TensorFlow搭建了一个4层的卷积神经网络（2个卷积层，两个全连接层，注意，最后的Softmax层实际上也是一个全连接层）

接下来，我们用Keras来搭建一个一模一样的模型，来对比一下二者在实现上的差异。


## 二、使用Keras框架


```python
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
```


```python
model = Sequential()
# 第一个卷积层（后接池化层）：
model.add(Conv2D(32,kernel_size=(5,5),padding='same',activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

# 第二个卷积层（后接池化层）：
model.add(Conv2D(64,kernel_size=(5,5),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

# 将上面的结果扁平化，然后接全连接层：
model.add(Flatten())
model.add(Dense(128,activation='relu'))

#最后一个Softmax输出：
model.add(Dense(10,activation='softmax'))
```


```python
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
```


```python
X_train_image = X_train.reshape(X_train.shape[0],28,28,1)
X_test_image = X_test.reshape(X_test.shape[0],28,28,1)
```


```python
model.fit(X_train_image,Y_train,epochs=6,batch_size=64)
```

    Epoch 1/6
    55000/55000 [==============================] - 138s 3ms/step - loss: 0.0093 - acc: 0.9968
    Epoch 2/6
    55000/55000 [==============================] - 140s 3ms/step - loss: 0.0064 - acc: 0.9979
    Epoch 3/6
    55000/55000 [==============================] - 141s 3ms/step - loss: 0.0052 - acc: 0.9982
    Epoch 4/6
    55000/55000 [==============================] - 141s 3ms/step - loss: 0.0049 - acc: 0.9984
    Epoch 5/6
    55000/55000 [==============================] - 140s 3ms/step - loss: 0.0060 - acc: 0.9981
    Epoch 6/6
    55000/55000 [==============================] - 141s 3ms/step - loss: 0.0034 - acc: 0.9991





    <keras.callbacks.History at 0xcd1f9be898>




```python
result = model.evaluate(X_test_image,Y_test)
print("Test accuracy:",result[1])
```

    10000/10000 [==============================] - 10s 969us/step
    Test accuracy: 0.9926

