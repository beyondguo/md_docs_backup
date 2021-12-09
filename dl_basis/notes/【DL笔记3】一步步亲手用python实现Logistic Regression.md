---
title: 【DL笔记3】一步步亲手用python实现Logistic Regression
published: 2021-6-24
sidebar: auto
---

# 【DL笔记3】一步步亲手用python实现Logistic Regression

>前面的[【DL笔记1】Logistic回归：最基础的神经网络](https://www.jianshu.com/p/4cf34bf158a1)和[【DL笔记2】神经网络编程原则&Logistic Regression的算法解析](https://www.jianshu.com/p/c67548909e99)讲解了Logistic regression的基本原理，并且我提到过这个玩意儿在我看来是学习神经网络和深度学习的基础，学到后面就发现，其实只要这个东西弄清楚了，后面的就很好明白。
另外，虽然说现在有很多很多的机器学习包和深度学习框架，像sklearn、TensorFlow、Keras等等，让我们实现一个神经网络十分容易，但是如果你不了解原理，即使给你一个框架，里面的大量的函数和方法你依然不知道如何下手，不知道什么时候该使用什么，而这些框架里面经常提到的“前向传播”、“反向传播”、“计算图”、各种梯度下降、mini-batch、各种initialization方法等等你也难以理解，更别提如何针对你的实际场景在对症下药了。
因此，我的深度学习系列笔记，主要是讲解神经网络的思路、算法、原理，然后前期主要使用python和numpy来实现，只有到我们把神经网络基本讲完，才会开始使用诸如TensorFlow这样的框架来实现。当然，这也是我正在听的吴恩达的深度学习系列课程的特点，不急不躁，耐心地用最朴素的方法来实践所有的原理，这样才能融会贯通，玩转各种框架。

这次的前言有点啰嗦了。。。主要是怕有的读者说“明明可以用机器学习包几行代码搞定，为啥偏要用纯python费劲去实现”。
好了，进入正题：
## 用python实现Logistic Regression

### 一、算法搭建步骤
#### （一）数据预处理
- 搞清楚数据的形状、维度
- 将数据（例如图片）转化成向量（image to vector）方便处理
- 将数据标准化（standardize），这样更好训练

#### （二）构造各种辅助函数
- 激活函数（此处我们使用sigmoid函数）--activation function
- 参数初始化函数（用来初始化W和b）--initialization
- 传播函数（这里是用来求损失cost并对W、b求导，即dW、db）--propagate
- 优化函数（迭代更新W和b，来最小化cost）--optimize
- 预测函数（根据学习到的W和b来进行预测）--predict

#### （三）综合上面的辅助函数，结合成一个模型
- 可以直接输入训练集、预测集、超参数，然后给出模型参数和准确率

上面这么多辅助函数可能看的让人有点懵逼，因此我花了半小时在PowerPoint里面画了这个图(ヾﾉ꒪ཫ꒪)，以便更清楚地说明它们之间的关系：

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624595337891-image.png)


构造辅助函数（helper function）是为了让我们的结构更清晰，更容易调试和修改。下面我们按照上面的步骤一个一个来。

### 二、开始编程吧
下面我们采用**“展示代码和注释+重点地方详解”**的方式来一步步实现：

#### （一）数据导入和预处理
```python
# 导入数据，“_orig”代表这里是原始数据，我们还要进一步处理才能使用：
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
#由数据集获取一些基本参数，如训练样本数m，图片大小：
m_train = train_set_x_orig.shape[0]  #训练集大小209
m_test = test_set_x_orig.shape[0]    #测试集大小209
num_px = train_set_x_orig.shape[1]  #图片宽度64，大小是64×64
#将图片数据向量化（扁平化）：
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
#对数据进行标准化：
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
```
上面的代码有几点要说明：
1.  数据导入是直接用吴恩达网课中的数据集，他提供了一个接口load_dataset()可以直接导入数据，如果**需要数据的话可以在文章下方留言获取**。这里主要是展示方法，完全可以用自己的数据集来操作。
数据集是一些图片，我们要训练一个识别猫的分类器。
**train_set_x_orig，也就是我们的原始数据**的**形状**是**(209, 64, 64, 3)**，**第一维代表m，即样本数量，第二维第三维分别是图片的长和宽，第四维代表图片的RGB三个通道**。
2. numpy包有重要的关于矩阵“形状”的方法：**.shape**和**.reshape()**
.shape可以获取一个矩阵的形状，于是我们可以通过[i]来知道每一维的大小；
.reshape()用来重构矩阵的形状，直接在里面填写维度即可，还有一些特殊用法，比如此处的用法：
当我们要把一个向量X(m,a,b,c)这个**四维向量**扁平化成X_flatten(m,a* b* c)的**二维向量**，可以写***X_flatten=X.reshape(X.shape[0],-1)***即可，其中“-1”代表把剩余维度压扁的模式。而代码中还有一个.T,代表转置，因为我们希望把训练样本压缩成（64* 64 *3，m）的形式。
3. **为什么需要标准化**？
在说明为什么要标准化前，我们不妨说说一般的标准化是怎么做的：先求出数据的均值和方差，然后对每一个样本数据，先**减去均值**，然后**除以方差**，也就是(x-μ)/σ<sup>2</sup>,说白了就是**转化成标准正态分布**！这样，每个特征都转化成了同样的分布，不管原来的范围是什么，现在都基本限定在同样的范围内了。
这样做的好处是什么呢？且看下面两个等高线图：

未标准化:
![未标准化](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624595362006-image.png)

标准化之后:
![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624595385826-image.png)

上面两个图展示了数据在未标准化和标准化之后的情形。**原数据的不同特征的范围可能会有很大差别**，比如一批数据中“年龄”的范围就比较小，可能20岁 ~ 60岁之间，但是另一个特征“年收入”可能波动范围就很大，也许0.5万 ~ 1000万，这种情况下回导致我们的**等高线图变得十分“扁平”**，在梯度下降的时候会很**容易走弯路**，因此**梯度下降会比较慢，精度也不高**。但是经过标准化（也称归一化）之后，**等高线就变规矩了，就很容易梯度下降了**。
另外，对于图片数据的话，进行标准化很简单，因为RGB三个通道的范围都是255，我们对图片的处理就是直接除以255即可。

至此，数据预处理就完成了，我们进入下一步：

#### （二）构建辅助函数们
**1.  激活函数/sigmoid函数：**
```python
def sigmoid(z):
    a = 1/(1+np.exp(-z))
    return a
```
就这么easy，sigmoid的公式就是1/(1+e<sup>-x</sup>)，这里用**np.exp()**就可以轻松构建。

**2. 参数初始化函数（给参数都初始化为0）：**
```python
def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    return w,b
```
W是一个列向量，传入维度dim，返回shape为（dim,1）的W，b就是一个数。
这里用到的方法是**np.zeros(shape)**.

**3.propagate函数：**
这里再次解释一下这个propagate，它包含了forward-propagate和backward-propagate，即正向传播和反向传播。正向传播求的是cost，反向传播是从cost的表达式倒推W和b的偏导数，当然我们会先求出Z的偏导数。这两个方向的传播也是神经网络的精髓。
具体倒数怎么求，这里就不推导了，就是很简单的求导嘛，公式请参见上一篇文章：[【DL笔记2】神经网络编程原则&Logistic Regression的算法解析](https://www.jianshu.com/p/c67548909e99)
那么我就直接上代码了：
```python
def propagate(w, b, X, Y):
    """
    传参:
    w -- 权重, shape： (num_px * num_px * 3, 1)
    b -- 偏置项, 一个标量
    X -- 数据集，shape： (num_px * num_px * 3, m),m为样本数
    Y -- 真实标签，shape： (1,m)

    返回值:
    cost， dw ，db，后两者放在一个字典grads里
    """
    #获取样本数m：
    m = X.shape[1]
    
    # 前向传播 ：
    A = sigmoid(np.dot(w.T,X)+b)    #调用前面写的sigmoid函数    
    cost = -(np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))/m                 
    
    # 反向传播：
    dZ = A-Y
    dw = (np.dot(X,dZ.T))/m
    db = (np.sum(dZ))/m
  
    #返回值：
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
```
这里需要额外说明的就是，**numpy中矩阵的点乘**，也就是内积运算，是用**np.dot(A,B)**，它要求前一个矩阵的列数等于后一个矩阵的行数。但矩阵也可以进行**元素相乘（element product）**，就是两个相同形状的矩阵对于元素相乘得到一个新的相同形状的矩阵，可以直接用**A * B**，或者用**np.multiply(A,B)**。
上面的代码中，既有点乘，也有元素相乘，我们在写的时候，先搞清楚形状，再确定用什么乘法。
上面还有各种numpy的数学函数，对矩阵求log就用**np.log()**，对矩阵元素求和就用**np.sum()**，贼方便。

**4.optimize函数：**
有了上面这些函数的加持，optimize函数就很好写了，就是在迭代中调用各个我们刚刚写的函数就是：
```python
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    #定义一个costs数组，存放每若干次迭代后的cost，从而可以画图看看cost的变化趋势：
    costs = []
    #进行迭代：
    for i in range(num_iterations):
        # 用propagate计算出每次迭代后的cost和梯度：
        grads, cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]
        
        # 用上面得到的梯度来更新参数：
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
        # 每100次迭代，保存一个cost看看：
        if i % 100 == 0:
            costs.append(cost)
        
        # 这个可以不在意，我们可以每100次把cost打印出来看看，从而随时掌握模型的进展：
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    #迭代完毕，将最终的各个参数放进字典，并返回：
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs
```
这个函数就没什么好解释的了。

**5.predict函数：**
预测就很简单了，我们已经学到了参数W和b，那么让我们的数据经过配备这些参数的模型就可得到预测值。注意，X->Z->激活得到A，此时还并不是预测值，由sigmoid函数我们知道，A的范围是0~1，但是我们的标签值是0和1，因此，我们可以设立规则：0.5~1的A对于预测值1,小于0.5的对应预测值0：
```python
def predict(w,b,X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))

    A = sigmoid(np.dot(w.T,X)+b)
    for  i in range(m):
        if A[0,i]>0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0

    return Y_prediction
```
恭喜，如果你有耐心看到这里了。。。那。。。我真的忍不住送你一朵fa了：

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624595430010-image.png)

毕竟我自己都不相信会有几个人真的去看这么枯燥的过程。但是我相信，每一份耐心和付出都有回报吧，学习这事儿，急不来。

至此，我们已经构建好了所有的辅助函数。接下来就是结合在一起，然后用我们的数据去训练、预测了！

#### （三）结合起来，搭建模型！
```python
def logistic_model(X_train,Y_train,X_test,Y_test,learning_rate=0.1,num_iterations=2000,print_cost=False):
    #获特征维度，初始化参数：
    dim = X_train.shape[0]
    W,b = initialize_with_zeros(dim)

    #梯度下降，迭代求出模型参数：
    params,grads,costs = optimize(W,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    W = params['w']
    b = params['b']

    #用学得的参数进行预测：
    prediction_train = predict(W,b,X_test)
    prediction_test = predict(W,b,X_train)

    #计算准确率，分别在训练集和测试集上：
    accuracy_train = 1 - np.mean(np.abs(prediction_train - Y_train))
    accuracy_test = 1 - np.mean(np.abs(prediction_test - Y_test))
    print("Accuracy on train set:",accuracy_train )
    print("Accuracy on test set:",accuracy_test )

   #为了便于分析和检查，我们把得到的所有参数、超参数都存进一个字典返回出来：
    d = {"costs": costs,
         "Y_prediction_test": prediction_test , 
         "Y_prediction_train" : prediction_train , 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations,
         "train_acy":train_acy,
         "test_acy":test_acy
        }
    return d
```
就是这么easy，只要我们一步步把前面的辅助函数搭建好，这里就可以很轻松很清晰地构造模型。
**唯一值得一提的是这个准确率怎么计算**的问题，我们的predict函数得到的是一个列向量（1，m），这个跟我们的标签Y是一样的形状。我们首先可以让**两者相减**：
**prediction_test  - Y_test**，
如果对应位置相同，则变成0，不同的话要么是1要么是-1，于是再**取绝对值**：
**np.abs**(prediction_test  - Y_test)，
就相当于得到了“哪些位置预测错了”的一个向量，于是我们再求一个**均值**：
**np.mean**(np.abs(prediction_test  - Y_test))，
就是**“错误率”**了，然后用**1来减**去它，就是**正确率**了！

### 大功告成！试试效果：
```python
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
```
运行模型就很简单了，把我们的数据集穿进去，设置我们想要的超参数，主要是学习率（learning rate）、迭代数（num_iterations），然后把print_cost设为True，这样可以在模型训练过程中打印cost的变化趋势。

运行，查看结果：
```
Cost after iteration 0: 0.693147
Cost after iteration 100: 0.584508
Cost after iteration 200: 0.466949
Cost after iteration 300: 0.376007
Cost after iteration 400: 0.331463
Cost after iteration 500: 0.303273
Cost after iteration 600: 0.279880
Cost after iteration 700: 0.260042
Cost after iteration 800: 0.242941
Cost after iteration 900: 0.228004
Cost after iteration 1000: 0.214820
Cost after iteration 1100: 0.203078
Cost after iteration 1200: 0.192544
Cost after iteration 1300: 0.183033
Cost after iteration 1400: 0.174399
Cost after iteration 1500: 0.166521
Cost after iteration 1600: 0.159305
Cost after iteration 1700: 0.152667
Cost after iteration 1800: 0.146542
Cost after iteration 1900: 0.140872
---------------------
train accuracy: 99.04306220095694 %
test accuracy: 70.0 %
```
可以看到，随着训练的进行，cost在不断地降低，这说明的参数在变得越来越好。
最终，在训练集上的准确率达到了99%以上，测试集准确率为70%。
哈哈，很明显，我们的模型**过拟合了**，测试集的准确率还有待提高。**但是这个不重要！重要的是我们亲手再没有用任何框架的情况下用python把Logistic regression给实现了一遍，每一个细节都明明白白！**٩(๑>◡<๑)۶ 
况且，这才仅仅是一个Logistic regression，相当于1层的只有一个神经元的神经网络，能对图片分类达到70%的准确率，我们已经很棒了!

---
**其实**，神经网络无非就是在Logistic regression的基础上，多了几个隐层，每层多了一些神经元，卷积神经网络无非就是再多了几个特殊的filter，多了一些有特定功能的层，但是核心都是跟Logistic Regression一样的：
>前向传播求损失，\
反向传播求倒数；\
不断迭代和更新，\
调参预测准确度。

哟嗬！才发现自己还有写诗的天赋。

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624595496062-image.png)

---
>本文就到此结束，终于结束了，出去吃串串了~





