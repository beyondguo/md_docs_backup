# Bayesian Optimization: AutoML, SMBO, GP and TPE



> AutoML这几年十分火热，本身我是不感兴趣的，但是最近读到一篇自己领域的论文，发现别人使用的AutoML中的Bayesian Optimization的方法来调参发了顶会，再加上最近自己跑实验调参总感觉好慢，很耽误我的实验分析，因此也准备了解这个领域的一些基本概念和技术，看能不能帮助我来提高科研的效率。本文的定位是一个科普，重点在于搞清楚这里面的概念和关系，然后对贝叶斯优化有一个直觉上的认识。



 ## 理清基本概念的关系

在开始看，我们先澄清几个该领域常见的容易混淆的概念：

AutoML, Bayesian Optimization (BO), Sequential Model Based Optimisation (SMBO), Gaussian Process Regression (GPR), Tree Parzen Estimator (TPE).

这些概念是啥关系呢：

AutoML $\supset$ BO $\supset$ SMBO $\supset$ {GPR, TPE}

**AutoML**是最大的一个概念，涵盖了贝叶斯优化（Bayesian Optimization, BO）、神经结构搜索（ Neural Architecture Search, NAS）以及很多工程性的东西。AutoML的目标就是让没有机器学习（这里包括深度学习）经验的人，可以使用某一个平台来构造、训练、使用机器学习模型，这个平台负责进行数据管理、模型结构设计、**模型超参数调节**、模型的评估与使用等等。

本文，也是我们搞深度学习的同学最关心的，就是模型超参数调节（hyper-parameter tuning）了。下面的所有概念都是围绕这个展开的。

**贝叶斯优化（BO）**则是AutoML中进行超参数调节的一种先进的方法，并列于人工调参（manul）、网格搜索（Grid Search）、随机搜索（Rrandom Search）。

**SMBO**则**是贝叶斯优化的一种具体实现方法**，是一种迭代式的优化方法，每一次的迭代就是一次新的超参数组合实验，而每次的迭代都是基于前面的历史。其实**SMBO可以认为就是BO的标准实现，二者在很多语境下我感觉是可以等价的**（这里我其实也不太确定，因为看了很多资料仿佛都是混着用，如果有朋友更清楚麻烦告诉我）。

最后就是**Gaussian Process Regression（GPR）和Tree Parzen Estimator (TPE)**了，这俩玩意儿是并列的概念，**是SMBO方法中的两种建模策略**。

以上，我们就先理清了这些概念的关系，然后，我们就可以更轻松地学习贝叶斯优化了。

下面，我们主要讲解这几个内容：

1. 各种超参数调节方法的对比
2. 贝叶斯优化/SMBO方法的基本流程
3. 基于GPR的SMBO方法的原理
4. 基于TPE的SMBO方法的原理

## 1. 各种超参数调节方法的对比

超参数调节（hyper-parameter tuning），主要有下面四种方法：

* 人工调参（manul tuning）
* 网格搜索（grid search）
* 随机搜索（random search）
* 贝叶斯优化（Bayesian Optimization）




人工调参就不用说了，跑几组，看看效果，然后再调整继续跑。

### Grid Search & Random Search

这里简单对比一下grid search和random search：

![Grid Search vs. Random Search 图源ResearchGate](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220115201939.png)

上面是grid search和random search的示意图。简单地讲，random search在相同的超参数组合次数中，探索了更多的空间（这句话也有歧义，更准确得说应该是**每个维度的参数，都尝试了更多的可能**），因此从平均意义上看可以比grid search更早得找到较好的区域。


| Grid Search | Random Search |
| --- | ---|
|<img src="https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220115204314.gif" alt="grid search, photo from SigOpt"  />|<img src="https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220115204545.gif" alt="random search, photo from SigOpt"  />|

上面这两个图（来自SigOpt）更加清楚一些，比方我们要找一个曲面的最高点，那我们按部就班地地毯式搜索，（有时候）就比我们东一榔头西一棒子地随机找要更慢，**从理论上将，random search可以更快地将整个参数空间都覆盖到，而不是像grid search一样从局部一点点去搜索。**

但是，grid search和random search，都属于无先验式的搜索，有些地方称之为**Uninformed** search，即每一步的搜索，都不考虑已经探索的点的情况，这也是grid/random search的主要问题，都是“偷懒式”搜索，闭着眼睛乱找一通。

而**贝叶斯优化**，则是一种**informed** search，**会利用前面已经搜索过的参数的表现，来推测下一步怎么走会比较好，从而减少搜索空间，大大提升搜索效率。**某种意义上，贝叶斯优化跟人工调参比较像，因为我们调参师傅也会根据已有的结果以及自己的经验来判断下一步如何调参。

>我们不要一提到贝叶斯就头疼，就预感到一大推数学公式、统计学推导（虽然事实上确实是这样），我们应该换一种思路：**看到贝叶斯，就想到先验（prior）信息。**所以贝叶斯优化，**就是一种基于先验的优化，一种根据历史信息来决定后面的路怎么走的优化方法。**

所以贝叶斯优化的关键在于：用什么样的标准来判断下一步怎么走比较好。



## 2. 贝叶斯优化/SMBO的基本流程

其实本文的主要内容，来自于超参数优化的经典论文"Algorithms for Hyper-Parameter Optimization"，发表在2011年的NIPS上，作者之一是图灵奖得住Yoshua Bengio：

![Algorithms for Hyper-Parameter Optimization](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220116144045.png)

该文章介绍了SMBO的基本框架，以及GPR和TPE两种优化策略。

### **SMBO**

SMBO全名Sequential model-based optimization，序列化基于模型的优化。所谓序列化，是指通过迭代的方式，通过一次一次试验来进行优化。SMBO是贝叶斯优化的一种具体实现形式，所以下文中我们可能会直接把SMBO跟贝叶斯优化混着说。

SMBO的框架是这样的：

![SMBO框架伪代码](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220116162538.png)

其中各个符号的意思如下：

- $f$是我们要去优化的函数，$x$是超参数组合，比方我们要训练一个图片分类模型，那么每一个$x$的选择，都会对应该分类模型的一个损失，这个函数关系就是$f$.  一般，$f$的计算都是十分耗时的;
- $S$是surrogate的缩写，意为“代理”，我们使用$S$作为$f$的代理函数，一般通过最小化$S$来寻找下一步超参数怎么选，$S$的计算一般会容易得多。一般这一步的最小化，我们是通过最大化一个acquisition function来进行的;
- $\mathcal{H}$是history，是前面所有的{$x$, $f(x)$}的记录，我们要对$\mathcal{H}$进行建模，得到它们的概率分布模型$M$.

所以，总体步骤如下：

1. 根据已有的调参历史$\mathcal{H}=(x_{1:k},f(x_{1:k}))$，建立概率分布模型$M$;
2. 根据acquisition function来挑选下一步超参数$x_{k+1}$;
3. 将新的观测$(x_{k+1},f(x_{k+1}))$加入到$\mathcal{H}$中.
4. 重复1-3步骤，直到达到最大迭代数

所以，不同的贝叶斯优化方法，主要区别在：

- 用何种概率模型对历史进行建模
- acquisition function如何选

### **对历史观测进行建模**

![我画的示意图](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220117145433.png)

假设我们已经尝试了一些超参数$x$，也得到了一系列$f$的值（即上图中的$y$），这些点就是上图中的黑点，这些就是我们的历史信息。

我们要优化的目标$f$，肯定会经过这些历史观测点，但是其他的位置我们是未知的， 有无数种可能的$f$会经过这些点，上图中的每一条虚线，都是可能的$f$。所以**各种可能的$f$会形成一个函数的分布**。我们虽然无法准确地知道$f$的具体形式，但如果我们能够抓住其分布，那我们就可以了解很多该函数的性质，就有利于我们$f$的趋势做一定的判断，从而帮助我们选择超参数。

这就好比对于随机变量，虽然我们抓不住其确切的值，但如果知道其分布的话，那我们就可以得到其均值、方差，也就知道了该变量会在什么样的一个范围内波动，对该随机变量也就有了一定的掌握。

所以，贝叶斯优化中的一个关键步骤，就是对**要优化的目标函数进行建模，得到该函数的分布 **$p(y|x)$，**从而了解该函数可能会在什么范围内波动**。

另外，我们有了函数的分布，实际上也就有了y在给定x的时候的条件分布：

![y在给定x时会有一个概率分布](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220117145805.png)

当给定一个新的超参数$x_{new}$，$y_{new}$的取值可以根据$p(y|x_{new})$给出。

具体的建模方法，最经典的包括高斯过程回归（GPR）和Tree Parzen Estimator（TPE），它们的细节会在后面的部分讲解。

### **Acquisition function / Expected Improvement (EI)**

我们继续看这个例子，假设我们的objective function就是loss，那么我们肯定希望找超参数使得loss最小，那么我们该如何根据该分布，来选择下一个超参数呢？

看下图：

![Exploration vs. Exploitation](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220117150829.png)

已经观测到的点中，$x_2$是最佳的超参数，那我们下一个点往哪个方向找？我们有两种策略：

- Exploitation（剥削？挖掘？使劲利用？）：我们既然发现了$x_2$最好，那估计周围的点也不错，所以我们在图中的area 1里面搜索；
- Exploration（探索）：虽然$x_2$处目前来看比较好，但是我们还没有探索到的空间实际还有好多好多，比方说图中的area 2区域，这里面很可能藏着惊喜！

实际上，上面两种策略各有各的道理，所以我们一般选择一个折中的方案，把这两方面都能考虑到，一种最常用的方案就是**Expected Improvement (EI)**.

Expected Improvement (EI)的公式如下：
$$
EI_{y^*}(x) = \int _{-\infty}^{y^*}max(y^*-y,0)p_M(y|x)dy
$$
其中$y^*$是某个阈值，EI就是一个期望，该期望是$x$的函数。当给定$x$的时候，EI(x)就是$y$相对于阈值$y^*$平均提升了多少。（notice：这里我们都默认我们是要minimize目标函数，因此y的降低就是效果的提升；另外，$y^*$实际上是一个我们需要指定的值，又是一个超参）

也就是，我们首先有一个baseline——$y^*$，我们就是通过计算EI——相对于$y^*$的期望提升，来评价一个超参数$x$的好坏。所以，我们下一步要找的超参数，就是:
$$
x_{new} = argmax_{x} EI_{y^*}(x)
$$
EI的公式决定了其偏好于选择均值小的、方差大的区域，因此就提现了"exploration vs. exploitation trade-off".



## 3. 不同的概率分布建模策略

其实了解了上面的内容，我们基本上对贝叶斯优化就了解的差不多了，接下来就是一些更加细节的内容，即如何对历史观测进行概率分布建模。

### ① 基于GPR的贝叶斯优化
高斯过程回归，是基于高斯过程的贝叶斯推断方法。

高斯过程，就是一个高斯分布的随机过程。我们对x和y做的一个先验假设：每一个$x$对应的$y$，都是一个高斯分布。

那么当我们还没有任何观测点时，$y$实际上服从一个无限维的高斯分布（这里是借用知乎作者@石溪的说法。因为$x$有无限种取法，所以$y$有无限种可能，无数个$y$的多维高斯分布，因此是一个无限维的高斯分布），一般我们假设其均值为0，协方差矩阵则是由我们指定的核函数来给出。这样的分布，就是一个**先验分布**。借用sklearn的图来辅助说明这个先验分布，它展示了一个高斯过程：

![高斯过程的先验分布](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220117211254.png)

上图中的黑线就是均值曲线，灰色区域代表一倍标准差范围，而各种虚线则代表从这个先验分布中随机采样得到的函数。

然后我们再来看高斯过程回归，所谓回归，就是根据一些观测点（也可以称为训练数据），来进行一些推断。上面的高斯过程描述了$p(y)$的过程，高斯过程回归就是想基于我们得到的一些观测点$x$来得到条件分布$p(y|x)$。由于多维高斯分布的良好性质，条件分布也会是一个高斯分布，所以可以根据$x$的分布以及$p(y)$来直接推出$p(y|x)$，具体就不赘述了。

还是看sklearn的例子，在得到一些观测点之后，我们就可以推出**后验分布**：

![高斯过程的后验分布](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220117212948.png)

这样，每观测到一个新的点，就可以更新一次我们的总体分布，这个过程就叫高斯过程回归（GPR），下面的例子来自Distill上的一个很棒的博客（https://distill.pub/2019/visual-exploration-gaussian-processes/）：

![高斯过程回归示意，来源https://distill.pub/2019/visual-exploration-gaussian-processes/](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220117212447.gif)

相信从上面的描述，我们就明白了GP以及GPR的概念，具体公式以及推导这里不展开，感兴趣的读者可在文末推荐资料中查询。



好，上面是知道了高斯过程回归是咋回事，现在的问题是：已知了一些超参数的观测结果，如何选择下一个超参数在何处取？

回顾前面的章节，我们需要最大化Expected Improvement：
$$
maximize_x\ EI_{y^*}(x) \\
EI_{y^*}(x) = \int _{-\infty}^{y^*}max(y^*-y,0)p_M(y|x)dy
$$
在GPR中，我们选取当前观测结果中最好的的$y$来作为$y^*$ (根据论文的说法，在实际使用时，好像会选择一个比best y稍差一点的值)，那么$p(y|x)$咱们也有了，就可以直接求解上面这个优化问题了，求得的使EI最大的x即为下一步我们要去搜索的超参数。



### ② 基于TPE的贝叶斯优化

TPE，Tree Parzer Estimator，采用了一种不同的思路来进行概率分布的建模。

根据贝叶斯定理：
$$
p(y|x) = \frac{p(x|y)p(y)}{p(x)}
$$
我们可以把想求的p(y|x)分解成p(x|y)和p(y)。

关键是这一步，TPE对p(x|y)进行了这么一个划分：
$$
p(x|y) = \left\{
\begin{aligned}
l(x),\ \ \ y < y^*\\
g(x),\ \ \ y >y^*
\end{aligned}
\right.
$$
即TPE对于在阈值$y^*$两侧的观测点$x$，构造不一样的分布，可认为是一个好成绩的超参数概率分布，和一个坏成绩的超参数概率分布。

通过上面这样的划分，我们可以得到：
$$
asdf
$$



---

推荐资料：

- YouTube上关于BO的概念性理解（使用的GP方法）：https://www.youtube.com/watch?v=K_qNcLY3XUI
- 关于BO以及TPE的概念性的解读（初次看可能没法理解TPE）：https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f
- 一个写的很好的关于基于GP的BO的教程：https://arxiv.org/pdf/1807.02811.pdf
- NIPS上介绍GP、TPE、SMBO的经典论文：https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf
- 对比GP和TPE的博客：https://towardsdatascience.com/algorithms-for-hyperparameter-optimisation-in-python-edda4bdb167