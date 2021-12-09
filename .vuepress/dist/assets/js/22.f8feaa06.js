(window.webpackJsonp=window.webpackJsonp||[]).push([[22],{387:function(t,n,s){"use strict";s.r(n);var _=s(44),a=Object(_.a)({},(function(){var t=this,n=t.$createElement,s=t._self._c||n;return s("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[s("h1",{attrs:{id:"「杂谈」神经网络参数初始化的学问"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#「杂谈」神经网络参数初始化的学问"}},[t._v("#")]),t._v(" 「杂谈」神经网络参数初始化的学问")]),t._v(" "),s("blockquote",[s("p",[t._v("我们已经知道，神经网络的参数主要是权重（weights）：W， 和偏置项（bias）：b。\n训练神经网络的时候需先给定一个初试值，才能够训练，然后一点点地更新，但是"),s("strong",[t._v("不同的初始化方法，训练的效果可能会截然不同")]),t._v("。本文主要记录一下不同的初始化的方法，以及相应的效果。")])]),t._v(" "),s("p",[t._v("笔者正在学习的Andrew Ng的DeepLearning.ai提供了相应的模型框架和数据，我们这里要自己设置的就是不同的初值。")]),t._v(" "),s("p",[t._v("数据可视化之后是这样的：")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624598961457-image.png",alt:""}})]),t._v(" "),s("p",[t._v("我们需要做的就是把上面的红点和蓝点分类。")]),t._v(" "),s("h2",{attrs:{id:"一、直接把参数都初始化为0"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#一、直接把参数都初始化为0"}},[t._v("#")]),t._v(" 一、直接把参数都初始化为0")]),t._v(" "),s("p",[t._v("这是大家可以想到的最简单的方法，也确实很多其他的地方都采用0初值，那神经网络中这样做是否可行呢？\n在python中，可以用***np.zeros((维度))*** 来给一个向量/矩阵赋值0，\n于是，对于L层神经网络，可这样进行0-initialization：")]),t._v(" "),s("div",{staticClass:"language- line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[t._v("for l in range(1,L): #总共L层，l为当前层\n    W = np.zeros((num_of_dim[l],num_of_dim[l-1])) # W的维度是（当前层单元数，上一层单元数）\n    b = np.zeros((num_of_dim[l],1)) # b的维度是（当前层单元数，1）\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br")])]),s("p",[t._v("通过这样的初值，我们run一下模型，得到的cost-iteration曲线以及在训练集、测试集上面的准确率如下：")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624598999800-image.png",alt:""}})]),t._v(" "),s("p",[t._v("可以发现，"),s("strong",[t._v("压根就没训练")]),t._v("！得到的模型跟瞎猜没有区别。")]),t._v(" "),s("h4",{attrs:{id:"为什么呢"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#为什么呢"}},[t._v("#")]),t._v(" 为什么呢？")]),t._v(" "),s("p",[t._v("我们看看神经网络的结构图：")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624599015156-image.png",alt:""}})]),t._v(" "),s("p",[t._v("这是一个3层神经网络，可以看出，神经网络结构是十分"),s("strong",[t._v("对称")]),t._v("的，不管有几层。\n当我们把所有的参数都设成0的话，那么上面的每一条边上的权重就都是0，那么神经网络就还是对称的，对于同一层的每个神经元，它们就一模一样了。\n这样的后果是什么呢？我们知道，"),s("strong",[t._v("不管是哪个神经元，它的前向传播和反向传播的算法都是一样的，如果初始值也一样的话，不管训练多久，它们最终都一样，都无法打破对称（fail to break the symmetry）")]),t._v(",那每一层就相当于只有一个神经元，"),s("strong",[t._v("最终L层神经网络就相当于一个线性的网络")]),t._v("，如Logistic regression，线性分类器对我们上面的非线性数据集是“无力”的，所以最终训练的结果就瞎猜一样。")]),t._v(" "),s("p",[t._v("因此，我们决不能把所有参数初始化为0，同样也不能初始化为任何相同的值，因为我们必须“"),s("strong",[t._v("打破对称性")]),t._v("”！")]),t._v(" "),s("h2",{attrs:{id:"二、随机初始化"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#二、随机初始化"}},[t._v("#")]),t._v(" 二、随机初始化")]),t._v(" "),s("p",[t._v("好，不用0，咱们随机给一批值总可以吧。确实可以！咱们看看：\n【下面的演示会试试多种参数或超参数，为了方便大家看，我分4步：①②③④】")]),t._v(" "),s("p",[t._v("####①随机初始化\npython中，随机初始化可以用 "),s("em",[s("strong",[t._v("np.random.randn(维度)")])]),t._v(" 来随机赋值：\n于是前面的代码改成：")]),t._v(" "),s("div",{staticClass:"language- line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[t._v("for l in range(1,L): #总共L层，l为当前层\n    W = np.random.randn(num_of_dim[l],num_of_dim[l-1]) # W的维度是（当前层单元数，上一层单元数）\n    b = np.zeros((num_of_dim[l],1)) # b的维度是（当前层单元数，1）\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br")])]),s("p",[s("strong",[t._v("这里有三点需要说明一下：")])]),t._v(" "),s("ol",[s("li",[t._v("b不用随机初始化，因为w随机之后，已经打破对称，b就一个常数，无所谓了")]),t._v(" "),s("li",[t._v("random.rand()是在0~1之间随机，random.randn()是标准正态分布中随机，有正有负")]),t._v(" "),s("li",[t._v("np.zeros(())这里是两个括号，random.randn()是一个括号，奇怪的很，就记着吧")])]),t._v(" "),s("p",[t._v("那看看run出来的效果如何呢：")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624599026562-image.png",alt:""}})]),t._v(" "),s("p",[s("strong",[t._v("效果明显比0初始化要好多了")]),t._v("，cost最后降的也比较低，准确率也不错，92%。给分类效果可视化：")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624599036100-image.png",alt:""}})]),t._v(" "),s("p",[t._v("我们接着试试，如果把随机初始化的值放大一点会出现什么：")]),t._v(" "),s("h3",{attrs:{id:"_2放大版随机初始化"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#_2放大版随机初始化"}},[t._v("#")]),t._v(" ②放大版随机初始化")]),t._v(" "),s("div",{staticClass:"language- line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[t._v("for l in range(1,L): #总共L层，l为当前层\n    W = np.random.randn(num_of_dim[l],num_of_dim[l-1])*10 # W的维度是（当前层单元数，上一层单元数）\n    b = np.zeros((num_of_dim[l],1)) # b的维度是（当前层单元数，1）\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br")])]),s("p",[t._v("上面的代码中，我们给W最后多"),s("strong",[t._v("乘以10")]),t._v("，run的效果：\n"),s("strong",[t._v("【注意啊，乘以10不一定就是变大，因为我们的w的随机取值可正可负，所以乘以10之后，正数更大，负数更小】")])]),t._v(" "),s("p",[s("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624599045935-image.png",alt:""}})]),t._v(" "),s("p",[s("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624599053039-image.png",alt:""}})]),t._v(" "),s("p",[t._v("咦~~ 真o心 ~~")]),t._v(" "),s("p",[t._v("准确率明显降低了许多，到86%。")]),t._v(" "),s("p",[t._v("####为什么把随机初始化的值放大就不好了呢？")]),t._v(" "),s("p",[t._v("我们看看神经网络中常用的sigmoid函数：")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624599063592-image.png",alt:""}})]),t._v(" "),s("p",[t._v("这家伙，中间的斜率大，两边的斜率小还趋于零。所以当我们把随机的值乘以10了之后，我们的初值会往两边跑，那么我们的"),s("strong",[t._v("梯度下降就会显著变慢，可能迭代半天，才下降一点点。")])]),t._v(" "),s("p",[t._v("这就是问题的症结。")]),t._v(" "),s("p",[t._v("我们上面的实验，可以从图的横坐标看出，都是设定的一样的迭代次数（iteration number）：15000次，因此，"),s("strong",[t._v("在相同的迭代次数下，放大版的随机初始化的模型的学习就像一个“笨学生”，没别人学的多，因此效果就更差")]),t._v("。")]),t._v(" "),s("p",[t._v("为了验证我说的，我们可以试试吧迭代次数加大，看看我说的是不是对的：")]),t._v(" "),s("h3",{attrs:{id:"_3增大迭代次数"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#_3增大迭代次数"}},[t._v("#")]),t._v(" ③增大迭代次数")]),t._v(" "),s("p",[t._v("测试了好久。。。\n然后打脸了。。。")]),t._v(" "),s("p",[t._v("不过还是值得玩味~~")]),t._v(" "),s("p",[t._v("我把迭代次数显示设为60000，也就是增大了4,5倍，结果cost function后来下降十分十分缓慢，最后效果还不如之前的。然后我再把"),s("strong",[t._v("迭代次数增加到了160000")]),t._v("，相当于比一开始增大了10倍多，结果....")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624599072182-image.png",alt:""}})]),t._v(" "),s("p",[t._v("可以看到，cost基本从20000次迭代之后就稳定了，怎么都降不下去了，实际上是在降低，但是十分十分十分X10地缓慢。难道这就是传说中的梯度消失？？？\n所以结果并没有我想象地把迭代次数加大，就可以解决这个问题，实际上，可以看到，在训练集上准确度确实上升了，所以说明确实模型有所改进，只不过改进的太缓慢，相当于没有改进。")]),t._v(" "),s("p",[t._v("仔细分析了一下，由于W太大或者太小，导致激活函数对w的倒数趋于零，那么计算cost对w的导数也会趋于零，所以下降如此缓慢也是可以理解。")]),t._v(" "),s("p",[t._v("好，放大的效果如此差，我们缩小试试？")]),t._v(" "),s("h3",{attrs:{id:"_4缩小版随机初始化"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#_4缩小版随机初始化"}},[t._v("#")]),t._v(" ④缩小版随机初始化")]),t._v(" "),s("p",[t._v("还是回到迭代14000次，这次把"),s("strong",[t._v("w除以10")]),t._v("看看：")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624599085309-image.png",alt:""}})]),t._v(" "),s("p",[s("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624599093300-image.png",alt:""}})]),t._v(" "),s("p",[t._v("嘿~缩小结果甚至更差！连圈圈都没有了。")]),t._v(" "),s("p",[t._v("上面这个图，说明学习到的模型太简单了，因为我们把"),s("strong",[t._v("w都除以10，实际上就接近0了")]),t._v("，深度学习中我们认为"),s("strong",[t._v("参数越大，模型越复杂；参数越小，模型越简单")]),t._v("。所以除以10之后，参数太小了，模型就too simple了，效果当然不好。")]),t._v(" "),s("p",[s("strong",[t._v("最后再试一次吧，再多的话大家都烦了我也烦了。")])]),t._v(" "),s("p",[t._v("上面乘以10和除以10，效果都很差，那我们试一个中间的，比如："),s("strong",[t._v("除以3")]),t._v("（真的是随便试试）")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624599103632-image.png",alt:""}})]),t._v(" "),s("p",[t._v("可见，只要找到一个恰当的值来缩小，是可以提高准确率的。但是，这里除以三是我拍脑门出来的，不能每次都这么一个个地试吧，"),s("strong",[t._v("有没有一个稳健的，通用的")]),t._v("方法呢？")]),t._v(" "),s("p",[t._v("有！接着看：")]),t._v(" "),s("h2",{attrs:{id:""}},[s("a",{staticClass:"header-anchor",attrs:{href:"#"}},[t._v("#")])]),t._v(" "),s("h2",{attrs:{id:"三、he-initialization"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#三、he-initialization"}},[t._v("#")]),t._v(" 三、He Initialization")]),t._v(" "),s("p",[t._v("上面试了各种方法，放大缩小都不好，无法把握那个度。还好，总有大神为我们铺路，何凯明大佬提出了一种方法，我们称之为"),s("strong",[t._v("He Initialization")]),t._v("，它就是在我们随机初始化了之后，"),s("strong",[t._v("乘以")]),t._v("!\n"),s("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624599230018-image.png",alt:""}})]),t._v(" "),s("p",[t._v("这样就避免了参数的初始值过大或者过小，因此可以取得比较好的效果，代码也很简单，用***np.sqrt()***来求平方根：")]),t._v(" "),s("div",{staticClass:"language- line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[t._v("for l in range(1,L): #总共L层，l为当前层\n    W = np.random.randn(num_of_dim[l],num_of_dim[l-1])**np.sqrt(2/num_of_dim[l-1]) # W的维度是（当前层单元数，上一层单元数）\n    b = np.zeros((num_of_dim[l],1)) # b的维度是（当前层单元数，1）\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br")])]),s("p",[t._v("取得的效果如下：")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624599239506-image.png",alt:""}})]),t._v(" "),s("p",[s("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624599246127-image.png",alt:""}})]),t._v(" "),s("p",[t._v("啧啧啧，看这效果，看这优美的损失曲线，看着卓越的准确率... ...")]),t._v(" "),s("p",[s("strong",[t._v("以后就用你了，He Initialization ！")])]),t._v(" "),s("p",[t._v("其实吧，He Initialization是推荐针对使用"),s("strong",[t._v("ReLU")]),t._v("激活函数的神经网络使用的，不过对其他的激活函数，效果也不错。")]),t._v(" "),s("p",[t._v("还有其他的类似的一些好的初始化方法，例如：")]),t._v(" "),s("p",[t._v("推荐给sigmoid的"),s("strong",[t._v("Xavier Initialization")]),t._v("：随机化之后"),s("strong",[t._v("乘以")])]),t._v(" "),s("p",[s("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624599262151-image.png",alt:""}})]),t._v(" "),s("h2",{attrs:{id:"总结一下"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#总结一下"}},[t._v("#")]),t._v(" 总结一下：")]),t._v(" "),s("ul",[s("li",[t._v("神经网络不可用0来初始化参数！")]),t._v(" "),s("li",[t._v("随机赋值是为了打破对称性，使得不同的神经元可以有不同的功能")]),t._v(" "),s("li",[t._v("推荐在初始化的时候使用He Initialization")])])])}),[],!1,null,null,null);n.default=a.exports}}]);