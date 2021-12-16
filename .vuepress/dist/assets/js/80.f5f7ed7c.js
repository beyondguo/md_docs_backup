(window.webpackJsonp=window.webpackJsonp||[]).push([[80],{450:function(t,s,a){"use strict";a.r(s);var i=a(44),e=Object(i.a)({},(function(){var t=this,s=t.$createElement,a=t._self._c||s;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"在特征空间增强数据集"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#在特征空间增强数据集"}},[t._v("#")]),t._v(" 在特征空间增强数据集")]),t._v(" "),a("ul",[a("li",[t._v("论文标题：DATASET AUGMENTATION IN FEATURE SPACE")]),t._v(" "),a("li",[t._v("发表会议：ICLR workshop 2017")]),t._v(" "),a("li",[t._v("组织机构：University of Guelph")])]),t._v(" "),a("blockquote",[a("p",[a("strong",[t._v("一句话评价")]),t._v("：")]),t._v(" "),a("p",[t._v("一篇简单的workshop文章，但是有一些启发性的实验结论。")])]),t._v(" "),a("h2",{attrs:{id:"简介"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#简介"}},[t._v("#")]),t._v(" 简介")]),t._v(" "),a("p",[t._v("最常用的数据增强方法，无论是CV还是NLP中，都是直接对原始数据进行各种处理。比如对图像的剪切、旋转、变色等，对文本数据的单词替换、删除等等。**对于原始数据进行处理，往往是高度领域/任务相关的，即我们需要针对数据的形式、任务的形式，来设计增强的方法，这样就不具有通用性。比如对于图像的增强方法，就没法用在文本上。**因此，本文提出了一种“领域无关的”数据增强方法——特征空间的增强。具体的话就是对可学习的样本特征进行 1) adding noise, 2) interpolating, 3) extrapolating 来得到新的样本特征。")]),t._v(" "),a("p",[t._v("文本提到的一个想法很有意思：")]),t._v(" "),a("blockquote",[a("p",[t._v("When traversing along the manifold it is more likely to encounter realistic samples in feature space than compared to input space.\n在样本所在的流形上移动，在特征空间上会比在原始输入空间上移动，更容易遇到真实的样本点。")])]),t._v(" "),a("p",[t._v("我们知道，对原始的数据进行数据增强，很多时候就根本不是真实可能存在的样本了，比如我们在NLP中常用的对文本进行单词随机删除，这样得到的样本，虽然也能够提高对模型学习的鲁棒性，但这种样本实际上很难在真实样本空间存在。本文提到的这句话则提示我们，如果我们把各种操作，放在高维的特征空间进行，则更可能碰到真实的样本点。文章指出，这种思想，Bengio等人也提过：“higher level representations expand the relative volume of plausible data points within the feature space, conversely shrinking the space allocated for unlikely data points.” 这里我们暂且不讨论这个说法背后的原理，先不妨承认其事实，这样的话就启示我们，在特征空间进行数据增强，我们有更大的探索空间。")]),t._v(" "),a("h2",{attrs:{id:"具体方法"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#具体方法"}},[t._v("#")]),t._v(" 具体方法")]),t._v(" "),a("p",[t._v("其实这个文章具体方法很简单，它使用的是encoder-decoder的框架，在（预训练好的）encoder之后的样本特征上进行增强，然后进行下游任务。所以是先有了一个表示模型来得到样本的特征，再进行增强，而不是边训练边增强。框架结构如下：")]),t._v(" "),a("p",[a("img",{attrs:{src:"https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20211128152721.png",alt:"image-20211128152702006"}})]),t._v(" "),a("p",[t._v("下面我们来看作者具体是怎么增强的：")]),t._v(" "),a("h3",{attrs:{id:"_1-adding-noise-加噪音"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_1-adding-noise-加噪音"}},[t._v("#")]),t._v(" 1. Adding Noise（加噪音）")]),t._v(" "),a("p",[t._v("直接在encoder得到的特征上添加高斯噪声：")]),t._v(" "),a("p",[a("span",{staticClass:"katex-display"},[a("span",{staticClass:"katex"},[a("span",{staticClass:"katex-mathml"},[a("math",{attrs:{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"}},[a("semantics",[a("mrow",[a("msubsup",[a("mi",[t._v("c")]),a("mi",[t._v("i")]),a("msup",[a("mrow"),a("mo",{attrs:{mathvariant:"normal",lspace:"0em",rspace:"0em"}},[t._v("′")])],1)],1),a("mo",[t._v("=")]),a("msub",[a("mi",[t._v("c")]),a("mi",[t._v("i")])],1),a("mo",[t._v("+")]),a("mi",[t._v("γ")]),a("mi",[t._v("X")]),a("mo",{attrs:{separator:"true"}},[t._v(",")]),a("mi",[t._v("X")]),a("mo",[t._v("∼")]),a("mi",{attrs:{mathvariant:"script"}},[t._v("N")]),a("mo",{attrs:{stretchy:"false"}},[t._v("(")]),a("mn",[t._v("0")]),a("mo",{attrs:{separator:"true"}},[t._v(",")]),a("msubsup",[a("mi",[t._v("σ")]),a("mi",[t._v("i")]),a("mn",[t._v("2")])],1),a("mo",{attrs:{stretchy:"false"}},[t._v(")")])],1),a("annotation",{attrs:{encoding:"application/x-tex"}},[t._v("c^{'}_i = c_i + \\gamma X, X \\sim \\mathcal{N}(0,\\sigma ^2_i)\n")])],1)],1)],1),a("span",{staticClass:"katex-html",attrs:{"aria-hidden":"true"}},[a("span",{staticClass:"base"},[a("span",{staticClass:"strut",staticStyle:{height:"1.23948em","vertical-align":"-0.247em"}}),a("span",{staticClass:"mord"},[a("span",{staticClass:"mord mathnormal"},[t._v("c")]),a("span",{staticClass:"msupsub"},[a("span",{staticClass:"vlist-t vlist-t2"},[a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.9924799999999999em"}},[a("span",{staticStyle:{top:"-2.4530000000000003em","margin-left":"0em","margin-right":"0.05em"}},[a("span",{staticClass:"pstrut",staticStyle:{height:"2.7em"}}),a("span",{staticClass:"sizing reset-size6 size3 mtight"},[a("span",{staticClass:"mord mathnormal mtight"},[t._v("i")])])]),a("span",{staticStyle:{top:"-3.113em","margin-right":"0.05em"}},[a("span",{staticClass:"pstrut",staticStyle:{height:"2.7em"}}),a("span",{staticClass:"sizing reset-size6 size3 mtight"},[a("span",{staticClass:"mord mtight"},[a("span",{staticClass:"mord mtight"},[a("span"),a("span",{staticClass:"msupsub"},[a("span",{staticClass:"vlist-t"},[a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.8278285714285715em"}},[a("span",{staticStyle:{top:"-2.931em","margin-right":"0.07142857142857144em"}},[a("span",{staticClass:"pstrut",staticStyle:{height:"2.5em"}}),a("span",{staticClass:"sizing reset-size3 size1 mtight"},[a("span",{staticClass:"mord mtight"},[a("span",{staticClass:"mord mtight"},[t._v("′")])])])])])])])])])])])])]),a("span",{staticClass:"vlist-s"},[t._v("​")])]),a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.247em"}},[a("span")])])])])]),a("span",{staticClass:"mspace",staticStyle:{"margin-right":"0.2777777777777778em"}}),a("span",{staticClass:"mrel"},[t._v("=")]),a("span",{staticClass:"mspace",staticStyle:{"margin-right":"0.2777777777777778em"}})]),a("span",{staticClass:"base"},[a("span",{staticClass:"strut",staticStyle:{height:"0.73333em","vertical-align":"-0.15em"}}),a("span",{staticClass:"mord"},[a("span",{staticClass:"mord mathnormal"},[t._v("c")]),a("span",{staticClass:"msupsub"},[a("span",{staticClass:"vlist-t vlist-t2"},[a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.31166399999999994em"}},[a("span",{staticStyle:{top:"-2.5500000000000003em","margin-left":"0em","margin-right":"0.05em"}},[a("span",{staticClass:"pstrut",staticStyle:{height:"2.7em"}}),a("span",{staticClass:"sizing reset-size6 size3 mtight"},[a("span",{staticClass:"mord mathnormal mtight"},[t._v("i")])])])]),a("span",{staticClass:"vlist-s"},[t._v("​")])]),a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.15em"}},[a("span")])])])])]),a("span",{staticClass:"mspace",staticStyle:{"margin-right":"0.2222222222222222em"}}),a("span",{staticClass:"mbin"},[t._v("+")]),a("span",{staticClass:"mspace",staticStyle:{"margin-right":"0.2222222222222222em"}})]),a("span",{staticClass:"base"},[a("span",{staticClass:"strut",staticStyle:{height:"0.8777699999999999em","vertical-align":"-0.19444em"}}),a("span",{staticClass:"mord mathnormal",staticStyle:{"margin-right":"0.05556em"}},[t._v("γ")]),a("span",{staticClass:"mord mathnormal",staticStyle:{"margin-right":"0.07847em"}},[t._v("X")]),a("span",{staticClass:"mpunct"},[t._v(",")]),a("span",{staticClass:"mspace",staticStyle:{"margin-right":"0.16666666666666666em"}}),a("span",{staticClass:"mord mathnormal",staticStyle:{"margin-right":"0.07847em"}},[t._v("X")]),a("span",{staticClass:"mspace",staticStyle:{"margin-right":"0.2777777777777778em"}}),a("span",{staticClass:"mrel"},[t._v("∼")]),a("span",{staticClass:"mspace",staticStyle:{"margin-right":"0.2777777777777778em"}})]),a("span",{staticClass:"base"},[a("span",{staticClass:"strut",staticStyle:{height:"1.1141079999999999em","vertical-align":"-0.25em"}}),a("span",{staticClass:"mord mathcal",staticStyle:{"margin-right":"0.14736em"}},[t._v("N")]),a("span",{staticClass:"mopen"},[t._v("(")]),a("span",{staticClass:"mord"},[t._v("0")]),a("span",{staticClass:"mpunct"},[t._v(",")]),a("span",{staticClass:"mspace",staticStyle:{"margin-right":"0.16666666666666666em"}}),a("span",{staticClass:"mord"},[a("span",{staticClass:"mord mathnormal",staticStyle:{"margin-right":"0.03588em"}},[t._v("σ")]),a("span",{staticClass:"msupsub"},[a("span",{staticClass:"vlist-t vlist-t2"},[a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.8641079999999999em"}},[a("span",{staticStyle:{top:"-2.4530000000000003em","margin-left":"-0.03588em","margin-right":"0.05em"}},[a("span",{staticClass:"pstrut",staticStyle:{height:"2.7em"}}),a("span",{staticClass:"sizing reset-size6 size3 mtight"},[a("span",{staticClass:"mord mathnormal mtight"},[t._v("i")])])]),a("span",{staticStyle:{top:"-3.113em","margin-right":"0.05em"}},[a("span",{staticClass:"pstrut",staticStyle:{height:"2.7em"}}),a("span",{staticClass:"sizing reset-size6 size3 mtight"},[a("span",{staticClass:"mord mtight"},[t._v("2")])])])]),a("span",{staticClass:"vlist-s"},[t._v("​")])]),a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.247em"}},[a("span")])])])])]),a("span",{staticClass:"mclose"},[t._v(")")])])])])])]),t._v(" "),a("h3",{attrs:{id:"_2-interpolation-内插值"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_2-interpolation-内插值"}},[t._v("#")]),t._v(" 2. Interpolation（内插值）")]),t._v(" "),a("p",[t._v("在"),a("strong",[t._v("同类别")]),t._v("点中，寻找K个最近邻，然后任意两个邻居间，进行内插值：")]),t._v(" "),a("p",[a("span",{staticClass:"katex-display"},[a("span",{staticClass:"katex"},[a("span",{staticClass:"katex-mathml"},[a("math",{attrs:{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"}},[a("semantics",[a("mrow",[a("msup",[a("mi",[t._v("c")]),a("msup",[a("mrow"),a("mo",{attrs:{mathvariant:"normal",lspace:"0em",rspace:"0em"}},[t._v("′")])],1)],1),a("mo",[t._v("=")]),a("mo",{attrs:{stretchy:"false"}},[t._v("(")]),a("msub",[a("mi",[t._v("c")]),a("mi",[t._v("k")])],1),a("mo",[t._v("−")]),a("msub",[a("mi",[t._v("c")]),a("mi",[t._v("j")])],1),a("mo",{attrs:{stretchy:"false"}},[t._v(")")]),a("mi",[t._v("λ")]),a("mo",[t._v("+")]),a("msub",[a("mi",[t._v("c")]),a("mi",[t._v("j")])],1)],1),a("annotation",{attrs:{encoding:"application/x-tex"}},[t._v("c^{'} = (c_k - c_j)\\lambda + c_j\n")])],1)],1)],1),a("span",{staticClass:"katex-html",attrs:{"aria-hidden":"true"}},[a("span",{staticClass:"base"},[a("span",{staticClass:"strut",staticStyle:{height:"0.99248em","vertical-align":"0em"}}),a("span",{staticClass:"mord"},[a("span",{staticClass:"mord mathnormal"},[t._v("c")]),a("span",{staticClass:"msupsub"},[a("span",{staticClass:"vlist-t"},[a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.99248em"}},[a("span",{staticStyle:{top:"-2.99248em","margin-right":"0.05em"}},[a("span",{staticClass:"pstrut",staticStyle:{height:"2.57948em"}}),a("span",{staticClass:"sizing reset-size6 size3 mtight"},[a("span",{staticClass:"mord mtight"},[a("span",{staticClass:"mord mtight"},[a("span"),a("span",{staticClass:"msupsub"},[a("span",{staticClass:"vlist-t"},[a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.8278285714285715em"}},[a("span",{staticStyle:{top:"-2.931em","margin-right":"0.07142857142857144em"}},[a("span",{staticClass:"pstrut",staticStyle:{height:"2.5em"}}),a("span",{staticClass:"sizing reset-size3 size1 mtight"},[a("span",{staticClass:"mord mtight"},[a("span",{staticClass:"mord mtight"},[t._v("′")])])])])])])])])])])])])])])])])]),a("span",{staticClass:"mspace",staticStyle:{"margin-right":"0.2777777777777778em"}}),a("span",{staticClass:"mrel"},[t._v("=")]),a("span",{staticClass:"mspace",staticStyle:{"margin-right":"0.2777777777777778em"}})]),a("span",{staticClass:"base"},[a("span",{staticClass:"strut",staticStyle:{height:"1em","vertical-align":"-0.25em"}}),a("span",{staticClass:"mopen"},[t._v("(")]),a("span",{staticClass:"mord"},[a("span",{staticClass:"mord mathnormal"},[t._v("c")]),a("span",{staticClass:"msupsub"},[a("span",{staticClass:"vlist-t vlist-t2"},[a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.33610799999999996em"}},[a("span",{staticStyle:{top:"-2.5500000000000003em","margin-left":"0em","margin-right":"0.05em"}},[a("span",{staticClass:"pstrut",staticStyle:{height:"2.7em"}}),a("span",{staticClass:"sizing reset-size6 size3 mtight"},[a("span",{staticClass:"mord mathnormal mtight",staticStyle:{"margin-right":"0.03148em"}},[t._v("k")])])])]),a("span",{staticClass:"vlist-s"},[t._v("​")])]),a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.15em"}},[a("span")])])])])]),a("span",{staticClass:"mspace",staticStyle:{"margin-right":"0.2222222222222222em"}}),a("span",{staticClass:"mbin"},[t._v("−")]),a("span",{staticClass:"mspace",staticStyle:{"margin-right":"0.2222222222222222em"}})]),a("span",{staticClass:"base"},[a("span",{staticClass:"strut",staticStyle:{height:"1.036108em","vertical-align":"-0.286108em"}}),a("span",{staticClass:"mord"},[a("span",{staticClass:"mord mathnormal"},[t._v("c")]),a("span",{staticClass:"msupsub"},[a("span",{staticClass:"vlist-t vlist-t2"},[a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.311664em"}},[a("span",{staticStyle:{top:"-2.5500000000000003em","margin-left":"0em","margin-right":"0.05em"}},[a("span",{staticClass:"pstrut",staticStyle:{height:"2.7em"}}),a("span",{staticClass:"sizing reset-size6 size3 mtight"},[a("span",{staticClass:"mord mathnormal mtight",staticStyle:{"margin-right":"0.05724em"}},[t._v("j")])])])]),a("span",{staticClass:"vlist-s"},[t._v("​")])]),a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.286108em"}},[a("span")])])])])]),a("span",{staticClass:"mclose"},[t._v(")")]),a("span",{staticClass:"mord mathnormal"},[t._v("λ")]),a("span",{staticClass:"mspace",staticStyle:{"margin-right":"0.2222222222222222em"}}),a("span",{staticClass:"mbin"},[t._v("+")]),a("span",{staticClass:"mspace",staticStyle:{"margin-right":"0.2222222222222222em"}})]),a("span",{staticClass:"base"},[a("span",{staticClass:"strut",staticStyle:{height:"0.716668em","vertical-align":"-0.286108em"}}),a("span",{staticClass:"mord"},[a("span",{staticClass:"mord mathnormal"},[t._v("c")]),a("span",{staticClass:"msupsub"},[a("span",{staticClass:"vlist-t vlist-t2"},[a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.311664em"}},[a("span",{staticStyle:{top:"-2.5500000000000003em","margin-left":"0em","margin-right":"0.05em"}},[a("span",{staticClass:"pstrut",staticStyle:{height:"2.7em"}}),a("span",{staticClass:"sizing reset-size6 size3 mtight"},[a("span",{staticClass:"mord mathnormal mtight",staticStyle:{"margin-right":"0.05724em"}},[t._v("j")])])])]),a("span",{staticClass:"vlist-s"},[t._v("​")])]),a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.286108em"}},[a("span")])])])])])])])])])]),t._v(" "),a("h3",{attrs:{id:"_3-extrapolating-外插值"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_3-extrapolating-外插值"}},[t._v("#")]),t._v(" 3. Extrapolating（外插值）")]),t._v(" "),a("p",[t._v("跟内插的唯一区别在于插值的位置：")]),t._v(" "),a("p",[a("span",{staticClass:"katex-display"},[a("span",{staticClass:"katex"},[a("span",{staticClass:"katex-mathml"},[a("math",{attrs:{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"}},[a("semantics",[a("mrow",[a("msup",[a("mi",[t._v("c")]),a("msup",[a("mrow"),a("mo",{attrs:{mathvariant:"normal",lspace:"0em",rspace:"0em"}},[t._v("′")])],1)],1),a("mo",[t._v("=")]),a("mo",{attrs:{stretchy:"false"}},[t._v("(")]),a("msub",[a("mi",[t._v("c")]),a("mi",[t._v("j")])],1),a("mo",[t._v("−")]),a("msub",[a("mi",[t._v("c")]),a("mi",[t._v("k")])],1),a("mo",{attrs:{stretchy:"false"}},[t._v(")")]),a("mi",[t._v("λ")]),a("mo",[t._v("+")]),a("msub",[a("mi",[t._v("c")]),a("mi",[t._v("j")])],1)],1),a("annotation",{attrs:{encoding:"application/x-tex"}},[t._v("c^{'} = (c_j - c_k)\\lambda + c_j\n")])],1)],1)],1),a("span",{staticClass:"katex-html",attrs:{"aria-hidden":"true"}},[a("span",{staticClass:"base"},[a("span",{staticClass:"strut",staticStyle:{height:"0.99248em","vertical-align":"0em"}}),a("span",{staticClass:"mord"},[a("span",{staticClass:"mord mathnormal"},[t._v("c")]),a("span",{staticClass:"msupsub"},[a("span",{staticClass:"vlist-t"},[a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.99248em"}},[a("span",{staticStyle:{top:"-2.99248em","margin-right":"0.05em"}},[a("span",{staticClass:"pstrut",staticStyle:{height:"2.57948em"}}),a("span",{staticClass:"sizing reset-size6 size3 mtight"},[a("span",{staticClass:"mord mtight"},[a("span",{staticClass:"mord mtight"},[a("span"),a("span",{staticClass:"msupsub"},[a("span",{staticClass:"vlist-t"},[a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.8278285714285715em"}},[a("span",{staticStyle:{top:"-2.931em","margin-right":"0.07142857142857144em"}},[a("span",{staticClass:"pstrut",staticStyle:{height:"2.5em"}}),a("span",{staticClass:"sizing reset-size3 size1 mtight"},[a("span",{staticClass:"mord mtight"},[a("span",{staticClass:"mord mtight"},[t._v("′")])])])])])])])])])])])])])])])])]),a("span",{staticClass:"mspace",staticStyle:{"margin-right":"0.2777777777777778em"}}),a("span",{staticClass:"mrel"},[t._v("=")]),a("span",{staticClass:"mspace",staticStyle:{"margin-right":"0.2777777777777778em"}})]),a("span",{staticClass:"base"},[a("span",{staticClass:"strut",staticStyle:{height:"1.036108em","vertical-align":"-0.286108em"}}),a("span",{staticClass:"mopen"},[t._v("(")]),a("span",{staticClass:"mord"},[a("span",{staticClass:"mord mathnormal"},[t._v("c")]),a("span",{staticClass:"msupsub"},[a("span",{staticClass:"vlist-t vlist-t2"},[a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.311664em"}},[a("span",{staticStyle:{top:"-2.5500000000000003em","margin-left":"0em","margin-right":"0.05em"}},[a("span",{staticClass:"pstrut",staticStyle:{height:"2.7em"}}),a("span",{staticClass:"sizing reset-size6 size3 mtight"},[a("span",{staticClass:"mord mathnormal mtight",staticStyle:{"margin-right":"0.05724em"}},[t._v("j")])])])]),a("span",{staticClass:"vlist-s"},[t._v("​")])]),a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.286108em"}},[a("span")])])])])]),a("span",{staticClass:"mspace",staticStyle:{"margin-right":"0.2222222222222222em"}}),a("span",{staticClass:"mbin"},[t._v("−")]),a("span",{staticClass:"mspace",staticStyle:{"margin-right":"0.2222222222222222em"}})]),a("span",{staticClass:"base"},[a("span",{staticClass:"strut",staticStyle:{height:"1em","vertical-align":"-0.25em"}}),a("span",{staticClass:"mord"},[a("span",{staticClass:"mord mathnormal"},[t._v("c")]),a("span",{staticClass:"msupsub"},[a("span",{staticClass:"vlist-t vlist-t2"},[a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.33610799999999996em"}},[a("span",{staticStyle:{top:"-2.5500000000000003em","margin-left":"0em","margin-right":"0.05em"}},[a("span",{staticClass:"pstrut",staticStyle:{height:"2.7em"}}),a("span",{staticClass:"sizing reset-size6 size3 mtight"},[a("span",{staticClass:"mord mathnormal mtight",staticStyle:{"margin-right":"0.03148em"}},[t._v("k")])])])]),a("span",{staticClass:"vlist-s"},[t._v("​")])]),a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.15em"}},[a("span")])])])])]),a("span",{staticClass:"mclose"},[t._v(")")]),a("span",{staticClass:"mord mathnormal"},[t._v("λ")]),a("span",{staticClass:"mspace",staticStyle:{"margin-right":"0.2222222222222222em"}}),a("span",{staticClass:"mbin"},[t._v("+")]),a("span",{staticClass:"mspace",staticStyle:{"margin-right":"0.2222222222222222em"}})]),a("span",{staticClass:"base"},[a("span",{staticClass:"strut",staticStyle:{height:"0.716668em","vertical-align":"-0.286108em"}}),a("span",{staticClass:"mord"},[a("span",{staticClass:"mord mathnormal"},[t._v("c")]),a("span",{staticClass:"msupsub"},[a("span",{staticClass:"vlist-t vlist-t2"},[a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.311664em"}},[a("span",{staticStyle:{top:"-2.5500000000000003em","margin-left":"0em","margin-right":"0.05em"}},[a("span",{staticClass:"pstrut",staticStyle:{height:"2.7em"}}),a("span",{staticClass:"sizing reset-size6 size3 mtight"},[a("span",{staticClass:"mord mathnormal mtight",staticStyle:{"margin-right":"0.05724em"}},[t._v("j")])])])]),a("span",{staticClass:"vlist-s"},[t._v("​")])]),a("span",{staticClass:"vlist-r"},[a("span",{staticClass:"vlist",staticStyle:{height:"0.286108em"}},[a("span")])])])])])])])])])]),t._v(" "),a("p",[t._v("下图表示了内插跟外插的区别：")]),t._v(" "),a("p",[a("img",{attrs:{src:"https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20211128154757.png",alt:"image-20211128154757628"}})]),t._v(" "),a("p",[t._v("在文本中，内插和外插都选择"),a("span",{staticClass:"katex"},[a("span",{staticClass:"katex-mathml"},[a("math",{attrs:{xmlns:"http://www.w3.org/1998/Math/MathML"}},[a("semantics",[a("mrow",[a("mi",[t._v("λ")]),a("mo",[t._v("=")]),a("mn",[t._v("0.5")])],1),a("annotation",{attrs:{encoding:"application/x-tex"}},[t._v("\\lambda = 0.5")])],1)],1)],1),a("span",{staticClass:"katex-html",attrs:{"aria-hidden":"true"}},[a("span",{staticClass:"base"},[a("span",{staticClass:"strut",staticStyle:{height:"0.69444em","vertical-align":"0em"}}),a("span",{staticClass:"mord mathnormal"},[t._v("λ")]),a("span",{staticClass:"mspace",staticStyle:{"margin-right":"0.2777777777777778em"}}),a("span",{staticClass:"mrel"},[t._v("=")]),a("span",{staticClass:"mspace",staticStyle:{"margin-right":"0.2777777777777778em"}})]),a("span",{staticClass:"base"},[a("span",{staticClass:"strut",staticStyle:{height:"0.64444em","vertical-align":"0em"}}),a("span",{staticClass:"mord"},[t._v("0.5")])])])]),t._v(".")]),t._v(" "),a("p",[t._v("论文作者为了更加形象地展示这三种增强方式，使用正弦曲线（上的点）作为样本，来进行上述操作，得到新样本：")]),t._v(" "),a("p",[a("img",{attrs:{src:"https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20211128160256.png",alt:"image-20211128160256809"}})]),t._v(" "),a("p",[t._v("作者还借用一个手写字母识别的数据集进行了可视化，进一步揭示interpolation和extrapolation的区别：")]),t._v(" "),a("p",[a("img",{attrs:{src:"https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20211128160514.png",alt:"image-20211128160514332"}})]),t._v(" "),a("p",[t._v("作者没有具体说可视化的方法，猜测是通过autoencoder来生成的。可以看到，extrapolation得到的样本，更加多样化，而interpolation则跟原来的样本对更加接近。")]),t._v(" "),a("h2",{attrs:{id:"实验"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#实验"}},[t._v("#")]),t._v(" 实验")]),t._v(" "),a("p",[t._v("下面我们来看看使用这三种方式的增强效果。本文的实验部分十分混乱，看得人头大，所以我只挑一些稍微清楚一些的实验来讲解。")]),t._v(" "),a("h3",{attrs:{id:"实验1-一个阿拉伯数字语音识别任务"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#实验1-一个阿拉伯数字语音识别任务"}},[t._v("#")]),t._v(" 实验1：一个阿拉伯数字语音识别任务")]),t._v(" "),a("p",[a("img",{attrs:{src:"https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20211128161250.png",alt:"实验1"}})]),t._v(" "),a("h3",{attrs:{id:"实验2-另一个序列数据集"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#实验2-另一个序列数据集"}},[t._v("#")]),t._v(" 实验2：另一个序列数据集")]),t._v(" "),a("p",[a("img",{attrs:{src:"https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20211128161607.png",alt:"image-20211128161607738"}})]),t._v(" "),a("p",[t._v("注：interpolation和extrapolation都是在同类别间进行的。")]),t._v(" "),a("p",[t._v("实验结果：")]),t._v(" "),a("ul",[a("li",[t._v("Adding Noise，效果一般般。")]),t._v(" "),a("li",[t._v("Interpolation，降低性能！")]),t._v(" "),a("li",[t._v("Extrapolation，效果最好！")])]),t._v(" "),a("h3",{attrs:{id:"实验3-跟再input-space进行增强对比"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#实验3-跟再input-space进行增强对比"}},[t._v("#")]),t._v(" 实验3：跟再input space进行增强对比")]),t._v(" "),a("p",[a("img",{attrs:{src:"https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20211128162546.png",alt:"image-20211128162546341"}})]),t._v(" "),a("p",[t._v("实验结果：")]),t._v(" "),a("ul",[a("li",[t._v("在特征空间进行extrapolation效果更好")]),t._v(" "),a("li",[t._v("特征空间的增强跟input空间的增强可以互补")])]),t._v(" "),a("h3",{attrs:{id:"实验4-把增强的特征重构回去-得到的新样本有用吗"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#实验4-把增强的特征重构回去-得到的新样本有用吗"}},[t._v("#")]),t._v(" 实验4：把增强的特征重构回去，得到的新样本有用吗")]),t._v(" "),a("p",[a("img",{attrs:{src:"https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20211128163337.png",alt:"image-20211128163337691"}})]),t._v(" "),a("p",[t._v("这个实验还是有一点意思的，一个重要的结论就是，在特征空间这么增强得到的特征，重构回去，屁用没有。可以看到，都比最基础的baseline要差，但是如果把测试集都换成重构图，那么效果就不错。这其实不能怪特征增强的不好，而是重构的不好，因为重构得到的样本，跟特征真实代表的样本，肯定是有差距的，因此效果不好是可以理解的。")]),t._v(" "),a("h2",{attrs:{id:"重要结论-讨论"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#重要结论-讨论"}},[t._v("#")]),t._v(" 重要结论&讨论")]),t._v(" "),a("p",[t._v("综上所有的实验，最重要的实验结论就是：**在特征空间中，添加噪音，或是进行同类别的样本的interpolation，是没什么增益的，甚至interpolation还会带来泛化性能的降低。相反，extrapolation往往可以带来较为明显的效果提升。**这个结论还是很有启发意义的。")]),t._v(" "),a("p",[t._v("究其原因，i"),a("strong",[t._v("nterpolation实际上制造了更加同质化的样本，而extrapolation得到的样本则更加有个性")]),t._v("，却还保留了核心的特征，更大的多样性有利于提高模型的泛化能力。")]),t._v(" "),a("p",[t._v("作者在结尾还提到他们做过一些进一步的探究，发现了一个有意思的现象：**对于类别边界非常简单的任务（例如线性边界、环状边界），interpolation是有帮助的，而extrapolation则可能有害。**这也是可以理解的，因为在这种场景下，extrapolation很容易“越界”，导致增强的样本有错误的标签。但是现实场景中任务多数都有着复杂的类别边界，因此extrapolation一般都会更好一些。作者认为，"),a("strong",[t._v("interpolation会让模型学习到一个更加“紧密”、“严格”的分类边界，从而让模型表现地过于自信，从而泛化性能不佳。")])]),t._v(" "),a("p",[t._v("总之，虽然这仅仅是一篇workshop的论文，实验也做的比较混乱，可读性较差，但是揭示的一些现象以及背后原理的探讨还是十分有意义的。经典的数据增强方法mixup也部分受到了这篇文章的启发，因此还是值得品味的。")])])}),[],!1,null,null,null);s.default=e.exports}}]);