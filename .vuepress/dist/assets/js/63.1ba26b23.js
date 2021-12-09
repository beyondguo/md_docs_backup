(window.webpackJsonp=window.webpackJsonp||[]).push([[63],{432:function(v,_,t){"use strict";t.r(_);var e=t(44),a=Object(e.a)({},(function(){var v=this,_=v.$createElement,t=v._self._c||_;return t("ContentSlotsDistributor",{attrs:{"slot-key":v.$parent.slotKey}},[t("h1",{attrs:{id:"我的第一篇论文诞生的故事"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#我的第一篇论文诞生的故事"}},[v._v("#")]),v._v(" 我的第一篇论文诞生的故事")]),v._v(" "),t("center",[v._v("作者：郭必扬")]),v._v(" "),t("center",[v._v("时间：2020-12-16")]),v._v(" "),t("blockquote",[t("p",[t("strong",[v._v("前言：")]),v._v("\n离上一次写博文已经快半年了，这半年我主要在忙两件事，一个是组里的企业项目，一个是我的第一篇学术论文。时间飞逝，转眼半年过去，从项目中诞生的一个想法最终转换成了我的第一篇学术论文，成功被AAAI接收，这对于刚刚开始博士生涯的我是莫大的鼓励。本文尝试回忆一下这篇论文诞生的全过程，算是给这段难忘的时光画上一个句号。")])]),v._v(" "),t("h2",{attrs:{id:"一、idea的产生"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#一、idea的产生"}},[v._v("#")]),v._v(" 一、Idea的产生")]),v._v(" "),t("h3",{attrs:{id:"_1-灵感来临的夜晚"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#_1-灵感来临的夜晚"}},[v._v("#")]),v._v(" 1. 灵感来临的夜晚")]),v._v(" "),t("p",[v._v("2020年7月份的某个周五夜晚，刚刚开完组会的我陷入沉思，博士一年级还没开学，怎么大家都已经在讲自己的论文和研究工作了？我还什么都没有呢，每天就只做做项目。之前曾经有一个研究工作，因为陷入了瓶颈，也好久没有继续推进了，因此从硕士毕业以来，一直不知道我的学术之路到底在哪里。")]),v._v(" "),t("p",[v._v("我们组做研究主要的方向，还是偏信管的方向，这意味着首先是以实际的应用为导向的，另外主要还是做计量的分析，即发现一些“什么对什么有什么影响”的这样的研究。说实话，我一直不太感兴趣。而看计算机的论文，我就觉得很有意思，而且计算机方向的研究，让我感觉很有创造力，所以我对计算机的方向，尤其是当今的人工智能方向，一直有执念。曾经也跟导师讨论过，他跟我建议还是做信管方向的研究，毕竟是我们组的专长；曾经也有一些稚嫩的计算机方面的研究点子，也被组里的老师轻松戳破（早被人做过了）。所以，好久以来，我还是挺迷茫的。")]),v._v(" "),t("p",[v._v("那个夜晚，结束组会后，我没有像往常一样直接去休息，而是拿出iPad开始构思一些想法，不知为何当时就有一种预感：“今晚我会想出一个可以做的想法”。近期在企业项目中，一直在做文本分类相关的工作，为了分析模型的结果，常常盯着混淆矩阵看，从中我也发现了一些问题，那就是有一些类别很容易搞混，就一直再想办法解决它。当时就觉得，各个类别彼此之间的相似性是很不一样的，然而分类的时候我们却是假设他们是一样的来分类（one-hot target），这样自然不太好，如果能让模型来训练的时候意识到不同的输出维度存在相似性就好了，于是我立马画了一个草稿：")]),v._v(" "),t("p",[t("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-24/1624540392571-image.png",alt:""}})]),v._v(" "),t("p",[v._v("草稿展示的实际上是一个很简单的想法，那就是手动构造一个soft target，然后用那个去训练模型。写完这个想法之后，我十分地兴奋（虽然后来通过文献阅读发现，学术界早就有类似的想法了），马上开始写相关的代码。第二天也是起的很早就在家里做实验。让我高兴的是，这样朴素的想法，确实是有效的，我在我们项目的数据集上，发现了些许的效果提高。")]),v._v(" "),t("h3",{attrs:{id:"_2-雏形初现"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#_2-雏形初现"}},[v._v("#")]),v._v(" 2. 雏形初现")]),v._v(" "),t("p",[v._v("在我的朴素的想法得到实验的印证之后，我就开始想这个如何能有“学术含量”。因为“人工的”的发不了论文，“智能的”才可以发论文。所以我需要设计一个方法，让我通过人工手段得到的效果提升，可以智能地实现。这个实际上不难，只要是有机器学习、深度学习经验的人，都知道基本的思路就是：")]),v._v(" "),t("center",[v._v("“人工构造了什么，就让模型去学习什么。”")]),v._v(" "),t("p",[v._v("所以我前面是自己手工构造的soft target，那我就需要设计模型去学习这个soft target。能达到这个目的的模型设计有很多，但可行的不多，所以我画了很多草图，从原理的角度去查看其可行性。这个时候，之前看过的几篇论文对我产生了极大的提示作用，因此也借鉴了其模型结构。现在回想起来，如果我没看过那篇论文，我也设计不出这样的模型。")]),v._v(" "),t("p",[v._v("设计出来的模型，我又赶紧用代码去实现它，我使用项目的数据集做试验，发现构造出来的模型，也有效果！")]),v._v(" "),t("p",[v._v("此时我的心中已经开始有些按捺不住喜悦了，但我还不急着去找老师分享我的实验结果。为了证明不光光是项目的数据集有效果，我开始在网上收集各种各样的公共的文本分类数据集，中文的，英文的，赶紧都拿来试一试。当我测试到第三个数据集，发现我的方法依然有稳定的提高的时候，我知道，我这个论文，要诞生了！但那个时候，我只敢仰望顶会，感觉像AAAI这种，我还遥不可期，我的想法是把这个idea，写成文章投一投国内的一些新兴的NLP会议，博士开学前先试个水。")]),v._v(" "),t("p",[v._v("我开始整理这几天的实验数据、绘制模型的草图、写下自己的思路，并预约老师讨论。")]),v._v(" "),t("p",[v._v("我依然清晰地记得那一天在小会议室，我激动地、一口气讲完我的研究问题来源、思路、模型设计、实验结果，我感觉无比舒畅。老师认真地听完，没有犹豫，直接告诉我：“我觉得这是一个很有价值的研究，思路很清晰。”")]),v._v(" "),t("p",[v._v("对于这样的评价，我既惊喜，又感觉再意料之中，毕竟我是有备而来。然后我跟老师说，我想投一投SMP（全国社会媒体处理大会）试试，结果老师说：“这么好的想法，不投一个顶会？那岂不是浪费了！投AAAI吧！”\n这一句话，比任何对我想法的好评都更加鼓舞人心。因此，我放弃了随便投一投的想法，立志冲击AAAI！")]),v._v(" "),t("h2",{attrs:{id:"二、疯狂做实验"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#二、疯狂做实验"}},[v._v("#")]),v._v(" 二、疯狂做实验")]),v._v(" "),t("p",[v._v("写论文有两种套路：先写出来，然后补实验。或者先做实验，再根据结果写论文。")]),v._v(" "),t("p",[v._v("一般前一种套路适合学术大佬或者老油条，他们相信自己的思路一定可以出做好结果，做实验只不过是为了找证据给别人看，而它们内心早已知道答案。但我还不太自信，我需要实验结果来给我信心，在idea产生后的一个月里，我几乎都在做实验，没有动笔开始写。")]),v._v(" "),t("p",[v._v("要想让自己的论文有说服力，自然是需要找大家都使用的、公开的benchmark数据集，收集了大量的数据集之后，还需要进行一些挑选，不可能所有的都放到论文里去。另外，因为要跟baseline模型进行对比，所以我还需要复现一些baseline模型。有些数据集，我发现我复现的结果，完全比不上baseline论文中声称的，那么这种数据集我就放弃了，不然容易被他人怀疑是不是故意把baseline做的很差。")]),v._v(" "),t("p",[v._v("数据集收集完毕之后，就进入到艰苦的“实验-调参-实验”循环了。这个过程是最无聊、最容易让人开始怀疑自己的阶段了。调参调到后期，甚至完全不知道自己在干嘛，开始怀疑自己做实验的意义，怀疑自己到底是在做科学研究，还是只是在费电而已。实验的记录我也比较随意，常常会忘记记录具体参数，导致结果无法复现。")]),v._v(" "),t("p",[v._v("这段时间的生活枯燥而乏味，每天在项目工作之余，把一组组的实验提交到服务器上跑。睡前如果发现跑出好结果了，我可以兴奋地晚上睡不着；要是效果差，我会啥都不想做，对女朋友都不耐烦，导致她有一次对我说：“我觉得自己对你根本不重要！你的喜怒哀乐完全由实验结果支配！”，哎，她这么一说，我发现还真是，完全被实验控制了。")]),v._v(" "),t("h2",{attrs:{id:"三、论文的成型"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#三、论文的成型"}},[v._v("#")]),v._v(" 三、论文的成型")]),v._v(" "),t("p",[v._v("时间到了8月份，基础的实验基本都做完了。这段时间主要是组会上跟老师们讨论实验结果、尝试进行理论解释的过程。深度学习的黑箱之处在于，你希望它怎么做，它还真不一定按你的做。所以我虽然按照自己的特定的意图设计的模型，但模型却不一定实现了我的意图。从实验结果也可以看出，有一些结果是跟我的预想不一致的。这段时间也很让人头疼，因为有些实验结果，真的不知道怎么解释，要解释也只能解释个大概，而大概的文字是无法写进论文里的。")]),v._v(" "),t("p",[v._v("然而随着AAAI投稿deadline的逼近，我也开始有点慌张了。本来还设计了很多补充实验和拓展研究，但按照目前的进度，是无法完成了。于是我决定不管那么多了，先根据现有的结果写出来再说！")]),v._v(" "),t("p",[v._v("这不写不知道，一开始写英文论文，就发现写起来总是很别扭，感觉自己词汇量就那么屈指可数的几个词。另外，论文的结构我也不知道怎么组织，我看的各种论文，都有各自的结构，有的related work在前，有的在后，没有一个标准。")]),v._v(" "),t("p",[v._v("这个时候咋办？我的办法是——抄！哦，不对，读书人的事儿怎么能叫“抄”呢？应该叫借鉴。我又是挑选了我那个时候正在看的我觉得格式写的特别工整的、又正好是AAAI的文章——TextGCN。上来先把人家的组织结构给“借鉴”了一遍，借鉴完这个，还不够，还要借鉴人家的遣词造句，诶，随着借鉴的深入，我发现我写的也慢慢“有内味儿了”，于是慢慢就上道了。除了TextGCN这一篇，我还借鉴了好几篇其他的优秀论文，包括别人画了写什么图、怎么画的、表格怎么设计的。总之，不管内容优不优秀，至少我先让我的论文看上去像一篇正经的AAAI论文。")]),v._v(" "),t("p",[v._v("我住的附近有个商场，楼上有家钢琴店，楼下有家星巴克。差不多在我刚刚产生这篇论文idea的时候，女朋友开始在那家琴房学钢琴。于是那些日子，经常是她在楼上练琴，我在楼下写论文，然后一起回家。令人怀恋的岁月啊！")]),v._v(" "),t("h2",{attrs:{id:"四、提交前的紧急修改"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#四、提交前的紧急修改"}},[v._v("#")]),v._v(" 四、提交前的紧急修改")]),v._v(" "),t("p",[v._v("deadline前的两周，另两位老师也加入了论文的讨论，听完论文后，他们首先肯定我的研究的完整性和规范性，但也提出了一些比较尖锐的问题，比如缺失了一个重要的baseline，另外模型的设计存在不合理之处。这两个都是十分严重的问题，对于只有两周就要提交的我来说，是十分可怕的。但同时，我也感到十分幸运在提价前能收到这样重要的反馈。")]),v._v(" "),t("p",[v._v("时间再紧也没有办法，只能硬着头皮去继续做实验、修改。说实话那个时候我已经有点疲惫了，首先，增加一个baseline意为着我至少要跑25组实验。另外设计上的不合理，这属于根本性问题了，但我已经没有时间去重新设计、重跑全部实验了，只能把这种不太合理的地方给淡化，毕竟效果上是提升的，只是模型结构可以设计的更有说服力。")]),v._v(" "),t("p",[v._v("最后我的折中方案是，新增几组实验实验，把baseline加上去，并使用一个更好的模型结构跑实验。最后火急火燎地做完了补充实验，算是把论文的一个大窟窿给填上了。")]),v._v(" "),t("p",[v._v("deadline分两个，一个是摘要的ddl，一个是正文的ddl。提交完摘要之后，一周之内我和老师们又一起把正文来来回回修改了N版，才放心地提交了。这个时候我感觉前面的“借鉴”还是发挥了很大作用，总体上本身已经比较规范了，但是多数是一些小问题，所以改起来还比较快。")]),v._v(" "),t("h2",{attrs:{id:"五、顺利挺过第一轮筛选"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#五、顺利挺过第一轮筛选"}},[v._v("#")]),v._v(" 五、顺利挺过第一轮筛选")]),v._v(" "),t("p",[v._v("论文提交后，就是接近一个月的空窗期。这段时间就是整理整理数据、代码，为github做准备。")]),v._v(" "),t("p",[v._v("10月14号的晚上，收到了顺利通过第一轮的邮件：")]),v._v(" "),t("p",[t("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-24/1624540431135-image.png",alt:""}})]),v._v(" "),t("p",[v._v("我也忘了当时是什么心情，应该也不是太激动了。但我确实是很满意的，第一次冲击顶会，我内心的小目标就是能过第一轮，那说明至少有一个国际上的该领域的审稿人认同我的工作了，那也是对我的一个巨大鼓励了。主要此时我也看不到任何的具体评论，所以我依然什么都做不了，只能耐心等待第二轮的结果一起放出来。")]),v._v(" "),t("h2",{attrs:{id:"六、为rebuttal鏖战到最后一刻"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#六、为rebuttal鏖战到最后一刻"}},[v._v("#")]),v._v(" 六、为Rebuttal鏖战到最后一刻")]),v._v(" "),t("p",[v._v("如果说第一轮的时候我内心还十分淡定，那么第二轮时我就真的开始紧张了。第一轮筛掉了38%的论文，一大半都还在呢，竞争只能是更加激烈了，而第二轮的结果，基本上就决定了最终的结果。")]),v._v(" "),t("p",[v._v("到了临近第二轮结果公布的时间点，我又开始像热锅上的蚂蚁了，反复地刷我的邮箱，就是看不到结果。直到第二天晚上，正骑车回寝室，手机振动了一下，一看“From Microsoft CMT”，我立马手抖着点开：")]),v._v(" "),t("p",[t("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-24/1624540444415-image.png",alt:""}})]),v._v(" "),t("p",[v._v("reviews已经在系统上公布，而且只有72小时的时间供我回复（rebuttal）。我一身冷汗，对着自行车踏板一顿狂踩冲到了寝室，迅速打开电脑查看reviews。")]),v._v(" "),t("p",[v._v("一打开review界面，密密麻麻的英文评论把我看蒙了，我直接去翻到最下面，应该会有打分，果真：")]),v._v(" "),t("p",[t("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-24/1624540460956-image.png",alt:""}})]),v._v(" "),t("p",[v._v("看到这一句“5-Below threshold of acceptance”，我的心凉了，哎。。。实力不济啊！")]),v._v(" "),t("p",[v._v("不过我马上又转悲为喜，因为我发现这个只是其中的一个打分。。。果真第一次投论文，完全没经验啊。最后的结果是两个6分，两个5分，完完全全地、不多不少地、恰到好处地踩到了分界线上！")]),v._v(" "),t("p",[v._v("这真是比走钢丝都要刺激啊！听说只要rebuttal做的好，是有可能让reviewer改分的！72小时倒计时，最后一搏，开始！")]),v._v(" "),t("p",[v._v("我赶紧联系老师们的时间，结果最晚只能到第二天晚上才能讨论，所以我得先自己整理好问题，然后明天做一次讨论后，写写就要提交了。")]),v._v(" "),t("p",[v._v("deadline前夕，我跟老师们在线上会议里碰面，仔细地讨论4个reviewer提出的几十条意见、建议。当时我最大的感受就是，这些reviewer的水平，真的是高啊！没有一个无聊的问题，每一个都一针见血，所以我曾经担心的、遗漏的问题，全部被他们给提了出来，很痛，又很爽（？？？）。而怎么回复他们的提问，又真的是一个技术活儿，这里面不仅仅要靠计算机知识，更需要借助心理学，不同的说法，给reviewer的感觉是完全不一样的，既要承认他们提出的问题（给审稿人面子），又要讲明自己的贡献（给自己面子），还要给出一个明确的解答或者计划（再次给审稿人面子），边想如何回复他们，其实我对自己的工作也理解的更加深刻。我和老师们从晚上8点，一直讨论到12点才散会。")]),v._v(" "),t("p",[v._v("那个晚上，我知道我是没有时间再睡觉了，我一个人留在实验室，开始整理rebuttal。夜晚的实验楼静悄悄，只有空调静静得吹，没有其他人，终于可以外放音乐，我写到了凌晨3点，实在写不动了，睡了一会，醒来后继续写，一直写到阳光从实验室窗户照射到我的电脑上，终于写完了！")]),v._v(" "),t("p",[v._v("发给老师们后，我赶紧回寝室睡觉，回去的路上碰到了刚刚出发去实验室的同学们。")]),v._v(" "),t("p",[v._v("睡醒后离deadline还有几小时，又根据老师们的意见修改了一部分，终于在系统上隆重地点击“submit”。那一刻，对我而言，仿佛是一个时代的结束。如果这是一场比赛，那么提交的那一刻，比赛就已经结束，我努力到了最后一刻，已经没有什么可以遗憾了。")]),v._v(" "),t("h2",{attrs:{id:"七、不抱希望到柳暗花明"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#七、不抱希望到柳暗花明"}},[v._v("#")]),v._v(" 七、不抱希望到柳暗花明")]),v._v(" "),t("p",[v._v("最终的放榜要等待12月1号，这是一个特别的日子，因为我的生日也在12月。我一直憧憬着，如果这次可以终稿，那么这将是我的收到的最好的生日礼物了。")]),v._v(" "),t("p",[v._v("这一个月来，我一直徘徊在“还是很有希望的！5566改成6666就有戏了！”和“没戏了，别幻想了！”之间。有时候看到知乎网友分享自己低分过线的经历，感觉自己也可以。后来看到知乎上有人开了一个AAAI21的专题讨论，很多人在分享自己的得分，我看了一圈，天哪，都比我高！7分8分的遍地都是，我这个边缘分，肯定没戏了！这时，我基本也不再幻想能中了。开始安安心心地根据review的看论文，准备年底的IJCAI。")]),v._v(" "),t("p",[v._v("11月底，越来越近了，虽然不抱希望了，但心中总不免还是会想“万一踩了狗屎运中了呢？”，所以我又开始焦虑了起来。到了放榜的那一天，我又开始每5分钟刷一次邮箱了。老婆也有点紧张了，总是提醒我看邮箱。一直等到晚上，还是没有放榜。")]),v._v(" "),t("p",[v._v("那一夜，真的很难睡着，辗转反侧，虽然我知道希望不大，但只要有一丝的希望，我就无法安心入睡。我一遍遍地刷新邮箱，什么也没有。然而第二天上午还有一门课的考试，我必须想办法睡着了。没办法，我打开手机记事本，在Todo list上写下这段话：")]),v._v(" "),t("p",[t("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-24/1624540482050-image.png",alt:""}})]),v._v(" "),t("p",[v._v("写完之后，顿时内心平静了下来。难道一个结果，就要把人的心态一直吊着吗？做好每天的事，管他结果怎么样！终于，我慢慢沉入梦乡......")]),v._v(" "),t("p",[v._v("第二天，太阳照常升起，我和室友照常骑着共享单车到教学楼，照常吃午饭、敲代码、看论文、调bug。结果依然迟迟不出现，老婆又问了几遍，我给她发了一个Twitter的截图，说国外网友也都在催呢，哈哈，不管了！")]),v._v(" "),t("p",[v._v("下午5时许，同实验室的同学喊我去吃饭，我没啥胃口，下午看别人Github的代码，一直还没看懂呢，正烦呢，于是我接着看代码。又过了一会儿，手机振动了一下，我火速打开手机，我知道，它来了。")]),v._v(" "),t("p",[v._v("边打开邮箱，我边自言自语：“好啦，我知道没过啦~~~不过万一真的能看到一个Congratulation单词呢？”")]),v._v(" "),t("p",[t("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-24/1624540500334-image.png",alt:""}})]),v._v(" "),t("center",[v._v('"Congratulations!"')]),v._v(" "),t("p",[v._v("是的，美好的事情发生了！")]),v._v(" "),t("p",[v._v('刚刚还眉头紧锁地，现在脸上的两坨肉已经开始止不住地上扬了，我反复地盯着邮件看，确认是我，确认是"congratulation"，确认是"delighted"，确认是"success"，我终于确认了，我中稿了！不管实验室里其他人了，一句“我艹！起飞了！”脱口而出。')]),v._v(" "),t("p",[v._v("我马上出门，给老婆打电话，开口我竟然激动地不知道怎么说，只是大声的说："),t("br"),v._v("\n“乖，快出来！快点！”"),t("br"),v._v("\n老婆听了有点吃惊：“啥？你不会来公司了吧？？好好，我出来了！”"),t("br"),v._v("\n我已经语无伦次：“不是不是，那个。。。告诉你一个天大的好消息！！！”"),t("br"),v._v("\n老婆迟疑了一下，马上反应过来了：“啊？不会吧！啊！！！！中稿啦！！！哇~~~！！”"),t("br"),v._v("\n我装作淡定地说：“是的，中稿了”，只有实验楼窗户的倒影知道我笑得多么灿烂。"),t("br"),v._v("\n......")]),v._v(" "),t("p",[v._v("跟老婆分享完喜悦之后，我又准备赶紧跟老师们说，结果老师们已经知道了，已经群里恭喜我了。于是我又赶紧跟我爸妈分享，他们也非常激动，还让我把邮件的截图、论文的大概意思都发给他们，看样子是要发朋友圈了，哈哈。")]),v._v(" "),t("p",[v._v("老婆约我去正大广场见面，请我吃大餐，在赴约的路上，我感觉自己好久好久没有这么轻松快乐了，也许从研究生起吧，感觉自己终于做成了一件事儿。")]),v._v(" "),t("h2",{attrs:{id:"∞、回顾与总结"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#∞、回顾与总结"}},[v._v("#")]),v._v(" ∞、回顾与总结")]),v._v(" "),t("p",[v._v("回顾从idea产生到论文接收的的过程，我觉得值得我记录的有这么几个点：")]),v._v(" "),t("ol",[t("li",[v._v("随时记录自己突然冒出来的想法，尤其是从实际问题中产生的想法，往往比较有价值")]),v._v(" "),t("li",[v._v("做模型的时候要多对进行细致的分析，比如分析混淆矩阵就比分析classification report要细，更能发现问题")]),v._v(" "),t("li",[v._v("从简单的想法入手，快速印证自己的想法的可行性")]),v._v(" "),t("li",[v._v("如果真的喜欢某个事物，就不要怕被泼冷水，坚持去想去做")]),v._v(" "),t("li",[v._v("先自己动手，找到证据说服自己，才能说服老师")]),v._v(" "),t("li",[v._v("多看经典论文和思路较为新颖的论文。例如对我有重大启发的那个论文，是TextGCN论文中的一个baseline方法，叫LEAM（Joint Embedding of Words and Labels for Text Classification），这个标题一看就是一个很新颖的结构，所以我特地去看了看，没想到后来用上了")]),v._v(" "),t("li",[v._v("认真地进行实验记录，不要偷懒。一个参数忘了写，等于后面要把一组参数都跑一遍。我就常常因为忘记某个实验结果是怎么跑出来的而不得不哭着重跑实验。。。")]),v._v(" "),t("li",[v._v("参数多了容易迷失自己，每次实验前，先想清楚自己的目的是什么，最好能记下来，实验结束后也马上记录一下结果印证了什么。")]),v._v(" "),t("li",[v._v("用云计算平台，记得即使保存代码和结果。我就因为忘了及时续费，导致中途两次代码和数据丢失，浪费了不少时间。")]),v._v(" "),t("li",[v._v("如果英文学术写作不熟悉，照葫芦画瓢是一个很有效的做法。")]),v._v(" "),t("li",[v._v("rebuttal really matters！好好写，有时候真的可以起死回生！")]),v._v(" "),t("li",[v._v("多找几个老师讨论，集思广益。这篇论文受到了四位老师的耐心帮助，每位老师都发挥了无可替代的贡献。在此再次感谢他们！")])]),v._v(" "),t("hr"),v._v(" "),t("blockquote",[t("p",[t("strong",[v._v("后记：")]),v._v("\n也许对于很多人来说，发表一篇AAAI是小菜一碟，尤其那些计算机名校、AI大组。但是对于我这种既不是计算机科班出身，学校学院乃至组里也不是专门做这个方向的人来说，这段经历对我来说已经十分珍贵了，遂记录下来，一来纪念这段时光，重温那段岁月的学术激情，二来鼓励自己继续努力，未来做出更多的学术贡献，三来给跟我类似背景的同学一些勉励，大家共同加油！")]),v._v(" "),t("p",[v._v("另外，细心的读者会发现，在论文诞生的这几个月里，我的女朋友也变成了我的老婆，是的，感谢她一直陪伴着我写完这篇论文，见证了我这期间的喜怒哀乐，也见证我人生中又一段独特的时光，未来还有好多美好的事情等待我们我一起经历。")])]),v._v(" "),t("blockquote",[t("p",[t("strong",[v._v("后记plus：")]),v._v("\n其实论文被接收与否，不影响这个论文的本质贡献。如果这篇论文没中，我也可以列举出各种没有中的理由。因此，论文本身是否真的有贡献，使我们更应该关注的。如果让我对自己这篇论文给出一个真实的评价，我会说，有一些创新和贡献，但还确实不够，革命尚未成功，同志任需努力！")])])],1)}),[],!1,null,null,null);_.default=a.exports}}]);