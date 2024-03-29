# ChatGPT的前身——InstructGPT论文解读



ChatGPT的论文尚未放出，也不知道会不会有论文放出，但是根据公开资料显示，其训练方式，跟OpenAI之前的一个工作——InstructGPT基本无异，主要是训练数据上有小的差异，因此我们可以从InstructGPT的论文中，窥探ChatGPT强大的秘密。本文主要介绍（粗略）解读一下InstructGPT的论文——***Training language models to follow instructions with human feedback***.

![image-20230101155304432](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202301011553460.png)

## 背景（Instruct? Align? 社会化?）

InstructGPT和后面的ChatGPT，都是OpenAI在**大模型alignment问题**上的研究成果，什么是模型的alignment呢？在InstructGPT论文中作者是这么说的：

> For example, large language models can **generate outputs that are untruthful, toxic, or simply not helpful to the user**. In other words, these models are not **aligned** with their users.
> （ChatGPT翻译：大型语言模型可以生成不真实、有毒、或者对用户没有帮助的输出。换句话说，这些模型与用户不匹配。）



就是说，模型的输出，跟我们期待的，可能有所不一致。这个跟人类的需求的对齐问题，就是所谓的alignment问题。李宏毅老师的视频（Chat GPT (可能)是怎麼煉成的 - GPT 社會化的過程，https://www.youtube.com/watch?v=e0aKI2GGZNg）中把对大模型的跟人类需求一致性的改善过程，称为大模型的“社会化”过程，我认为十分的形象，大模型在预训练过程中见识了各种各样的数据，因此针对一个prompt会输出什么东西，也可能是多种多样的，但是预训练数据中出现的数据模式，不代表都是人类在使用模型时希望看到的模式，因此需要一个社会化的过程，来规范模型的“言行举止”。

<img src="https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202301011631986.png" alt="image-20230101163114940" style="zoom:50%;" />



下面我举个例子：

![image-20230101165951854](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202301011659889.png)

对于GPT这样的自回归式生成模型，也就是大家常见的“续写”模型，我们给一个输入：“ACL会议的主题是什么”，我们自然是希望模型直接告诉我们问题的答案，也就是上图中蓝色机器人的回答。但是模型的输出可能跟我们期待的差别巨大，输出一连串的问题，即图中红色机器人的输出。为什么呢？因为无论是“一个问题后面接一个回答”，还是“一个问题后面接另一个问题”，都是训练语料中可能经常出现的**模式**，因此，你让模型根据一个问题来续写，那无论是续写问题的答案，还是续写更多的问题，对于模型来说都是合理的。这就是问题所在，如果让经过大规模语料（可能也没任何人知道数据集里到底都有些啥乱七八糟玩意儿）预训练的模型，在输出时符合人类的期待？



## InstructGPT的方法

下面直接讲一讲OpenAI是如何处理alignment问题的，论文中的这个图就已经十分清楚：

![img](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202301011718133.svg)

这里顺便也放出ChatGPT训练的流程图，基本可以等于复制粘贴：

![img](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202301011719566.svg)

（不能说很像，只能说是一模一样了，所以我们可以当做是一对孪生姐妹了，可能吃的饲料略有不同，再就是ChatGPT出生的晚，家庭条件相对来说更好一些，是从GPT-3.5出发的，而InstructGPT是从GPT-3继续训练的。）



我们先不看上面的三个步骤，自己想一想，通过前文对问题背景的介绍，我们应该如何解决模型跟人类期待不匹配的问题？最直接的办法，就是我们人工构造一大批数据（人们自己写prompt和期待的输出），完全符合人类的期待的模式，然后交给模型去学。然而，这显然代价太大了。因此，我们得想办法怎么让这个过程变得更轻松一点：

- 称初始模型为**V0**，也就是GPT-3。我们可以先人工构造一批数据，不用数量很大，尽其所能吧，然后先让模型学一学，我们这个时候模型为**V1**。

- 然后让模型再根据一堆prompt输出，看看效果咋地，我们让模型**V1**对一个prompt进行多个输出，然后让人对多个输出进行打分排序，*排序的过程虽然也需要人工，但是比直接让人写训练数据，还是要方便的多*，因此这个过程可以更轻松地标注更多数据。然而，这个标注数据，并不能直接拿来训练模型，因为这是一个排序，但我们可以训练一个打分模型，称为**RM**（reward-model），RM的作用就是可以对一个`<prompt,output>`pair打分，评价这个output跟prompt搭不搭。

- 接下来，我们继续训练**V1**模型，给定一些prompt，得到输出之后，把prompt和output输入给**RM**，得到打分，然后借助强化学习的方法，来训练**V1**模型，如此反复迭代，最终修炼得到**V2**模型，也就是最终的InstructGPT。



上面的三步，就是图中展示的三个步骤，可以看出就是老师（人类）先注入一些精华知识，然后让模型试着模仿老师的喜好做出一些尝试，然后老师对模型的这些尝试进行打分，打分之后，学习一个打分机器，最后打分机器就可以和模型配合，自动化地进行模型的迭代，总体思路称为**基于人类反馈的强化学习，RLHF**。

能实现这样的方式，我觉得前提就是——这个模型本身已经比较强大了。只有模型本身就比较强大了，才能人类提供少量的精华数据，就可以开始进行模仿，同时在第二步产出较为合理的输出供人类打分。所以这里的GPT-3作为出发点，是这一套流程能行得通的保证之一，而ChatGPT又是从GPT-3.5出发的，那效果肯定更加好了。

InstructGPT论文中，给出了上述三个步骤，分别制造/标注了多少样本：

- **SFT数据集**（即第一步人类根据prompt自己写理想的输出，SFT：supervised fine-tuning），包含**13K**的prompts；
- **RM数据集**（即第二步用来训练打分模型的数据），包含**33K**的prompts；
- **PPO数据集**（即第三步用来训练强化学习PPO模型的数据），包含**31K**的prompts。

前两步的prompts，来自于OpenAI的在线API上的用户使用数据，以及雇佣的标注者手写的。最后一步则全都是从API数据中采样的，下表的具体数据：

![image-20230101182048706](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202301011820739.png)

总共加起来，也就**77K**的数据，而其中**涉及人工的，只有46K**。真心不多！也就是GPT-3继续在77K的数据上进行了进一步微调，就得到了InstructGPT。



**初始的种子数据集，需要标注者来编写prompts，而不是从API数据中采样**，这是因为API接口中的prompts数据，多数都不是那种”人类要求模型干什么事儿“这类**instruction-like prompts**，多数都是续写之类的，这跟本文的出发点——希望模型能按照人类的要求做事儿，有点不匹配，所以需要标注者现场编写。具体这些**标注者**被要求写这么三种数据：

- **Plain**：自己随便拍脑袋想一些prompts，同时尽可能保证任务的多样性。（比方随便写”请给我写个段子“，”请给我把这段话翻译成德语“，”啥是马尔科夫链啊？“等等各种问题、要求）
- **Few-shot**：不仅仅需要需要写prompts，还需要写对应的outputs。（这部分应该是最耗费人力的了，也是SFT数据的主要组成部分）
- **User-based**：OpenAI的用户希望OpenAI未来能提供哪些服务，有一个waitlist，然后这些标注者，就根据这个waitlist里面的task来编写一些prompts。（相当于告诉标注者，你们看看用户们都期待些什么功能，你们可以作为参考）



下表则展示了OpenAI的客户在日常使用时的用途分布，即**API数据的分布**（这也是RM数据集的大致分布）：

![image-20230101182729853](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202301011827884.png)

（但奇怪的是，论文中找不到**SFT数据集**的分布，感觉应该跟RM的分布差别挺大的，而从数据质量上讲，这部分应该是质量最高的，可能是属于商业机密？）



以上就是InstructGPT的方法论，以及大家最关心的数据收集过程。至于模型怎么训练，这些不重要，毕竟99.99%的人都没法训练GPT-3，更别提GPT-3.5了。但是这里有一个地方确实也需要说一嘴，打分模型（RM模型）也是基于GPT-3进行训练的，使用的是6B的版本，具体就是在进行SFT训练之后，把最后的embedding层去掉，改成输出一个标量。



## InstructGPT工作的主要结论

效果其实不必多说，大家已经十分熟悉ChatGPT多么强大，InstructGPT其实类似。最终的结论就是，**在”听指挥“方面，1.3B版本的InstructGPT，就可以超过比自己大100倍的175B版本的GPT-3了**：

![image-20230101190610194](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202301011906219.png)

下面是一个例子：

![image-20230101190727500](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202301011907527.png)



总的来说，这篇文章就是在介绍OpenAI是怎么把GPT-3这个野孩子调教得听人类指挥的，而且这个调教成本，并没有那么大，相比于GPT-3预训练的成本，InstructGPT仅使用了77K的数据进行微调，这基本不值一提。最终，InstructGPT生成的结果，在真实性、无害性、有用性方面都有了很大的提高（但是对偏见这种问题依然没有改善）。



除此之外，另外作者团队通过大量的实践，总结了一下几个重要结论：

- 这种“调教”，会降低模型在常见NLP任务上的效果，作者称之为“对齐税”——alignment tax（实际上之前很多研究都发现了这个问题）。但是，通过改善RLHF的过程，比如在预训练过程也混合RLHF的方法。
- 常见的公开NLP数据集，跟人类实际使用语言模型的场景，差别很大。因此单纯在公开NLP数据集进行指令微调，效果依然不够。
- 虽然人类标注只有几十K，远远不能覆盖所有可能的prompts，但是实验发现InstructGPT的域外泛化能力很强，对于没有见过的prompt类型，依然有比较好的泛化能力。
- 革命尚未成功，InstructGPT依然会犯错，依然可能瞎编乱造、啰里吧嗦、不听指挥、黑白不分。。。测试过ChatGPT的同学肯定也发现及时是ChatGPT也难以避免这个问题。所以InstructGPT、ChatGPT是开启了一扇门，让人看到了巨大的希望，也看到了巨大的困难，依然有很多有挑战性的问题等着我们解决。





以上。

