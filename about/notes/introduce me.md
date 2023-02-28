---
title:  About 郭必扬
published: 2022-01-11
sidebar: auto
---

# 郭必扬 Biyang Guo

- E-Mail: guo_biyang(@)163(dot)com
- [Github ](https://github.com/beyondguo) | [Google Scholar](https://scholar.google.co.uk/citations?hl=zh-CN&pli=1&user=B7l02PQAAAAJ) | [知乎](https://www.zhihu.com/people/guo-bi-yang-78/posts) | [SimpleAI公众号](https://mp.weixin.qq.com/s/v35g-p7wK2MkuM-SqjkF3g)

我目前是[上海财经大学信息管理与工程学院](https://sime.sufe.edu.cn/main.htm) AI Lab 三年级博士生（2020~2024），师从[黄海量](https://sime.sufe.edu.cn/5b/79/c10574a154489/page.htm)教授。硕士、本科均就读于上海财经大学信管学院。博士期间主要研究NLP中的数据增强、以数据为中心的 AI、更鲁棒的文本分类等。相关成果发表于 AAAI 会议，并有多篇工作在审稿中。

曾在[微软亚洲研究院（MSRA）](https://www.msra.cn/) NLC 组进行 9 个月（2022.3~2022.11）的研究实习，由[宫叶云](https://www.microsoft.com/en-us/research/people/yegong/)研究员指导。实习期间提出 [GENIUS 模型](https://arxiv.org/abs/2211.10330)，一个强大的基于草稿的文本生成预训练模型，可用于多种NLP任务的数据增强。

作为 [SimpleAI 社区](https://huggingface.co/Hello-SimpleAI)的创始人，在 ChatGPT 推出 10 天之后就组建了一个博士生、工程师团队，开展一项名为 [ChatGPT 对比与检测](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection)项目，推出首个开源的[人类-ChatGPT问答对比语料集（HC3）](https://huggingface.co/datasets/Hello-SimpleAI/HC3)和首个中英双语 [ChatGPT 内容检测器](https://huggingface.co/spaces/Hello-SimpleAI/chatgpt-detector-qa)，推出一个月累计访问量超过 2 万次。



## 科研工作/项目经历

#### ChatGPT 对比与检测 (preprint) ![](https://img.shields.io/github/stars/Hello-SimpleAI/chatgpt-comparison-detection?style=social)

- 论文： **[How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection](https://arxiv.org/abs/2301.07597)** (*Biyang Guo*, Xin Zhang, Ziyuan Wang, Minqi Jiang, Jinran Nie, Yuxuan Ding, Jianwei Yue, Yupeng Wu)
- 角色：项目发起人、负责人 | SimpleAI 社区，SUFE AI Lab
- <small>简介：ChatGPT的推出引起了学术界、业界的巨大轰动，ChatGPT生成的内容开始充斥各大UGC平台，并开始被用于作假、作弊，对互联网、教育等等行业产生了巨大威胁。基于此，我发起 ChatGPT 对比与检测项目，组建由8位国内外高校、企业的博士生、工程师，共同收集人类-ChatGPT对比数据，进行丰富的统计、语言学等分析，并基于深度学习、机器学习等技术，开发了一系列ChatGPT 内容检测器。据我们了解，**我们是学术、产业界最早开源对比数据集、检测器模型的团队**，目前检测器demo全球访问量已突破**2万**，用户覆盖5大洲，开源模型**月均下载量超过3K**，数据集月均下载量超过1K，Github **Stars 超过 512**，受到广大用户的认可和产业界的关注。相关论文预印版已发布于Arxiv平台。</small>

#### GENIUS – 基于草稿的文本生成模型 (preprint)  ![](https://img.shields.io/github/stars/beyondguo/genius?style=social)

- 论文：**[GENIUS: Sketch-based Language Model Pre-training via Extreme and Selective Masking for Text Generation and Augmentation](https://arxiv.org/abs/2211.10330)** (*Biyang Guo*, Yeyun Gong, Yelong Shen, Songqiao Han, Hailiang Huang, Nan Duan, Weizhu Chen)
- 角色：第一作者 | SUFE AI Lab，MSRA
- <small>简介：GENIUS 我在MSRA访问实习期间做的工作。是一个**基于草稿的生成式语言模型**（sketch-based generative language model），在超过2700万语料上进行大规模预训练，从而可以**基于少量的关键词、短语等信息生成内容丰富的文本段落**。GENIUS可用于**写作辅助、残缺信息填补，更是一个开箱即用的通用的 NLP 数据增强工具**，我们在分类、实体抽取、机器阅读等任务上验证了GENIUS作为数据增强工具的有效性。相关模型开源在Huggingface平台，模型**月均下载量超过 500 次**。</small>

#### STA – 针对性文本增强技术 (preprint)  ![](https://img.shields.io/github/stars/beyondguo/STA?style=social)

- 论文：**[Selective Text Augmentation with Word Roles for Low-Resource Text Classification](https://arxiv.org/abs/2209.01560)** (*Biyang Guo*, Songqiao Han, Hailiang Huang)
- 角色：第一作者 | SUFE AI Lab
- <small>简介：STA 是对传统NLP数据增强技术的一个改进，使用语义相似度和统计相关度对一个词的角色进行区分，然后针对性地进行数据增强。使用了STA方法的改进，**传统的基于规则的方法可以媲美甚至超越基于大型语言模型**（BERT、BART、GPT-2）的方法，且计算成本显著更低。</small>

#### LCM – 标签混淆学习，更鲁棒的文本分类 (AAAI-21)  ![](https://img.shields.io/github/stars/beyondguo/label_confusion_learning?style=social)

- 论文：**[Label Confusion Learning to Enhance Text Classification Models](https://ojs.aaai.org/index.php/AAAI/article/view/17529)** (*Biyang Guo*, Songqiao Han, Xiao Han, Hailiang Huang, Ting Lu)
- 角色：第一作者 | SUFE AI Lab
- <small>简介：我们提出在经典深度学习分类器的基础上添加一个LCM 插件，LCM 可以**在模型训练的过程中学习不同标签之间的重叠、相似关系，从而模拟一个比 one-hot 分布更加合理的标签分布**，使用这个改进后的标签分布来指导模型训练可以使模型在数据有噪音、标签易混淆的场景下获得显著性能提升。</small>



## 科研之外

我是一名技术科普爱好者，<u>**喜欢并追求将艰深复杂的理论知识用通俗易懂的语言描绘出来**</u>。在科研之外的时间，我喜欢撰写技术博客，进行模型、论文解读。代表作品如下：

- <small><a href='https://zhuanlan.zhihu.com/p/147310766'>整理了12小时，只为让你20分钟搞懂Seq2seq「知乎900+赞」</a></small>
- <small><a href='https://zhuanlan.zhihu.com/p/42559190'>从此明白了卷积神经网络（CNN）「知乎2600+赞」</a></small>
- <small><a href='https://zhuanlan.zhihu.com/p/71200936'>何时能懂你的心——图卷积神经网络（GCN）「知乎3500+赞」</a></small>
- <small><a href='https://zhuanlan.zhihu.com/p/74242097'>GraphSAGE：我寻思GCN也没我牛逼「知乎1600+赞」</a></small>


包含上述作品在内，我在知乎上的专栏[DeepLearning学习笔记](https://www.zhihu.com/column/deeplearningnotes)和[NLP学习笔记](https://www.zhihu.com/column/pythontricks)累计被收藏**3.5W次**，获得众多深度学习和自然语言处理领域同学的认可。



