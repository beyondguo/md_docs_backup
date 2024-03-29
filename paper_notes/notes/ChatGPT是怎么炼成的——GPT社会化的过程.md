# ChatGPT是怎么炼成的&带来的一些启示

> 本文前半部分内容，是对李宏毅老师的视频（Chat GPT (可能)是怎麼煉成的 - GPT 社會化的過程，https://www.youtube.com/watch?v=e0aKI2GGZNg）的笔记记录。然后后半部分写了一些受到的启发和个人的思考。

---



## 一、ChatGPT是怎么炼成的

根据OpenAI的博客，ChatGPT跟InstructGPT训练过程十分相似（查重率90%以上）。由于ChatGPT还没有放出论文，但InstructGPT有论文，下面主要是InstructGPT的训练方法，以此来窥探ChatGPT的训练方式：



**四阶段学习：**

1. **学习文字接龙**
causal language modeling
![image-20221208233943144](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202212082339385.png)

2. **人类老师来引导文字接龙的方向**
    前面单纯的“文字接龙”的问题就是，回答可能是多种多样的，有一些是我们希望的，有些则是我们不希望得到的。所以第二步，就引入人工干预，告诉模型我们人类的偏好。
    ![image-20221208234856388](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202212082348415.png)

  （人工标注的正确答案只有数万则）

3. **模仿人类老师的喜好**
    人类对GPT的多个输出进行反馈。然后借助这个反馈，去学习一个teacher mode。
    ![image-20221208235130016](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202212082351039.png) 

4. **用强化学习方式，向teacher model学习**

![image-20221208235346988](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202212082353018.png)

![image-20221208235409573](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202212082354601.png)



---

![image-20221208235614355](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/typora/202212082356374.png)



