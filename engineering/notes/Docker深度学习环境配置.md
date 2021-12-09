---
title: Docker深度学习环境配置
published: 2021-6-24
sidebar: auto
---

# Docker深度学习环境配置

Author：郭必扬
Time：2019-05-02

### Why Docker？
导师提供了一台高性能GPU机器，但是装系统的老师对深度学习不大了解，所以环境需要我自己安装。在折腾了一两周后若干次失败后，我是在忍不住发了一条朋友圈：

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624615888380-image.png)

评论也是十分热烈，激起了大家的共鸣：

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624615895772-image.png)

但是，在茫茫评论中，一位大佬留言：

**“docker了解一下”**

顿时给我一线希望。docker这个玩意儿之前也听说过，但是一直没去一探究竟，因为一直没有痛点。这一次装环境可让我深深体会到这个痛点。


![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624615906032-image.png)


我们为什么学编程的时候，一提到配置环境就头疼？因为没有一个确定的安装流程可以保证你的环境安装不出错。每一个人的电脑都不同，软硬件环境都不一样，所以可能同样的步骤在我这里可以顺利安装好，但是到你那里就各种bug满天飞。
Docker这个玩意儿，就是专门解决这个痛点的。大家把自己配置好的环境打包成镜像，我们可以直接使用别人的镜像，进入镜像之后，就进入了别人搭建好的环境，我们只需要提供硬件支持即可。这个就相当于一个虚拟机，我们可以在Windows系统里安装一个linux的虚拟机，但是docker相比虚拟机来说占用内存更小，转移起来更加方便。

---

写下上面这些话时，我使用docker也有几周了，刚开始对docker的各种操作很懵逼，所以决定记录下来经常使用的各种操作和对应的说明，方便日后的使用。后来发现记录的还挺不错，所以决定整理成体系分享给大家。

>**注:**
本文的安装、使用，均在**Linux-Ubuntu**系统下进行。不同系统安装过程会有不同，但是安装好后的操作基本相同。

## 一、Docker、深度学习镜像、Nvidia-docker的安装
安装这种事儿，真不想详细写。因为这里确实没有很多坑。
#### 1. Docker的安装
链接：https://docs.docker.com/install/linux/docker-ce/ubuntu/
跟着教程一路复制粘贴回车即可。
唯一的难点就是看懂英文的安装教程，看清楚段落层次结构。

反正，最后如果你运行`sudo docker run hello-world`，可以跑通，看到：

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624615920252-image.png)

就说明Docker已经被你成功安装了！

#### 2. Nvidia-docker的安装
为何又蹦出来一个nvidia-docker？因为原本的docker不支持GPU加速，所以NVIDIA单独做了一个docker，来让docker镜像可以使用NVIDIA的gpu。
链接： https://github.com/NVIDIA/nvidia-docker
也是直接找对应的操作系统的命令，一行行复制粘贴回车就搞定了。

反正，最后当你运行`docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi`时，如果看到：

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624615929769-image.png)

恭喜，安装成功了！

#### 3. 深度学习镜像的安装
我这里使用镜像是**deepo**一款咱们中国人做出来的深度学习镜像，包含了现在多数流行的深度学习框架，而且版本也很新，所以我这个小白第一次就选择了这个。
链接：https://hub.docker.com/r/ufoym/deepo
只有安装好了前面的docker和nvidia-docker，这里就很方便了。
直接通过命令`docker pull ufoym/deepo`就可以把各种框架都下载下来。但是这样比较大，费时较长，所以教程里面也提供了值安装某一种框架的方式：

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624615940586-image.png)


另外，还提供了jupyter notebook版的镜像，我这里就是安装的这个，因为我日常基本都是使用jupyter notebook，这里贴一下我的命令：
```
sudo docker pull ufoym/deepo:all-jupyter-py36-cu100
```
这里的`all-jupyter-py36-cu100`也是deepo提供的jupyter notebook镜像的tag。
安装好之后，通过`docker images`命令，可以查看已经下载好的镜像：

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624615948308-image.png)


好了，该装的东西都装好了，下面进入操作部分了！

---

## 二、Docker最常用操作
### （一）基本概念
image，镜像，是一个个配置好的环境。
container，容器，是image的具体实例。
image和container的关系，相当于面向对象中类与对象的关系。

**如何查询命令参数：**
`docker`可以看docker客户端有那些基本命令；
对应每一条命令，想看看具体是做什么的，可以在后面加一个`--help`查看具体用法，例如对于run命令：
`docker run --help`

### （二）容器的相关操作
#### 1.容器的创建、查看、删除
`docker run [-it] some-image` 创建某个镜像的容器。**注意，同一个镜像可以通过这种方式创建任意多个container.**
加上`-it`之后，可以创建之后，马上进入交互模式。

`docker ps`列出当前运行的容器

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624615956747-image.png)


`docker ps -a`列出所有的容器，包括运行的和不运行的

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624615963725-image.png)


`docker rm container-id`删除某个容器

#### 2.容器的启动、进入、退出：
`docker start [-i] container-id`**启动**某个容器，必须是已经创建的。
加上`-i` 参数之后，可以直接**进入**交互模式：

注意到，交互模式下前缀会变化：
![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624615984226-image.png)

除了通过`-i`**进入**交互模式，还有一种方法，那就是通过`attach`:
`docker attach container-id`

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624615998363-image.png)


进入交互模式之后，怎么**退出**呢：
- 想退出但是保持容器运行，按`CTRL+Q+P`三个键
- 退出，并关闭停止容器，按`CTRL+D`或者输入`exit`再回车

注：Ctrl+P+Q按的时候有时候会不灵，多按几次！

容器的停止、重启：
`docker stop container-id`
`docker restart container-id`

### （三）Docker jupyter notebook 服务 [力荐!] 
深度学习jupyter notebook镜像已经创建：

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616007523-image.png)

#### 1.如何创建自己的可以远程访问的容器：

```
sudo nvidia-docker run -it -p 7777:8888 --ipc=host -v /home/shcd/Documents/gby:/gby --name gby-notebook  90be7604e476
```

其中：
- `-it`为直接进入交互式
- `-p 7777:8888`是把主机的7777端口映射到容器的8888端口
- `-ipc=host`可以让容器与主机共享内存
- 还可以加一个`--name xxxxx`给容器定义一个个性化名字
- `-v /home/shcd/Documents/gby:/gby`可以讲主机上的/home/shcd/Documents/gby地址挂载到容器里，并命名为/data文件夹，这样这个文件夹的内容可以在容器和主机之间共享了。因为容器一旦关闭，容器中的所有改动都会清除，所以这样挂载一个地址可以吧容器内的数据保存到本地。
- `90be7604e476`则是你安装的jupyter镜像的id，可以在刚刚docker images命令后面查看，当然你也可以直接写全名`ufoym/deepo:all-py36-jupyter`

经过上面的操作，你应该可以直接进入容器了，这时你用`ls`命令，应该可以看到一个新的文件夹gby产生了！

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616016228-image.png)


#### 2.创建了容器之后，我们可以进而启动jupyter notebook：
```
jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir='/gby'
```
其中：
- `--no-browser`即不通过浏览器启动，`--ip`指定容器的ip，`--allow-root`允许root模型运行
- `--NotebookApp.token`可以指定jupyter 登录密码，可以为空
- `--notebook-dir='/gby'`指定jupyter的根目录

#### 3.开启本地与服务器的端口映射，从而远程登录jupyter：
在**本地机器**上，执行如下命令：
```
 ssh username@host-ip -L 1234:127.0.0.1:7777
```
这样，可以将本地的1234端口，映射到服务器的localhost的7777端口（即你前面创建jupyter容器时候的指定的服务器端口）
这样，你在本地电脑的浏览器里输入'localhost:1234'，即可登录到服务器上的jupyter notebook了！

服务器的jupyter容器内我的文件夹：
![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616030322-image.png)

本地访问服务器jupyter：
![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616045632-image.png)


当我第一次看到这个画面的时候，简直激动地要跳起来！
**既能远程访问高性能服务器，又可以像在本地一样便捷地操作**，你说激动不激动你说激动不激动？

### （四）容器的备份
之前好不容易配置好的环境，突然被学校服务器要重装！？怎么办？
你想到的一定是：**能不能把配置好的环境备份一份，后面直接重新加载进来？**

方法也很简单：
一般情况下，我们想备份的是容器，因为我们具体的配置都是在容器中进行的，而镜像一般都是直接在网上下载的，我们不做什么改动。

先通过`docker ps`或者`docker ps -a`来查看你想备份的容器的id，
然后通过：
```
docker commit -p [your-container-id] [your-backup-name]
```
来将id为your-container-id的容器创建成一个镜像快照。

接着，你通过`docker images`就可以查看到刚刚创建好的镜像快照了。
然后，通过：
```
docker save -o [path-you-want-to-save/your-backup-name.tar]] [your-backup-name]
```
把那个镜像打包成tar文件，保存到服务器上。
后面就可以把服务器上打包好的tar文件，下载到本地了。

恢复：
`docker load -i your-backup-name.tar`
`docker run -d -p 80:80 your-backup-name`

---

以上就是我目前使用到的最常用的用法了，至少对我目前的需求来说是够用了，随着我使用次数的变多，我也会不断更新。希望能够减少大家在环境搭建之路上的折磨吧！
