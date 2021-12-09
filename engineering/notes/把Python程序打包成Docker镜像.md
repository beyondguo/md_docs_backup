---
title: 把Python程序打包成Docker镜像
published: 2021-6-24
sidebar: auto
---

# 把Python程序打包成Docker镜像

## 前提：
1. 母机上已安装anaconda
2. 了解基本的Docker概念、最基本的命令

本文中涉及到的主要命令：
- `docker pull` ：从docker hub拉取某个镜像
- `docker image ls`：查看当前系统中的所有镜像
- `docker build`：根据Dockerfile创建一个镜像
- `docker run`：启动某个镜像，运行一个容器
- `docker ps`：查看当前系统中所有运行中的容器
- `docker ps -a`：查看所有容器，不管是否运行

---

# 1. 母机上创建python虚拟环境
创建：
`conda create -n your_project_name python=3.6`
这样会创建一个只有pyhton3.6的环境。

进入虚拟环境：
`conda activate your_project_name `

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616128547-image.png)


# 2. 在虚拟环境中，部署自己的程序和相关文件
安装python依赖，**记得记录下来安装了那些包**：

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616135761-image.png)

上传自己的代码、文件，**确保在当前环境下可以正常运行（记住一定要在自己的虚拟环境中）**。

# 3.编写dockerfile
先安装一个基础的镜像，如python3.6环境
`docker pull silverlogic/python3.6`

直接pull可能会比较慢，可以通过阿里云镜像加速：

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616144570-image.png)

直接在命令行复制图中的命令即可。


![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616165196-image.png)



**然后就开始编写Dockerfile了！**
首先我以我的项目为例，来明确几个概念：
项目根目录： `\trp_service\`
在根目录下，我的文件有这些：

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616174763-image.png)
257d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中，start.sh是我的启动文件，通过`sh start.sh`即可运行我的服务trp_service.py
这个python程序启动后，会生成一个api，提供词向量计算服务。内部的api端口是9000.

接下来我们编写Dockerfile文件：
在项目的根目录 `\trp_service\`里，我创建了一个文件，名为`Dockerfile`，**注意只能是这个名字**。然后我们进入Dockerfile文件，按照如下内容编写：
```shell
FROM silverlogic/python3.6
MAINTAINER gby

ENV BUILD_HOME /trp_service

RUN pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple/;\
        ... ...
	pip install bert4keras==0.8.4 -i https://pypi.tuna.tsinghua.edu.cn/simple/

COPY . $BUILD_HOME
RUN chmod 777 $BUILD_HOME/start.sh

WORKDIR $BUILD_HOME

CMD $BUILD_HOME/start.sh
```

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616184842-image.png)


# 4. 开始创建我的Docker镜像
在当前的项目目录中，执行如下命令：
`docker build -t trp:v1 .`
- `-t`参数后，输入`名字：版本`
- 注意命令最后有一个 `.` ，这是指在当前的目录下去寻找Dockerfile文件。


![build成功](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616193016-image.png)


创建之后，可以通过`docker image ls`查看系统镜像，发现trp已经创建好了：

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616205864-image.png)




# 5.启动镜像，生成容器，调用容器内的服务
通过命令：
`docker run -itd -p 9000:9000 trp:v1`
即可启动服务。
此时通过`docker ps`查看当前运行的容器：

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616216662-image.png)

能看到，就说明容器已经启动成功。

在命令中，我通过-p来设置宿主机和容器内的端口映射。故现在我在宿主机，也可以通过9000端口来访问我容器内的服务了：

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616228668-image.png)






