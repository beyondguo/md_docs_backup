---
title: 【随时记录】Linux命令
published: 2021-6-24
sidebar: auto
---

# 【随时记录】Linux命令

笔者之前学习linux时候，学得七窍生烟。现在七窍通了六窍，一窍不通。
linux命令看了又忘，每次都令人抓狂，因此决定在此记录一些最常用的命令，和一些最基本的概念。



### 目录与层级

linux的目录层次可参考：
https://www.cnblogs.com/silence-hust/p/4319415.html

`ls`列出当前文件夹中的所有文件名称
`cd [folder_name]`进入某个文件夹
`cd ..`回到上一级目录

这里我就一直有一个问题搞不懂：
进入linux系统的时候，我的目录是：
![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624615810953-image.png)

我一直以为这是根目录，但是后来在反复的`cd ..`后，发现自己进入了这个目录：

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624615818372-image.png)

一下子把我搞懵了。
后来查了查，才发现，`～`是我们用户的目录，一个系统可能会有多个用户，当前用户的目录就是`～`。而系统根目录是`/`。

要回到用户的目录，只需要输入`cd ～`即可；
要访问根目录，也是只需要输入`cd /`即可。

**Linux系统上传文件到linux服务器，不能通过rz命令，因为需要端支持，如Xshell**
此时，应该使用`scp`命令：
例如：
```
scp -r /media/x1c/文档/Jupyter/--NLP/big_things/w2v/GoogleNews-vectors-negative300.bin root@202.121.138.168:/root/Documents/gby2019/wvmodels
```
即：
`scp -r 本地文件地址 username@hostname:服务器目标地址`

这个命令还可以上传**文件夹**！

如果反过来，想把服务器上的文件传到本地，则把服务器地址和本地地址颠倒过来即可：
`scp -r username@hostname:服务器文件地址 本地目标地址`

**确认有可用CUDA的gpu：**
`lspci | grep -i nvidia`

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624615831415-image.png)


**查看系统信息:**
`uname -m && cat /etc/*release`


**修改文件、目录权限**
`sudo chmod -R 777 /home/gby/Documents`
777代表权限模式，不用具体管，反正777就是最大的权限了。


### 查看磁盘占用
`df  -h`
`du -h`
https://www.runoob.com/w3cnote/linux-view-disk-space.html


### Vim
https://www.runoob.com/linux/linux-vim.html
#### vim的粘贴模式
`:set paste`
https://blog.csdn.net/wzy_1988/article/details/50264285?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task

#### vim全选，全复制，全删除
`ggvGd`
https://blog.csdn.net/ztf312/article/details/83025297


### 让程序后台运行，不随中断终端
`nohup 命令 &`

### 查看后台运行的所有程序
`ps x`
通过`grep`则可以进一步搜索具体的进程，xxx就写想查的进程中包含的字符
`ps -aux | grep xxx`

### 查看cpu实时使用情况
`top`

### tail命令
`tail` 命令可用于查看文件的内容，有一个常用的参数` -f `常用于查阅正在改变的日志文件。

`tail -f filename` 会把 filename 文件里的最尾部的内容显示在屏幕上，并且不断刷新，只要 filename 更新就可以看到最新的文件内容。
这个时候，对应目录会生成一个`nohup.out`的文件，可以在里面看到后台程序运行时的一些日志。


### 压缩/解压
- zip打包文件夹：zip -r 打包后的名字 要打包的文件/文件夹
- 解压.gz文件：(https://linuxize.com/post/how-to-unzip-gz-file/)
  `gzip -d file.gz` (The command will restore the compressed file to its original state and remove the `.gz` file.)
  `gzip -dk file.gz` (To keep the compressed file pass the `-k` option to the command:)

### 生成项目的python依赖
pip install pipreqs
使用的时候也很简单，进入项目的根目录
pipreqs ./

### 可以显示进度的文件传输
`rsync --progress 文件路径 ~/`



### Screen命令

通过`apt-get update` 和`apt-get install screen`安装

```
sudo apt-get update
sudo apt-get install screen
```



创建新screen：`screen -S screen_name`

查看所有的screen：`screen -ls`

退出当前screen（程序继续运行）：按Ctrl+a+d三个键

有时候忘记在哪里开了某个screen，导致直接 screen -r无法进入，可以使用 `screen -d name/id`来先把那个screen退出，就可以进去，放心这不会kill那个screen

关闭（kill）某个screen：进入之后，输入exit即可

重新进入某个screen：`screen -r screen_name/screen_id`

更多的一些例子参见：https://www.cnblogs.com/mchina/archive/2013/01/30/2880680.html

**这个挺好**：https://www.liquidweb.com/kb/how-to-use-the-screen-command-in-linux/

在screen模式下进行**上下滚动**：先同时按CTRL+A键，然后再按esc，就可以滚动了。要退出滚动，就再esc一下。

### ps 和 kill

查看后台程序 `ps -x`

后面还可以加一些筛选，比如查询后台所有包含 "beyond"的程序：

`ps -x | grep beyond`

kill命令，除了直接`kill [PID]`之外，还可以按照名字来kill一大批：

`pkill -f *beyond*` 就会把所有名字里包含beyond的程序都关掉。





### conda environment

创建环境（直接拷贝现有环境）：

`conda create -n your_env_name --clone some_other_env`

创建全新环境，并指定python版本：

`conda create -n your_env_name python=3.8`

删除某环境:

`conda env remove -n your_env_name`



### CUDA

check cuda version:  `nvcc --version`

运行程序时指定CUDA device：

① 在命令行指定：

`CUDA_VISIBLE_DEVICES=1 python my_script.py`

② 在程序内设定：

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
```

或者对于torch：

```python
import torch
torch.cuda.set_device(id)
```



### 文件访问权限

例如对于私钥，你要设置成只有你自己可读：

Keys need to be only readable by you:

```
chmod 400 ~/.ssh/id_rsa
```

If Keys need to be read-writable by you:

```
chmod 600 ~/.ssh/id_rsa
```

最高权限: `chmod 777`

### jupyter

发现需要密码的话，可以通过这个命令设置自己的密码，不需要之前的密码是啥：

`jupyter notebook password`



## 下载Google Drive文件

`pip install gdown`

`gdown <file_id>`

file_id的位置，一般在分享链接的这里：`https://drive.google.com/file/d/<file_id>/view?usp=sharing`



## 环境变量

https://www.baeldung.com/linux/path-variable

To append a new path, we reassign PATH with the **new path at the end**:

```bash
export PATH=$PATH:/some/new/path
```

export PATH=$PATH:/home/v-biyangguo/.local/bin

## 查看python位置

查看所有python路径：

`whereis python`

查看当前用的哪个python：

`which python`



## 打包python环境依赖

https://towardsdatascience.com/stop-using-pip-freeze-for-your-python-projects-9c37181730f9

使用`pipreqs`包:

`pip install pipreqs `

然后直接在你的project目录下执行`**pipreqs**`即可得到requirements.txt。这个比使用`pip freeze > requirements.txt`的方式更好，详情见连接。