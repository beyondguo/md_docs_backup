---
title: 【随时记录】Git笔记
published: 2021-6-24
sidebar: auto
---

# 【随时记录】Git笔记

## 最基本

初始化一个Git仓库，使用`git init`命令。
**添加文件**到Git仓库，分两步：

- 使用命令`git add <file>`，注意，可反复多次使用，添加多个文件；
- 使用命令`git commit -m <message>`，完成。

**状态查看：**
- 要随时掌握工作区的状态，使用`git status`命令。
- 如果`git status`告诉你有文件被修改过，用`git diff`可以查看修改内容。

## **版本回退：**

- `HEAD`指向的版本就是当前版本，因此，Git允许我们在版本的历史之间穿梭，使用命令`git reset --hard commit_id`。
- 穿梭前，用`git log`可以查看提交历史，以便确定要回退到哪个版本。
- 要重返未来，用`git reflog`查看命令历史，以便确定要回到未来的哪个版本。


![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624615687282-image.png)


>Git跟踪并管理的是`修改`，而非`文件`。
每次修改，如果不用`git add`到暂存区，那就不会加入到`commit`中。


`git dif`f 是只比较比较工作区和暂存区（最后一次add）的区别，`git diff --cached `是只比较暂存区和版本库的区别，`git diff HEAD -- filename` 是只比较工作区和版本库（最后一次commit）的区别。

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624615698020-image.png)


**撤销修改：**
- 场景1：当你改乱了工作区某个文件的内容，想直接丢弃工作区的修改时，用命令`git checkout -- file`。
- 场景2：当你不但改乱了工作区某个文件的内容，还添加到了暂存区时，想丢弃修改，分两步，第一步用命令`git reset HEAD <file>`，就回到了场景1，第二步按场景1操作。
- 场景3：已经提交了不合适的修改到版本库时，想要撤销本次提交，参考[版本回退](https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000/0013744142037508cf42e51debf49668810645e02887691000)一节，不过前提是没有推送到远程库。

## **删除文件：**

当你要删除文件的时候，可以采用命令：  rm test.txt  
这个时候（也就是说这个时候只执行了  rm test.txt  ）有两种情况:
**第一种情况**:
的确要把test.txt删掉，那么可以执行
                  ` git rm test.txt`
                   `git commit -m "remove test.txt"`
                   然后文件就被删掉了
**第二种情况**:
删错文件了，不应该删test.txt，注意这时只执行了`rm test.txt`，还没有提交，所以可以执行`git checkout test.txt`将文件恢复。

并不是说执行完`git commit -m "remove test.txt"`后还能用`checkout`恢复，`commit`之后版本库里的文件也没了，自然没办法用`checkout`恢复，而是要用其他的办法.

## **添加远程库关联：**
`$ git remote add origin git@github.com:beyondguo/JD_CV_Match.git`
后面git@github.com:beyondguo/JD_CV_Match.git从GitHub上复制。

然后用
`git add .`来添加本地目录的所有文件。再`git commit -m "some comments"`提交一下。
最后，通过`git push -u origin master`把文件全部推送到远程库。`-u`在第一次推送的时候添加，之后就不用了。
如果需要把远程库的东西同步到本地，用`git pull origin master`。

当你本地修改了文件，放心地把所有文件都add，一次性commit，一次性push，git只会提交你的修改，所有不用担心每次都全部重新上传。而且你在commit中写的comment，也只会在修改的文件的备注里显示。

## 报错

>fatal: remote origin already exists.

解决方法：把远程库给删了：

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624615711843-image.png)


>hint: Updates were rejected because the remote contains work that you do
hint: not have locally. 

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624615719883-image.png)


解决方法：一般是由于远程库里面有你本地库没有的，比如你创建的时候多了一个readme文件，但是本地没有。

![](https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624615727956-image.png)



## 使用本地强制覆盖远程仓库

git push origin 分支名 --force



## 添加多个远程仓库

`git remote add [name] 地址`

原来之前的`git add origin 地址`以及`git push origin master`中的orgin都是一个名称，默认的初始仓库都是origin。所以新建的远程仓库，需要换个别的名字，比如lab，然后push的时候也使用新的名字，比如：

`git remote add git@github.com:SUFE-AILAB/STA.git`

`git push lab master`

可以通过`git remote -v`来查看远程仓库的情况。



## 多人/多设备协作

首先一个地方（人、设备）建立仓库。确保两个地方都添加了ssh key。

另一个地方直接git clone下来，然后就可以直接进行修改，然后git add-commit-push了，不会有冲突。

在一个地方修改并push之后，另一个地方在修改前，必须先进行git pull。

几个细节：

1. git clone之后，不需要git remote add远程仓库，可以直接修改提交。
2. git add *似乎只会提交可见文件，所以用它的话，就不会把 .ipynb_checkpoints 和 .gitignore 还有 .DS_store这些文件同步。而使用git add -A则是把所有文件包括隐藏文件都同步。一般.gitignore我还是想同步的，所以就需要在.gitignore里面把我们不想上传的隐藏文件都添加进来，然后再使用git add -A
3. git add -A 跟 git add * 还有个不同在于，如果我对文件夹或者文件进行的位置的移动，或者删除了某个文件。用 * 的话，原本的文件是不会被删掉的，相当于只是复制了一份到新地址了，但是 -A 的话，得到的文件目录就跟你本地是一样的了。



## 关于ssh key

ssh key的生成：

`ssh-keygen -t rsa`

会告诉你默认地址，去那里复制，也可以直接通过`vim ~/.ssh/id_rsa.pub`命令查看，然后贴到github。**注意：每次运行，生成的都是不一样的！**

同一个pub key，只能放在一个账号里，所以同一个设备如果pub key希望对应多个账号，就需要多生成pub key。

## Git lfs大文件

先得作如下的准备，才能进行lfs下载：

```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash

sudo apt-get install git-lfs
```



---

# 常见错误



> error: The following untracked working tree files would be overwritten by merge:

The error above is often triggered when we do not clone the repository we are trying to pull from. The projects may be identical, but we may be working on one locally while trying to pull it from the repo on Github because it may have other files or features we’d like to incorporate on our local version.

https://careerkarma.com/blog/error-the-following-untracked-working-tree-files-would-be-overwritten-by-merge/

> fatal: refusing to merge unrelated histories

https://itsmycode.com/fatal-refusing-to-merge-unrelated-histories-solved/



force overwrite local branch:

https://stackoverflow.com/questions/1125968/how-do-i-force-git-pull-to-overwrite-local-files



使用vscode进行commit的时候，会出现一个傻逼问题，明明写了comment，却还是显示：

> Aborting commit due to empty commit message.

这个时候，可以执行：`git config --global core.editor "code -w"`

然后直接`git commit`

这时，会弹出一个新窗口，让你写comment，写完之后，保存，关闭，vscode就会帮你进行commit