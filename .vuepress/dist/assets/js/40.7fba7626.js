(window.webpackJsonp=window.webpackJsonp||[]).push([[40],{407:function(t,s,e){"use strict";e.r(s);var n=e(44),a=Object(n.a)({},(function(){var t=this,s=t.$createElement,e=t._self._c||s;return e("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[e("h1",{attrs:{id:"把python程序打包成docker镜像"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#把python程序打包成docker镜像"}},[t._v("#")]),t._v(" 把Python程序打包成Docker镜像")]),t._v(" "),e("h2",{attrs:{id:"前提"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#前提"}},[t._v("#")]),t._v(" 前提：")]),t._v(" "),e("ol",[e("li",[t._v("母机上已安装anaconda")]),t._v(" "),e("li",[t._v("了解基本的Docker概念、最基本的命令")])]),t._v(" "),e("p",[t._v("本文中涉及到的主要命令：")]),t._v(" "),e("ul",[e("li",[e("code",[t._v("docker pull")]),t._v(" ：从docker hub拉取某个镜像")]),t._v(" "),e("li",[e("code",[t._v("docker image ls")]),t._v("：查看当前系统中的所有镜像")]),t._v(" "),e("li",[e("code",[t._v("docker build")]),t._v("：根据Dockerfile创建一个镜像")]),t._v(" "),e("li",[e("code",[t._v("docker run")]),t._v("：启动某个镜像，运行一个容器")]),t._v(" "),e("li",[e("code",[t._v("docker ps")]),t._v("：查看当前系统中所有运行中的容器")]),t._v(" "),e("li",[e("code",[t._v("docker ps -a")]),t._v("：查看所有容器，不管是否运行")])]),t._v(" "),e("hr"),t._v(" "),e("h1",{attrs:{id:"_1-母机上创建python虚拟环境"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#_1-母机上创建python虚拟环境"}},[t._v("#")]),t._v(" 1. 母机上创建python虚拟环境")]),t._v(" "),e("p",[t._v("创建：\n"),e("code",[t._v("conda create -n your_project_name python=3.6")]),t._v("\n这样会创建一个只有pyhton3.6的环境。")]),t._v(" "),e("p",[t._v("进入虚拟环境：\n"),e("code",[t._v("conda activate your_project_name")])]),t._v(" "),e("p",[e("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616128547-image.png",alt:""}})]),t._v(" "),e("h1",{attrs:{id:"_2-在虚拟环境中-部署自己的程序和相关文件"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#_2-在虚拟环境中-部署自己的程序和相关文件"}},[t._v("#")]),t._v(" 2. 在虚拟环境中，部署自己的程序和相关文件")]),t._v(" "),e("p",[t._v("安装python依赖，"),e("strong",[t._v("记得记录下来安装了那些包")]),t._v("：")]),t._v(" "),e("p",[e("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616135761-image.png",alt:""}})]),t._v(" "),e("p",[t._v("上传自己的代码、文件，"),e("strong",[t._v("确保在当前环境下可以正常运行（记住一定要在自己的虚拟环境中）")]),t._v("。")]),t._v(" "),e("h1",{attrs:{id:"_3-编写dockerfile"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#_3-编写dockerfile"}},[t._v("#")]),t._v(" 3.编写dockerfile")]),t._v(" "),e("p",[t._v("先安装一个基础的镜像，如python3.6环境\n"),e("code",[t._v("docker pull silverlogic/python3.6")])]),t._v(" "),e("p",[t._v("直接pull可能会比较慢，可以通过阿里云镜像加速：")]),t._v(" "),e("p",[e("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616144570-image.png",alt:""}})]),t._v(" "),e("p",[t._v("直接在命令行复制图中的命令即可。")]),t._v(" "),e("p",[e("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616165196-image.png",alt:""}})]),t._v(" "),e("p",[e("strong",[t._v("然后就开始编写Dockerfile了！")]),t._v("\n首先我以我的项目为例，来明确几个概念：\n项目根目录： "),e("code",[t._v("\\trp_service\\")]),t._v("\n在根目录下，我的文件有这些：")]),t._v(" "),e("p",[e("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616174763-image.png",alt:""}}),t._v("\n257d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)\n其中，start.sh是我的启动文件，通过"),e("code",[t._v("sh start.sh")]),t._v("即可运行我的服务trp_service.py\n这个python程序启动后，会生成一个api，提供词向量计算服务。内部的api端口是9000.")]),t._v(" "),e("p",[t._v("接下来我们编写Dockerfile文件：\n在项目的根目录 "),e("code",[t._v("\\trp_service\\")]),t._v("里，我创建了一个文件，名为"),e("code",[t._v("Dockerfile")]),t._v("，"),e("strong",[t._v("注意只能是这个名字")]),t._v("。然后我们进入Dockerfile文件，按照如下内容编写：")]),t._v(" "),e("div",{staticClass:"language-shell line-numbers-mode"},[e("pre",{pre:!0,attrs:{class:"language-shell"}},[e("code",[t._v("FROM silverlogic/python3.6\nMAINTAINER gby\n\nENV BUILD_HOME /trp_service\n\nRUN pip "),e("span",{pre:!0,attrs:{class:"token function"}},[t._v("install")]),t._v(" numpy -i https://pypi.tuna.tsinghua.edu.cn/simple/"),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("\\")]),t._v("\n        "),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("..")]),t._v(". "),e("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("..")]),t._v(".\n\tpip "),e("span",{pre:!0,attrs:{class:"token function"}},[t._v("install")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token assign-left variable"}},[t._v("bert4keras")]),e("span",{pre:!0,attrs:{class:"token operator"}},[t._v("==")]),e("span",{pre:!0,attrs:{class:"token number"}},[t._v("0.8")]),t._v(".4 -i https://pypi.tuna.tsinghua.edu.cn/simple/\n\nCOPY "),e("span",{pre:!0,attrs:{class:"token builtin class-name"}},[t._v(".")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token variable"}},[t._v("$BUILD_HOME")]),t._v("\nRUN "),e("span",{pre:!0,attrs:{class:"token function"}},[t._v("chmod")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token number"}},[t._v("777")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token variable"}},[t._v("$BUILD_HOME")]),t._v("/start.sh\n\nWORKDIR "),e("span",{pre:!0,attrs:{class:"token variable"}},[t._v("$BUILD_HOME")]),t._v("\n\nCMD "),e("span",{pre:!0,attrs:{class:"token variable"}},[t._v("$BUILD_HOME")]),t._v("/start.sh\n")])]),t._v(" "),e("div",{staticClass:"line-numbers-wrapper"},[e("span",{staticClass:"line-number"},[t._v("1")]),e("br"),e("span",{staticClass:"line-number"},[t._v("2")]),e("br"),e("span",{staticClass:"line-number"},[t._v("3")]),e("br"),e("span",{staticClass:"line-number"},[t._v("4")]),e("br"),e("span",{staticClass:"line-number"},[t._v("5")]),e("br"),e("span",{staticClass:"line-number"},[t._v("6")]),e("br"),e("span",{staticClass:"line-number"},[t._v("7")]),e("br"),e("span",{staticClass:"line-number"},[t._v("8")]),e("br"),e("span",{staticClass:"line-number"},[t._v("9")]),e("br"),e("span",{staticClass:"line-number"},[t._v("10")]),e("br"),e("span",{staticClass:"line-number"},[t._v("11")]),e("br"),e("span",{staticClass:"line-number"},[t._v("12")]),e("br"),e("span",{staticClass:"line-number"},[t._v("13")]),e("br"),e("span",{staticClass:"line-number"},[t._v("14")]),e("br"),e("span",{staticClass:"line-number"},[t._v("15")]),e("br")])]),e("p",[e("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616184842-image.png",alt:""}})]),t._v(" "),e("h1",{attrs:{id:"_4-开始创建我的docker镜像"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#_4-开始创建我的docker镜像"}},[t._v("#")]),t._v(" 4. 开始创建我的Docker镜像")]),t._v(" "),e("p",[t._v("在当前的项目目录中，执行如下命令：\n"),e("code",[t._v("docker build -t trp:v1 .")])]),t._v(" "),e("ul",[e("li",[e("code",[t._v("-t")]),t._v("参数后，输入"),e("code",[t._v("名字：版本")])]),t._v(" "),e("li",[t._v("注意命令最后有一个 "),e("code",[t._v(".")]),t._v(" ，这是指在当前的目录下去寻找Dockerfile文件。")])]),t._v(" "),e("p",[e("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616193016-image.png",alt:"build成功"}})]),t._v(" "),e("p",[t._v("创建之后，可以通过"),e("code",[t._v("docker image ls")]),t._v("查看系统镜像，发现trp已经创建好了：")]),t._v(" "),e("p",[e("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616205864-image.png",alt:""}})]),t._v(" "),e("h1",{attrs:{id:"_5-启动镜像-生成容器-调用容器内的服务"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#_5-启动镜像-生成容器-调用容器内的服务"}},[t._v("#")]),t._v(" 5.启动镜像，生成容器，调用容器内的服务")]),t._v(" "),e("p",[t._v("通过命令：\n"),e("code",[t._v("docker run -itd -p 9000:9000 trp:v1")]),t._v("\n即可启动服务。\n此时通过"),e("code",[t._v("docker ps")]),t._v("查看当前运行的容器：")]),t._v(" "),e("p",[e("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616216662-image.png",alt:""}})]),t._v(" "),e("p",[t._v("能看到，就说明容器已经启动成功。")]),t._v(" "),e("p",[t._v("在命令中，我通过-p来设置宿主机和容器内的端口映射。故现在我在宿主机，也可以通过9000端口来访问我容器内的服务了：")]),t._v(" "),e("p",[e("img",{attrs:{src:"https://cdn.jsdelivr.net/gh/beyondguo/mdnice_pictures/2021-6-25/1624616228668-image.png",alt:""}})])])}),[],!1,null,null,null);s.default=a.exports}}]);