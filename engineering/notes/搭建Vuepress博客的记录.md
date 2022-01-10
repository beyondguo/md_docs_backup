# 搭建Vuepress博客的记录



参考B站视频：

https://www.bilibili.com/video/BV1vb411m7NY?p=2&spm_id_from=pageDriver



1. 安装node.js/yarn/vscode/vewpress

![image-20220110181918945](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220110181918.png)



在进行yarn init初始化的过程中，会自动创建一个package.json文件，记录博客的基本配置

2. 跟着Vuepress教程走，创建一个demo文档：

   ![image-20220110181938404](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220110181938.png)

然后在package.json添加scripts字段的一些属性：

![截屏2021-06-24 下午12.10.07](/Users/beyond/Desktop/截屏2021-06-24 下午12.10.07.png)

接下来就可以启动了：

![image-20220110181956735](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220110181956.png)

访问localhost 8080端口：

![image-20220110182008801](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220110182008.png)



这里的scripts就是记录命令行的启动命令和对应打开的文档。



3. 给Markdown文档添加主题。使用yaml的模板

在md文件的最上面添加yaml样式：

![image-20220110182020063](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220110182020.png)

再启动，就可以看到炫酷的界面了：

![image-20220110182028514](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220110182028.png)

4. 目录结构

![image-20220110182037499](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220110182037.png)

如果在scrips中设置了docs就是根目录，那么docs下的README.md就是根路由/

接着创建一个about文件夹，分别创建上面三个文件。那么，这三个文件的路由分别是：

- /about --> about/README.md
- /about/about.html --> about/about.md
- /about/关于我.html --> about/关于我.md



5. 配置网站导航栏和logo

   ![image-20220110182046072](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220110182046.png)

   这里的图片要放在.vuepress文件夹下的public文件夹：

   ![image-20220110182054787](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220110182054.png)

   

   

   操蛋，这玩意儿也没那么容易搞清楚，尤其对我这种前端小白来说。

   

   - 如果要文章列表，就需要自己写组件（components），就涉及到一些vue语法了，使用pages属性。然后再对应页面渲染出来
- 比方，你在.vuepress/components中写了一个ListDL.vue，那么想让这个组件在某个页面生效，就到对应的页面的md里写一个 `<ListDL />`，就可以生效了。
  

  

  

导航栏配置在.vuepress/config.json中写：

![image-20220110182106728](https://gitee.com/beyond_guo/typora_pics/raw/master/typora/20220110182106.png)

   





https://softchris.github.io/pages/vue-vuepress.html#a-list-control

http://www.inode.club/webframe/tool/vuepressBlog.html



渲染数学公式：

https://juejin.cn/post/6844903764546043911

https://blog.chgtaxihe.top/pages/4f9f4f/#%E6%B8%B2%E6%9F%93%E6%95%B0%E5%AD%A6%E5%85%AC%E5%BC%8F



vue修改默认主题（继承）：

https://vuepress.vuejs.org/zh/theme/inheritance.html#%E4%BD%BF%E7%94%A8

我的需求是修改Home的基本组件，具体就是在.vuepress/theme/components文件夹下，去官网上把Home.vue拷贝一份，然后进行修改。这时就涉及到下面这个文章：



vue往HTML标签中插入参数：

https://www.cnblogs.com/smiler/p/8514395.html

