module.exports = {
  title: 'SimpleAI',
  description: '来了，就别空着手走！',
  themeConfig: {
    logo: '/assets/img/logo_no_words.png',
    nav: [
      { text: 'Home', link: '/' },
      { text: '深度学习笔记', link: '/dl_basis/' },
      { text: 'NLP笔记', link: '/nlp_basis/' },
      { text: '吃点儿论文', link: '/paper_notes/' },
      { text: '工程', link: '/engineering/'},
      { text: '随笔', link: '/opinions/' },
      { text: 'Me', link: '/about/' },
      {
        text: '在别处~',
        items: [
          { text: 'Github', link: 'https://github.com/beyondguo' },
          { text: '微信公众号「SimpleAI」', link: 'https://mp.weixin.qq.com/s/v35g-p7wK2MkuM-SqjkF3g' },
          { text: '知乎「蝈蝈」', link: 'https://www.zhihu.com/people/guo-bi-yang-78' },
          { text: '简书', link: 'https://www.jianshu.com/u/f4fe92da869c' }
        ]
      }
    ]
  },
  markdown: {
    lineNumbers: true, // 代码行号
    extendMarkdown: md => {
            md.set({
                html: true
            })
            md.use(require('@neilsustc/markdown-it-katex'), {"throwOnError" : false, "errorColor" : " #cc0000"})
        }
  },
  head: [
        ['link', {rel: 'stylesheet',href: 'https://cdn.jsdelivr.net/npm/katex@0.10.0-alpha/dist/katex.min.css'}],
        ['link', {rel: "stylesheet",href: "https://cdn.jsdelivr.net/github-markdown-css/2.2.1/github-markdown.css"}],
        ['link', { rel: "icon", type: "image/png", sizes: "32x32", href: "/assets/img/logo_no_words.png"}],
        [
    'script', {}, `
    var _hmt = _hmt || [];
(function() {
  var hm = document.createElement("script");
  hm.src = "https://hm.baidu.com/hm.js?5aca48f844181444aea941eb9d707584";
  var s = document.getElementsByTagName("script")[0]; 
  s.parentNode.insertBefore(hm, s);
})();
    `
  ]
    ]
}