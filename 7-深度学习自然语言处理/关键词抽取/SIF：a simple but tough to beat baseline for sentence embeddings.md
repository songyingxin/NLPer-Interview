# SIF：a simple but tough to beat baseline for sentence embeddings

https://zhuanlan.zhihu.com/p/44534561

https://zhuanlan.zhihu.com/p/111710604

https://blog.csdn.net/qq_42491242/article/details/105381771

核心思想：

- **词频加权：** 对组成句子的词进行线性加权，其中每个词的权重为 $\alpha/(\alpha + p_w)$ ，其中$p_w$ 是词频， $\alpha$ 是超参数。
- **语义无关向量去除：**句子向量的生成要去除文本中与语义无关的向量。具体做法是从数据集中抽样一些句子，然后计算这些句子向量对应的最大的奇异值向量，这样的向量被认为代表了文本中的语法或停用词这些和语义无关的内容。将这些向量去除可以增强文本对语义本身的表达能力。



## 模型解释

在潜变量生成模型(latent variable generative model)中，语料的生成是一个动态过程，也就是说每个时刻生成一个单词。这个动态过程是由一个话语向量 $c_t$ (discourse vector)的随机游走来控制的。这里的话语向量代表着当前句子正在表达的意思(what is being talked about)代表着句子的状态或者一种潜在的语义信息。而单词的词向量 $v_w$ 和 $c_t$ 的内积则表示了单词和句子的联系。



两个问题：

- 有些在单词上下文之外的词对句向量也是有影响的
- 有些频繁出现的高频词(和，的，and，the等)是对话语本身没有贡献的