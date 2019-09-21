# 奈何本人没文化，一句卧槽走天下

tags: 博客文章

---

## 前言

又又又屠榜了，一开始，我以为只是噱头，直到鄙人打开了 RACE 排行榜：

![](http://ww1.sinaimg.cn/large/006gOeiSly1g47usj10xmj30qq0bvaaz.jpg)

我近几个月来一直希望能够通过设计一个适合 RACE 数据集的 Attention 机制融合长距离文本来实现 **终极反杀**，然而，看到这一幕，我对 Attention 的必要性产生了严重的怀疑。

虽然说我们依旧可以在 XLNet 上加上 Attention 来实现超越 XLNet 本身的效果，但这真的有意义吗，或许再过几个月，就又会有屠榜的消息传来， 那么设计精巧的 Attention 机制又被 **大力出奇迹** 的模型按在地上摩擦的时候，意义何在？ 

我觉得我近期可能要深入思考一下： **Attention 的存在意义**，有大佬理解深刻的话，希望能指点我一下。

## XLNet 为何如此之屌？

XLNet 在多个任务上超出了 Bert 许多，我们先来看看 XLNet 是怎么做的，然后分析相对于 Bert 而言， XLNet 为何会如此优秀。

推荐张俊林大佬的讲解，真的是深入浅出： [XLNet:运行机制及和Bert的异同比较](https://zhuanlan.zhihu.com/p/70257427)

**1， AR LM **

AR 即 AutoRegressive language model， 其代表了一类从上文内容预测之后下一个词的语言模型，可以分为两种，一种是从左到右的语言模型(left-to-right)， 一种是从右到左的语言模型(right-to-left)。 语言模型的公式分别如下：
$$
\text{left to right: } p(x) = \prod_{t=1}^T p(x_t | x_{<t}) \\
\text{right to left: } p(x) = \prod_{t=T}^1 p(x_t | x_{>t}) 
$$
结合之前的模型， GPT 就是一个典型的 left-to-right 的语言模型， ELMO 是 left-to-right + right-to-left 的结合体。

AR 模型的缺点很明显：**无法同时利用上下文信息**， ELMO 的这种简单拼接的方式的确有点粗糙，效果也不是很好，但的确目前没有预训练模型采用 ELMO 这种方式，但估计效果并不会比 Bert 更好。而针对具体的任务尤其如阅读理解而言， AR 模型变现不佳也就在意料之中了。

AR 模型的优点很重要：**对自然语言生成任务很友好。** 这一点从 GPT 2.0 与 Bert 的对比就可以看出， GPT 2.0 在生成任务上的表现可以说惊艳， 因为生成语言的过程就是从左到右的， AR 模型天然的符合这个过程。

**2， AE LM**

AE 即 AutoEncoding Language Model， 其代表一类从被 mask 的输入中来学习上下文信息的方式， Bert 就是其中的典型代表，就我理解来看，其目标函数大概长这样：
$$
p(x) = \prod_{i=1}^{mask \, num} p(x_i | X) \quad x_i \in mask
$$
目前很多改进的文章都是按照这一套路来的，比如百度的 ERNIE ， 微软的 MASS等。

Bert 文章中已经提到， 这种 AE 模型的最大缺点是： **由于引入了 [MASK] 造成的预训练阶段与 fine-tuning 阶段不一致的问题。** 而 Bert 还特意提出一个 Trick 来减轻这个问题（参见：[Bert 改进： 如何融入知识](https://zhuanlan.zhihu.com/p/69941989)。  且另一方面， 这种 AE 模型对于生成并不友好， MASS 一定程度上减轻了这一问题，但我认为依旧存在改进空间。

另外，文章还提到了 Bert 会忽略 **被 mask 的各个 token 之间的相关性**，额，，那为啥不像 Word2Vec 的 CBOW 那样一句话只 mask 一个token， 这样不就解决了吗？ 当然，这样做训练起来效率会很低，的确有点得不偿失。

而AE 模型的最大优点也很清楚了： **能够很好的融入上下文信息。** 这也是为啥 bert 能够在自然语言理解问题上实现巨大突破的根本原因。

**3， 如何 AR + AE ？ **

XLNet 出发点就是结合 AR 与 AR ， 从而实现二者的互补，从结果看，它做的很棒， 我们先抠模块细节，然后分析为什么这么做就能够避免某种问题。

**4， 预训练模型： PLM**

PLM 全称为 Permutation Language Modeling， 通过 PLM， 模型不仅能够保持 AR 模型的优势， 又能够捕捉到双向的上下文信息， 其核心思想如下：



## Reference

[1]  XLNet: Generalized Autoregressive Pretraining for Language Understanding

[2]  Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

[3]  [XLNet:运行机制及和Bert的异同比较](https://zhuanlan.zhihu.com/p/70257427)

