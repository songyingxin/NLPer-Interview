# BERT 可解释性

## BERT 向量 vs Glove 向量[4]

这篇文章对比了 BERT ，Glove，random 三种向量， 我们都知道， BERT 相对于其他两种向量，其效果提升非常明显，本文基于此，探讨与传统词向量相比，BERT向量优异在何处呢？

 为了对比不同词向量的在下游任务的表现，本文采用了三个任务： 

- NER： 词汇级别的任务
- sentiment analysis：句子级别的任务
- GLUE：句子对级别的任务

为了更加纯粹对比三种向量，三种词向量在训练时均不微调，如果微调的话，就会难以判断是模型的作用还是词向量本身的作用。

### 1.  数据规模的影响

实验表明，下游任务的训练数据对于不同的向量影响是十分不同的， 结果如下图所示：

![屏幕快照 2020-08-04 下午8.36.28.png](image/006gOeiSly1ghf294k7rcj30qw0k2tc6.jpg)

从上图中我们可以看出

- 随着数据规模的扩大，Glove 向量的表现与 BERT 向量的表现差距越来越小，但在绝大多数情况下依旧比BERT 向量差很多，这说明 BERT 对于小数据集的优越性。

- 在简单任务上，随着数据量的增加， Glove 能达到 BERT 十分接近的效果

### 2. 语言特性

本节主要分析，相对于 Glove ，BERT 向量主要在哪些方面获得了提升。本节采用 GLUE diagnostic task，该任务主要从四个方面考察语言特性：

- lexical semantics (LS)
- predicate-argument structure (PAS)
- logic (L)
- knowledge and common sense (KCS)

BERT，Glove，Random 向量的结果对比如下：

![屏幕快照 2020-08-04 下午9.17.53.png](image/006gOeiSly1ghf3d4qgg0j30cn064t92.jpg)

从上表中得出如下结论：

- Glove 与 Random 在四个语言特性上表现相似
- 在 PAS 上，BERT 明显好于 其余两个向量。 PAS 主要评测模型是否了解句子的短语是如何组合到一起的，比如：介词短语， 识别主语和宾语

接下来， 文章从三个角度来评测不同的向量：

- **the complexity of text structure：**句子结构的复杂性
- **Ambiguity in word usage**: 单词的歧义性。
- **Prevalence of unseen words**：未登录词出现的概率

![bert vs random](image/006gOeiSly1ghgbkmpmgfj30fb0753z7.jpg)

![bert vs glove](image/006gOeiSly1ghgwbnmicdj30in099gmn.jpg)

从结果来看，以 BERT 为代表的 Contextual embeddings 在解决一些文本结构复杂度高和单词歧义性方面有显著的效果，但是在未登录词方面 GloVe 代表的Non-Contextual embeddings 有不错的效果。

从上面的结论可以看出，

- 在对于拥有大量训练数据和简单任务中，考虑算力和设备等，GloVe 代表的 Non-Contextual embeddings 是个不错的选择。
- 但是对于文本复杂度高和单词语义歧义比较大的任务，BERT代表的 Contextual embeddings 却有明显的优势。

https://zhuanlan.zhihu.com/p/145695511

## BERT vs ELMO vs Flair [6]

本文主要探讨这三种上下文向量在 WSD（word sense disambiguation）上的表现，进而判断上下文向量能否解决 WSD 问题。



![屏幕快照 2020-08-06 上午11.14.15.png](image/006gOeiSly1ghgx5p4aitj30fa07bwf6.jpg)

|      |       |       |       |       |
| ---- | ----- | ----- | ----- | ----- |
| MFS  | 54.79 | 58.95 | 70.94 | 48.44 |

从上表中看出，BERT 要远高于baseline 模型，且要比其他上下文向量表现更佳。

![屏幕快照 2020-08-06 上午11.31.21.png](image/006gOeiSly1ghgxnodew6j30u00e9wj4.jpg)

从上图中我们得出如下结论：

- The Flair embeddings hardly allow to dis- tinguish any clusters as most senses are scattered across the entire plot.
- In the ELMo embedding space, the major senses are slightly more separated in different regions of the point cloud
- Only in the BERT embedding space, some senses form clearly separable clusters



**BERT 的不足之处**

看 Error analysis



## BERT vs ELMO vs GPT-2

https://zhuanlan.zhihu.com/p/106892122





## BERT 各层向量学到了什么？[1]

https://zhuanlan.zhihu.com/p/74515580

本文分析：

- We first show that the lower layers capture phrase-level information, which gets diluted in the upper layers. 
-  BERT captures a rich hierarchy of linguistic information, with sur- face features in lower layers, syntactic features in middle layers and semantic features in higher layers.
- we test the ability of BERT representa- tions to track subject-verb agreement and find that BERT requires deeper layers for handling harder cases involving long-distance dependencies.
- to explore different hypothe- ses about the compositional nature of BERT’s rep- resentation and find that BERT implicitly captures classical, tree-like structures.



### 1. Phrasal Syntax

这部分主要验证 BERT 能否扑捉到短语级别的结构信息。 采用将短语片段在 l 层的隐层向量进行 concatenating，得到该短语的向量表示。

![屏幕快照 2020-08-06 下午1.35.04.png](image/006gOeiSly1ghh183ecndj30s30csq5r.jpg)



从上图中可以得到：

- BERT mostly captures phrase-level information in the lower layers 
- Phrase-level information gets gradually diluted in higher layers. 

### 2. 探测任务

https://zhuanlan.zhihu.com/p/149730830





https://zhuanlan.zhihu.com/p/63746935











## BERT Attention 学到了什么

https://zhuanlan.zhihu.com/p/148729018



## Reference

[1] What does BERT learn about the structure of language?  -- 2019-6-4

[2] What Does BERT Look At? An Analysis of BERT's Attention -- 2019-6-11

[3] A multiscale visualization of attention in the transformer model  --2019-6-12

[4] Contextual Embeddings: When Are They Worth It?  -- 2020-3-18

[5] A Primer in BERTology: What we know about how BERT works --2020-2-27

[6] Does BERT make any sense? interpretable word sense disambiguation with contextualized embeddings

 https://www.jiqizhixin.com/articles/2019-09-09-6

https://zhuanlan.zhihu.com/p/148729018

https://zhuanlan.zhihu.com/p/74515580

https://lsc417.com/2020/06/19/paper-reading3/#prevalence-of-unseen-words

https://zhuanlan.zhihu.com/p/145695511



https://zhuanlan.zhihu.com/p/110085059
