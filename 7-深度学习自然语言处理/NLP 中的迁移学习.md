# NLP中的迁移学习

tags：笔记

---

## 1. 为何NLP中要引入迁移学习

- 大多数NLP任务都共享语言信息
- 标注数据很稀少
- 经验上，迁移学习能帮助许多NLP任务获得SOTA效果

## 2. 迁移学习的类型

![1](..\img\迁移学习\1.png)

## 3. Sequential transfer learning

Learn on one task / dataset, then transfer to another task / dataset

### 1. 为何选择语言模型作为预训练阶段的任务？

对于预训练阶段的任务来说，要求一点：数据量大且质量高。对比图像而言，图像中具有十分庞大的分类图像数据集ImageNet，而对于自然语言领域来说，缺乏大规模的有监督学习数据，这是选择语言模型作为预训练阶段的任务的重要因素。此外，分布式假设：You shall know a word by the company it keeps， 为语言模型作为预训练阶段任务奠定了理论基础。

总的来说，语言模型有三大优势：

- 不需要人类打标签
- 大多数语言都有着大量的文本内容来训练模型
- 能够学习到句子的表示与单词的表示

### 2. NLP中迁移学习演变

迁移学习最初在NLP中是以词向量的方式出现的，它将NLP带入到分布式表示时代，而词向量有一个很大的缺陷：它无法解决词的多多义性问题。于是研究人员开始研究，如何采用句子级别的向量来解决这种多义性问题。直至现在，无论是ELMO还是BERT都通过深层模型解决了这一问题，即词在上下文中向量的表示。至此，预训练语言模型真正成为主流。

在解决多义性问题上，迁移学习的演变本质上是采用深度模型来取代浅层模型的过程。

### 3. 为何语言模型表现极佳

- 对于人类来说，语言模型是一个十分困难的任务
- Language models are expected to compress any possible context into a vector that generalizes over possible
  completions.
- To have any chance at solving this task, a model is forced to learn syntax, semantics, encode facts about the world, etc.
- Given enough data, a huge model,and enough compute, can do a reasonable job!
- Empirically works better than translation, autoencoding: “Language Modeling Teaches You More Syntax than
  Translation Does” 

### 4. 数据越多，模型越大越好吗



### 5. 跨语言预训练语言模型

- 核心思想：Share vocabulary and representations across languages by training one model on many languages
- 优点：: Easy to implement, enables cross-lingual pretraining by itself
- 缺点：Leads to under-representation of low-resource languages

