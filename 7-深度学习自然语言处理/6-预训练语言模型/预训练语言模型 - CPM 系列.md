# 预训练语言模型 - CPM 系列

## CPM 1.0

CPM 1.0 本质上是一个中文版的 GPT 1.0，因此这里就不详细谈设计思路了，这里只聊一聊与GPT 1.0 不同的地方。

### 唯一创新点：Vocabulary Construction

BERT 的中文版本中采用的是 char-level 来做词向量表征， 但在中文中往往采用word 来表示词义，如何在预训练中融合 char-level 与 word-level 的信息是本文的创新点。

文中建立了一个包含 words 与 characters 的词表，但是没有说在具体的分词方式与模型输入处理上是怎么处理的。

总的来说，这篇文章价值一般，没有看的必要了。

## CPM 2.0

### 1. 模型架构

模型的整体架构遵循 双向Transformer encoder - 单向 Transformer decoder，文中针对 encoder-decoder 设计了特殊的mask机制。

思想：encoder中用特殊字符随机替换 几个 spans，然后在decoder中预测这些spans

```
input: These are [X] which [Y] may seek to address
output: [X]    issues [Y] future studies [Z]    
```

其中，[X]，[Y]是被 mask 的spans，[Z]表示 output 的结尾。

15% 的token 被mask

平均span长度为10

### 2. 数据预处理

数据： 2.3TB的中文数据，300GB的英文数据。

优化 sentencepiece ：

### 3. 知识继承

Chinese Stage：采用中文语料预训练

Bilingual Stage：采用中英文语料混合训练

### 4. Prompt Tuning





## Reference

[1] CPM: A Large-scale Generative Chinese Pre-trained Language Model

[2] CPM-2: Large-scale Cost-effective Pre-trained Language Models