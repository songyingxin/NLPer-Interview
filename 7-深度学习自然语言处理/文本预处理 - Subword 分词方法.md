# Subword ，你值得拥有

---

## 前言

 在进入预训练语言模型时代，Subword方法就已经开始大行其道了，虽然目前在英文领域应用广泛，但似乎还没有在中文领域中出现很重量级的 Subword 方法，我个人觉得可以适当探讨一下。因此，我就把最近看过的所有的预训练语言模型中所用的 Subword 方法提取出来，分析了一波，感兴趣的可以看看。

目前来看，方法大致可以分为三种，分别是 BPE[2]，WordPiece[1][4]，以及SentencePiece[3]。本文先对这三大算法进行论述，然后谈谈中文方向的分词方法发展，最后，在Github上维护一个实现仓库来帮助各位更好的理解。

## BPE [2]









## 中文分词 - BERT-wwm

其实，早在之前就有写文章谈到我个人对于中文分词的看法：[深度学习时代，分词真的有必要吗](<https://zhuanlan.zhihu.com/p/66155616>)，最近看Subword方法时，又想到中文分词的问题，于是我提了一个小问题：[预训练语言模型时代，还需要做分词吗？](<https://www.zhihu.com/question/357757060>)，希望各位大佬能够分享看法。

BERT-wwm其实从另一个角度阐述了分词的问题，其实其与百度的那篇ERNIE差不多，都是通过 mask 词而非字来实现的。具体的是，**如果词中的某个字被mask掉，那么该词需要被完全mask。**且相同的是，BERT-wwm与ERNIE都是在BERT已训练好的基础上进行再训练的，其实本质上是**词的粒度信息与字的粒度信息的融合**，而这似乎是一种很不错的方式。 而这结果再次验证了：**不同粒度的信息对于预训练语言模型的提升是有用的。**

从 Mask 这种操作来看，分词似乎已经完全没有必要了，当然，如果你想去训练一个中文预训练语言模型的话，那么词粒度的信息似乎是要被考虑进去的。

## Reference

[1] BERT，RoBERTa，UNILM：Google’s Neural Machine Translation System:Bridging the Gap Between Human and Machine Translation

[2] GPT 1.0，GPT 2,.0，MASS，XLMs：Neural machine translation of rare words with subword units.

[3] XLNet，ALBERT：Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing.

[4] Subword regularization: Improving neural network translation models with multiple subword candidates.

[5] BERT-WWM：Pre-Training with Whole Word Masking for Chinese BERT