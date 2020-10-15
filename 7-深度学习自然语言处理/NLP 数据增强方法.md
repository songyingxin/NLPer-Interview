# NLP 数据增强方法

https://zhuanlan.zhihu.com/p/75207641

https://www.qbitai.com/2020/06/16103.html

https://www.dataapplab.com/enhance-nlp-what-are-the-easiest-use-augmentation-techniques/

## 文本替代

- 同义词替代：
- 词嵌入替换： 采用嵌入空间中最近的邻接词作为句子中某些单词的替换
- 掩码语言模型：通过bert等这种MLM模型来预测被mask掉的词来做替换，需要注意的是决定哪一个单词被mask是比较重要的