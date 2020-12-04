# SIFRank: A New Baseline for Unsupervised Keyphrase Extraction Based on Pre-trained Language Model

## Abstract











## Model Overview



步骤：

- 首先先分词，并标注词性
- 采用 NP-chunker（用正则编写） 提取句子中的 NPs 
- 用预训练语言模型获取每个 token 的表示
- 分别将document 与 NPs 表示成向量
- 计算NPs embedding 与 document的 cos距离， 选择topk



### SIFRank

- document d 的embedding 为$v_d$
- 候选NP的embedding 为 $v_{NP}$

$$
SIFRank(v_{NP_i}, v_d) = Sim(v_{NP_i}, v_d) = cos(v_{NP_i}, v_d) = \frac{\overrightarrow{v_{NP_i}} \cdot \overrightarrow{v_d} }{||\overrightarrow{v_{NP_i}}||\,||\overrightarrow{v_d}||}
$$





