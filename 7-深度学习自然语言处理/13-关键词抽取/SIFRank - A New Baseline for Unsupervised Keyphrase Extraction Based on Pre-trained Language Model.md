# SIFRank: A New Baseline for Unsupervised Keyphrase Extraction Based on Pre-trained Language Model

## Abstract



## Model Overview

### Overall Structure

- 首先先分词，并标注词性
- 采用 NP-chunker（用正则编写） 提取句子中的 NPs 
- 用预训练语言模型获取每个 token 的表示
- 分别将document 与 NPs 表示成向量
- 计算NPs embedding 与 document的 cos距离， 选择topk



### SIF

通过SIF 来分别获得 NPs 与 document 的 embedding 。

引入了两个平滑项，来解释（1）有些词是在上下文之外出现（2）某些高频词如『the』是没有语境限制的：

- $\alpha f_w$：$\alpha$ 是标量， $p(w)$ 是整个语料库中单词 $w$ 的词频。


$$
Pr(s|c_d) = \prod_{w \in s} Pr(w|c_d) = \prod_{w \in s} \alpha f_w + (1-\alpha) \frac{exp(<v_w, \widetilde{c}_d>)}{Z_{\widetilde{c}_d}} \\
Z_{\widetilde{c}_d} = \sum_{w \in V} exp(<\widetilde{c}_d, v_w>) \\
\widetilde{c}_d = \beta c_0 + (1-\beta)c_d, c_0 \bot c_d \\
$$

### SIFRank

- document d 的embedding 为$v_d$
- 候选NP的embedding 为 $v_{NP}$

$$
SIFRank(v_{NP_i}, v_d) = Sim(v_{NP_i}, v_d) = cos(v_{NP_i}, v_d) = \frac{\overrightarrow{v_{NP_i}} \cdot \overrightarrow{v_d} }{||\overrightarrow{v_{NP_i}}||\,||\overrightarrow{v_d}||}
$$



