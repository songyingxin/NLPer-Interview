# 词向量 - Word2Vec

---

## 什么是Word2Vec[3]？

Word2Vec是Google发布的一个工具， 用于训练词向量，其提供了两种语言模型来供选择， 且Google 基于大规模语料集上训练出了预训练词向量来供开发者或研究者使用。 一般情况下，我们是没有必要自己去训练词向量的，但如果要求特殊，且语料集庞大，自己训练也是可以的。

在Word2Vec中，实现了两个模型：**CBOW** 与 **Skip-Gram**。

## CBOW模型

CBOW，全称Continuous Bag-of-Word，中文叫做连续词袋模型：**以上下文来预测当前词** $w_t$ 。



![img](https://pic4.zhimg.com/v2-27f3e577618f84c0026968d273d823ef_b.jpg)

如上图是一个两层的神经网络，其实在训练语言模型的过程中考虑到效率等问题，常常采用浅层的神经网络来训练，并取第一层的参数如上图就是 $W_{V \times N}$ 来作为最终的词向量矩阵（参考 [语言模型：从n元模型到NNLM](https://zhuanlan.zhihu.com/p/43453548)）。

CBOW模型的目的是预测 $P(w_t| w_{t-k}, \cdots, w_{t-1}, w_{t+1}, \cdots, w_{t+k}) $，我们先来走一遍CBOW的前向传播过程 。

### 1. 前向传播过程

- 输入层: 输入C个单词： $x_{1k}, \cdots, x_{Ck} $，并且每个 $x$ 都是用 One-hot 编码表示，每一个 $x$ 的维度为 V（词表长度）。

- 输入层到隐层:  共享矩阵为 $W_{V \times N}$ ，V表示词表长度，W的每一行表示的就是一个N维的向量（训练结束后，W的每一行就表示一个词的词向量）。在隐藏层中，我们的所有输入的词转化为对应词向量，然后取平均值，这样我们就得到了隐层输出值 ( 注意，隐层中无激活函数，也就是说这里是线性组合)。 其中，隐层输出 ![h](https://www.zhihu.com/equation?tex=h)h 是一个N维的向量 。
  $$
   h = \frac{1}{C} W^T(x_1 + x_2 + \cdots + x_c) 
  $$

- 隐层到输出层：隐层的输出为N维向量 $h$ ， 隐层到输出层的权重矩阵为  $W'_{N \times V}$ 。然后，通过矩阵运算我们得到一个 $V \times 1 $ 维向量
  $$
  u = W'^{T} * h
  $$


其中，向量 $u$  的第 $i$  行表示词汇表中第 $i$  个词的可能性，然后我们的目的就是取可能性最高的那个词。因此，在最后的输出层是一个softmax 层获取分数最高的词，那么就有我们的最终输出：
$$
P(w_j| context)  =y_i =  \frac{exp({u_j})}{\sum_{k \in V} exp({u_k})}    
$$

### 2. 损失函数 

我们假定 $j^*$ 是真实单词在词汇表中的下标，那么根据极大似然法，则目标函数定义如下：
$$
\begin{split} O &= max \, p(W_O|W_I) \\ &= max \, y_{j*}  \\ &:=  max \, log \, y_{j*}   \\ &= max \, log(\frac{exp(u_{j*})}{\sum_{k \in V} exp(u_k)} )  \\ &= u_{j*} - log \sum_{k \in V} exp(u_{k})  \\ &:= -E \end{split}
$$
其中，损失函数为 $E = -log \, p(W_O| W_I) $， 我们需要最小化 $E$ 。

## Skip-gram模型

Skip-Gram的基本思想是：已知当前词 $w_t$ 的前提下，预测其上下文 $w_{t-i}, \cdots , w_{t+i}$ ，模型如下图所示：

![img](https://pic2.zhimg.com/v2-42ef75691c18a03cfda4fa85a8409e19_b.jpg)

### 1. 前向传播过程：

- 输入层：   输入的是一个单词，其表示形式为 **One-hot** ，我们将其表示为V维向量 $x_k$ ，其中 $V$ 为词表大小。然后，通过词向量矩阵 $W_{V \times N}$ 我们得到一个N维向量  
  $$
  h = W^T * x_k = v^{T}_{w_I}
  $$


- 隐层： 而隐层中没有激活函数，也就是说输入=输出，因此隐藏的输出也是 $h$ 。

- 隐层到输出层：

  > - 首先，因为要输出C个单词，因此我们此时的输出有C个分布： $y_1, \cdots y_C $，且每个分布都是独立的，我们需要单独计算， 其中 $y_i$  表示窗口的第 $i$  个单词的分布。 
  > - 其次， 因为矩阵 $W'_{N \times V}$ 是共享的，因此我们得到的 $V \times 1$ 维向量 $u$ 其实是相同的，也就是有 $u_{c,j} = u_j$ ，这里 $u$ 的每一行同 CBOW 中一样，表示的也是评分。
  > - 最后，每个分布都经过一个 softmax 层，不同于 CBOW，我们此处产生的是第 $i$ 个单词的分布（共有C个单词），如下：

  $$
    P(w_{i,j}| context)  =y_i =  \frac{exp({u_j})}{\sum_{k \in V} exp({u_k})}    
  $$


### 2. 损失函数

假设 $j^*$ 是真实单词在词汇表中的下标，那么根据极大似然法，则目标函数定义如下：
$$
\begin{split} E &= - log \, p(w_1, w_2, \cdots, w_C | w_I)   \\ &= - log \prod_{c=1}^C P(w_c|w_i) \\ &= - log  \prod_{c=1}^{C} \frac{exp(u_{c, j})}{\sum_{k=1}^{V} exp(u_{c,k}) } \\ &= - \sum_{c=1}^C u_{j^*_c} + C \cdot log \sum_{k=1}^{V} exp(u_k) \end{split}
$$


## 模型复杂度

本节中我们来分析一下模型训练时的复杂度，无论是在CBOW还是Skip-Gram模型中，都需要学习两个词向量矩阵： $W, W'$ 。

对于矩阵 $W$ ， 从前向传播中可以看到， 可以看到对于每一个样本(或mini-batch)，CBOW更新 $W$  的 C 行（h与C个x相关）， 而Skip-Gram 更新W中的其中一行（h与1个x相关），这点训练量并不算大。

对于 $W'$  而言， 无论是 CBOW 还是 Skip-Gram 模型，每个训练样本(mini-batch)都更新 $W'$ 的所有 $V \times N$ 个元素。

在现实中，用于语言模型训练的数据集通常都很大，此外词表也是巨大的，这就导致对于 $W'$ 的更新所花费的计算成本是很大的，真的是验证了一个道理：穷逼必要搞语言模型。

为了解决优化起来速度太慢的问题， Word2Vec 中提供了两种策略来对这方面进行优化。

## Hierarchical Softmax ，Negative Sampling

两者都是通过对softmax进行优化来降低模型训练时的复杂度， 从而提高效率的。

### 1. Hierarchical Softmax

<https://shomy.top/2017/07/28/word2vec-all/>

## Reference Papers

[1] Mikolov, T.(2013). Distributed Representations of Words and Phrases and their Compositionality.

[2] Mikolov, T.(2013). Efficient Estimation of Word Representations in Vector Space.

[3] Rong, X. (2014). word2vec Parameter Learning Explained.


  