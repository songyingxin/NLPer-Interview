# Attention 机制 -- 基础篇

## Hard vs Soft [1]

Attention首先分为两大类：**Hard Attention 与 Soft Attention**， 两者的区别在于 Hard Attention 关注一个很小的区域，而soft Attention 关注的相对要发散。 举个机器翻译方面的例子：

> 我是小明   -->  I am XiaoMing

- 对于 Hard Attention而言，在第1时刻翻译时，只关注“我”这个词，我们翻译得到“I”，在第2时刻翻译时，关注“是”这个词，翻译结果为“am”，以此直到 t 时刻结束。 它是采用one-hot编码的方式对位置进行标记，比如第1时刻，编号信息就是[1,0,0...]， 第二时刻，编码信息就是 [0, 1, 0, ...]， 以此类推。这样会带来一个缺点：**无法采用常规优化方法来进行优化，具体的训练细节很复杂，不推荐深入了解。**
- 而对于soft attention 而言，在第一时刻翻译时， “我是小明” 都对 “I” 做出了贡献，只不过贡献有大小之分，也就是说，虽然“我”这个词很重要，但是我们也不能放过其他词所带来的信息。

比较二者而言，很显然，soft attention有很大的优势，因此，对于NLP领域而言，目前大多数的研究都基于 soft Attention 进行扩展。

虽然 [1] 具有很强的开创意义，但其毕竟是关于CV领域的，不推荐精读，因此我没有写任何公式，个人十分推荐下面这篇文章来作为 Attention 的第一篇精读论文。

## Global vs Local [2] 

在 soft attention阵营中，很快又划分为两大阵营： **Glocal attention** 与 **Local attention**， **二者的区别在于关注的范围大小不同**， 其中，Global attention 关注**全部的文字序列**，而 Local attention 关注的是**固定的窗口中的所有文字序列**。

比较二者， Global attention 的计算量要比 Local Attention 要大，尤其是对于长句子而言，效率变得很低； 而 Local Attention 只与窗口内的文字相关，因此窗口的大小就显得至关重要了，且在local attention 中多了一个**预测中心词 $p_t$ 的过程**，这有可能会忽略一些重要的词， 但同时，如果选择适当，那么 local attention 理应会降低无关词的干扰，当然，所带来的收益并不大。 而对于窗口的设置，论文中采用**高斯分布**来实现，如下：
$$
\hat{a}_{i,j} = a_{i,j} \, e^{- \frac{(s - p_t)^2}{2 \sigma^2}}, \sigma = \frac{D}{2}
$$
另一方面，由于Global Attention考虑的信息较多，因此从原理上讲要更好一些，毕竟local attention 可能会忽略对当前输出很重要的词，且 Local Attention 的表现与窗口的大小密切相关，如果设置小了，可能会导致效果变得很差。 

而考虑到NLP中问题的复杂性（如句子长短不一，句子之间可能有很强的相关性），因此**后来的很多论文[3][4]中很少考虑采用 Local Attention 方法**，且我自己在做阅读理解任务时，也基本不会考虑Local Attention， 毕竟窗口大小的设置实在太考验人了。

- Global Attention

![Global Attention](http://ww1.sinaimg.cn/large/006gOeiSly1g0tereh268j30zk0k00te.jpg)

- local attention

![Local Attention](http://ww1.sinaimg.cn/large/006gOeiSly1g0terwkhqmj30zk0k0wf1.jpg)

## Attention的本质思想 [5][6]

![](http://ww1.sinaimg.cn/large/006gOeiSly1g0tf397umyj30kn08uq34.jpg)

在上图中，Query 表示我们的问题信息，在阅读理解问题中，其对应的就是 question，而在机器翻译中，常常采用上一时刻的输出信息 $S_{i-1}$ 作为 Query；

对于 { key:value }而言，大多数情况下key与value是相同的，一般指的是第 j 个词的表示, 比如在机器翻译中， key 与 value 通常采用词的隐层输出 $h_j$  。我们通过计算 Query 与各个 key 之间的相似性或相关性来获得 $a_{i,j}$，然后对各个进行加权求和：
$$
\alpha_{i,j} = \frac{e^{score(Query, Key(j))}}{\sum_{k=1}^t e^{score(Query, Key(k))}} \\
c_i= \sum_{i=1}^n = \alpha_{i,j} h_j
$$
由上式可以看到，对于Attention机制的整个计算过程，可以总结为以下三个过程：

- **socre 函数：** 根据 Query 与 Key 计算两者之间的相似性或相关性， 即 socre 的计算。
- **注意力权重计算：**通过一个softmax来对值进行归一化处理获得注意力权重值， 即$a_{i,j}$ 的计算。
- **加权求和生成注意力值：**通过注意力权重值对value进行加权求和， 即 $c_i$ 的计算。

总的来说，Attention 无论如何变化，总是万变不离其宗。 对于大多数 Attention 的文章来说，其变化主要在于 Query， Key， Value 的定义以及第一阶段 Score 的计算方法，下面我们来详细讨论一下。

## Score 函数的选择 [6][2]

Score 函数本质的思想就是度量两个向量的相似度。

常见的方式主要有以下三种：

- 求点积：学习快，适合向量再同一空间中，如 Transformer 。

$$
score(Query, Key(j)) = Query \cdot Key(j)
$$

- Cosine 相似性

$$
score(Query, Key(j)) = \frac{Query \cdot Key(j)}{||Query|| \cdot ||Key(j)||}
$$

- MLP网络

$$
score(Query, Key(j)) = MLP(Query,  Key(j)) \\
general: score(Query, Key(j)) = Query \, W \, Key(j) \\
concat: score(Query, key(j)) = W \, [Query;Key(j) ]
$$

一般情况下，采用MLP网络更加灵活一些，且可以适当的扩展层以及改变网络结构，这对于一些任务来说是很有帮助的。

## Query, Key, Value 的定义

对于一个 Attention 机制而言，定义好 Query， Key， Value 是至关重要的，这一点我个人认为是一个经验工程，看的多了，自然就懂了。 我这里简单举阅读理解与机器翻译的例子：

- 对于机器翻译而言，常见的是： Query 就是上一时刻 Decoder 的输出 $S_{i-1}$， 而Key，Value 是一样的，指的是 Encoder 中每个单词的上下文表示。
- 对于英语高考阅读理解而言， Query 可以是**问题的表示**，也可以是**问题+选项的表示**， 而对于Key， Value而言，往往也都是一样的，都指的是**文章**。而此时Attention的目的就是找出**文章**中与**问题**或**问题+选项**的相关片段，以此来判断我们的问题是否为正确的选项。

由此我们可以看出， Attention 中 Query， Key， Value 的定义都是很灵活的，不同的定义可能产生不一样的化学效果，比如 **Self-Attention** ，下面我就好好探讨探讨这一牛逼思想。

## Self-Attention [5]

Self-Attention 可以说是最火的 Attention 模型了，其在最近很火的 Bert 中也起到了重要的作用，最关键的是，**其可与 LSTM 一较高低**。 

这篇文章是十分值得精读，反复看的，因为其真正的将 Attention 用到了另一个新的高度，膜拜 Google。鉴于篇幅所限，本文就不赘述其中详细原理了，而是简述一下其核心思想。

Self-Attention 的本质就是**自己注意自己**， 粗暴点来说，就是，Q，K，V是一样的，即：
$$
Attention \, value = Attention(W_QX,W_KX,W_VX)
$$
它的内部含义是对序列本身做 Attention，来获得序列内部的联系，如下图所示 [7]。 

![](http://ww1.sinaimg.cn/large/006gOeiSly1g0u0j6zj7hg30go0er1kx.gif)

这其实是有点类似于我们在 Embedding 层的时候采用 LSTM 来获得输入序列的上下文表示，但与 LSTM 不同之处在于**Self - Attention 更能够把握句子中词与词的句法特征或语义特征**，但另一方面其对于**序列的位置信息**不能很好的表示，这也是为什么会采用 Postition Embedding 来对位置信息做一个补充，但对于一些对位置信息敏感的任务，position  Embedding 所带来的信息可能会不够。

之所以说这篇文章具有开创意义，是因为其将Attention用到了一个基础单元上， 为取代LSTM提供了一种可能。

## Reference

[1]  Show, Attend and Tell: Neural Image Caption Generation with Visual Attention -- 不推荐

[2]  Effective Approaches to Attention-based Neural Machine Translation

[3] Neural Machine Translation by Jointly Learning to Align and Translate

[4] Neural Responding Machine for Short-Text Conversation

[5] Attention is all you need

[深度学习中的注意力机制](https://blog.csdn.net/qq_40027052/article/details/78421155)

[7] https://zhuanlan.zhihu.com/p/47282410