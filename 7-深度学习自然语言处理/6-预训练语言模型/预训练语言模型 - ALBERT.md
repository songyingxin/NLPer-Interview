# ALBERT 的一些问题

---

需要注意的一点是：**ALBERT降低了模型参数量从而降低了模型的训练时间（通信开销降低），但是，模型的预测推理时间并没有降低。**这点需要牢记，并贯穿全文。

## ALBERT 的目的

论文开篇就提到，在预训练语言模型领域，增大模型往往能够到来不错的效果提升，但这种提升是无止境的吗？[2]中进行了详细的实验，相当程度上解答了这一问题。这里先埋一个坑，过几天再填。

预训练语言模型已经很大了，大到绝大多数的实验室和公司都没有资格参与这场游戏，对于大模型而言，一个很浅的idea就是：**如何对大模型进行压缩？** ALBERT 本质就是对 BERT 模型压缩后的产物。

如果对模型压缩有了解的同学肯定知道，模型压缩有很多手段，包括**剪枝，参数共享，低秩分解，网络结构设计，知识蒸馏**等。ALBERT 也没能逃出这一框架，它其实是一个相当工程化的思想，它的两大 压缩Trick 也很容易想到，下面就细聊一下。

## 三大创新

### 1. Factorized embedding parameterization

该 Trick 本质上就是一个**低秩分解**的操作，其通过对Embedding 部分降维来达到降低参数的作用。在最初的BERT中，以Base为例，Embedding层的维度与隐层的维度一样都是768，但是我们知道，对于词的分布式表示，往往并不需要这么高的维度，比如在Word2Vec时代就多采用50或300这样的维度。那么一个很简单的思想就是，通过将Embedding部分分解来达到降低参数量的作用，其以公式表示如下：
$$
O(V \times H) \to O(V \times E + E \times H)
$$

- V：词表大小；H：隐层维度；E：词向量维度

我们以 BERT-Base 为例。Base中的Hidden size 为768， 词表大小为3w，此时的参数量为：768 * 3w = 23040000。 如果将 Embedding 的维度改为 128，那么此时Embedding层的参数量为： 128 * 3w + 128 * 768 = 3938304。二者的差为19101696，大约为19M。

我们看到，其实Embedding参数量从原来的23M变为了现在的4M，似乎变化特别大，然而当我们放到全局来看的话，BERT-Base的参数量在110M，降低19M也不能产生什么革命性的变化。因此，可以说**Embedding层的因式分解其实并不是降低参数量的主要手段。**

注意，我在这里特意忽略了Position Embedding的那部分参数量， 主要是考虑到512相对于3W显得有点微不足道。

文章在4.4中，对词向量维度的选择做了详细的探讨：

![1](image\ALBERT_1.PNG)

从上图中，我们可以看出，增大词向量维度所带来的收益在128之后十分少，这也呼应了上面的观点。

### 2. Cross-layer parameter sharing

该Trick本质上就是对**参数共享机制**在Transformer内的探讨。在Transfor中有两大主要的组件：**FFN**与**多头注意力机制**。ALBERT主要是对这两大组件的共享机制进行探讨。

![2](image/ALBERT_2.png)

论文里采用了四种方式： **all-shared，shared-attention，shared-FFN以及 not-shared。**

首选关注一下参数量，not-shared与all-shared的参数量相差极为明显，因此可以得出共享机制才是参数量大幅减少的根本原因。然后，我们看到，只共享Attention参数能够获得参数量与性能的权衡。最后，很明显的就是，随着层数的加深，基于共享机制的 ALBERT 参数量与BERT参数量相比下降的更加明显。

此外，文章还说道，通过共享机制，能够帮助模型稳定网络的参数。这点是通过L2距离与 cosine similarity 得出的，俺也不太懂，感兴趣的可以找其他文章看看：

![3](image/ALBERT_3.PNG)

### 3. SOP 替代 NSP

SOP 全称为 Sentence Order Prediction，其用来取代 NSP 在 BERT 中的作用，毕竟一些实验表示NSP非但没有作用，反而会对模型带来一些损害。SOP的方式与NSP相似，其也是判断第二句话是不是第一句话的下一句，但对于负例来说，SOP并不从不相关的句子中生成，而是将原来连续的两句话翻转形成负例。

很明显的就可以看出，SOP的设计明显比NSP更加巧妙，毕竟NSP任务的确比较简单，不相关句子的学习不要太容易了。论文也比较了二者：

![4](image/ALBERT_4.PNG)

## BERT vs ALBERT

### 1. 从参数量级上看

![5](image\ALBERT_5.PNG)

首先，参数量级上的对比如上表所示，十分明显。这里需要提到的是ALBERT-xxlarge，它只有12层，但是隐层维度高达4096，这是考虑到深层网络的计算量问题，其本质上是一个浅而宽的网络。

### 2. 从模型表现上看

![6](image\ALBERT_6.PNG)

首先，我们看到 ALBERT-xxlarge的表现完全超过了BERT-large的表现，但是BERT-large的速度是要比ALBERT-xxlarge快3倍左右的。

其次，BERT-xlarge的表现反而变差，这点在[2]中有详细探讨，先略过不表。

## Questions



## Reference

[1] ALBERT: A Lite BERT for Self-supervised Learning of Language Representations

[2] T5 - Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer