# Normalization

tags: 深度学习

---



## 1. Batch Normalization

假设一个 batch 为 m 个输入 $B = \{x_{1, \cdots, m}\}$ , BN 在这 m 个数据之间做Normalization， 并学习参数 $\gamma , \beta$：
$$
\mu_B \leftarrow \frac{1}{m} \sum_{i=1}^{m} x_i    \\
\sigma_B^2 \leftarrow \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2  \\
\hat{x}_i  \leftarrow \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \\
y_i \leftarrow \gamma \hat{x}_i + \beta \equiv BN_{\gamma, \beta}{(x_i)}
$$

## 2. Layer Normalization

LN 不同于 BN， 其在层内进行 Normalization， 即直接对隐层单元的输出做 Normalization。最大的好处是不再依赖 batch size。 


$$
u^l = \frac{1}{H} \sum_{i=1}^H a_i^l \\
\sigma^l = \sqrt{\frac{1}{H}\sum_{i=1}^H(a_i^l - u^l)^2} \\
\hat{a}_i^l  \leftarrow \frac{a_i^l - \mu^l}{\sqrt{\sigma_l^2 + \epsilon}} \\
y_i^l \leftarrow \gamma \, \hat{a}_i^l + \beta \equiv LN_{\gamma, \beta}{(a_i^l)}
$$

- H： 某层中的隐层单元数

## 3. Weight Normalization

---

## QA

### 0. 什么是 ICS？

ICS 指的是在训练过程中由于网络参数的变化而导致各层网络的输入分布发生了变化。

从数学角度来看，它指的是源空间与目标空间的条件概率是一致的，但是其边缘概率不同。 即对所有 $x \in X, P_s(Y|X=x) = P_t(Y|X=x)$ ， 但是 $P_s(X) \neq P_t(X)$ 。

其实，通俗理解来说，就是对于神经网络的各层输出，由于经过了一系列的隐层操作，其分布显然与各层对应的输入数据分布不同，且这种差距会随着网络深度的增加而加大。而在训练过程中， 网络参数的变化会使得各层网络的输入分布发生变化，越深的网络，变化可能越大。

### 1. BN 到底解决了什么问题？

- 解释1： 原论文说 BN 解决了 ICS 问题，但后续有论文推翻了这个结论，参见：《How Does Batch Normalization Help Optimization》

- 解释2： 为了防止梯度消失，这也很好理解， BN 将激活函数的输入数据压缩在 N(0,1) 空间内，的确能够很大程度上减轻梯度消失问题。

- 解释3： 来源于《How Does Batch Normalization Help Optimization》 ， BN 使得优化空间更加平滑，具体来说，BN实际上是改变了损失函数的 lipschitz 性质， 使得梯度的改变变得很轻微，这使得我们可以采用更大的步长且仍然能够保持对实际梯度方向的精确估计。

  通俗来讲， 不进行BN， 损失函数不仅仅非凸且趋向于坑洼，平坦区域和极小值，这使得优化算法极不稳定，使得模型对学习率的选择和初始化方式极为敏感，而BN大大减少了这几种情况发生。

我个人更倾向于第三种解释。

### 2. BN 的优点与缺点

- BN 的优点：

  > - 加速网络训练（缓解梯度消失，支持更大的学习率）
  > - 抑制过拟合
  > - 降低了**参数初始化**的要求。

- BN 的缺点：

  > - **对 batch size的要求较高**。这是因为如果 batch size 过小，无法估计出全局的样本分布
  > - **训练和预测时有些差别。**训练时一个 batch 之间进行 Normalization， 预测时需要依靠训练时获得的 均值和方差来进行预测。

### 3. 为何训练时不采用移动平均？

参见： 《Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models》

- 使用 BN 的目的就是为了保证每批数据的分布稳定，使用全局统计量反而违背了这个初衷；
- BN 的作者认为在训练时采用移动平均可能会与梯度优化存在冲突；

### 4. BN 与 LN 的区别是什么？

LN 对层进行 Normalization ， BN 对 batch 进行 Normalization。 LN 拜托了对 batch size 的依赖， 在 NLP 领域，使用极为广泛， 基本不用 BN。 

我个人的认为是， BN 是对batch进行操作，然而， 语言的复杂度使得 一个 batch 的数据对于全局的数据分布估计极为不准，这使得 BN 的效果变得很差。

### 5. 什么时候使用 BN 或 LN？

一般只在深层网络中使用， 比如在深层的 Transformer 中， LN 就起到了很关键的作用。 

当你发现你的网络训练速度慢，梯度消失，爆炸等问题时， 不妨考虑加入 BN 或 LN 来试试。

### 6. BN 在何处做？

BN 可以对**第 L 层激活函数输出值**或**对第 L层激活函数的输入值**进行 Normalization， 对于两个不同的位置，不少研究已经表明：**放在第 L 层激活函数输出值会更好**。



## Reference

[1] Batch Normalization

[2] Layer Normalization

[3] How Does Batch Normalization Help Optimization

[4]  Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models

[深度学习中 Batch Normalization为什么效果好？](https://www.zhihu.com/question/38102762/answer/85238569) - 知乎 

[深度学习中的Normalization模型](<http://www.cvmart.net/community/article/detail/368>)