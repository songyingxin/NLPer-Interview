# Normalization

tags: 深度学习

---

## 0 . 归一化

### 1. 归一化手段

- **Min-max 归一化：**当有新数据加入时， 可能导致max和min的变化， 需要重新定义。
  $$
   x^* = \frac{x -min } {max - min} 
  $$

- **Zero-mean 归一化：**均值为0，标准差为1的标准正态分布。 z-score标准化方法适用于属性A的最大值和最小值未知的情况，或有超出取值范围的离群数据的情况。该种标准化方式要求原始数据的分布可以近似为高斯分布，否则效果会变得很糟糕。
  $$
  x^* = \frac{x- \mu }{\sigma}
  $$


### 2. Min-max 与 Zero-mean 区别

- 对于输出结果范围有要求， 使用Min-max normalization 
- 数据较为稳定， 不存在极端最大值，最小值， 用归一化
- 如果数据存在异常值或较多噪音， 使用标准化。

### 3. 为何归一化为何如此优秀？

归一化的本质就是**线性变换**。 线性变化的诸多良好性质，决定了为什么对数据进行改变后不会造成“失效”，还能提高数据的表现。

**归一化加快了梯度下降求最优解的速度。**

假定一个预测房价的例子，两个特征： 面积与房间数，那么则有： $y = \theta_1 x_1 + \theta_2x_2$， $x_1$ 表房间数，$x_2$ 表面积 ， 现实中，面积范围往往在 $1-1000$， 而房间通常为 $1-10$， 那么面积对模型的影响要更大一些，此时寻求最优解的过程为：

![1](..\img\Normalization\1.jpg)

归一化后，寻求最优解的过程为：

![2](D:\Github\NLPer-Interview\img\Normalization\2.jpg)

在未归一化的时候， 由于 $\theta_1$ 的更新幅度要比  $\theta_2$ 小， 因此  $\theta_2$ 的应该要比  $\theta_1$ 要大，但是在实际中，我们使用常规梯度下降法时，我们各个的学习率都是一样的，这也就造成了 $\theta_2$ 的更新会比较慢，结果就是寻求最优解的过程会走很多弯路导致模型收敛速度缓慢。

我们来实际举例， 假设在未归一化的时候， 我们的损失函数为：
$$
J = (3 \theta_1 + 600 \theta_2 - \hat{y})^2
$$
那么经过归一化后，我们的损失函数可能就变为：
$$
J = (0.5 \theta_1 + 0.55 \theta_2 - \hat{y})^2 
$$
很明显可以看到，数据归一化后， 最优解的寻找过程会很平缓，更容易正确收敛到最优解。

其次，还有一些博客中提到，**归一化有可能提高精度**， 这在涉及到一些距离计算的算法时效果显著。

## 1. Batch Normalization

假设一个 batch 为 m 个输入 $B = \{x_{1, \cdots, m}\}$ , BN 在这 m 个数据之间做 Normalization， 并学习参数 $\gamma , \beta$：
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

## 3. Weight Normalization -- TODO

---

## QA

### 0. 什么是 ICS？

ICS 指的是在训练过程中由于网络参数的变化而导致各层网络的输入分布发生了变化， 而深层网络通过层层叠加，高层的输入分布变化会非常剧烈，这就需要高层不断去重新适应底层的参数更新。而为了训好模型，我们需要非常谨慎的去设定学习率， 初始化权重等。

从数学角度来看，它指的是源空间与目标空间的条件概率是一致的，但是其边缘概率不同。 即对所有 $x \in X, P_s(Y|X=x) = P_t(Y|X=x)$ ， 但是 $P_s(X) \neq P_t(X)$ 。

其实，通俗理解来说，就是对于神经网络的各层输出，由于经过了一系列的隐层操作，其分布显然与各层对应的输入数据分布不同，且这种差距会随着网络深度的增加而加大。而在训练过程中， 网络参数的变化会使得各层网络的输入分布发生变化，越深的网络，变化可能越大。

### 0. ICS 会导致什么问题？

- 上层参数需要不断适应新的输入数据分布，降低学习速度
- 下层输入的变化可能趋向于变大或变小，导致上层落入饱和区，使得学习过早停止。
- 每层的更新都会影响到其它层，因此每层的参数更新策略需要尽可能的谨慎。

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

- 一般只在深层网络中使用， 比如在深层的 Transformer 中， LN 就起到了很关键的作用。 
- BN 与 LN 应用在非线性映射前效果更佳。
- 当你发现你的网络训练速度慢，梯度消失，爆炸等问题时， 不妨考虑加入 BN 或 LN 来试试。
- 使用 BN 时， 对 batch size 要求较高， 且在训练前需要对数据进行 shuffle。

### 6. BN 在何处做？

BN 可以对**第 L 层激活函数输出值**或**对第 L层激活函数的输入值**进行 Normalization， 对于两个不同的位置，不少研究已经表明：**放在第 L 层激活函数输出值会更好**。

### 7. 为什么要归一化？

- 归一化的确可以避免一些不必要的数值问题。
- 为了程序运行时收敛加快。 
- 统一量纲。样本数据的评价标准不一样，需要对其量纲化，统一评价标准。这算是应用层面的需求。
- 避免神经元饱和。啥意思？就是当神经元的激活在接近 0 或者 1 时会饱和，在这些区域，梯度几乎为 0，这样，在反向传播过程中，局部梯度就会接近 0，这会有效地“杀死”梯度。
- 保证输出数据中数值小的不被吞食。 







## Reference

[1] Batch Normalization

[2] Layer Normalization

[3] How Does Batch Normalization Help Optimization

[4]  Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models

[深度学习中 Batch Normalization为什么效果好？](https://www.zhihu.com/question/38102762/answer/85238569) - 知乎 

[深度学习中的Normalization模型](<http://www.cvmart.net/community/article/detail/368>)