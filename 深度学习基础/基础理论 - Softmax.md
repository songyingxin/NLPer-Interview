# 基础理论 - Softmax

---

## 1. Softmax 定义

$$
P(i) = \frac{e^{a_i}}{\sum_{k=1}^T e^{a_k}} \in [0,1]
$$





## 2. Softmax 损失

$$
L = - \sum_{j=1}^T y_j \, log \, s_j  \\
$$





---

## QA

### 1. 为何一般选择 softmax 为多分类的输出层

虽然能够将输出范围概率限制在 [0,1]之间的方法有很多，但 Softmax 的好处在于， 它使得输出两极化：正样本的结果趋近于 1， 负样本的结果趋近于 0。 可以说， Softmax 是 logistic 的一种泛化。

