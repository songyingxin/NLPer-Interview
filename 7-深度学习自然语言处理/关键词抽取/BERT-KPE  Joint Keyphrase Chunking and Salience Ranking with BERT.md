# BERT-KPE - Joint Keyphrase Chunking and Salience Ranking with BERT









## Methodology




$$
文档： D = \{w_1, \cdots, w_i, \cdots, w_n\} \\
BERT表示文档： H = BERT(D) = \{h_1, \cdots, h_i, \cdots, h_n \} \\
ngram \,的 \, phrase : c_i^k = w_{i:i+k-1}  \\
CNN \,表示 \, ngram: g_i^k = CNN^k \{h_i, \cdots, h_{i+k-1} \}
$$

- a chunking network： 识别出有意义的n-grams， 直接采用全连接接网络+softmax判断该n-gram $c_i^k$ 是否为合适的chunk

$$
p(c_i^k=y_i^k) = softmax(Linear(g_i^k))
$$

- a ranking network：为 phrase 评分
  $$
  f(c_i^k, D) = Linear(g_i^k)
  $$
  