# Pytorch-Transformer 

---



## BERT

- 配置类： BertConfig
- Layer Normalization：BertLayerNorm = torch.nn.LayerNorm
- Bert 输入： BertEmbeddings
- 多头注意力机制/自注意力机制： BertSelfAttention
- Bert 一层的输出： BertSelfOutput 依赖于 BertLayerNorm
- Bert 一层： BertAttention 依赖于 BertSelfAttention， BertSelfOutput
- BertIntermediate ： 无依赖
- Bert 最终输出： BertOutput， 依赖于 BertLayerNorm
- Bert一层 ： BertLayer 依赖于 BertAttention， BertIntermediate， BertOutput
- Bert 的Encoder： BertEncoder， 依赖于 BertLayer
- BertPooler： Bert [CLS] 输出
- BertPredictionHeadTransform：依赖于 BertLayerNorm
- BertLMPredictionHead： 依赖于 BertPredictionHeadTransform
- BertOnlyMLMHead： 依赖于 BertLMPredictionHead
- BertOnlyNSPHead： 
- BertPreTrainingHeads： 依赖于 BertLMPredictionHead
- BertPreTrainedModel： 继承于 PreTrainedModel
- BertModel:  依赖于 BertEmbedding  BertEncoder， BertPooler
- BertForPreTraining： 继承于 BertPreTrainedModel， 依赖于 BertModel， BertPreTrainingHeads
- BertForMaskedLM： 依赖于 BertModel， BertOnlyMLMHead
-  BertForNextSentencePrediction： 依赖于 BertModel， BertOnlyNSPHead

---

- BertForSequenceClassification： 用于分类

-  BertForMultipleChoice：用于多选

- BertForTokenClassification： 用于命名实体识别

- BertForQuestionAnswering： 用于 QA
