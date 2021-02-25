# language+vision 预训练模型





## idea -- 多模态融合

- 如何将图片特征与语言特征进行有效融合

现有的预训练方法采用句子级别与图片进行联合训练，我认为这种方式是不恰当的，因为图片本身就是一个 token 粒度的，而用token粒度去与句子粒度去一起训练是不合适的，我认为首要的任务还是如何将token粒度进行有效的融合。 此外，数据量也是需要考虑的事情。







unicoder-vl：

https://www.msra.cn/zh-cn/news/features/machine-reasoning-unicoder-vl





## Reference

[1] Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training  2019-12

[2] VideoBERT: A Joint Model for Video and Language Representation Learning  2019-9

ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks  2019-8

ERNIE-ViL: Knowledge Enhanced Vision-Language Representations Through Scene Graph  - 2020-6

Vl-bert: Pre-training of generic visual-linguistic representations  2019-8

Visualbert: A simple and performant baseline for vision and language  2019-8

Uniter: Learning universal image-text representations   2019-9

Imagebert: Cross-modal pre-training with large-scale weak-supervised image-text data  2020-1

Unified vision-language pre-training for image captioning and vqa  -- 2019-9

Pixel-bert: Aligning image pixels with text by deep multi-modal transformers  2020-4

Lxmert: Learning cross-modality encoder representations from transformers. 2019-8

FashionBERT- Text and Image Matching with Adaptive Loss for Cross-modal Retrieval   2020-5

https://zhuanlan.zhihu.com/p/80917186  -- 多篇

https://zhuanlan.zhihu.com/p/150261938





vl-bert： https://zhuanlan.zhihu.com/p/113700984

OSCAR： https://zhuanlan.zhihu.com/p/141089155

https://zhuanlan.zhihu.com/p/142935893  - 这里面几篇

https://www.jiqizhixin.com/articles/2020-03-05



https://www.zhihu.com/question/374681272

ERNIE-VIL: KNOWLEDGE ENHANCED VISION-LANGUAGE REPRESENTATIONS THROUGH SCENE GRAPH