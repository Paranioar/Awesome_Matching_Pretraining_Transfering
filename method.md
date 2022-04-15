Method Summary of Cross-modal Retrieval
==============================

## ``Catalogue ``
* [Algorithm-oriented Works](#algorithm-oriented-works)
    * [Vision-Language Pretraining](#vision-language-pretraining)
    * [Generic-Feature Extraction](#generic-feature-extraction)
    * [Cross-Modal Interaction](#cross-modal-interaction)
    * [Similarity Measurement](#similarity-measurement)
    * [Commonsense Learning](#commonsense-learning)
    * [Adversarial Learning](#adversarial-learning)
    * [Loss Function](#loss-function)
* [Task-oriented Works](#task-oriented-works)
    * [Un-Supervised or Semi-Supervised](#un-supervised-or-semi-supervised)
    * [Zero-Shot or Fewer-Shot](#zero-shot-or-fewer-shot)
    * [Identification Learning](#identification-learning)
    * [Scene-Text Learning](#scene-text-learning)
    * [Related Works](#related-works)  
    * [Posted in](#posted-in)


## ``Algorithm-oriented Works`` 

### ``*Vision-Language Pretraining*``

**(*ICCV2019_VideoBERT*) VideoBERT: A Joint Model for Video and Language Representation Learning.** <br>
*Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, Cordelia Schmid.*<br>
[[paper]](https://arxiv.org/abs/1904.01766)

**(*ICCV2019_HowTo100M*) HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips.** <br>
*Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, Josef Sivic.*<br>
[[paper]](https://arxiv.org/abs/1906.03327)
[[code]](http://www.di.ens.fr/willow/research/howto100m/)

**(*arXiv2019_CBT*) Learning Video Representations using Contrastive Bidirectional Transformer.** <br>
*Chen Sun, Fabien Baradel, Kevin Murphy, Cordelia Schmid.*<br>
[[paper]](https://arxiv.org/abs/1906.05743)

**(*NeurIPS2019_ViLBERT*) ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks.** <br>
*Jiasen Lu, Dhruv Batra, Devi Parikh, Stefan Lee.*<br>
[[paper]](https://arxiv.org/abs/1908.02265)
[[code]](https://github.com/jiasenlu/vilbert_beta)

**(*ACL2020_VisualBERT*) VisualBERT: A Simple and Performant Baseline for Vision and Language.** <br>
*Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, Kai-Wei Chang.*<br>
[[paper]](https://arxiv.org/abs/1908.03557)
[[code]](https://github.com/uclanlp/visualbert)

**(*EMNLP2019_B2T2*) Fusion of Detected Objects in Text for Visual Question Answering.** <br>
*Chris Alberti, Jeffrey Ling, Michael Collins, David Reitter.*<br>
[[paper]](https://arxiv.org/abs/1908.05054)
[[code]](https://github.com/google-research/language/tree/master/language/question_answering/b2t2)

**(*AAAI2020_Unicoder-VL*) Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training.** <br>
*Gen Li, Nan Duan, Yuejian Fang, Ming Gong, Daxin Jiang, Ming Zhou.*<br>
[[paper]](https://arxiv.org/abs/1908.06066)

**(*EMNLP2019_LXMERT*) LXMERT: Learning Cross-Modality Encoder Representations from Transformers.** <br>
*Hao Tan, Mohit Bansal.*<br>
[[paper]](https://arxiv.org/abs/1908.07490)
[[code]](https://github.com/airsplay/lxmert)

**(*ICLR2020_VL-BERT*) VL-BERT: Pre-training of Generic Visual-Linguistic Representations.** <br>
*Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, Jifeng Dai.*<br>
[[paper]](https://arxiv.org/abs/1908.08530)
[[code]](https://github.com/jackroos/VL-BERT)

**(*AAAI2020_Unified-VLP*) Unified Vision-Language Pre-Training for Image Captioning and VQA.** <br>
*Luowei Zhou, Hamid Palangi, Lei Zhang, Houdong Hu, Jason J. Corso, Jianfeng Gao.*<br>
[[paper]](https://arxiv.org/abs/1909.11059)
[[code]](https://github.com/LuoweiZhou/VLP)

**(*ECCV2020_UNITER*) UNITER: UNiversal Image-TExt Representation Learning.** <br>
*Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, Jingjing Liu.*<br>
[[paper]](https://arxiv.org/abs/1909.11740)
[[code]](https://github.com/ChenRocks/UNITER)

**(*CVPR2020_M4C*) Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA.** <br>
*Ronghang Hu, Amanpreet Singh, Trevor Darrell, Marcus Rohrbach.*<br>
[[paper]](https://arxiv.org/abs/1911.06258)

**(*CVPR2020_12-in-1*) 12-in-1: Multi-Task Vision and Language Representation Learning.** <br>
*Jiasen Lu, Vedanuj Goswami, Marcus Rohrbach, Devi Parikh, Stefan Lee.*<br>
[[paper]](https://arxiv.org/abs/1912.02315)
[[code]](https://github.com/facebookresearch/vilbert-multi-task)

**(*ECCV2020_VisDial-BERT*) Large-scale Pretraining for Visual Dialog: A Simple State-of-the-Art Baseline.** <br>
*Vishvak Murahari, Dhruv Batra, Devi Parikh, Abhishek Das.*<br>
[[paper]](https://arxiv.org/abs/1912.02379)
[[code]](https://github.com/vmurahari3/visdial-bert)

**(*arXiv2020_ImageBERT*) ImageBERT: Cross-modal Pre-training with Large-scale Weak-supervised Image-Text Data.** <br>
*Di Qi, Lin Su, Jia Song, Edward Cui, Taroon Bharti, Arun Sacheti.*<br>
[[paper]](https://arxiv.org/abs/2001.07966)

**(*arXiv2020_MSB*) Measuring Social Biases in Grounded Vision and Language Embeddings.** <br>
*Candace Ross, Boris Katz, Andrei Barbu.*<br>
[[paper]](https://arxiv.org/abs/2002.08911)
[[code]](https://github.com/candacelax/bias-in-vision-and-language)

**(*CVPR2020_PREVALENT*) Towards Learning a Generic Agent for Vision-and-Language Navigation via Pre-training.** <br>
*Weituo Hao, Chunyuan Li, Xiujun Li, Lawrence Carin, Jianfeng Gao.*<br>
[[paper]](https://arxiv.org/abs/2002.10638)
[[code]](https://github.com/weituo12321/PREVALENT)

**(*INLG2020_VQG-BERT*) What BERT Sees: Cross-Modal Transfer for Visual Question Generation.** <br>
*Thomas Scialom, Patrick Bordes, Paul-Alexis Dray, Jacopo Staiano, Patrick Gallinari.*<br>
[[paper]](https://arxiv.org/abs/2002.10832)

**(*arXiv2020_XGPT*) XGPT: Cross-modal Generative Pre-Training for Image Captioning.** <br>
*Qiaolin Xia, Haoyang Huang, Nan Duan, Dongdong Zhang, Lei Ji, Zhifang Sui, Edward Cui, Taroon Bharti, Xin Liu, Ming Zhou.*<br>
[[paper]](https://arxiv.org/abs/2003.01473)

**(*arXiv2020_InterBERT*) InterBERT: Vision-and-Language Interaction for Multi-modal Pretraining.** <br>
*Junyang Lin, An Yang, Yichang Zhang, Jie Liu, Jingren Zhou, Hongxia Yang.*<br>
[[paper]](https://arxiv.org/abs/2003.13198)

**(*arXiv2020_Pixel-BERT*) Pixel-BERT: Aligning Image Pixels with Text by Deep Multi-Modal Transformers.** <br>
*Zhicheng Huang, Zhaoyang Zeng, Bei Liu, Dongmei Fu, Jianlong Fu.*<br>
[[paper]](https://arxiv.org/abs/2004.00849)

**(*ECCV2020_Oscar*) Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks.** <br>
*Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, Yejin Choi, Jianfeng Gao.*<br>
[[paper]](https://arxiv.org/abs/2004.06165)
[[code]](https://github.com/microsoft/Oscar)

**(*arXiv2020_MMF*) Are we pretraining it right? Digging deeper into visio-linguistic pretraining.** <br>
*Amanpreet Singh, Vedanuj Goswami, Devi Parikh.*<br>
[[paper]](https://arxiv.org/abs/2004.08744)
[[code]](https://github.com/facebookresearch/mmf)

**(*ACMMM2020_MMNas*) Deep Multimodal Neural Architecture Search.** <br>
*Zhou Yu, Yuhao Cui, Jun Yu, Meng Wang, Dacheng Tao, Qi Tian.*<br>
[[paper]](https://arxiv.org/abs/2004.12070)
[[code]](https://github.com/MILVLG/mmnas/)

**(*EMNLP2020_VD-BERT*) VD-BERT: A Unified Vision and Dialog Transformer with BERT.** <br>
*Yue Wang, Shafiq Joty, Michael R. Lyu, Irwin King, Caiming Xiong, Steven C.H. Hoi.*<br>
[[paper]](https://arxiv.org/abs/2004.13278)
[[code]](https://github.com/salesforce/VD-BERT)

**(*ECCV2020_VALUE*) Behind the Scene: Revealing the Secrets of Pre-trained Vision-and-Language Models.** <br>
*Jize Cao, Zhe Gan, Yu Cheng, Licheng Yu, Yen-Chun Chen, Jingjing Liu.*<br>
[[paper]](https://arxiv.org/abs/2005.07310)
[[code]](https://github.com/JizeCao/VALUE)

**(*ACLSRW2020_AT*) Adaptive Transformers for Learning Multimodal Representations.** <br>
*Prajjwal Bhargava.*<br>
[[paper]](https://arxiv.org/abs/2005.07486)
[[code]](https://github.com/prajjwal1/adaptive_transformer)

**(*NeurIPS2020_VILLA*) Large-Scale Adversarial Training for Vision-and-Language Representation Learning.** <br>
*Zhe Gan, Yen-Chun Chen, Linjie Li, Chen Zhu, Yu Cheng, Jingjing Liu.*<br>
[[paper]](https://arxiv.org/abs/2006.06195)
[[code]](https://github.com/zhegan27/VILLA)

**(*CVPR2021_VirTex*) VirTex: Learning Visual Representations from Textual Annotations.** <br>
*Karan Desai, Justin Johnson.*<br>
[[paper]](https://arxiv.org/abs/2006.06666)
[[code]](https://github.com/kdexd/virtex)

**(*AAAI2021_ERNIE-ViL*) ERNIE-ViL: Knowledge Enhanced Vision-Language Representations Through Scene Graph.** <br>
*Fei Yu, Jiji Tang, Weichong Yin, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang.*<br>
[[paper]](https://arxiv.org/abs/2006.16934)

**(*ACMMM2020_DeVLBert*) DeVLBert: Learning Deconfounded Visio-Linguistic Representations.** <br>
*Shengyu Zhang, Tan Jiang, Tan Wang, Kun Kuang, Zhou Zhao, Jianke Zhu, Jin Yu, Hongxia Yang, Fei Wu.*<br>
[[paper]](https://arxiv.org/abs/2008.06884)
[[code]](https://github.com/shengyuzhang/DeVLBert)

**(*Access2021_RVL-BERT*) Visual Relationship Detection With Visual-Linguistic Knowledge From Multimodal Representations.** <br>
*Meng-Jiun Chiou, Roger Zimmermann, Jiashi Feng.*<br>
[[paper]](https://arxiv.org/abs/2009.04965)
[[code]](https://github.com/coldmanck/RVL-BERT)

**(*EMNLP2020_X-LXMERT*) X-LXMERT: Paint, Caption and Answer Questions with Multi-Modal Transformers.** <br>
*Jaemin Cho, Jiasen Lu, Dustin Schwenk, Hannaneh Hajishirzi, Aniruddha Kembhavi.*<br>
[[paper]](https://arxiv.org/abs/2009.11278)
[[code]](https://prior.allenai.org/projects/x-lxmert)

**(*arXiv2020_CAPT*) CAPT: Contrastive Pre-Training for Learning Denoised Sequence Representations.** <br>
*Fuli Luo, Pengcheng Yang, Shicheng Li, Xuancheng Ren, Xu Sun.*<br>
[[paper]](https://arxiv.org/abs/2010.06351)

**(*EMNLP2020_STL-CQA*) STL-CQA: Structure-based Transformers with Localization and Encoding for Chart Question Answering.** <br>
*Hrituraj Singh, Sumit Shekhar.*<br>
[[paper]](https://aclanthology.org/2020.emnlp-main.264.pdf)

**(*CVPR2020_ActBERT*) ActBERT: Learning Global-Local Video-Text Representations.** <br>
*Linchao Zhu, Yi Yang.*<br>
[[paper]](https://arxiv.org/abs/2011.07231)

**(*TACL2021_MPU*) Multimodal Pretraining Unmasked: A Meta-Analysis and a Unified Framework of Vision-and-Language BERTs.** <br>
*Emanuele Bugliarello, Ryan Cotterell, Naoaki Okazaki, Desmond Elliott.*<br>
[[paper]](https://arxiv.org/abs/2011.15124)
[[code]](https://github.com/e-bug/mpre-unmasked)

**(*arXiv2020_LAMP*) LAMP: Label Augmented Multimodal Pretraining.** <br>
*Jia Guo, Chen Zhu, Yilun Zhao, Heda Wang, Yao Hu, Xiaofei He, Deng Cai.*<br>
[[paper]](https://arxiv.org/abs/2012.04446)

**(*arXiv2020_MANGO*) A Closer Look at the Robustness of Vision-and-Language Pre-trained Models.** <br>
*Linjie Li, Zhe Gan, Jingjing Liu.*<br>
[[paper]](https://arxiv.org/abs/2012.08673)

**(*ACL2021_UNIMO*) UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contrastive Learning.** <br>
*Wei Li, Can Gao, Guocheng Niu, Xinyan Xiao, Hao Liu, Jiachen Liu, Hua Wu, Haifeng Wang.*<br>
[[paper]](https://arxiv.org/abs/2012.15409)
[[code]](https://github.com/PaddlePaddle/Research/tree/master/NLP/UNIMO)

**(*CVPR2021_VinVL*) VinVL: Revisiting Visual Representations in Vision-Language Models.** <br>
*Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, Jianfeng Gao.*<br>
[[paper]](https://arxiv.org/abs/2101.00529)
[[code]](https://github.com/pzzhang/VinVL)

**(*AAAI2021_VisualMRC*) VisualMRC: Machine Reading Comprehension on Document Images.** <br>
*Ryota Tanaka, Kyosuke Nishida, Sen Yoshida.*<br>
[[paper]](https://arxiv.org/abs/2101.11272)
[[code]](https://github.com/nttmdlab-nlp/VisualMRC)

**(*AAAI2021_TDEN*) Scheduled Sampling in Vision-Language Pretraining with Decoupled Encoder-Decoder Network.** <br>
*Yehao Li, Yingwei Pan, Ting Yao, Jingwen Chen, Tao Mei.*<br>
[[paper]](https://arxiv.org/abs/2101.11562)
[[code]](https://github.com/YehLi/TDEN)

**(*ICML2021_VL-BART*) Unifying Vision-and-Language Tasks via Text Generation.** <br>
*Jaemin Cho, Jie Lei, Hao Tan, Mohit Bansal.*<br>
[[paper]](https://arxiv.org/abs/2102.02779)
[[code]](https://github.com/j-min/VL-T5)

**(*ICML2021_ViLT*) ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision.** <br>
*Wonjae Kim, Bokyung Son, Ildoo Kim.*<br>
[[paper]](https://arxiv.org/abs/2102.03334)
[[code]](https://github.com/dandelin/vilt)

**(*ICML2021_ALIGN*) Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision.** <br>
*Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig.*<br>
[[paper]](https://arxiv.org/abs/2102.05918)
[[blog]](https://ai.googleblog.com/2021/05/align-scaling-up-visual-and-vision.html)

**(*CVPR2021_ClipBERT*) Less is More: ClipBERT for Video-and-Language Learning via Sparse Sampling.** <br>
*Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L. Berg, Mohit Bansal, Jingjing Liu.*<br>
[[paper]](https://arxiv.org/abs/2102.06183)
[[code]](https://github.com/jayleicn/ClipBERT)

**(*ICCV2021_UniT*) UniT: Multimodal Multitask Learning with a Unified Transformer.** <br>
*Ronghang Hu, Amanpreet Singh.*<br>
[[paper]](https://arxiv.org/abs/2102.10772)
[[code]](https://github.com/facebookresearch/mmf/tree/main/projects/unit)

**(*arXiv2021_CLIP*) Learning Transferable Visual Models From Natural Language Supervision.** <br>
*Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever.*<br>
[[paper]](https://arxiv.org/abs/2103.00020)
[[code]](https://github.com/OpenAI/CLIP)

**(*arXiv2021_SemVLP*) SemVLP: Vision-Language Pre-training by Aligning Semantics at Multiple Levels.** <br>
*Chenliang Li, Ming Yan, Haiyang Xu, Fuli Luo, Wei Wang, Bin Bi, Songfang Huang.*<br>
[[paper]](https://arxiv.org/abs/2103.07829)

**(*NAACL2021_LightningDOT*) LightningDOT: Pre-training Visual-Semantic Embeddings for Real-Time Image-Text Retrieval** <br>
*Siqi Sun, Yen-Chun Chen, Linjie Li, Shuohang Wang, Yuwei Fang, Jingjing Liu.*<br>
[[paper]](https://arxiv.org/abs/2103.08784)
[[code]](https://github.com/intersun/LightningDOT)

**(*CVPR2021_Fast&Slow*) Thinking Fast and Slow: Efficient Text-to-Visual Retrieval with Transformers.** <br>
*Antoine Miech, Jean-Baptiste Alayrac, Ivan Laptev, Josef Sivic, Andrew Zisserman.*<br>
[[paper]](https://arxiv.org/abs/2103.16553)

**(*CVPR2021_UC2*) UC2: Universal Cross-lingual Cross-modal Vision-and-Language Pre-training.** <br>
*Mingyang Zhou, Luowei Zhou, Shuohang Wang, Yu Cheng, Linjie Li, Zhou Yu, Jingjing Liu.*<br>
[[paper]](https://arxiv.org/abs/2104.00332)

**(*ICCV2021_DistillVLM*) Compressing Visual-linguistic Model via Knowledge Distillation.** <br>
*Zhiyuan Fang, Jianfeng Wang, Xiaowei Hu, Lijuan Wang, Yezhou Yang, Zicheng Liu.*<br>
[[paper]](https://arxiv.org/abs/2104.02096)

**(*CVPR2021_SOHO*) Seeing Out of tHe bOx: End-to-End Pre-training for Vision-Language Representation Learning.** <br>
*Zhicheng Huang, Zhaoyang Zeng, Yupan Huang, Bei Liu, Dongmei Fu, Jianlong Fu.*<br>
[[paper]](https://arxiv.org/abs/2104.03135)
[[code]](https://github.com/researchmm/soho)

**(*EMNLP2021_GLUE*) Effect of Visual Extensions on Natural Language Understanding in Vision-and-Language Models.** <br>
*Taichi Iki, Akiko Aizawa.*<br>
[[paper]](https://arxiv.org/abs/2104.08066)
[[code]](https://github.com/alab-nii/eval_vl_glue)

**(*CVPR2021_MCT*) Multimodal Contrastive Training for Visual Representation Learning.** <br>
*Xin Yuan, Zhe Lin, Jason Kuen, Jianming Zhang, Yilin Wang, Michael Maire, Ajinkya Kale, Baldo Faieta.*<br>
[[paper]](https://arxiv.org/abs/2104.12836)

**(*ACLIJCNLP2021_IAIS*) Learning Relation Alignment for Calibrated Cross-modal Retrieval.** <br>
*Shuhuai Ren, Junyang Lin, Guangxiang Zhao, Rui Men, An Yang, Jingren Zhou, Xu Sun, Hongxia Yang.*<br>
[[paper]](https://arxiv.org/abs/2105.13868)
[[code]](https://github.com/lancopku/IAIS)

**(*arXiv2021_CLIP2Video*) CLIP2Video: Mastering Video-Text Retrieval via Image CLIP.** <br>
*Han Fang, Pengfei Xiong, Luhui Xu, Yu Chen.*<br>
[[paper]](https://arxiv.org/abs/2106.11097)
[[code]](https://github.com/CryhanFang/CLIP2Video)

**(*SIGIR2021_GilBERT*) GilBERT: Generative Vision-Language Pre-Training for Image-Text Retrieval.** <br>
*Weixiang Hong, Kaixiang Ji, Jiajia Liu, Jian Wang, Jingdong Chen, Wei Chu.*<br>
[[paper]](https://dl.acm.org/doi/pdf/10.1145/3404835.3462838)

**(*NeurIPS2021_ALBEF*) Align before Fuse: Vision and Language Representation Learning with Momentum Distillation.** <br>
*Junnan Li, Ramprasaath R. Selvaraju, Akhilesh Deepak Gotmare, Shafiq Joty, Caiming Xiong, Steven Hoi.*<br>
[[paper]](https://arxiv.org/abs/2107.07651)
[[code]](https://github.com/salesforce/ALBEF)

**(*NeurIPS2021_Frozen*) Multimodal Few-Shot Learning with Frozen Language Models.** <br>
*Maria Tsimpoukelli, Jacob Menick, Serkan Cabi, S. M. Ali Eslami, Oriol Vinyals, Felix Hill.*<br>
[[paper]](https://openreview.net/pdf?id=WtmMyno9Tq2)
[[project]](https://fh295.github.io/frozen.html)

**(*arXiv2021_SimVLM*) SimVLM: Simple Visual Language Model Pretraining with Weak Supervision.** <br>
*Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, Yuan Cao.*<br>
[[paper]](https://arxiv.org/abs/2108.10904)

**(*arXiv2021_CAMoE*) Improving Video-Text Retrieval by Multi-Stream Corpus Alignment and Dual Softmax Loss.** <br>
*Xing Cheng, Hezheng Lin, Xiangyu Wu, Fan Yang, Dong Shen.*<br>
[[paper]](https://arxiv.org/abs/2109.04290)

**(*arXiv2021_MURAL*) MURAL: Multimodal, Multitask Retrieval Across Languages.** <br>
*Aashi Jain, Mandy Guo, Krishna Srinivasan, Ting Chen, Sneha Kudugunta, Chao Jia, Yinfei Yang, Jason Baldridge.*<br>
[[paper]](https://arxiv.org/abs/2109.05125)

**(*arXiv2021_KD-VLP*) KD-VLP: Improving End-to-End Vision-and-Language Pretraining with Object Knowledge Distillation.** <br>
*Yongfei Liu, Chenfei Wu, Shao-yen Tseng, Vasudev Lal, Xuming He, Nan Duan.*<br>
[[paper]](https://arxiv.org/abs/2109.10504)

**(*CIKM2021_TDMR*) Student Can Also be a Good Teacher: Extracting Knowledge from Vision-and-Language Model for Cross-Modal Retrieval.** <br>
*Jun Rao, Tao Qian, Shuhan Qi, Yulin Wu, Qing Liao, Xuan Wang.*<br>
[[paper]](https://dl.acm.org/doi/pdf/10.1145/3459637.3482194)

**(*ICCV2021_COOKIE*) COOKIE: Contrastive Cross-Modal Knowledge Sharing Pre-Training for Vision-Language Representation.** <br>
*Keyu Wen, Jin Xia, Yuanyuan Huang, Linyang Li, Jiayan Xu, Jie Shao.*<br>
[[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wen_COOKIE_Contrastive_Cross-Modal_Knowledge_Sharing_Pre-Training_for_Vision-Language_Representation_ICCV_2021_paper.pdf)
[[code]](https://github.com/kywen1119/COOKIE)

**(*arXiv2021_DeCLIP*) Supervision Exists Everywhere: A Data Efficient Contrastive Language-Image Pre-training Paradigm.** <br>
*Yangguang Li, Feng Liang, Lichen Zhao, Yufeng Cui, Wanli Ouyang, Jing Shao, Fengwei Yu, Junjie Yan.*<br>
[[paper]](https://arxiv.org/abs/2110.05208)

**(*arXiv2021_VLDeformer*) VLDeformer: Learning Visual-Semantic Embeddings by Vision-Language Transformer Decomposing.** <br>
*Lisai Zhang, Hongfa Wu, Qingcai Chen, Yimeng Deng, Zhonghua Li, Dejiang Kong, Zhao Cao, Joanna Siebert, Yunpeng Han.*<br>
[[paper]](https://arxiv.org/abs/2110.11338)

**(*arXiv2021_VLMo*) VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts.** <br>
*Wenhui Wang, Hangbo Bao, Li Dong, Furu Wei.*<br>
[[paper]](https://arxiv.org/abs/2111.02358)
[[code]](https://github.com/microsoft/unilm)

**(*arXiv2021_TAGS*) Negative Sample is Negative in Its Own Way: Tailoring Negative Sentences for Image-Text Retrieval.** <br>
*Zhihao Fan, Zhongyu Wei, Zejun Li, Siyuan Wang, Jianqing Fan.*<br>
[[paper]](https://arxiv.org/abs/2111.03349)
[[code]](https://github.com/LibertFan/TAGS)

**(*arXiv2021_CLIP2TV*) CLIP2TV: An Empirical Study on Transformer-based Methods for Video-Text Retrieval.** <br>
*Zijian Gao, Jingyu Liu, Sheng Chen, Dedan Chang, Hao Zhang, Jinwei Yuan.*<br>
[[paper]](https://arxiv.org/abs/2111.05610)

**(*arXiv2021_FILIP*) FILIP: Fine-grained Interactive Language-Image Pre-Training.** <br>
*Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, Chunjing Xu.*<br>
[[paper]](https://arxiv.org/abs/2111.07783)

**(*arXiv2021_LiT*) LiT: Zero-Shot Transfer with Locked-image Text Tuning.** <br>
*Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, Lucas Beyer.*<br>
[[paper]](https://arxiv.org/abs/2111.07991)

**(*arXiv2021_X-VLM*) Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts.** <br>
*Yan Zeng, Xinsong Zhang, Hang Li.*<br>
[[paper]](https://arxiv.org/abs/2111.08276)
[[code]](https://github.com/zengyan-97/x-vlm)

**(*arXiv2021_Florence*) Florence: A New Foundation Model for Computer Vision.** <br>
*Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, Ce Liu, Mengchen Liu, Zicheng Liu, Yumao Lu, Yu Shi, Lijuan Wang, Jianfeng Wang, Bin Xiao, Zhen Xiao, Jianwei Yang, Michael Zeng, Luowei Zhou, Pengchuan Zhang.*<br>
[[paper]](https://arxiv.org/abs/2111.11432)

**(*arXiv2021_OA-Transformer*) Object-aware Video-language Pre-training for Retrieval.** <br>
*Alex Jinpeng Wang, Yixiao Ge, Guanyu Cai, Rui Yan, Xudong Lin, Ying Shan, Xiaohu Qie, Mike Zheng Shou.*<br>
[[paper]](https://arxiv.org/abs/2112.00656)
[[code]](https://github.com/FingerRec/OA-Transformer)

**(*arXiv2021_RegionLearner*) Video-Text Pre-training with Learned Regions.** <br>
*Rui Yan, Mike Zheng Shou, Yixiao Ge, Alex Jinpeng Wang, Xudong Lin, Guanyu Cai, Jinhui Tang.*<br>
[[paper]](https://arxiv.org/abs/2112.01194)
[[code]](https://github.com/ruiyan1995/Region_Learner)

**(*arXiv2021_ViT-BERT*) Towards a Unified Foundation Model: Jointly Pre-Training Transformers on Unpaired Images and Text.** <br>
*Qing Li, Boqing Gong, Yin Cui, Dan Kondratyuk, Xianzhi Du, Ming-Hsuan Yang, Matthew Brown.*<br>
[[paper]](https://arxiv.org/abs/2112.07074)

**(*ACMMM2021_CoCo-BERT*) CoCo-BERT: Improving Video-Language Pre-training with Contrastive Cross-modal Matching and Denoising.** <br>
*Jianjie Luo, Yehao Li, Yingwei Pan, Ting Yao, Hongyang Chao, Tao Mei.*<br>
[[paper]](https://arxiv.org/abs/2112.07515)

**(*ACLARR_PromptFuse*) Prompting as Multimodal Fusing.** <br>
[[paper]](https://openreview.net/pdf?id=wWZCNLkK-FK)

**(*arXiv2022_LoopITR*) LoopITR: Combining Dual and Cross Encoder Architectures for Image-Text Retrieval.** <br>
*Jie Lei, Xinlei Chen, Ning Zhang, Mengjiao Wang, Mohit Bansal, Tamara L. Berg, Licheng Yu.*<br>
[[paper]](https://arxiv.org/abs/2203.05465)

**(*ACL2022_VLKD*) Enabling Multimodal Generation on CLIP via Vision-Language Knowledge Distillation.** <br>
*Wenliang Dai, Lu Hou, Lifeng Shang, Xin Jiang, Qun Liu, Pascale Fung.*<br>
[[paper]](https://arxiv.org/abs/2203.06386)

**(*ACL2022_CMKT*) Leveraging Visual Knowledge in Language Tasks: An Empirical Study on Intermediate Pre-training for Cross-modal Knowledge Transfer.** <br>
*Woojeong Jin, Dong-Ho Lee, Chenguang Zhu, Jay Pujara, Xiang Ren.*<br>
[[paper]](https://arxiv.org/abs/2203.07519)
[[code]](https://github.com/INK-USC/CMKT)

**(*arXiv2022_DemoVLP*) Revitalize Region Feature for Democratizing Video-Language Pre-training.** <br>
*Guanyu Cai, Yixiao Ge, Alex Jinpeng Wang, Rui Yan, Xudong Lin, Ying Shan, Lianghua He, Xiaohu Qie, Jianping Wu, Mike Zheng Shou.*<br>
[[paper]](https://arxiv.org/abs/2203.07720)
[[code]](https://github.com/showlab/DemoVLP)

**(*CVPR2022_X-Pool*) X-Pool: Cross-Modal Language-Video Attention for Text-Video Retrieval.** <br>
*Satya Krishna Gorti, Noel Vouitsis, Junwei Ma, Keyvan Golestan, Maksims Volkovs, Animesh Garg, Guangwei Yu.*<br>
[[paper]](https://arxiv.org/abs/2203.15086)
[[code]](https://github.com/layer6ai-labs/xpool)
[[project]](https://layer6ai-labs.github.io/xpool/)

**(*CVPR2022_ViSTA*) ViSTA: Vision and Scene Text Aggregation for Cross-Modal Retrieval.** <br>
*Mengjun Cheng, Yipeng Sun, Longchao Wang, Xiongwei Zhu, Kun Yao, Jie Chen, Guoli Song, Junyu Han, Jingtuo Liu, Errui Ding, Jingdong Wang.*<br>
[[paper]](https://arxiv.org/abs/2203.16778)


### ``*Generic-Feature Extraction*``

**(*NeurIPS2013_DeViSE*) DeViSE: A Deep Visual-Semantic Embedding Model.** <br>
*Andrea Frome, Greg S. Corrado, Jonathon Shlens, Samy Bengio, Jeffrey Dean, Marc’Aurelio Ranzato, Tomas Mikolov.*<br>
[[paper]](https://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model.pdf)

**(*TACL2014_SDT-RNN*) Grounded Compositional Semantics for Finding and Describing Images with Sentences.**<br>
*Richard Socher, Andrej Karpathy, Quoc V. Le, Christopher D. Manning, Andrew Y. Ng.*<br>
[[paper]](https://www.aclweb.org/anthology/Q14-1017.pdf)

**(*NeurIPSws2014_UVSE*) Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models.**<br>
*Ryan Kiros, Ruslan Salakhutdinov, Richard S. Zemel.*<br>
[[paper]](https://arxiv.org/abs/1411.2539)
[[code]](https://github.com/ryankiros/visual-semantic-embedding)
[[demo]](http://www.cs.toronto.edu/~rkiros/lstm_scnlm.html)

**(*NeurIPS2014_DeFrag*) Deep fragment embeddings for bidirectional image sentence mapping.**<br>
*Andrej Karpathy, Armand Joulin, Li Fei-Fei.*<br>
[[paper]](https://cs.stanford.edu/people/karpathy/nips2014.pdf)

**(*ICCV2015_m-CNN*) Multimodal Convolutional Neural Networks for Matching Image and Sentence.**<br>
*Lin Ma, Zhengdong Lu, Lifeng Shang, Hang Li.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7410658)

**(*CVPR2015_DCCA*) Deep Correlation for Matching Images and Text.**<br>
*Fei Yan, Krystian Mikolajczyk.*<br>
[[paper]](http://openaccess.thecvf.com/content_cvpr_2015/papers/Yan_Deep_Correlation_for_2015_CVPR_paper.pdf)

**(*CVPR2015_FV*) Associating Neural Word Embeddings with Deep Image Representationsusing Fisher Vectors.**<br>
*Benjamin Klein, Guy Lev, Gil Sadeh, Lior Wolf.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7299073)

**(*CVPR2015_DVSA*) Deep Visual-Semantic Alignments for Generating Image Descriptions.**<br>
*Andrej Karpathy, Li Fei-Fei.*<br>
[[paper]](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)

**(*NeurIPS2015_STV*) Skip-thought Vectors.**<br>
*Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, Sanja Fidler.*<br>
[[paper]](https://arxiv.org/abs/1506.06726)

**(*CVPR2016_SPE*) Learning Deep Structure-Preserving Image-Text Embeddings.**<br>
*Liwei Wang, Yin Li, Svetlana Lazebnik.*<br>
[[paper]](http://slazebni.cs.illinois.edu/publications/cvpr16_structure.pdf)

**(*ICCV2017_HM-LSTM*) Hierarchical Multimodal LSTM for Dense Visual-Semantic Embedding.**<br>
*Zhenxing Niu, Mo Zhou, Le Wang, Xinbo Gao, Gang Hua.*<br>
[[paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Niu_Hierarchical_Multimodal_LSTM_ICCV_2017_paper.pdf)

**(*ICCV2017_RRF-Net*) Learning a Recurrent Residual Fusion Network for Multimodal Matching.**<br>
*Yu Liu, Yanming Guo, Erwin M. Bakker, Michael S. Lew.*<br>
[[paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_a_Recurrent_ICCV_2017_paper.pdf)

**(*CVPR2017_2WayNet*) Linking Image and Text with 2-Way Nets.**<br>
*Aviv Eisenschtat, Lior Wolf.*<br>
[[paper]](https://arxiv.org/abs/1608.07973)

**(*ACMMM2018_WSJE*) Webly Supervised Joint Embedding for Cross-Modal Image-Text Retrieval.**<br>
*Niluthpol Chowdhury Mithun, Rameswar Panda, Evangelos E. Papalexakis, Amit K. Roy-Chowdhury.*<br>
[[paper]](https://arxiv.org/abs/1808.07793)

**(*WACV2018_SEAM*) Fast Self-Attentive Multimodal Retrieval.**<br>
*Jônatas Wehrmann, Maurício Armani Lopes, Martin D More, Rodrigo C. Barros.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8354311&tag=1)
[[code]](https://github.com/jwehrmann/seam-retrieval)

**(*CVPR2018_CSE*) End-to-end Convolutional Semantic Embeddings.**<br>
*Quanzeng You, Zhengyou Zhang, Jiebo Luo.*<br>
[[paper]](https://ai.tencent.com/ailab/media/publications/cvpr/End-to-end_Convolutional_Semantic_Embeddings.pdf)

**(*CVPR2018_CHAIN-VSE*) Bidirectional Retrieval Made Simple.**<br>
*Jonatas Wehrmann, Rodrigo C. Barros.*<br>
[[paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wehrmann_Bidirectional_Retrieval_Made_CVPR_2018_paper.pdf)
[[code]](https://github.com/jwehrmann/chain-vse)

**(*CVPR2018_SCO*) Learning Semantic Concepts and Order for Image and Sentence Matching.**<br>
*Yan Huang, Qi Wu, Liang Wang.*<br>
[[paper]](https://arxiv.org/abs/1712.02036)

**(*NC2019_MDM*) Bidirectional image-sentence retrieval by local and global deep matching.**<br>
*Lin Ma, Wenhao Jiang, Zequn Jie, Xu Wang.*<br>
[[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0925231219301390)

**(*ACMMM2019_SAEM*) Learning Fragment Self-Attention Embeddings for Image-Text Matching.**<br>
*Yiling Wu, Shuhui Wang, Guoli Song, Qingming Huang.*<br>
[[paper]](https://dl.acm.org/doi/pdf/10.1145/3343031.3350940)
[[code]](https://github.com/yiling2018/saem)

**(*ICCV2019_VSRN*) Visual Semantic Reasoning for Image-Text Matching.**<br>
*Kunpeng Li, Yulun Zhang, Kai Li, Yuanyuan Li, Yun Fu.*<br>
[[paper]](https://arxiv.org/abs/1909.02701.pdf)
[[code]](https://github.com/KunpengLi1994/VSRN)

**(*ICCV2019_LIWE*) Language-Agnostic Visual-Semantic Embeddings.**<br>
*Jonatas Wehrmann, Maurício Armani Lopes, Douglas Souza, Rodrigo Barros.*<br>
[[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wehrmann_Language-Agnostic_Visual-Semantic_Embeddings_ICCV_2019_paper.pdf)
[[code]](https://github.com/jwehrmann/lavse)
[[demo]](https://jwehrmann.github.io/projects.lavse/)

**(*CVPR2019_Personality*) Engaging Image Captioning via Personality.**<br>
*Kurt Shuster, Samuel Humeau, Hexiang Hu, Antoine Bordes, Jason Weston.*<br>
[[paper]](https://arxiv.org/abs/1810.10665)

**(*CVPR2019_PVSE*) Polysemous Visual-Semantic Embedding for Cross-Modal Retrieval.**<br>
*Yale Song, Mohammad Soleymani.*<br>
[[paper]](https://arxiv.org/abs/1906.04402)
[[code]](https://github.com/yalesong/pvse)

**(*Access2020_GSLS*) Combining Global and Local Similarity for Cross-Media Retrieval.**<br>
*Zhixin Li, Feng Ling, Canlong Zhang, Huifang Ma.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8970540)

**(*Access2020_M3A*) Multi-Modal Memory Enhancement Attention Network for Image-Text Matching.**<br>
*Zhong Ji, Zhigang Lin, Haoran Wang, Yuqing He.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9006782)

**(*ICPR2020_TERN*) Transformer Reasoning Network for Image-Text Matching and Retrieval.**<br>
*Nicola Messina, Fabrizio Falchi, Andrea Esuli, Giuseppe Amato.*<br>
[[paper]](https://arxiv.org/abs/2004.09144)
[[code]](https://github.com/mesnico/TERN)

**(*TOMM2020_TERAN*) Fine-grained Visual Textual Alignment for Cross-Modal Retrieval using Transformer Encoders.**<br>
*Nicola Messina, Giuseppe Amato, Andrea Esuli, Fabrizio Falchi, Claudio Gennaro, Stéphane Marchand-Maillet.*<br>
[[paper]](https://arxiv.org/abs/2008.05231)
[[code]](https://github.com/mesnico/TERAN)

**(*TOMM2020_NIS*) Upgrading the Newsroom: An Automated Image Selection System for News Articles.**<br>
*Fangyu Liu, Rémi Lebret, Didier Orel, Philippe Sordet, Karl Aberer.*<br>
[[paper]](https://arxiv.org/abs/2004.11449)
[[slides]](http://fangyuliu.me/media/others/lsir_talk_final_version_0.3.pdf)
[[demo]](https://modemos.epfl.ch/article)

**(*TCSVT2020_MFM*) Matching Image and Sentence With Multi-Faceted Representations.**<br>
*Lin Ma, Wenhao Jiang, Zequn Jie, Yu-Gang Jiang, Wei Liu.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8712424)

**(*TCSVT2020_DSRAN*) Learning Dual Semantic Relations with Graph Attention for Image-Text Matching.**<br>
*Keyu Wen, Xiaodong Gu, Qingrong Cheng.*<br>
[[paper]](https://arxiv.org/abs/2010.11550)
[[code]](https://github.com/kywen1119/DSRAN)

**(*ICMR2020_VRACR*) Visual Relations Augmented Cross-modal Retrieval.**<br>
*Yutian Guo, Jingjing Chen, Hao Zhang, Yu-Gang Jiang.*<br>
[[paper]](http://vireo.cs.cityu.edu.hk/jingjing/papers/guo_visual.pdf)

**(*WACV2020_SGM*) Cross-modal Scene Graph Matching for Relationship-aware Image-Text Retrieval.**<br>
*Sijin Wang, Ruiping Wang, Ziwei Yao, Shiguang Shan, Xilin Chen.*<br>
[[paper]](https://arxiv.org/abs/1910.05134)

**(*ACMMM2020_CAMERA*) Context-Aware Multi-View Summarization Network for Image-Text Matching.**<br>
*Leigang Qu, Meng Liu, Da Cao, Liqiang Nie, Qi Tian.*<br>
[[paper]](https://www.researchgate.net/profile/Meng-Liu-67/publication/346201037_Context-Aware_Multi-View_Summarization_Network_for_Image-Text_Matching/links/6052a29f299bf173674e0be6/Context-Aware-Multi-View-Summarization-Network-for-Image-Text-Matching.pdf)
[[code]](https://github.com/LgQu/CAMERA)

**(*arXiv2021_DXR*) Cross-Modal Retrieval Augmentation for Multi-Modal Classification.**<br>
*Shir Gur, Natalia Neverova, Chris Stauffer, Ser-Nam Lim, Douwe Kiela, Austin Reiter.*<br>
[[paper]](https://arxiv.org/abs/2104.08108)

**(*arXiv2021_T-EMDE*) T-EMDE: Sketching-based global similarity for cross-modal retrieval.**<br>
*Barbara Rychalska, Mikolaj Wieczorek, Jacek Dabrowski.*<br>
[[paper]](https://arxiv.org/abs/2105.04242)

**(*arXiv2021_LGSGM*) A Deep Local and Global Scene-Graph Matching for Image-Text Retrieval.**<br>
*Manh-Duy Nguyen, Binh T. Nguyen, Cathal Gurrins.*<br>
[[paper]](https://arxiv.org/abs/2106.02400)

**(*ACLIJCNLP2021_HEI*) Hashing based Efficient Inference for Image-Text Matching.** <br>
*Rong-Cheng Tu, Lei Ji, Huaishao Luo, Botian Shi, Heyan Huang, Nan Duan, Xian-Ling Mao.*<br>
[[paper]](https://aclanthology.org/2021.findings-acl.66.pdf)

**(*CSAE2021_SVSEN*) Super Visual Semantic Embedding for Cross-Modal Image-Text Retrieval.** <br>
*Zhixian Zeng, Jianjun Cao, Guoquan Jiang, Nianfeng Weng, Yuxin Xu, Zibo Nie.*<br>
[[paper]](https://dl.acm.org/doi/pdf/10.1145/3487075.3487167?casa_token=DDhsAGOgZhMAAAAA:o-4O3H_m8sBHd5YbUrGMcOGZIEGSh-N_fMHfJw8xZ_-_a99JJQ58_boxf159U2qp9--zyb2cNHA)

**(*ACMMM2021_SMFEA*) Structured Multi-modal Feature Embedding and Alignment for Image-Sentence Retrieval.**<br>
*Xuri Ge, Fuhai Chen, Joemon M. Jose, Zhilong Ji, Zhongqin Wu, Xiao Liu.*<br>
[[paper]](https://arxiv.org/abs/2108.02417)

**(*IJCAI2021_SSP*) Rethinking Label-Wise Cross-Modal Retrieval from A Semantic Sharing Perspective.**<br>
*Yang Yang, Chubing Zhang, Yi-Chu Xu, Dianhai Yu, De-Chuan Zhan, Jian Yang.*<br>
[[paper]](https://www.ijcai.org/proceedings/2021/0454.pdf)

**(*CVPR2021_PCME*) Probabilistic Embeddings for Cross-Modal Retrieval.**<br>
*Sanghyuk Chun, Seong Joon Oh, Rafael Sampaio de Rezende, Yannis Kalantidis, Diane Larlus.*<br>
[[paper]](https://arxiv.org/abs/2101.05068)
[[code]](https://github.com/naver-ai/pcme)

**(*CVPR2021_PG*) Discrete-continuous Action Space Policy Gradient-based Attention for Image-Text Matching.**<br>
*Shiyang Yan, Li Yu, Yuan Xie.*<br>
[[paper]](https://arxiv.org/abs/2104.10406)
[[code]](https://github.com/Shiyang-Yan/Discrete-continous-PG-for-Retrieval)

**(*CVPR2021_GPO*) Learning the Best Pooling Strategy for Visual Semantic Embedding.**<br>
*Jiacheng Chen, Hexiang Hu, Hao Wu, Yuning Jiang, Changhu Wang.*<br>
[[paper]](https://arxiv.org/abs/2011.04305)
[[code]](https://github.com/woodfrog/vse_infty)

**(*ICCV2021_AACH*) Adversarial Attack on Deep Cross-Modal Hamming Retrieval.**<br>
*Chao Li, Shangqian Gao, Cheng Deng, Wei Liu, Heng Huang.*<br>
[[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Adversarial_Attack_on_Deep_Cross-Modal_Hamming_Retrieval_ICCV_2021_paper.pdf)

**(*ICCV2021_WCGL*) Wasserstein Coupled Graph Learning for Cross-Modal Retrieval.**<br>
*Yun Wang, Tong Zhang, Xueya Zhang, Zhen Cui, Yuge Huang, Pengcheng Shen, Shaoxin Li, Jian Yang.*<br>
[[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Wasserstein_Coupled_Graph_Learning_for_Cross-Modal_Retrieval_ICCV_2021_paper.pdf)

**(*TOMM2022_CGMN*) Cross-modal Graph Matching Network for Image-text Retrieval.**<br>
*Yuhao Cheng, Xiaoguang Zhu, Jiuchao Qian, Fei Wen, Peilin Liu.*<br>
[[paper]](https://dl.acm.org/doi/pdf/10.1145/3499027?casa_token=c93-XuNXLDMAAAAA:N9Z3SIxuEX7sK1cIZVT609ECpTQzSyRYmIX-LosblETTEcDo3ug36apmE_YitI9CVs5T0R3m_aYO)
[[code]](https://github.com/cyh-sj/CGMN)

**(*TPAMI2022_VSRN++*) Image-Text Embedding Learning via Visual and Textual Semantic Reasoning.**<br>
*Kunpeng Li, Yulun Zhang, Kai Li, Yuanyuan Li, Yun Fu.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9706340)


### ``*Cross-Modal Interaction*``

**(*arXiv2014_NIC*) Show and Tell: A Neural Image Caption Generator.**<br>
*Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan.*<br>
[[paper]](https://arxiv.org/abs/1411.4555)

**(*ICLR2015_m-RNN*) Deep Captioning with Multimodal Recurrent Neural Network(M-RNN).**<br>
*Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, Zhiheng Huang, Alan Yuille.*<br>
[[paper]](https://arxiv.org/abs/1412.6632)
[[code]](https://github.com/mjhucla/mRNN-CR)

**(*CVPR2015_LRCN*) Long-term Recurrent Convolutional Networks for Visual Recognition and Description.**<br>
*Jeff Donahue, Lisa Anne Hendricks, Marcus Rohrbach, Subhashini Venugopalan, Sergio Guadarrama, Kate Saenko, Trevor Darrell.*<br>
[[paper]](https://arxiv.org/abs/1411.4389)

**(*CVPR2017_DAN*) Dual Attention Networks for Multimodal Reasoning and Matching.**<br>
*Hyeonseob Nam, Jung-Woo Ha, Jeonghee Kim.*<br>
[[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Nam_Dual_Attention_Networks_CVPR_2017_paper.pdf)

**(*CVPR2017_sm-LSTM*) Instance-aware Image and Sentence Matching with Selective Multimodal LSTM.**<br>
*Yan Huang, Wei Wang, Liang Wang.*<br>
[[paper]](https://arxiv.org/abs/1611.05588)

**(*ECCV2018_CITE*) Conditional Image-Text Embedding Networks.**<br>
*Bryan A. Plummer, Paige Kordas, M. Hadi Kiapour, Shuai Zheng, Robinson Piramuthu, Svetlana Lazebnik.*<br>
[[paper]](https://arxiv.org/abs/1711.08389.pdf)

**(*ECCV2018_SCAN*) Stacked Cross Attention for Image-Text Matching.**<br>
*Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He.*<br>
[[paper]](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Kuang-Huei_Lee_Stacked_Cross_Attention_ECCV_2018_paper.pdf)
[[code]](https://github.com/kuanghuei/SCAN)

**(*CVPR2018_DSVE-Loc*) Finding beans in burgers: Deep semantic-visual embedding with localization.**<br>
*Martin Engilberge, Louis Chevallier, Patrick Pérez, Matthieu Cord.*<br>
[[paper]](https://arxiv.org/abs/1804.01720)

**(*arXiv2019_R-SCAN*) Learning Visual Relation Priors for Image-Text Matching and Image Captioning with Neural Scene Graph Generators.**<br>
*Kuang-Huei Lee, Hamid Palang, Xi Chen, Houdong Hu, Jianfeng Gao.*<br> 
[[paper]](https://arxiv.org/abs/1909.09953)

**(*arXiv2019_ParNet*) ParNet: Position-aware Aggregated Relation Network for Image-Text matching.**<br>
*Yaxian Xia, Lun Huang, Wenmin Wang, Xiaoyong Wei, Jie Chen.*<br> 
[[paper]](https://arxiv.org/abs/1906.06892)

**(*arXiv2019_TOD-Net*) Target-Oriented Deformation of Visual-Semantic Embedding Space.**<br>
*Takashi Matsubara.*<br>
[[paper]](https://arxiv.org/abs/1910.06514)

**(*ACML2019_SAVE*) Multi-Scale Visual Semantics Aggregation with Self-Attention for End-to-End Image-Text Matching.**<br>
*Zhuobin Zheng, Youcheng Ben, Chun Yuan.*<br>
[[paper]](http://proceedings.mlr.press/v101/zheng19a/zheng19a.pdf)

**(*ICMR2019_OAN*) Improving What Cross-Modal Retrieval Models Learn through Object-Oriented Inter- and Intra-Modal Attention Networks.**<br>
*Po-Yao Huang, Vaibhav, Xiaojun Chang, Alexander Georg Hauptmann.*<br>
[[paper]](https://dl.acm.org/doi/pdf/10.1145/3323873.3325043)
[[code]](https://github.com/idejie/OAN)

**(*ACMMM2019_BFAN*) Focus Your Attention: A Bidirectional Focal Attention Network for Image-Text Matching.**<br>
*Chunxiao Liu, Zhendong Mao, An-An Liu, Tianzhu Zhang, Bin Wang, Yongdong Zhang.*<br>
[[paper]](https://arxiv.org/abs/1909.11416)
[[code]](https://github.com/CrossmodalGroup/BFAN) 

**(*ACMMM2019_MTFN*) Matching Images and Text with Multi-modal Tensor Fusion and Re-ranking.**<br>
*Tan Wang, Xing Xu, Yang Yang, Alan Hanjalic, Heng Tao Shen, Jingkuan Song.*<br>
[[paper]](https://arxiv.org/abs/1908.04011)
[[code]](https://github.com/Wangt-CN/MTFN-RR-PyTorch-Code) 

**(*IJCAI2019_RDAN*) Multi-Level Visual-Semantic Alignments with Relation-Wise Dual Attention Network for Image and Text Matching.** <br>
*Zhibin Hu, Yongsheng Luo,Jiong Lin,Yan Yan, Jian Chen.*<br>
[[paper]](https://www.ijcai.org/proceedings/2019/0111.pdf)

**(*IJCAI2019_PFAN*) Position Focused Attention Network for Image-Text Matching.**<br>
*Yaxiong Wang, Hao Yang, Xueming Qian, Lin Ma, Jing Lu, Biao Li, Xin Fan.*<br>
[[paper]](https://arxiv.org/abs/1907.09748)
[[code]](https://github.com/HaoYang0123/Position-Focused-Attention-Network) 

**(*ICCV2019_CAMP*) CAMP: Cross-Modal Adaptive Message Passing for Text-Image Retrieval.**<br>
*Zihao Wang, Xihui Liu, Hongsheng Li, Lu Sheng, Junjie Yan, Xiaogang Wang, Jing Shao.*<br>
[[paper]](https://arxiv.org/abs/1909.05506)
[[code]](https://github.com/ZihaoWang-CV/CAMP_iccv19)

**(*ICCV2019_SAN*) Saliency-Guided Attention Network for Image-Sentence Matching.**<br>
*Zhong Ji, Haoran Wang, Jungong Han, Yanwei Pang.*<br>
[[paper]](https://arxiv.org/abs/1904.09471)
[[code]](https://github.com/HabbakukWang1103/SAN)

**(*TC2020_SMAN*) SMAN: Stacked Multimodal Attention Network for Cross-Modal Image-Text Retrieval.**<br>
*Zhong Ji, Haoran Wang, Jungong Han, Yanwei Pang.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9086164)

**(*TMM2020_PFAN++*) PFAN++: Bi-Directional Image-Text Retrieval with Position Focused Attention Network.**<br>
*Yaxiong Wang, Hao Yang, Xiuxiu Bai, Xueming Qian, Lin Ma, Jing Lu, Biao Li, Xin Fan.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9200698)
[[code]](https://github.com/HaoYang0123/Position-Focused-Attention-Network) 

**(*TNNLS2020_CASC*) Cross-Modal Attention With Semantic Consistence for Image-Text Matching.**<br>
*Xing Xu, Tan Wang, Yang Yang, Lin Zuo, Fumin Shen, Heng Tao Shen.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8994196)
[[code]](https://github.com/Wangt-CN/Code_CASC) 

**(*AAAI2020_DP-RNN*) Expressing Objects just like Words: Recurrent Visual Embedding for Image-Text Matching.** <br>
*Tianlang Chen, Jiebo Luo.*<br>
[[paper]](https://arxiv.org/abs/2002.08510)

**(*AAAI2020_ADAPT*) Adaptive Cross-modal Embeddings for Image-Text Alignment.**<br>
*Jonatas Wehrmann, Camila Kolling, Rodrigo C Barros.*<br>
[[paper]](https://ojs.aaai.org//index.php/AAAI/article/view/6915)
[[code]](https://github.com/jwehrmann/retrieval.pytorch) 

**(*CVPR2020_CAAN*) Context-Aware Attention Network for Image-Text Retrieval.**<br>
*Qi Zhang, Zhen Lei, Zhaoxiang Zhang, Stan Z. Li.*<br>
[[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Context-Aware_Attention_Network_for_Image-Text_Retrieval_CVPR_2020_paper.pdf)

**(*CVPR2020_MMCA*) Multi-Modality Cross Attention Network for Image and Sentence Matching.**<br>
*Xi Wei, Tianzhu Zhang, Yan Li, Yongdong Zhang, Feng Wu.*<br>
[[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Multi-Modality_Cross_Attention_Network_for_Image_and_Sentence_Matching_CVPR_2020_paper.pdf)

**(*CVPR2020_IMRAM*) IMRAM: Iterative Matching with Recurrent Attention Memory for Cross-Modal Image-Text Retrieval.**<br>
*Hui Chen, Guiguang Ding, Xudong Liu, Zijia Lin, Ji Liu, Jungong Han.*<br>
[[paper]](https://arxiv.org/abs/2003.03772)
[[code]](https://github.com/HuiChen24/IMRAM)

**(*arXiv2021_CCRS*) More Than Just Attention: Learning Cross-Modal Attentions with Contrastive Constraints.**<br>
*Yuxiao Chen, Jianbo Yuan, Long Zhao, Rui Luo, Larry Davis, Dimitris N. Metaxas.*<br>
[[paper]](https://arxiv.org/abs/2105.09597)

**(*arXiv2021_SSAMT*) Constructing Phrase-level Semantic Labels to Form Multi-Grained Supervision for Image-Text Retrieval.** <br>
*Zhihao Fan, Zhongyu Wei, Zejun Li, Siyuan Wang, Haijun Shan, Xuanjing Huang, Jianqing Fan.*<br>
[[paper]](https://arxiv.org/abs/2109.05523)

**(*arXiv2021_SwAMP*) SwAMP: Swapped Assignment of Multi-Modal Pairs for Cross-Modal Retrieval.** <br>
*Minyoung Kim.*<br>
[[paper]](https://arxiv.org/abs/2111.05814)

**(*BMVC2021_RELAX*) Image-Text Alignment using Adaptive Cross-attention with Transformer Encoder for Scene Graphs.** <br>
*Juyong Song, Sunghyun Choi.*<br>
[[paper]](https://www.bmvc2021-virtualconference.com/assets/papers/0117.pdf)

**(*ACMMM2021_CSCC*) Conceptual and Syntactical Cross-modal Alignment with Cross-level Consistency for Image-Text Matching.** <br>
*Pengpeng Zeng, Lianli Gao, Xinyu Lyu, Shuaiqi Jing, Jingkuan Song.*<br>
[[paper]](https://dl.acm.org/doi/pdf/10.1145/3474085.3475380)

**(*IJCAI2021_SHAN*) Step-Wise Hierarchical Alignment Network for Image-Text Matching.**<br>
*Zhong Ji, Kexin Chen, Haoran Wang.*<br>
[[paper]](https://arxiv.org/abs/2106.06509)

**(*EMNLP2021_ISERI*) Inflate and Shrink: Enriching and Reducing Interactions for Fast Text-Image Retrieval.**<br>
*Haoliang Liu, Tan Yu, Ping Li.*<br>
[[paper]](https://aclanthology.org/2021.emnlp-main.772.pdf)

**(*TIP2021_MEMBER*) Memorize, Associate and Match: Embedding Enhancement via Fine-Grained Alignment for Image-Text Retrieval.**<br>
*Jiangtong Li, Liu Liu, Li Niu, and Liqing Zhang.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9605203)

**(*SIGIR2021_HAN*) Heterogeneous Attention Network for Effective and Efficient Cross-modal Retrieval.** <br>
*Tan Yu, Yi Yang, Yi Li, Lin Liu, Hongliang Fei, Ping Li.*<br>
[[paper]](https://dl.acm.org/doi/pdf/10.1145/3404835.3462924?casa_token=j-BtmZk6d2YAAAAA:40RmrMEWtPibbu7E0bIj07LhFCDaXPCt95ZEMg-86pcVd4K9uAKKDMHWVbe9pxHc2Rhoh9t6NDorBw)

**(*SIGIR2021_CAEMCL*) Cross-Graph Attention Enhanced Multi-Modal Correlation Learning for Fine-Grained Image-Text Retrieval.** <br>
*Yi He, Xin Liu, Yiu-Ming Cheung, Shu-Juan Peng, Jinhan Yi, Wentao Fan.*<br>
[[paper]](https://dl.acm.org/doi/pdf/10.1145/3404835.3463031)

**(*SIGIR2021_DIME*) Dynamic Modality Interaction Modeling for Image-Text Retrieval.** <br>
*Leigang Qu, Meng Liu, Jianlong Wu, Zan Gao, Liqiang Nie.*<br>
[[paper]](https://dl.acm.org/doi/pdf/10.1145/3404835.3462829)
[[code]](https://github.com/LgQu/DIME)

**(*TMM2022_UARDAN*) Unified Adaptive Relevance Distinguishable Attention Network for Image-Text Matching.** <br>
*Kun Zhang, Zhendong Mao, An-An Liu, Yongdong Zhang.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9676463)

**(*WACV2022_GraDual*) GraDual: Graph-based Dual-modal Representation for Image-Text Matching.** <br>
*Siqu Long, Soyeon Caren Han, Xiaojun Wan, Josiah Poon.*<br>
[[paper]](https://openaccess.thecvf.com/content/WACV2022/papers/Long_GraDual_Graph-Based_Dual-Modal_Representation_for_Image-Text_Matching_WACV_2022_paper.pdf)

**(*AAAI2022_CMCAN*) Show Your Faith: Cross-Modal Confidence-Aware Network for Image-Text Matching.** <br>
*Huatian Zhang, Zhendong Mao, Kun Zhang, Yongdong Zhang.*<br>
[[paper]](https://www.aaai.org/AAAI22Papers/AAAI-2029.ZhangH.pdf)
[[code]](https://github.com/CrossmodalGroup/CMCAN)


### ``*Similarity Measurement*``

**(*ICLR2016_Order-emb*) Order-Embeddings of Images and Language.**<br>
*Ivan Vendrov, Ryan Kiros, Sanja Fidler, Raquel Urtasun.*<br>
[[paper]](https://arxiv.org/abs/1511.06361)
[[code]](https://github.com/ivendrov/order-embedding)

**(*CVPR2020_HOAD*) Visual-Semantic Matching by Exploring High-Order Attention and Distraction.**<br>
*Yongzhi Li, Duo Zhang, Yadong Mu.*<br>
[[paper]](https://pkumyd.github.io/paper/CVPR2020_LYZ.pdf)

**(*CVPR2020_GSMN*) Graph Structured Network for Image-Text Matching.**<br>
*Chunxiao Liu, Zhendong Mao, Tianzhu Zhang, Hongtao Xie, Bin Wang, Yongdong Zhang.*<br>
[[paper]](https://arxiv.org/abs/2004.00277)
[[code]](https://github.com/CrossmodalGroup/GSMN)

**(*ICML2020_GOT*) Graph Optimal Transport for Cross-Domain Alignment.**<br>
*Liqun Chen, Zhe Gan, Yu Cheng, Linjie Li, Lawrence Carin, Jingjing Liu.*<br>
[[paper]](https://arxiv.org/abs/2006.14744)
[[code]](https://github.com/LiqunChen0606/Graph-Optimal-Transport)

**(*EMNLP2020_WD-Match*) Wasserstein Distance Regularized Sequence Representation for Text Matching in Asymmetrical Domains.**<br>
*Weijie Yu, Chen Xu, Jun Xu, Liang Pang, Xiaopeng Gao, Xiaozhao Wang, Ji-Rong Wen.*<br>
[[paper]](https://arxiv.org/abs/2010.07717)
[[code]](https://github.com/RUC-WSM/WD-Match)

**(*AAAI2021_SGRAF*) Similarity Reasoning and Filtration for Image-Text Matching.**<br>
*Haiwen Diao, Ying Zhang, Lin Ma, Huchuan Lu.*<br>
[[paper]](https://arxiv.org/abs/2101.01368)
[[code]](https://github.com/Paranioar/SGRAF)

**(*arXiv2022_TSHSR*) Two-stream Hierarchical Similarity Reasoning for Image-text Matching.**<br>
*Ran Chen, Hanli Wang, Lei Wang, Sam Kwong.*<br>
[[paper]](https://arxiv.org/abs/2203.05349)


### ``*Commonsense Learning*``

**(*KSEM2019_SCKR*) Semantic Modeling of Textual Relationships in Cross-Modal Retrieval.**<br>
*Jing Yu, Chenghao Yang, Zengchang Qin, Zhuoqian Yang, Yue Hu, Weifeng Zhang.*<br>
[[paper]](https://arxiv.org/abs/1810.13151)
[[code]](https://github.com/yzhq97/SCKR)

**(*IJCAI2019_SCG*) Knowledge Aware Semantic Concept Expansion for Image-Text Matching.**<br>
*Botian Shi, Lei Ji, Pan Lu, Zhendong Niu, Nan Duan.*<br>
[[paper]](https://www.ijcai.org/Proceedings/2019/0720.pdf)

**(*ECCV2020_CVSE*) Consensus-Aware Visual-Semantic Embedding for Image-Text Matching.**<br>
*Haoran Wang, Ying Zhang, Zhong Ji, Yanwei Pang, Lin Ma.*<br>
[[paper]](https://arxiv.org/abs/2007.08883)
[[code]](https://github.com/BruceW91/CVSE)


### ``*Adversarial Learning*``

**(*ACMMM2017_ACMR*) Adversarial Cross-Modal Retrieval.**<br>
*Bokun Wang, Yang Yang, Xing Xu, Alan Hanjalic, Heng Tao Shen.*<br>
[[paper]](https://dl.acm.org/doi/pdf/10.1145/3123266.3123326?casa_token=mDis4biJCswAAAAA:fDohxTTh3xaD64uBjbnLAOK6T_nShFY803qbaF2mfRlx6Lcq90Ax4Lyo1Pk4lstv7XweURpjrhJd)
[[code]](https://github.com/sunpeng981712364/ACMR_demo)

**(*COLING2018_CAS*) Learning Visually-Grounded Semantics from Contrastive Adversarial Samples.**<br>
*Haoyue Shi, Jiayuan Mao, Tete Xiao, Yuning Jiang, Jian Sun.*<br>
[[paper]](https://aclweb.org/anthology/C18-1315)
[[code]](https://github.com/ExplorerFreda/VSE-C)

**(*CVPR2018_GXN*) Look, Imagine and Match: Improving Textual-Visual Cross-Modal Retrieval with Generative Models.**<br>
*Jiuxiang Gu, Jianfei Cai, Shafiq Joty, Li Niu, Gang Wang.*<br>
[[paper]](https://arxiv.org/abs/1711.06420)

**(*ICCV2019_TIMAM*) Adversarial Representation Learning for Text-to-Image Matching.**<br>
*Nikolaos Sarafianos, Xiang Xu, Ioannis A. Kakadiaris.*<br>
[[paper]](https://arxiv.org/abs/1908.10534)

**(*CVPR2019_UniVSE*) Unified Visual-Semantic Embeddings: Bridging Vision and Language with Structured Meaning Representations.**<br>
*Hao Wu, Jiayuan Mao, Yufeng Zhang, Yuning Jiang, Lei Li, Weiwei Sun, Wei-Ying Ma.*<br>
[[paper]](https://arxiv.org/abs/1904.05521)

**(*arXiv2020_ADDR*) Beyond the Deep Metric Learning: Enhance the Cross-Modal Matching with Adversarial Discriminative Domain Regularization.**<br>
*Li Ren, Kai Li, LiQiang Wang, Kien Hua.*<br>
[[paper]](https://arxiv.org/abs/2010.12126)

**(*PR2021_ITMeetsAL*) Integrating Information Theory and Adversarial Learning for Cross-modal Retrieval.**<br>
*Wei Chen, Yu Liu, Erwin M. Bakker, Michael S. Lew.*<br>
[[paper]](https://arxiv.org/abs/2104.04991)


### ``*Loss Function*``

**(*TPAMI2018_TBNN*) Learning Two-Branch Neural Networks for Image-Text Matching Tasks.**<br>
*Liwei Wang, Yin Li, Jing Huang, Svetlana Lazebnik.*<br>
[[paper]](https://arxiv.org/abs/1704.03470)
[[code]](https://github.com/lwwang/Two_branch_network)

**(*BMVC2018_VSE++*) VSE++: Improving Visual-Semantic Embeddings with Hard Negatives.**<br>
*Fartash Faghri, David J. Fleet, Jamie Ryan Kiros, Sanja Fidler.*<br>
[[paper]](https://arxiv.org/abs/1707.05612)
[[code]](https://github.com/fartashf/vsepp)

**(*ECCV2018_CMPL*) Deep Cross-Modal Projection Learning for Image-Text Matching.**<br>
*Ying Zhang, Huchuan Lu.*<br>
[[paper]](https://drive.google.com/file/d/1aiBuE1NjW83PGgYbP0eQDGEKr4fqMA6J/view)
[[code]](https://github.com/YingZhangDUT/Cross-Modal-Projection-Learning)

**(*ACLws2019_kNN-loss*) A Strong and Robust Baseline for Text-Image Matching.**<br>
*Fangyu Liu, Rongtian Ye.*<br> 
[[paper]](https://www.aclweb.org/anthology/P19-2023.pdf)

**(*ICASSP2019_NAA*) A Neighbor-aware Approach for Image-text Matching.**<br>
*Chunxiao Liu, Zhendong Mao, Wenyu Zang, Bin Wang.*<br> 
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8683869)

**(*CVPR2019_PVSE*) Polysemous Visual-Semantic Embedding for Cross-Modal Retrieval.**<br>
*Yale Song, Mohammad Soleymani.*<br>
[[paper]](https://arxiv.org/abs/1906.04402)
[[code]](https://github.com/yalesong/pvse)

**(*CVPR2019_SoDeep*) SoDeep: a Sorting Deep net to learn ranking loss surrogates.**<br>
*Martin Engilberge, Louis Chevallier, Patrick Pérez, Matthieu Cord.*<br>
[[paper]](https://arxiv.org/abs/1904.04272)
[[code]](https://github.com/kakaobrain/learning-loss-for-tta/tree/e9e5d79429f0eaec24b845f7b86ae68338dac718/sodeep)

**(*TOMM2020_Dual-Path*) Dual-path Convolutional Image-Text Embeddings with Instance Loss.**<br>
*Zhedong Zheng, Liang Zheng, Michael Garrett, Yi Yang, Mingliang Xu, YiDong Shen.*<br>
[[paper]](https://arxiv.org/abs/1711.05535)
[[code]](https://github.com/layumi/Image-Text-Embedding)

**(*AAAI2020_HAL*) HAL: Improved Text-Image Matching by Mitigating Visual Semantic Hubs.**<br>
*Fangyu Liu, Rongtian Ye, Xun Wang, Shuaipeng Li.*<br>
[[paper]](https://arxiv.org/abs/1911.10097)
[[code]](https://github.com/hardyqr/HAL) 

**(*AAAI2020_CVSE++*) Ladder Loss for Coherent Visual-Semantic Embedding.**<br>
*Mo Zhou, Zhenxing Niu, Le Wang, Zhanning Gao, Qilin Zhang, Gang Hua.*<br>
[[paper]](https://arxiv.org/abs/1911.07528)
[[code]](https://github.com/cdluminate/ladderloss) 

**(*CVPR2020_MPL*) Universal Weighting Metric Learning for Cross-Modal Matching.**<br>
*Jiwei Wei, Xing Xu, Yang Yang, Yanli Ji, Zheng Wang, Heng Tao Shen.*<br>
[[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Universal_Weighting_Metric_Learning_for_Cross-Modal_Matching_CVPR_2020_paper.pdf)
[[code]](https://github.com/wayne980/PolyLoss) 

**(*ECCV2020_PSN*) Preserving Semantic Neighborhoods for Robust Cross-modal Retrieval.**<br>
*Christopher Thomas, Adriana Kovashka.*<br>
[[paper]](https://arxiv.org/abs/2007.08617)
[[code]](https://github.com/CLT29/semantic_neighborhoods)

**(*ECCV2020_AOQ*) Adaptive Offline Quintuplet Loss for Image-Text Matching.**<br>
*Tianlang Chen, Jiajun Deng, Jiebo Luo.*<br>
[[paper]](https://arxiv.org/abs/2003.03669)
[[code]](https://github.com/sunnychencool/AOQ)

**(*ACMMM2021_Meta-SPN*) Meta Self-Paced Learning for Cross-Modal Matching.**<br>
*Jiwei Wei, Xing Xu, Zheng Wang, Guoqing Wang.*<br>
[[paper]](https://dl.acm.org/doi/pdf/10.1145/3474085.3475451)

**(*ICCVws2021_IMRL*) Hard-Negatives or Non-Negatives? A Hard-Negative Selection Strategy for Cross-Modal Retrieval Using the Improved Marginal Ranking Loss.**<br>
*Damianos Galanopoulos, Vasileios Mezaris.*<br>
[[paper]](https://openaccess.thecvf.com/content/ICCV2021W/ViRaL/papers/Galanopoulos_Hard-Negatives_or_Non-Negatives_A_Hard-Negative_Selection_Strategy_for_Cross-Modal_Retrieval_ICCVW_2021_paper.pdf)

**(*CVPR2021_MRL*) Learning Cross-Modal Retrieval with Noisy Labels.**<br>
*Peng Hu, Xi Peng, Hongyuan Zhu, Liangli Zhen, Jie Lin.*<br>
[[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Hu_Learning_Cross-Modal_Retrieval_With_Noisy_Labels_CVPR_2021_paper.pdf)

**(*NeurIPS2021_NCR*) Learning with Noisy Correspondence for Cross-modal Matching.**<br>
*Zhenyu Huang, Guocheng Niu, Xiao Liu, Wenbiao Ding, Xinyan Xiao, Hua Wu, Xi Peng.*<br>
[[paper]](https://papers.nips.cc/paper/2021/file/f5e62af885293cf4d511ceef31e61c80-Paper.pdf)
[[code]](https://github.com/XLearning-SCU/2021-NeurIPS-NCR)

**(*TPAMI2021_LESS*) Learning to Embed Semantic Similarity for Joint Image-text retrieval.**<br>
*Noam Malali, Yosi Keller.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9633145)

**(*ECIR2022_DLMLG*) Do Lessons from Metric Learning Generalize to Image-Caption Retrieval.**<br>
*Maurits Bleeker, Maarten de Rijke.*<br>
[[paper]](https://arxiv.org/abs/2202.07474)
[[code]](https://github.com/MauritsBleeker/ecir-2022-reproducibility-bleeker)

**(*WACV2022_SAM*) Is An Image Worth Five Sentences? A New Look into Semantics for Image-Text Matching.**<br>
*Ali Furkan Biten, Andres Mafla, Lluis Gomez, Dimosthenis Karatzas.*<br>
[[paper]](https://arxiv.org/abs/2110.02623)


## ``Task-oriented Works`` 

### ``*Un-Supervised or Semi-Supervised*``

**(*ECCV2018_VSA-AE-MMD*) Visual-Semantic Alignment Across Domains Using a Semi-Supervised Approach.**<br>
*Angelo Carraggi, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara.*<br>
[[paper]](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11134/Carraggi_Visual-Semantic_Alignment_Across_Domains_Using_a_Semi-Supervised_Approach_ECCVW_2018_paper.pdf)

**(*ACMMM2019_A3VSE*) Annotation Efficient Cross-Modal Retrieval with Adversarial Attentive Alignment.**<br>
*Po-Yao Huang, Guoliang Kang, Wenhe Liu, Xiaojun Chang, Alexander G Hauptmann.*<br>
[[paper]](http://www.cs.cmu.edu/~poyaoh/data/ann.pdf)


### ``*Zero-Shot or Fewer-Shot*``

**(*CVPR2017_DEM*) Learning a Deep Embedding Model for Zero-Shot Learning.**<br>
*Li Zhang, Tao Xiang, Shaogang Gong.*<br>
[[paper]](https://arxiv.org/abs/1611.05088)
[[code]](https://github.com/lzrobots/DeepEmbeddingModel_ZSL)

**(*AAAI2019_GVSE*) Few-shot image and sentence matching via gated visual-semantic matching.**<br>
*Yan Huang, Yang Long, Liang Wang.*<br>
[[paper]](https://ojs.aaai.org//index.php/AAAI/article/view/4866)

**(*ICCV2019_ACMM*) ACMM: Aligned Cross-Modal Memory for Few-Shot Image and Sentence Matching.**<br>
*Yan Huang, Liang Wang.*<br>
[[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_ACMM_Aligned_Cross-Modal_Memory_for_Few-Shot_Image_and_Sentence_Matching_ICCV_2019_paper.pdf)


### ``*Identification Learning*``

**(*ICCV2015_LSTM-Q+I*) VQA: Visual question answering.**<br>
*Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, MargaretMitchell, Dhruv Batra, C Lawrence Zitnick, Devi Parikh.*<br>
[[paper]](http://scholar.google.com.hk/scholar_url?url=http://openaccess.thecvf.com/content_iccv_2015/papers/Antol_VQA_Visual_Question_ICCV_2015_paper.pdf&hl=zh-CN&sa=X&ei=EDHkX9aDAY6CywTJ6a2ACw&scisig=AAGBfm2VHgUhZ4sZPI-ODBqcEdCd34_V8w&nossl=1&oi=scholarr)

**(*CVPR2016_Word-NN*) Learning Deep Representations of Fine-grained Visual Descriptions.**<br>
*Scott Reed, Zeynep Akata, Bernt Schiele, Honglak Lee.*<br>
[[paper]](https://arxiv.org/abs/1605.05395)

**(*CVPR2017_GNA-RNN*) Person search with natural language description.**<br>
*Shuang  Li, Tong Xiao, Hongsheng Li, Bolei Zhou, DayuYue, Xiaogang Wang.*<br>
[[paper]](https://arxiv.org/abs/1702.05729)
[[code]](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description)

**(*ICCV2017_IATV*) Identity-aware textual-visual matching with latent co-attention.**<br>
*Shuang Li, Tong Xiao, Hongsheng Li, Wei Yang, Xiaogang Wang.*<br>
[[paper]](https://arxiv.org/abs/1708.01988)

**(*WACV2018_PWM-ATH*) Improving text-based person search by spatial matching and adaptive threshold.**<br>
*Tianlang Chen, Chenliang Xu, Jiebo Luo.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8354312)

**(*ECCV2018_CMPL*) Deep Cross-Modal Projection Learning for Image-Text Matching.**<br>
*Ying Zhang, Huchuan Lu.*<br>
[[paper]](https://drive.google.com/file/d/1aiBuE1NjW83PGgYbP0eQDGEKr4fqMA6J/view)
[[code]](https://github.com/YingZhangDUT/Cross-Modal-Projection-Learning)

**(*ECCV2018_GLA*) Improving deep visual representation for person re-identification by global and local image-language association.**<br>
*Dapeng Chen, Hongsheng Li, Xihui Liu, Yantao Shen, JingShao, Zejian Yuan, Xiaogang Wang.*<br>
[[paper]](https://arxiv.org/abs/1808.01571)

**(*ICASSP2019_MCCL*) Language person search with mutually connected classification loss.**<br>
*Yuyu Wang, Chunjuan Bo, Dong Wang, Shuang Wang, Yunwei Qi, Huchuan Lu.*<br>
[[paper]](https://ieeexplore.ieee.org/document/8682456)

**(*ACMMM2019_A-GANet*) Deep adversarial graph attention convolution network for text-based person search.**<br>
*Jiawei Liu, Zheng-Jun Zha, Richang Hong, Meng Wang, Yongdong Zhang.*<br>
[[paper]](https://dl.acm.org/doi/pdf/10.1145/3343031.3350991?casa_token=P1d2TR_i_XUAAAAA:cYOVht8CMCuRGo1Q__knkyN-M76d9J0F3lqhuTXOi7VZTfl604QZjGtn_nZ4bqaD4qugCmKfIA)

**(*ICCV2019_TIMAM*) Adversarial Representation Learning for Text-to-Image Matching.**<br>
*Nikolaos Sarafianos, Xiang Xu, Ioannis A. Kakadiaris.*<br>
[[paper]](https://arxiv.org/abs/1908.10534)

**(*ICCV2019_FTD*) Fusing Two Directions in Cross-Domain Adaption for Real Life Person Search by Language.**<br>
*Kai Niu, Yan Huang, Liang Wang.*<br>
[[paper]](https://openaccess.thecvf.com/content_ICCVW_2019/papers/WIDER/Niu_Fusing_Two_Directions_in_Cross-Domain_Adaption_for_Real_Life_Person_ICCVW_2019_paper.pdf)

**(*CVPR2019_DSCMR*) Deep Supervised Cross-modal Retrieval.**<br>
*Liangli Zhen, Peng Hu, Xu Wang, Dezhong Peng.*<br>
[[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhen_Deep_Supervised_Cross-Modal_Retrieval_CVPR_2019_paper.pdf)
[[code]](https://github.com/penghu-cs/DSCMR)

**(*TOMM2020_Dual-Path*) Dual-path Convolutional Image-Text Embeddings with Instance Loss.**<br>
*Zhedong Zheng, Liang Zheng, Michael Garrett, Yi Yang, Mingliang Xu, YiDong Shen.*<br>
[[paper]](https://arxiv.org/abs/1711.05535)
[[code]](https://github.com/layumi/Image-Text-Embedding)

**(*TIP2020_MIA*) Improving Description-based Person Re-identification by Multi-granularity Image-text Alignment.**<br>
*Kai Niu, Yan Huang, Wanli Ouyang, Liang Wang.*<br>
[[paper]](https://arxiv.org/abs/1906.09610)

**(*WACV2020_CMAAM*) Text-based Person Search via Attribute-aided Matching.**<br>
*Surbhi Aggarwal, R. Venkatesh Babu, Anirban Chakraborty.*<br>
[[paper]](https://openaccess.thecvf.com/content_WACV_2020/papers/Aggarwal_Text-based_Person_Search_via_Attribute-aided_Matching_WACV_2020_paper.pdf)

**(*AAAI2020_PMA*) Pose-Guided Multi-Granularity Attention Network for Text-Based Person Search.**<br>
*Ya Jing, Chenyang Si, Junbo Wang, Wei Wang, Liang Wang, Tieniu Tan.*<br>
[[paper]](https://arxiv.org/abs/1809.08440)

**(*ECCV2020_ViTAA*) ViTAA: Visual-Textual Attributes Alignment in Person Search by Natural Language.**<br>
*Zhe Wang, Zhiyuan Fang, Jun Wang, Yezhou Yang.*<br>
[[paper]](https://arxiv.org/abs/2005.07327)
[[code]](https://github.com/Jarr0d/ViTAA)

**(*arXiv2021_NAFS*) Contextual Non-Local Alignment over Full-Scale Representation for Text-Based Person Search.**<br>
*Chenyang Gao, Guanyu Cai, Xinyang Jiang, Feng Zheng, Jun Zhang, Yifei Gong, Pai Peng, Xiaowei Guo, Xing Sun.*<br>
[[paper]](https://arxiv.org/abs/2101.03036)
[[code]](https://github.com/TencentYoutuResearch/PersonReID-NAFS)

**(*arXiv2021_SSAN*) Semantically Self-Aligned Network for Text-to-Image Part-aware Person Re-identification.**<br>
*Zefeng Ding, Changxing Ding, Zhiyin Shao, Dacheng Tao.*<br>
[[paper]](https://arxiv.org/abs/2107.12666)
[[code]](https://github.com/zifyloo/SSAN)

**(*PR2021_ITMeetsAL*) Integrating Information Theory and Adversarial Learning for Cross-modal Retrieval.**<br>
*Wei Chen, Yu Liu, Erwin M. Bakker, Michael S. Lew.*<br>
[[paper]](https://arxiv.org/abs/2104.04991)

**(*ACMMM2021_DSSL*) DSSL: Deep Surroundings-person Separation Learning for Text-based Person Retrieval.**<br>
*Aichun Zhu, Zijie Wang, Yifeng Li, Xili Wan, Jing Jin, Tian Wang, Fangqiang Hu, Gang Hua.*<br>
[[paper]](https://arxiv.org/abs/2109.05534)
[[code]](https://github.com/NjtechCVLab/RSTPReid-Dataset)

**(*IJCAI2021_MGEL*) Text-based Person Search via Multi-Granularity Embedding Learning.**<br>
*Chengji Wang, Zhiming Luo1, Yaojin Lin, Shaozi Li.*<br>
[[paper]](https://www.ijcai.org/proceedings/2021/0148.pdf)

**(*ICCV2021_LapsCore*) LapsCore: Language-Guided Person Search via Color Reasoning.**<br>
*Yushuang Wu, Zizheng Yan, Xiaoguang Han, Guanbin Li, Changqing Zou, Shuguang Cui.*<br>
[[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_LapsCore_Language-Guided_Person_Search_via_Color_Reasoning_ICCV_2021_paper.pdf)


### ``*Scene-Text Learning*``

**(*ECCV2018_SS*) Single Shot Scene Text Retrieval.**<br>
*Lluís Gómez, Andrés Mafla, Marçal Rusiñol, Dimosthenis Karatzas.*<br>
[[paper]](https://arxiv.org/abs/1808.09044)
[[code_Tensorflow]](https://github.com/lluisgomez/single-shot-str)[[code_Pytorch]](https://github.com/AndresPMD/Pytorch-yolo-phoc)

**(*WACV2020_PHOC*) Fine-grained Image Classification and Retrieval by Combining Visual and Locally Pooled Textual Features.**<br>
*Andres Mafla, Sounak Dey, Ali Furkan Biten, Lluis Gomez, Dimosthenis Karatzas.*<br>
[[paper]](https://arxiv.org/abs/2001.04732.pdf)
[[code]](https://github.com/AndresPMD/Fine_Grained_Clf)

**(*WACV2021_MMRG*) Multi-Modal Reasoning Graph for Scene-Text Based Fine-Grained Image Classification and Retrieval.**<br>
*Andres Mafla, Sounak Dey, Ali Furkan Biten, Lluis Gomez, Dimosthenis Karatzas.*<br>
[[paper]](https://arxiv.org/abs/2009.09809)
[[code]](https://github.com/AndresPMD/GCN_classification)

**(*WACV2021_StacMR*) StacMR: Scene-Text Aware Cross-Modal Retrieval.**<br>
*Andrés Mafla, Rafael Sampaio de Rezende, Lluís Gómez, Diane Larlus, Dimosthenis Karatzas.*<br>
[[paper]](https://arxiv.org/abs/2012.04329)
[[code]](http://europe.naverlabs.com/stacmr)


### ``*Related Works*``

**(*Machine Learning 2010*) Large scale image annotation: learning to rank with joint word-image embeddings.**<br>
*Jason Weston, Samy Bengio, Nicolas Usunier.*<br>
[[paper]](https://link.springer.com/content/pdf/10.1007%2Fs10994-010-5198-3.pdf)

**(*NeurIPS2013_Word2Vec*) Distributed Representations of Words and Phrases and their Compositionality.**<br>
*Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean.*<br>
[[paper]](https://arxiv.org/abs/1310.4546)

**(*CVPR2017_DVSQ*) Deep Visual-Semantic Quantization for Efficient Image Retrieval.**<br>
*Yue Cao, Mingsheng Long, Jianmin Wang, Shichen Liu.*<br>
[[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Cao_Deep_Visual-Semantic_Quantization_CVPR_2017_paper.pdf)

**(*ACL2018_ILU*) Illustrative Language Understanding: Large-Scale Visual Grounding with Image Search.**<br>
*Jamie Kiros, William Chan, Geoffrey Hinton.*<br>
[[paper]](https://aclweb.org/anthology/P18-1085)

**(*AAAI2018_VSE-ens*) VSE-ens: Visual-Semantic Embeddings with Efficient Negative Sampling.**<br>
*Guibing Guo, Songlin Zhai, Fajie Yuan, Yuan Liu, Xingwei Wang.*<br>
[[paper]](https://arxiv.org/abs/1801.01632)

**(*ECCV2018_HTG*) An Adversarial Approach to Hard Triplet Generation.**<br>
*Yiru Zhao, Zhongming Jin, Guo-jun Qi, Hongtao Lu, Xian-sheng Hua.*<br>
[[paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yiru_Zhao_A_Principled_Approach_ECCV_2018_paper.pdf)

**(*ECCV2018_WebNet*) CurriculumNet: Weakly Supervised Learning from Large-Scale Web Images.**<br>
*Sheng Guo, Weilin Huang, Haozhi Zhang, Chenfan Zhuang, Dengke Dong, Matthew R. Scott, Dinglong Huang.*<br>
[[paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Sheng_Guo_CurriculumNet_Learning_from_ECCV_2018_paper.pdf)
[[code]](https://github.com/MalongTech/research-curriculumnet)

**(*CVPR2018_BUTD*) Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering.**<br>
*Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, Lei Zhang.*<br>
[[paper]](https://arxiv.org/abs/1707.07998)
[[code]](https://github.com/peteanderson80/bottom-up-attention)

**(*EMNLP2019_GMMR*) Multi-Head Attention with Diversity for Learning Grounded Multilingual Multimodal Representations.**<br>
*Po-Yao Huang, Xiaojun Chang, Alexander Hauptmann.*<br>
[[paper]](https://www.aclweb.org/anthology/D19-1154.pdf)

**(*EMNLP2019_MIMSD*) Unsupervised Discovery of Multimodal Links in Multi-Image, Multi-Sentence Documents.**<br>
*Jack Hessel, Lillian Lee, David Mimno.*<br>
[[paper]](https://arxiv.org/abs/1904.07826)
[[code]](https://github.com/jmhessel/multi-retrieval)

**(*ICCV2019_DRNet*) Fashion Retrieval via Graph Reasoning Networks on a Similarity Pyramid.**<br>
*Zhanghui Kuang, Yiming Gao, Guanbin Li, Ping Luo, Yimin Chen, Liang Lin, Wayne Zhang.*<br>
[[paper]](https://arxiv.org/abs/1908.11754)

**(*ICCV2019_Align2Ground*) Align2Ground: Weakly Supervised Phrase Grounding Guided by Image-Caption Alignment.**<br>
*Samyak Datta, Karan Sikka, Anirban Roy, Karuna Ahuja, Devi Parikh, Ajay Divakaran.*<br>
[[paper]](https://arxiv.org/abs/1903.11649)

**(*CVPR2019_TIRG*) Composing Text and Image for Image Retrieval - An Empirical Odyssey.**<br>
*Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.*<br>
[[paper]](https://arxiv.org/abs/1812.07119)

**(*SIGIR2019_PAICM*) Prototype-guided Attribute-wise Interpretable Scheme for Clothing Matching.**<br>
*Xianjing Han, Xuemeng Song, Jianhua Yin, Yinglong Wang, Liqiang Nie.*<br>
[[paper]](https://xuemengsong.github.io/SIGIR2019_PAICM.pdf)

**(*SIGIR2019_NCR*) Neural Compatibility Ranking for Text-based Fashion Matching.**<br>
*Suthee Chaidaroon, Mix Xie, Yi Fang, Alessandro Magnani.*<br>
[[paper]](https://dl.acm.org/doi/pdf/10.1145/3331184.3331365)

**(*arXiv2020_Tweets*) Deep Multimodal Image-Text Embeddings for Automatic Cross-Media Retrieval.**<br>
*Hadi Abdi Khojasteh, Ebrahim Ansari, Parvin Razzaghi, Akbar Karimi.*<br>
[[paper]](https://arxiv.org/abs/2002.10016)

**(*arXiv2020_TIMNet*) Weakly-Supervised Feature Learning via Text and Image Matching.**<br>
*Gongbo Liang, Connor Greenwell, Yu Zhang, Xiaoqin Wang, Ramakanth Kavuluru, Nathan Jacobs.*<br>
[[paper]](https://arxiv.org/abs/2010.03060)
[[code]](http://www.gb-liang.com/TIMNet)

**(*ECCV2020_InfoNCE*) Contrastive Learning for Weakly Supervised Phrase Grounding.**<br>
*Tanmay Gupta, Arash Vahdat, Gal Chechik, Xiaodong Yang, Jan Kautz, Derek Hoiem.*<br>
[[paper]](https://arxiv.org/abs/2006.09920)
[[code]](https://github.com/BigRedT/info-ground)

**(*ECCV2020_JVSM*) Learning Joint Visual Semantic Matching Embeddings for Language-guided Retrieval.**<br>
*Yanbei Chen, Loris Bazzani.*<br>
[[paper]](https://assets.amazon.science/5b/db/440af26349adb83c77c85cd11922/learning-joint-visual-semantic-matching-embeddings-for-text-guided-retrieval.pdf)

**(*CVPR2020_POS-SCAN*) More Grounded Image Captioning by Distilling Image-Text Matching Model.**<br>
*Yuanen Zhou, Meng Wang, Daqing Liu, Zhenzhen Hu, Hanwang Zhang.*<br>
[[paper]](https://arxiv.org/abs/2004.00390)
[[code]](https://github.com/YuanEZhou/Grounded-Image-Captioning)

**(*COLING2020_VSE-Probing*) Probing Multimodal Embeddings for Linguistic Properties: the Visual-Semantic Case.**<br>
*Adam Dahlgren Lindström, Suna Bensch, Johanna Björklund, Frank Drewes.*<br>
[[paper]](https://www.aclweb.org/anthology/2020.coling-main.64.pdf)
[[code]](https://github.com/dali-does/vse-probing)

**(*TMM2021_ALGCN*) Adaptive Label-aware Graph Convolutional Networks for Cross-Modal Retrieval.**<br>
*Shengsheng Qian, Dizhan Xue, Quan Fang, Changsheng Xu.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9508809)

**(*ICASSP2021_DAQN*) Deep Adversarial Quantization Network for Cross-Modal Retrieval.**<br>
*Yu Zhou, Yong Feng, Mingliang Zhou, Baohua Qiang, Leong Hou U, Jiajie Zhu.*<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9414247)
[[code]](https://github.com/zhouyu1996/DAQN)

**(*ACMMM2021_DARR*) Database-adaptive Re-ranking for Enhancing Cross-modal Image Retrieval.**<br>
*Rintaro Yanagi, Ren Togo, Takahiro Ogawa, Miki Haseyama.*<br>
[[paper]](https://dl.acm.org/doi/pdf/10.1145/3474085.3475681)

**(*AAAI2021_DAGNN*) Dual Adversarial Graph Neural Networks for Multi-label Cross-modal Retrieval.**<br>
*Shengsheng Qian, Dizhan Xue, Huaiwen Zhang, Quan Fang, Changsheng Xu.*<br>
[[paper]](https://www.aaai.org/AAAI21Papers/AAAI-6207.QianS.pdf)

**(*ICCV2021_Ask&Confirm*) Ask&Confirm: Active Detail Enriching for Cross-Modal Retrieval with Partial Query.**<br>
*Guanyu Cai, Jun Zhang, Xinyang Jiang, Yifei Gong, Lianghua He, Fufu Yu, Pai Peng, Xiaowei Guo, Feiyue Huang, Xing Sun.*<br>
[[paper]](https://arxiv.org/abs/2103.01654)
[[code]](https://github.com/CuthbertCai/Ask-Confirm)


### ``Posted in``

**(*JImaging2021_Survey*) On the Limitations of Visual-Semantic Embedding Networks for Image-to-Text Information Retrieval.**<br>
*Yan Gong, Georgina Cosma, Hui Fang.*<br>
[[paper]](https://www.mdpi.com/2313-433X/7/8/125)
[[code]](https://github.com/yangong23/VSEnetworksIR)

**(*arXiv2022_Survey*) Where Does the Performance Improvement Come From? A Reproducibility Concern about Image-Text Retrieval.**<br>
*Jun Rao, Fei Wang, Liang Ding, Shuhan Qi, Yibing Zhan, Weifeng Liu, Dacheng Tao.*<br>
[[paper]](https://arxiv.org/abs/2203.03853)
[[code]](https://github.com/WangFei-2019/Image-text-Retrieval)

**(*arXiv2022_Survey*) Image-text Retrieval: A Survey on Recent Research and Development.**<br>
*Min Cao, Shiping Li, Juntao Li, Liqiang Nie, Min Zhang.*<br>
[[paper]](https://arxiv.org/abs/2203.14713)

**(*EACL2021_CxC*) Crisscrossed Captions: Extended Intramodal and Intermodal Semantic Similarity Judgments for MS-COCO.**<br>
*Zarana Parekh, Jason Baldridge, Daniel Cer, Austin Waters, Yinfei Yang.*<br>
[[paper]](https://arxiv.org/abs/2004.15020)
[[code]](https://github.com/google-research-datasets/Crisscrossed-Captions)

**(*arXiv2022_ECCVCaption*) ECCV Caption: Correcting False Negatives by Collecting Machine-and-Human-verified Image-Caption Associations for MS-COCO.**<br>
*Sanghyuk Chun, Wonjae Kim, Song Park, Minsuk Chang, Seong Joon Oh.*<br>
[[paper]](https://arxiv.org/abs/2204.03359)
[[code]](https://github.com/naver-ai/eccv-caption)


