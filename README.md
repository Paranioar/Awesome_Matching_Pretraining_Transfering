Cross-modal_Retrieval_Tutorial
==============================
The Tutorial of Image-Text Matching for Preliminary Insight. 
****
## ``Catalogue ``
* [Peformance comparison](#peformance-comparison)
    * [Flickr8K](#performance-of-flickr8k)
    * [Flickr30K](#performance-of-flickr30k)
    * [MSCOCO1K](#performance-of-mscoco1k)
    * [MSCOCO5K](#performance-of-mscoco5k)

* [Methods summary](#method-summary)
    * [Generic-feature extraction](#generic-feature-extraction)
    * [Cross-attention interaction](#cross-attention-interaction)
    * [Similarity measurement](#similarity-measurement)
    * [Loss function](#loss-function)
    * [Related works](#related-works)
    * [Posted in](#posted-in)
****

## ``Peformance comparison``

### *Performance of Flickr8K*
**(*\** indicates Ensemble models, *^* indicates questionable authen)**
<table>
   <tr> <td rowspan="3">Method_name</td> <td rowspan="3", align="center">Concise_note</td> 
        <td colspan="6", align="center">Flickr8K</td> </tr>
   <tr> <td colspan="3", align="center">Sentence retrieval</td> <td colspan="3", align="center">Image retrieval</td> </tr>
   <tr> <td>R@1</td><td>R@5</td><td>R@10</td> <td>R@1</td><td>R@5</td><td>R@10</td> </tr>
   <tr> <td>SDT-RNN</td><td></td> <td>4.5</td><td>18.0</td><td>28.6</td> <td>6.1</td><td>18.5</td><td>29.0</td> </tr> 
   <tr> <td>SDT-RNN</td><td>detection</td> <td>6.0</td><td>22.7</td><td>34.0</td> <td>6.6</td><td>21.6</td><td>31.7</td> </tr>
   <tr> <td>DeViSE</td><td>detection</td> <td>4.8</td><td>16.5</td><td>27.3</td> <td>5.9</td><td>20.1</td><td>29.6</td> </tr>
   <tr> <td>DeFrag</td><td></td> <td>5.9</td><td>19.2</td><td>27.3</td> <td>5.2</td><td>17.6</td><td>26.5</td> </tr>
   <tr> <td>DeFrag</td><td>detection</td> <td>12.6</td><td>32.9</td><td>44.0</td> <td>9.7</td><td>29.6</td><td>42.5</td> </tr>
   <tr> <td>m-RNN</td><td></td> <td>14.5</td><td>37.2</td><td>48.5</td> <td>11.5</td><td>31.0</td><td>42.4</td> </tr>
   <tr> <td>UVSE</td><td></td> <td>13.5</td><td>36.2</td><td>45.7</td> <td>10.4</td><td>31.0</td><td>43.7</td> </tr>
   <tr> <td>UVSE</td><td>OxfordNet</td> <td>18.0</td><td>40.9</td><td>55.0</td> <td>12.5</td><td>37.0</td><td>51.5</td> </tr>
</table> 

### *Performance of Flickr30K*
<table>
   <tr> <td rowspan="3">Method_name</td> <td rowspan="3", align="center">Concise_note</td> 
        <td colspan="6", align="center">Flickr30K</td> </tr>
   <tr> <td colspan="3", align="center">Sentence retrieval</td> <td colspan="3", align="center">Image retrieval</td> </tr>
   <tr> <td>R@1</td><td>R@5</td><td>R@10</td> <td>R@1</td><td>R@5</td><td>R@10</td> </tr>
   <tr> <td>DeViSE</td><td>detection</td> <td>4.5</td><td>18.1</td><td>29.2</td> <td>6.7</td><td>21.9</td><td>32.7</td> </tr>
   <tr> <td>SDT-RNN</td><td>detection</td> <td>9.6</td><td>29.8</td><td>41.1</td> <td>8.9</td><td>29.8</td><td>41.1</td> </tr>
   <tr> <td>DeFrag</td><td>detection</td> <td>14.2</td><td>37.7</td><td>51.3</td> <td>10.2</td><td>30.8</td><td>44.2</td> </tr>
   <tr> <td>DeFrag(ft_CNN)</td><td>detection</td> <td>16.4</td><td>40.2</td><td>54.7</td> <td>10.3</td><td>31.4</td><td>44.5</td> </tr>
   <tr> <td>m-RNN</td><td></td> <td>18.4</td><td>40.2</td><td>50.9</td> <td>12.6</td><td>31.2</td><td>41.5</td> </tr>
   <tr> <td>UVSE</td><td></td> <td>14.8</td><td>39.2</td><td>50.9</td> <td>11.8</td><td>34.0</td><td>46.3</td> </tr>
   <tr> <td>UVSE</td><td>OxfordNet</td> <td>23.0</td><td>50.7</td><td>62.9</td> <td>16.8</td><td>42.0</td><td>56.5</td> </tr>
</table> 

### *Performance of MSCOCO1K*
<table>
   <tr> <td rowspan="3">Method_name</td> <td rowspan="3", align="center">Concise_note</td> 
        <td colspan="6", align="center">MSCOCO1K</td> </tr>
   <tr> <td colspan="3", align="center">Sentence retrieval</td> <td colspan="3", align="center">Image retrieval</td> </tr>
   <tr> <td>R@1</td><td>R@5</td><td>R@10</td> <td>R@1</td><td>R@5</td><td>R@10</td> </tr>
   <tr> <td>SDT-RNN</td><td></td> <td>31.2</td><td>31.2</td><td>31.2</td> <td>31.2</td><td>31.2</td><td>31.2</td> </tr> 
   <tr> <td>SDT-RNN</td><td></td> <td>31.2</td><td>31.2</td><td>31.2</td> <td>31.2</td><td>31.2</td><td>31.2</td> </tr>
</table> 

### *Performance of MSCOCO5K*
<table>
   <tr> <td rowspan="3">Method_name</td> <td rowspan="3", align="center">Concise_note</td> 
        <td colspan="6", align="center">MSCOCO5K</td> </tr>
   <tr> <td colspan="3", align="center">Sentence retrieval</td> <td colspan="3", align="center">Image retrieval</td> </tr>
   <tr> <td>R@1</td><td>R@5</td><td>R@10</td> <td>R@1</td><td>R@5</td><td>R@10</td> </tr>
   <tr> <td>SDT-RNN</td><td></td> <td>31.2</td><td>31.2</td><td>31.2</td> <td>31.2</td><td>31.2</td><td>31.2</td> </tr> 
   <tr> <td>SDT-RNN</td><td></td> <td>31.2</td><td>31.2</td><td>31.2</td> <td>31.2</td><td>31.2</td><td>31.2</td> </tr>
</table> 

****

## ``Method summary`` 

### *Generic-feature extraction*
**(*DeViSE_NIPS2013*) DeViSE: A Deep Visual-Semantic Embedding Model.** <br>
*Andrea Frome, Greg S. Corrado, Jonathon Shlens, Samy Bengio, Jeffrey Dean, Marc’Aurelio Ranzato, Tomas Mikolov.*<br>
[[paper]](https://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model.pdf)

**(*UVSE_NIPS2014*) Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models.**<br>
*Ryan Kiros, Ruslan Salakhutdinov, Richard S. Zemel.*<br>
[[paper]](https://arxiv.org/pdf/1411.2539.pdf)
[[code]](https://github.com/ryankiros/visual-semantic-embedding)
[[demo]](http://www.cs.toronto.edu/~rkiros/lstm_scnlm.html)

**(*DFE_NIPS2014*) Deep fragment embeddings for bidirectional image sentence mapping.**<br>
*Andrej Karpathy, Armand Joulin, and Li Fei-Fei.*<br>
[[paper]](https://cs.stanford.edu/people/karpathy/nips2014.pdf)

### *Cross-attention interaction*
### *Similarity measurement*
### *Loss function*
### *Related works*
### *Posted in*


**Deep Visual-Semantic Alignments for Generating Image Descriptions.**<br>
*Andrej Karpathy, Li Fei-Fei.*<br>
**_(CVPR 2015)_**<br>
[[paper]](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)

**Deep Correlation for Matching Images and Text.**<br>
*Fei Yan, Krystian Mikolajczyk.*<br>
**_(CVPR 2015)_**<br>
[[paper]](http://openaccess.thecvf.com/content_cvpr_2015/papers/Yan_Deep_Correlation_for_2015_CVPR_paper.pdf)

**ORDER-EMBEDDINGS OF IMAGES AND LANGUAGE.**<br>
*Ivan Vendrov, Ryan Kiros, Sanja Fidler, Raquel Urtasun.*<br>
**_(ICLR 2016)_**<br>
[[paper]](https://arxiv.org/pdf/1511.06361.pdf)

**Learning Deep Structure-Preserving Image-Text Embeddings.**<br>
*Liwei Wang, Yin Li, Svetlana Lazebnik.*<br>
**_(CVPR 2016)_**<br>
[[paper]](http://slazebni.cs.illinois.edu/publications/cvpr16_structure.pdf)

**Learning a Deep Embedding Model for Zero-Shot Learning.**<br>
*Li Zhang, Tao Xiang, Shaogang Gong.*<br>
**_(CVPR 2017)_**<br>
[[paper]](https://arxiv.org/pdf/1611.05088.pdf)
[[code]](https://github.com/lzrobots/DeepEmbeddingModel_ZSL)(TF)

**Deep Visual-Semantic Quantization for Efficient Image Retrieval.**<br>
*Yue Cao, Mingsheng Long, Jianmin Wang, Shichen Liu.*<br>
**_(CVPR 2017)_**<br>
[[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Cao_Deep_Visual-Semantic_Quantization_CVPR_2017_paper.pdf)

**Dual Attention Networks for Multimodal Reasoning and Matching.**<br>
*Hyeonseob Nam, Jung-Woo Ha, Jeonghee Kim.*<br>
**_(CVPR 2017)_**<br>
[[paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Nam_Dual_Attention_Networks_CVPR_2017_paper.pdf)

**Sampling Matters in Deep Embedding Learning.**<br>
*Chao-Yuan Wu, R. Manmatha, Alexander J. Smola, Philipp Krähenbühl.*<br>
**_(ICCV 2017)_**<br>
[[paper]](https://arxiv.org/pdf/1706.07567.pdf)
[[zhihu discussion]](https://www.zhihu.com/question/61748966)

**Learning Robust Visual-Semantic Embeddings.**<br>
*Yao-Hung Hubert Tsai, Liang-Kang Huang, Ruslan Salakhutdinov.*<br>
**_(ICCV 2017)_**<br>
[[paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Tsai_Learning_Robust_Visual-Semantic_ICCV_2017_paper.pdf)

**Hierarchical Multimodal LSTM for Dense Visual-Semantic Embedding.**<br>
*Zhenxing Niu, Mo Zhou, Le Wang, Xinbo Gao, Gang Hua.*<br>
**_(ICCV 2017)_**<br>
[[paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Niu_Hierarchical_Multimodal_LSTM_ICCV_2017_paper.pdf)

**Learning a Recurrent Residual Fusion Network for Multimodal Matching.**<br>
*Yu Liu, Yanming Guo, Erwin M. Bakker, Michael S. Lew.*<br>
**_(ICCV 2017)_**<br>
[[paper]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_a_Recurrent_ICCV_2017_paper.pdf)

**VSE-ens: Visual-Semantic Embeddings with Efficient Negative Sampling.**<br>
*Guibing Guo, Songlin Zhai, Fajie Yuan, Yuan Liu, Xingwei Wang.*<br>
**_(AAAI 2018)_**<br>
[[paper]](https://arxiv.org/pdf/1801.01632.pdf)

**Incorporating GAN for Negative Sampling in Knowledge Representation Learning.**<br>
*Peifeng Wang, Shuangyin Li, Rong pan.*<br>
**_(AAAI 2018)_**<br>
[[paper]](https://arxiv.org/pdf/1809.11017.pdf)

**Fast Self-Attentive Multimodal Retrieval.**<br>
*Jônatas Wehrmann, Maurício Armani Lopes, Martin D More, Rodrigo C. Barros.*<br>
**_(WACV 2018)_**<br>
[[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8354311&tag=1)
[[code]](https://github.com/jwehrmann/seam-retrieval)(PyTorch)

**End-to-end Convolutional Semantic Embeddings.**<br>
*Quanzeng You, Zhengyou Zhang, Jiebo Luo.*<br>
**_(CVPR 2018)_**<br>
[[paper]](https://ai.tencent.com/ailab/media/publications/cvpr/End-to-end_Convolutional_Semantic_Embeddings.pdf)
 
**Bidirectional Retrieval Made Simple.**<br>
*Jonatas Wehrmann, Rodrigo C. Barros.*<br>
**_(CVPR 2018)_**<br>
[[paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wehrmann_Bidirectional_Retrieval_Made_CVPR_2018_paper.pdf)
[[code]](https://github.com/jwehrmann/chain-vse)(PyTorch)

**Illustrative Language Understanding: Large-Scale Visual Grounding with Image Search.**<br>
*Jamie Kiros, William Chan, Geoffrey Hinton.*<br>
**_(ACL 2018)_**<br>
[[paper]](https://aclweb.org/anthology/P18-1085)

**Learning Visually-Grounded Semantics from Contrastive Adversarial Samples.**<br>
*Haoyue Shi, Jiayuan Mao, Tete Xiao, Yuning Jiang, Jian Sun.*<br>
**_(COLING 2018)_**<br>
[[paper]](https://aclweb.org/anthology/C18-1315)
[[code]](https://github.com/ExplorerFreda/VSE-C)(PyTorch)

**VSE++: Improving Visual-Semantic Embeddings with Hard Negatives.**<br>
*Fartash Faghri, David J. Fleet, Jamie Ryan Kiros, Sanja Fidler.*<br>
**_(BMVC 2018)_**<br>
[[paper]](https://arxiv.org/pdf/1707.05612.pdf)
[[code]](https://github.com/fartashf/vsepp)(PyTorch)

**An Adversarial Approach to Hard Triplet Generation.**<br>
*Yiru Zhao, Zhongming Jin, Guo-jun Qi, Hongtao Lu, Xian-sheng Hua.*<br>
**_(ECCV 2018)_**<br>
[[paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yiru_Zhao_A_Principled_Approach_ECCV_2018_paper.pdf)

**Conditional Image-Text Embedding Networks.**<br>
*Bryan A. Plummer, Paige Kordas, M. Hadi Kiapour, Shuai Zheng, Robinson Piramuthu, Svetlana Lazebnik.*<br>
**_(ECCV 2018)_**<br>
[[paper]](https://arxiv.org/pdf/1711.08389.pdf)

**Visual-Semantic Alignment Across Domains Using a Semi-Supervised Approach.**<br>
*Angelo Carraggi, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara.*<br>
**_(ECCV 2018)_**<br>
[[paper]](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11134/Carraggi_Visual-Semantic_Alignment_Across_Domains_Using_a_Semi-Supervised_Approach_ECCVW_2018_paper.pdf)

**Stacked Cross Attention for Image-Text Matching.**<br>
*Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He.*<br>
**_(ECCV 2018)_**<br>
[[paper]](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Kuang-Huei_Lee_Stacked_Cross_Attention_ECCV_2018_paper.pdf)

**CurriculumNet: Weakly Supervised Learning from Large-Scale Web Images.**<br>
*Sheng Guo, Weilin Huang, Haozhi Zhang, Chenfan Zhuang, Dengke Dong, Matthew R. Scott, Dinglong Huang.*<br>
**_(ECCV 2018)_**<br>
[[paper]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Sheng_Guo_CurriculumNet_Learning_from_ECCV_2018_paper.pdf)
[[code]](https://github.com/MalongTech/research-curriculumnet)(Caffe)

**A Strong and Robust Baseline for Text-Image Matching.**<br>
*Fangyu Liu, Rongtian Ye.*<br>
**_(ACL Student Research Workshop 2019)_**<br>
[[paper]](https://www.aclweb.org/anthology/P19-2023.pdf)

**Unified Visual-Semantic Embeddings: Bridging Vision and Language with Structured Meaning Representations.**<br>
*Hao Wu, Jiayuan Mao, Yufeng Zhang, Yuning Jiang, Lei Li, Weiwei Sun, Wei-Ying Ma.*<br>
**_(CVPR 2019)_**<br>
[[paper]](https://arxiv.org/pdf/1904.05521.pdf)

**Engaging Image Captioning via Personality.**<br>
*Kurt Shuster, Samuel Humeau, Hexiang Hu, Antoine Bordes, Jason Weston.*<br>
**_(CVPR 2019)_**<br>
[[paper]](https://arxiv.org/pdf/1810.10665.pdf)

**Polysemous Visual-Semantic Embedding for Cross-Modal Retrieval.**<br>
*Yale Song, Mohammad Soleymani.*<br>
**_(CVPR 2019)_**<br>
[[paper]](https://arxiv.org/pdf/1906.04402.pdf)
 
**Composing Text and Image for Image Retrieval - An Empirical Odyssey.**<br>
*Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.*<br>
**_(CVPR 2019)_**<br>
[[paper]](https://arxiv.org/pdf/1812.07119.pdf)

**Annotation Efficient Cross-Modal Retrieval with Adversarial Attentive Alignment.**<br>
*Po-Yao Huang, Guoliang Kang, Wenhe Liu, Xiaojun Chang, Alexander G Hauptmann.*<br>
**_(ACM MM 2019)_**<br>
[[paper]](http://www.cs.cmu.edu/~poyaoh/data/ann.pdf)

**Visual Semantic Reasoning for Image-Text Matching.**<br>
*Kunpeng Li, Yulun Zhang, Kai Li, Yuanyuan Li, Yun Fu.*<br>
**_(ICCV 2019)_**<br>
[[paper]](https://arxiv.org/pdf/1909.02701.pdf)

**Adversarial Representation Learning for Text-to-Image Matching.**<br>
*Nikolaos Sarafianos, Xiang Xu, Ioannis A. Kakadiaris.*<br>
**_(ICCV 2019)_**<br>
[[paper]](https://arxiv.org/pdf/1908.10534.pdf)

**CAMP: Cross-Modal Adaptive Message Passing for Text-Image Retrieval.**<br>
*Zihao Wang, Xihui Liu, Hongsheng Li, Lu Sheng, Junjie Yan, Xiaogang Wang, Jing Shao.*<br>
**_(ICCV 2019)_**<br>
[[paper]](https://arxiv.org/pdf/1909.05506.pdf)

**Align2Ground: Weakly Supervised Phrase Grounding Guided by Image-Caption Alignment.**<br>
*Samyak Datta, Karan Sikka, Anirban Roy, Karuna Ahuja, Devi Parikh, Ajay Divakaran.*<br>
**_(ICCV 2019)_**<br>
[[paper]](https://arxiv.org/pdf/1903.11649.pdf)

**Multi-Head Attention with Diversity for Learning Grounded Multilingual Multimodal Representations.**<br>
*Po-Yao Huang, Xiaojun Chang, Alexander Hauptmann.*<br>
**_(EMNLP 2019)_**<br>
[[paper]](https://www.aclweb.org/anthology/D19-1154.pdf)

**Unsupervised Discovery of Multimodal Links in Multi-Image, Multi-Sentence Documents.**<br>
*Jack Hessel, Lillian Lee, David Mimno.*<br>
**_(EMNLP 2019)_**<br>
[[paper]](https://arxiv.org/pdf/1904.07826.pdf)
[[code]](https://github.com/jmhessel/multi-retrieval)

**HAL: Improved Text-Image Matching by Mitigating Visual Semantic Hubs.**<br>
*Fangyu Liu, Rongtian Ye, Xun Wang, Shuaipeng Li.*<br>
**_(AAAI 2020)_**<br>
[[paper]](https://arxiv.org/pdf/1911.10097v1.pdf)
[[code]](https://github.com/hardyqr/HAL) (PyTorch)

**Ladder Loss for Coherent Visual-Semantic Embedding.**<br>
*Mo Zhou, Zhenxing Niu, Le Wang, Zhanning Gao, Qilin Zhang, Gang Hua.*<br>
**_(AAAI 2020)_**<br>
[[paper]](https://arxiv.org/pdf/1911.07528.pdf)

**Expressing Objects just like Words: Recurrent Visual Embedding for Image-Text Matching.** <br>
*Tianlang Chen, Jiebo Luo.*<br>
**_(AAAI 2020)_**<br>
[[paper]](https://arxiv.org/pdf/2002.08510.pdf)

**Adaptive Cross-modal Embeddings for Image-Text Alignment.**<br>
*Jonatas Wehrmann, Camila Kolling, Rodrigo C Barros.*<br>
**_(AAAI 2020)_**<br>
[[paper]](https://ojs.aaai.org//index.php/AAAI/article/view/6915)
[[code]](https://github.com/jwehrmann/retrieval.pytorch) (PyTorch)

**Graph Structured Network for Image-Text Matching**.<br>
*Chunxiao Liu, Zhendong Mao, Tianzhu Zhang, Hongtao Xie, Bin Wang, Yongdong Zhang.*<br>
**_(CVPR 2020)_**<br>
[[paper]](https://arxiv.org/pdf/2004.00277.pdf)

**IMRAM: Iterative Matching with Recurrent Attention Memory for Cross-Modal Image-Text Retrieval.**<br>
*Hui Chen, Guiguang Ding, Xudong Liu, Zijia Lin, Ji Liu, Jungong Han.*<br>
**_(CVPR 2020)_**<br>
[[paper]](https://arxiv.org/pdf/2003.03772.pdf)

**Visual-Semantic Matching by Exploring High-Order Attention and Distraction.**<br>
*Yongzhi Li, Duo Zhang, Yadong Mu.*<br>
**_(CVPR 2020)_**<br>
[[paper]](https://pkumyd.github.io/paper/CVPR2020_LYZ.pdf)

**Multi-Modality Cross Attention Network for Image and Sentence Matching.**<br>
*Xi Wei, Tianzhu Zhang, Yan Li, Yongdong Zhang, Feng Wu.*<br>
**_(CVPR 2020)_**<br>
[[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Multi-Modality_Cross_Attention_Network_for_Image_and_Sentence_Matching_CVPR_2020_paper.pdf)

**Context-Aware Attention Network for Image-Text Retrieval.**<br>
*Qi Zhang, Zhen Lei, Zhaoxiang Zhang, Stan Z. Li.*<br>
**_(CVPR 2020)_**<br>
[[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Context-Aware_Attention_Network_for_Image-Text_Retrieval_CVPR_2020_paper.pdf)

**Universal Weighting Metric Learning for Cross-Modal Matching.**<br>
*Jiwei Wei, Xing Xu, Yang Yang, Yanli Ji, Zheng Wang, Heng Tao Shen.*<br>
**_(CVPR 2020)_**<br>
[[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Universal_Weighting_Metric_Learning_for_Cross-Modal_Matching_CVPR_2020_paper.pdf)

**Graph Optimal Transport for Cross-Domain Alignment.**<br>
*Liqun Chen, Zhe Gan, Yu Cheng, Linjie Li, Lawrence Carin, Jingjing Liu.*<br>
**_(ICML 2020)_**<br>
[[paper]](https://arxiv.org/pdf/2006.14744.pdf)

**Adaptive Offline Quintuplet Loss for Image-Text Matching.**<br>
*Tianlang Chen, Jiajun Deng, Jiebo Luo.*<br>
**_(ECCV 2020)_**<br>
[[paper]](https://arxiv.org/pdf/2003.03669.pdf)
[[code]](https://github.com/sunnychencool/AOQ)(PyTorch)

**Learning Joint Visual Semantic Matching Embeddings for Language-guided Retrieval.**<br>
*Yanbei Chen, Loris Bazzani.*<br>
**_(ECCV 2020)_**<br>
[[paper]](https://assets.amazon.science/5b/db/440af26349adb83c77c85cd11922/learning-joint-visual-semantic-matching-embeddings-for-text-guided-retrieval.pdf)

**Consensus-Aware Visual-Semantic Embeddingfor Image-Text Matching.**<br>
*Haoran Wang, Ying Zhang, Zhong Ji, Yanwei Pang, Lin Ma.*<br>
**_(ECCV 2020)_**<br>
[[paper]](https://arxiv.org/pdf/2007.08883.pdf)
[[code]](https://github.com/BruceW91/CVSE)(PyTorch)

**Contrastive Learning for Weakly Supervised Phrase Grounding.**<br>
*Tanmay Gupta, Arash Vahdat, Gal Chechik, Xiaodong Yang, Jan Kautz, Derek Hoiem.*<br>
**_(ECCV 2020)_**<br>
[[paper]](https://arxiv.org/pdf/2006.09920.pdf)
[[code]](https://github.com/BigRedT/info-ground)(PyTorch)

**Preserving Semantic Neighborhoods for Robust Cross-modal Retrieval.**<br>
*Christopher Thomas, Adriana Kovashka.*<br>
**_(ECCV 2020)_**<br>
[[paper]](https://arxiv.org/pdf/2007.08617.pdf)


**Probing Multimodal Embeddings for Linguistic Properties: the Visual-Semantic Case.**<br>
*Adam Dahlgren Lindström, Suna Bensch, Johanna Björklund, Frank Drewes.*<br>
**_(COLING 2020)_**<br>
[[paper]](https://www.aclweb.org/anthology/2020.coling-main.64.pdf)
[[code]](https://github.com/dali-does/vse-probing)(PyTorch)

## Journals
**Large scale image annotation: learning to rank with joint word-image embeddings.**<br>
*Jason Weston, Samy Bengio, Nicolas Usunier.*<br>
**_(Machine Learning 2010)_**<br>
[[paper]](https://link.springer.com/content/pdf/10.1007%2Fs10994-010-5198-3.pdf)

**(*SDT-RNN_TACL2014*) Grounded Compositional Semantics for Finding and Describing Images with Sentences.**<br>
*Richard Socher, Andrej Karpathy, Quoc V. Le, Christopher D. Manning, Andrew Y. Ng.*<br>
[[paper]](https://www.aclweb.org/anthology/Q14-1017.pdf)

**Learning Two-Branch Neural Networks for Image-Text Matching Tasks.**<br>
*Liwei Wang, Yin Li, Jing Huang, Svetlana Lazebnik.*<br>
**_(IPAMI 2019)_**<br>
[[paper]](https://arxiv.org/pdf/1704.03470.pdf)
[[code]](https://github.com/lwwang/Two_branch_network)(TF)

**Upgrading the Newsroom: An Automated Image Selection System for News Articles.**<br>
*Fangyu Liu, Rémi Lebret, Didier Orel, Philippe Sordet, Karl Aberer.*<br>
**_(ACM TOMM 2020)_**<br>
[[paper]](https://arxiv.org/pdf/2004.11449.pdf)
[[slides]](http://fangyuliu.me/media/others/lsir_talk_final_version_0.3.pdf)
[[demo]](https://modemos.epfl.ch/article)
