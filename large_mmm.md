Methods Summary of Large Multi-Modality Model
==============================

## ``Catalogue ``
* [Large Language Model](#large-language-model)
* [Large Vision Model](#large-vision-model)
* [Large Region Multimodal Model](#large-region-multimodal-model)
* [Large Image Multimodal Model](#large-image-multimodal-model)
* [Large Video Multimodal Model](#large-video-multimodal-model)
* [Large Model Distillation](#large-modal-distillation)
* [Related Survey](#related-survey)
* [Related Benchmark](#related-benchmark)


### ``*Large Language Model*``

**(*arXiv2018_GPT*) Improving Language Understanding by Generative Pre-Training.** <br>
*Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever.*<br>
[[paper]](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
[[code]](https://github.com/openai/finetune-transformer-lm)

**(*NAACL2019_BERT*) BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.** <br>
*Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova.*<br>
[[paper]](https://arxiv.org/abs/1810.04805)
[[code]](https://github.com/google-research/bert)

**(*arXiv2019_GPT-2*) Language Models are Unsupervised Multitask Learners.** <br>
*Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever.*<br>
[[paper]](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
[[code]](https://github.com/openai/gpt-2)

**(*NeurIPS2019_UniLM*) Unified Language Model Pre-training for Natural Language Understanding and Generation.** <br>
*Li Dong, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng Gao, Ming Zhou, Hsiao-Wuen Hon.*<br>
[[paper]](https://arxiv.org/abs/1905.03197)
[[code]](https://github.com/microsoft/unilm/tree/master/unilm-v1)

**(*NeurIPS2019_XLNet*) XLNet: Generalized Autoregressive Pretraining for Language Understanding.** <br>
*Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.*<br>
[[paper]](https://arxiv.org/abs/1906.08237)
[[code]](https://github.com/zihangdai/xlnet)

**(*ICML2020_UniLMv2*) UniLMv2: Pseudo-Masked Language Models for Unified Language Model Pre-Training.** <br>
*Hangbo Bao, Li Dong, Furu Wei, Wenhui Wang, Nan Yang, Xiaodong Liu, Yu Wang, Songhao Piao, Jianfeng Gao, Ming Zhou, Hsiao-Wuen Hon.*<br>
[[paper]](https://arxiv.org/abs/2002.12804)
[[code]](https://github.com/microsoft/unilm/tree/master/unilm)

**(*arXiv2020_GPT-3*) Language Models are Few-Shot Learners.** <br>
*OpenAI Team.*<br>
[[paper]](https://arxiv.org/abs/2005.14165)
[[code]](https://github.com/openai/gpt-3)

**(*arXiv2022_PaLM*) PaLM: Scaling Language Modeling with Pathways.** <br>
*Google Research.*<br>
[[paper]](https://arxiv.org/abs/2204.02311)
[[code]](https://github.com/lucidrains/PaLM-pytorch)

**(*arXiv2023_LLaMA*) LLaMA: Open and Efficient Foundation Language Models.** <br>
*Meta Team.*<br>
[[paper]](https://arxiv.org/abs/2302.13971)
[[code]](https://github.com/facebookresearch/llama)

**(*arXiv2023_RWKV*) RWKV: Reinventing RNNs for the Transformer Era.** <br>
*RWKV Team.*<br>
[[paper]](https://arxiv.org/abs/2305.13048)
[[code]](https://github.com/BlinkDL/RWKV-LM)

**(*arXiv2023_LLM-Judge*) Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.** <br>
*Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica.*<br>
[[paper]](https://arxiv.org/abs/2306.05685)
[[code]](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)

**(*arXiv2023_RETNET*) Retentive Network: A Successor to Transformer for Large Language Models.** <br>
*Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, Furu Wei.*<br>
[[paper]](https://arxiv.org/abs/2307.08621)
[[code]](https://aka.ms/retnet)

**(*arXiv2023_Llama 2*) Llama 2: Open Foundation and Fine-Tuned Chat Models.** <br>
*Meta Team.*<br>
[[paper]](https://arxiv.org/abs/2307.09288)
[[code]](https://github.com/facebookresearch/llama)

**(*arXiv2023_InternLM*) InternLM: A Multilingual Language Model with Progressively Enhanced Capabilities.** <br>
*InternLM Team.*<br>
[[paper]](https://github.com/InternLM/InternLM-techreport/blob/main/InternLM.pdf)
[[code]](https://github.com/InternLM/InternLM)

**(*arXiv2023_Qwen*) Qwen Technical Report.** <br>
*Qwen Team.*<br>
[[paper]](https://arxiv.org/abs/2309.16609)
[[code]](https://github.com/qwenlm/qwen)

**(*arXiv2023_LightSeq*) LightSeq: Sequence Level Parallelism for Distributed Training of Long Context Transformers.** <br>
*Dacheng Li, Rulin Shao, Anze Xie, Eric P. Xing, Joseph E. Gonzalez, Ion Stoica, Xuezhe Ma, Hao Zhang.*<br>
[[paper]](https://arxiv.org/abs/2310.03294v1)
[[code]](https://github.com/RulinShao/LightSeq)

**(*arXiv2023_Mamba*) Mamba: Linear-Time Sequence Modeling with Selective State Spaces.** <br>
*Albert Gu, Tri Dao.*<br>
[[paper]](https://arxiv.org/abs/2312.00752)
[[code]](https://github.com/state-spaces/mamba)

**(*arXiv2024_Mixtral*) Mixtral of Experts.** <br>
*Mistral.AI.*<br>
[[paper]](https://arxiv.org/abs/2401.04088)
[[code]](https://github.com/mistralai/mistral-src)

**(*arXiv2024_Phi-3*) Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone.** <br>
*Microsoft Team.*<br>
[[paper]](https://arxiv.org/abs/2404.14219)


### ``*Large Vision Model*``

**(*ICLR2021_ViT*) An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.** <br>
*Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.*<br>
[[paper]](https://arxiv.org/abs/2010.11929)
[[code]](https://github.com/google-research/vision_transformer)

**(*ICCV2021_ViViT*) ViViT: A Video Vision Transformer.** <br>
*Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lučić, Cordelia Schmid.*<br>
[[paper]](https://arxiv.org/abs/2103.15691)
[[code]](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit)

**(*arXiv2021_MLP-Mixer*) MLP-Mixer: An all-MLP Architecture for Vision.** <br>
*Ilya Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, Alexey Dosovitskiy.*<br>
[[paper]](https://arxiv.org/abs/2105.01601)
[[code]](https://github.com/google-research/vision_transformer)

**(*ICLR2022_BEiT*) BEiT: BERT Pre-Training of Image Transformers.** <br>
*Hangbo Bao, Li Dong, Songhao Piao, Furu Wei.*<br>
[[paper]](https://arxiv.org/abs/2106.08254)
[[code]](https://aka.ms/beit)

**(*CVPR2022_MAE*) Masked Autoencoders Are Scalable Vision Learners.** <br>
*Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick.*<br>
[[paper]](https://arxiv.org/abs/2111.06377)
[[code]](https://github.com/facebookresearch/mae)

**(*ECCV2022_MVP*) MVP: Multimodality-guided Visual Pre-training.** <br>
*Longhui Wei, Lingxi Xie, Wengang Zhou, Houqiang Li, Qi Tian.*<br>
[[paper]](https://arxiv.org/abs/2203.05175)

**(*arXiv2022_BEiTv2*) BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers.** <br>
*Zhiliang Peng, Li Dong, Hangbo Bao, Qixiang Ye, Furu Wei.*<br>
[[paper]](https://arxiv.org/abs/2208.06366)
[[code]](https://github.com/microsoft/unilm/tree/master/beit2)

**(*ICLR2023_ToME*) Token Merging: Your ViT But Faster.** <br>
*Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, Judy Hoffman.*<br>
[[paper]](https://arxiv.org/abs/2210.09461)
[[code]](https://github.com/facebookresearch/tome)

**(*CVPR2023_EVA*) EVA: Exploring the Limits of Masked Visual Representation Learning at Scale.** <br>
*Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, Yue Cao.*<br>
[[paper]](https://arxiv.org/abs/2211.07636)
[[code]](https://github.com/baaivision/EVA)

**(*CVPR2023_Painter*) Images Speak in Images: A Generalist Painter for In-Context Visual Learning.** <br>
*Xinlong Wang, Wen Wang, Yue Cao, Chunhua Shen, Tiejun Huang.*<br>
[[paper]](https://arxiv.org/abs/2212.02499)
[[code]](https://github.com/baaivision/Painter)

**(*CVPR2023_MAGVIT*) MAGVIT: Masked Generative Video Transformer.** <br>
*Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G. Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, Lu Jiang.*<br>
[[paper]](https://arxiv.org/abs/2212.05199)
[[code]](https://magvit.cs.cmu.edu/)

**(*ICML2023_ViT-22B*) Scaling Vision Transformers to 22 Billion Parameters.** <br>
*Google Research.*<br>
[[paper]](https://arxiv.org/abs/2302.05442)

**(*arXiv2023_EVA-02*) EVA-02: A Visual Representation for Neon Genesis.** <br>
*Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, Yue Cao.*<br>
[[paper]](https://arxiv.org/abs/2303.11331)
[[code]](https://github.com/baaivision/EVA/tree/master/EVA-02)

**(*arXiv2023_EVA-CLIP*) EVA-CLIP: Improved Training Techniques for CLIP at Scale.** <br>
*Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, Yue Cao.*<br>
[[paper]](https://arxiv.org/abs/2303.15389)
[[code]](https://github.com/baaivision/EVA/tree/master/EVA-CLIP)

**(*CVPR2023_VideoMAEv2*) VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking.** <br>
*Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, Yu Qiao.*<br>
[[paper]](https://arxiv.org/abs/2303.16727)
[[code]](https://github.com/OpenGVLab/VideoMAEv2)

**(*ICCV2023_SegGPT*) SegGPT: Segmenting Everything In Context.** <br>
*Xinlong Wang, Xiaosong Zhang, Yue Cao, Wen Wang, Chunhua Shen, Tiejun Huang.*<br>
[[paper]](https://arxiv.org/abs/2304.03284)
[[code]](https://github.com/baaivision/Painter)

**(*ICLR2024_MAGVITv2*) Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation.** <br>
*Lijun Yu, José Lezama, Nitesh B. Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G. Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A. Ross, Lu Jiang.*<br>
[[paper]](https://arxiv.org/abs/2310.05737)

**(*CVPR2024_LVM*) Sequential Modeling Enables Scalable Learning for Large Vision Models.** <br>
*Yutong Bai, Xinyang Geng, Karttikeya Mangalam, Amir Bar, Alan Yuille, Trevor Darrell, Jitendra Malik, Alexei A Efros.*<br>
[[paper]](https://arxiv.org/abs/2312.00785)
[[code]](https://github.com/ytongbai/LVM)

**(*arXiv2024_AIM*) Scalable Pre-training of Large Autoregressive Image Models.** <br>
*Alaaeldin El-Nouby, Michal Klein, Shuangfei Zhai, Miguel Angel Bautista, Alexander Toshev, Vaishaal Shankar, Joshua M Susskind, Armand Joulin.*<br>
[[paper]](https://arxiv.org/abs/2401.08541)
[[code]](https://github.com/apple/ml-aim)

**(*arXiv2024_VIM*) Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model.** <br>
*Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, Xinggang Wang.*<br>
[[paper]](https://arxiv.org/abs/2401.09417)
[[code]](https://github.com/hustvl/Vim)

**(*arXiv2024_EVA-CLIP-18B*) EVA-CLIP-18B: Scaling CLIP to 18 Billion Parameters.** <br>
*Quan Sun, Jinsheng Wang, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Xinlong Wang.*<br>
[[paper]](https://arxiv.org/abs/2402.04252)
[[code]](https://github.com/baaivision/EVA/tree/master/EVA-CLIP-18B)

**(*arXiv2024_VisionLLaMA*) VisionLLaMA: A Unified LLaMA Interface for Vision Tasks.** <br>
*Xiangxiang Chu, Jianlin Su, Bo Zhang, Chunhua Shen.*<br>
[[paper]](https://arxiv.org/abs/2403.00522)
[[code]](https://github.com/Meituan-AutoML/VisionLLaMA)

**(*arXiv2024_Vision-RWKV*) Vision-RWKV: Efficient and Scalable Visual Perception with RWKV-Like Architectures.** <br>
*Yuchen Duan, Weiyun Wang, Zhe Chen, Xizhou Zhu, Lewei Lu, Tong Lu, Yu Qiao, Hongsheng Li, Jifeng Dai, Wenhai Wang.*<br>
[[paper]](https://arxiv.org/abs/2403.02308)
[[code]](https://github.com/OpenGVLab/Vision-RWKV)

**(*arXiv2024_VideoMamba*) VideoMamba: State Space Model for Efficient Video Understanding.** <br>
*Kunchang Li, Xinhao Li, Yi Wang, Yinan He, Yali Wang, Limin Wang, Yu Qiao.*<br>
[[paper]](https://arxiv.org/abs/2403.06977)
[[code]](https://github.com/opengvlab/videomamba)

**(*arXiv2024_VAR*) Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction.** <br>
*Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, Liwei Wang.*<br>
[[paper]](https://arxiv.org/abs/2404.02905)
[[code]](https://github.com/FoundationVision/VAR)

**(*arXiv2024_Ctrl-Adapter*) Ctrl-Adapter: An Efficient and Versatile Framework for Adapting Diverse Controls to Any Diffusion Model.** <br>
*Han Lin, Jaemin Cho, Abhay Zala, Mohit Bansal.*<br>
[[paper]](https://arxiv.org/abs/2404.09967)
[[code]](https://ctrl-adapter.github.io/)


### ``*Large Region Multimodal Model*``

**(*NeurIPS2023_VisionLLM*) VisionLLM: Large Language Model is also an Open-Ended Decoder for Vision-Centric Tasks.** <br>
*Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, Jifeng Dai.*<br>
[[paper]](https://arxiv.org/abs/2305.11175)
[[code]](https://github.com/OpenGVLab/VisionLLM)

**(*NeurIPS2023_RECODE*) Zero-shot Visual Relation Detection via Composite Visual Cues from Large Language Models.** <br>
*Lin Li, Jun Xiao, Guikun Chen, Jian Shao, Yueting Zhuang, Long Chen.*<br>
[[paper]](https://arxiv.org/abs/2305.12476)
[[code]](https://github.com/hkust-longgroup/recode)

**(*arXiv2023_DetGPT*) DetGPT: Detect What You Need via Reasoning.** <br>
*Renjie Pi, Jiahui Gao, Shizhe Diao, Rui Pan, Hanze Dong, Jipeng Zhang, Lewei Yao, Jianhua Han, Hang Xu, Lingpeng Kong, Tong Zhang.*<br>
[[paper]](https://arxiv.org/abs/2305.14167)
[[code]](http://detgpt.github.io/)

**(*arXiv2023_DAC*) Dense and Aligned Captions (DAC) Promote Compositional Reasoning in VL Models.** <br>
*Sivan Doveh, Assaf Arbelle, Sivan Harary, Roei Herzig, Donghyun Kim, Paola Cascante-bonilla, Amit Alfassy, Rameswar Panda, Raja Giryes, Rogerio Feris, Shimon Ullman, Leonid Karlinsky.*<br>
[[paper]](https://arxiv.org/abs/2305.19595)

**(*arXiv2023_Kosmos-2*) Kosmos-2: Grounding Multimodal Large Language Models to the World.** <br>
*Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, Furu Wei.*<br>
[[paper]](https://arxiv.org/abs/2306.14824)
[[code]](https://aka.ms/kosmos-2)

**(*arXiv2023_BuboGPT*) BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs.** <br>
*Yang Zhao, Zhijie Lin, Daquan Zhou, Zilong Huang, Jiashi Feng, Bingyi Kang.*<br>
[[paper]](https://arxiv.org/abs/2307.08581)
[[code]](https://bubo-gpt.github.io/)

**(*arXiv2023_ChatSpot*) ChatSpot: Bootstrapping Multimodal LLMs via Precise Referring Instruction Tuning.** <br>
*Liang Zhao, En Yu, Zheng Ge, Jinrong Yang, Haoran Wei, Hongyu Zhou, Jianjian Sun, Yuang Peng, Runpei Dong, Chunrui Han, Xiangyu Zhang.*<br>
[[paper]](https://arxiv.org/abs/2307.09474)

**(*CVPR2024_LISA*) LISA: Reasoning Segmentation via Large Language Model.** <br>
*Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, Jiaya Jia.*<br>
[[paper]](https://arxiv.org/abs/2308.00692)
[[code]](https://github.com/dvlab-research/LISA)

**(*CVPR2024_LLM4SGG*) LLM4SGG: Large Language Models for Weakly Supervised Scene Graph Generation.** <br>
*Kibum Kim, Kanghoon Yoon, Jaehyeong Jeon, Yeonjun In, Jinyoung Moon, Donghyun Kim, Chanyoung Park.*<br>
[[paper]](https://arxiv.org/abs/2310.10404)
[[code]](https://github.com/rlqja1107/torch-LLM4SGG)

**(*arXiv2023_SoM*) Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V.** <br>
*Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, Jianfeng Gao.*<br>
[[paper]](https://arxiv.org/abs/2310.11441)
[[code]](https://github.com/microsoft/SoM)

**(*CVPR2024_GLaMM*) GLaMM: Pixel Grounding Large Multimodal Model.** <br>
*Hanoona Rasheed, Muhammad Maaz, Sahal Shaji Mullappilly, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M. Anwer, Erix Xing, Ming-Hsuan Yang, Fahad S. Khan.*<br>
[[paper]](https://arxiv.org/abs/2311.03356)
[[code]](https://mbzuai-oryx.github.io/groundingLMM)

**(*CVPR2024_LION*) LION: Empowering Multimodal Large Language Model with Dual-Level Visual Knowledge.** <br>
*Gongwei Chen, Leyang Shen, Rui Shao, Xiang Deng, Liqiang Nie.*<br>
[[paper]](https://arxiv.org/abs/2311.11860)
[[code]](https://github.com/rshaojimmy/JiuTian)

**(*arXiv2023_PG-Video-LLaVA*) PG-Video-LLaVA: Pixel Grounding Large Video-Language Models.** <br>
*Shehan Munasinghe, Rusiru Thushara, Muhammad Maaz, Hanoona Abdul Rasheed, Salman Khan, Mubarak Shah, Fahad Khan.*<br>
[[paper]](https://arxiv.org/abs/2311.13435)
[[code]](https://github.com/mbzuai-oryx/Video-LLaVA)

**(*arXiv2023_DINOv*) Visual In-Context Prompting.** <br>
*Feng Li, Qing Jiang, Hao Zhang, Tianhe Ren, Shilong Liu, Xueyan Zou, Huaizhe Xu, Hongyang Li, Chunyuan Li, Jianwei Yang, Lei Zhang, Jianfeng Gao.*<br>
[[paper]](https://arxiv.org/abs/2311.13601)
[[code]](https://github.com/UX-Decoder/DINOv)

**(*arXiv2023_TAP*) Tokenize Anything via Prompting.** <br>
*Ting Pan, Lulu Tang, Xinlong Wang, Shiguang Shan.*<br>
[[paper]](https://arxiv.org/abs/2312.09128)
[[code]](https://github.com/baaivision/tokenize-anything)

**(*CVPR2024_Emu2*) Generative Multimodal Models are In-Context Learners.** <br>
*Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, Xinlong Wang.*<br>
[[paper]](https://arxiv.org/abs/2312.13286)
[[code]](https://baaivision.github.io/emu2)

**(*arXiv2024_VisionLLMv2*) VisionLLM v2: An End-to-End Generalist Multimodal Large Language Model for Hundreds of Vision-Language Tasks.** <br>
*Jiannan Wu, Muyan Zhong, Sen Xing, Zeqiang Lai, Zhaoyang Liu, Wenhai Wang, Zhe Chen, Xizhou Zhu, Lewei Lu, Tong Lu, Ping Luo, Yu Qiao, Jifeng Dai.*<br>
[[paper]](https://arxiv.org/abs/2406.08394)
[[code]](https://github.com/opengvlab/visionllm)


### ``*Large Image Multimodal Model*``

**(*NeurIPS2022_Flamingo*) Flamingo: a Visual Language Model for Few-Shot Learning.** <br>
*DeepMind Team.*<br>
[[paper]](https://arxiv.org/abs/2204.14198)
[[code]](https://github.com/mlfoundations/open_flamingo)

**(*CVPR2023_BEiTv3*) Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks.** <br>
*Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, Furu Wei.*<br>
[[paper]](https://arxiv.org/abs/2208.10442)
[[code]](https://github.com/microsoft/unilm/tree/master/beit3)

**(*ICCV2023_DiT*) Scalable Diffusion Models with Transformers.** <br>
*William Peebles, Saining Xie.*<br>
[[paper]](https://arxiv.org/abs/2212.09748)
[[code]](https://www.wpeebles.com/DiT)

**(*ICML2023_mPLUG-2*) mPLUG-2: A Modularized Multi-modal Foundation Model Across Text, Image and Video.** <br>
*Haiyang Xu, Qinghao Ye, Ming Yan, Yaya Shi, Jiabo Ye, Yuanhong Xu, Chenliang Li, Bin Bi, Qi Qian, Wei Wang, Guohai Xu, Ji Zhang, Songfang Huang, Fei Huang, Jingren Zhou.*<br>
[[paper]](https://arxiv.org/abs/2302.00402v1)
[[code]](https://github.com/alibaba/AliceMind)

**(*ICCV2023_ControlNet*) Adding Conditional Control to Text-to-Image Diffusion Models.** <br>
*Lvmin Zhang, Anyi Rao, Maneesh Agrawala.*<br>
[[paper]](https://arxiv.org/abs/2302.05543)
[[code]](https://github.com/lllyasviel/ControlNet)

**(*arXiv2023_Kosmos-1*) Language Is Not All You Need: Aligning Perception with Language Models.** <br>
*Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, Qiang Liu, Kriti Aggarwal, Zewen Chi, Johan Bjorck, Vishrav Chaudhary, Subhojit Som, Xia Song, Furu Wei.*<br>
[[paper]](https://arxiv.org/abs/2302.14045)
[[code]](https://github.com/microsoft/unilm/tree/master/kosmos-1)

**(*arXiv2023_PaLM-E*) PaLM-E: An Embodied Multimodal Language Model.** <br>
*Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, Pete Florence.*<br>
[[paper]](https://arxiv.org/abs/2303.03378)
[[code]](https://github.com/kyegomez/PALM-E)

**(*arXiv2023_Visual-ChatGPT*) Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models.** <br>
*Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, Nan Duan.*<br>
[[paper]](https://arxiv.org/abs/2303.04671)
[[code]](https://github.com/microsoft/visual-chatgpt)

**(*CVPR2023_GigaGAN*) Scaling up GANs for Text-to-Image Synthesis.** <br>
*Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, Taesung Park.*<br>
[[paper]](https://arxiv.org/abs/2303.05511)
[[code]](https://mingukkang.github.io/GigaGAN/)

**(*ICCV2023_ViperGPT*) ViperGPT: Visual Inference via Python Execution for Reasoning.** <br>
*Dídac Surís, Sachit Menon, Carl Vondrick.*<br>
[[paper]](https://arxiv.org/abs/2303.08128)
[[code]](https://viper.cs.columbia.edu/)

**(*arXiv2023_MM-REACT*) MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action.** <br>
*Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, Lijuan Wang.*<br>
[[paper]](https://arxiv.org/abs/2303.11381)
[[code]](https://github.com/microsoft/MM-REACT)

**(*arXiv2023_LLaMA-Adapter*) LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention.** <br>
*Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Yu Qiao.*<br>
[[paper]](https://arxiv.org/abs/2303.16199)
[[code]](https://github.com/OpenGVLab/LLaMA-Adapter)

**(*NeurIPS2023_HuggingGPT*) HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face.** <br>
*Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, Yueting Zhuang.*<br>
[[paper]](https://arxiv.org/abs/2303.17580)
[[code]](https://github.com/microsoft/JARVIS)

**(*NeurIPS2023_LLaVA*) Visual Instruction Tuning.** <br>
*Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee.*<br>
[[paper]](https://arxiv.org/abs/2304.08485)
[[code]](https://github.com/haotian-liu/LLaVA)

**(*arXiv2023_MiniGPT-4*) MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models.** <br>
*Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, Mohamed Elhoseiny.*<br>
[[paper]](https://arxiv.org/abs/2304.10592)
[[code]](https://minigpt-4.github.io/)

**(*arXiv2023_mPLUG-Owl*) mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality.** <br>
*Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, Chenliang Li, Yuanhong Xu, Hehong Chen, Junfeng Tian, Qi Qian, Ji Zhang, Fei Huang, Jingren Zhou.*<br>
[[paper]](https://arxiv.org/abs/2304.14178)
[[code]](https://www.modelscope.cn/studios/damo/mPLUG-Owl)

**(*arXiv2023_LLaMA-AdapterV2*) LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model.** <br>
*Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, Hongsheng Li, Yu Qiao.*<br>
[[paper]](https://arxiv.org/abs/2304.15010)
[[code]](https://github.com/ZrrSkywalker/LLaMA-Adapter)

**(*arXiv2023_Otter*) Otter: A Multi-Modal Model with In-Context Instruction Tuning.** <br>
*Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, Ziwei Liu.*<br>
[[paper]](https://arxiv.org/abs/2305.03726)
[[code]](https://github.com/luodian/otter)

**(*arXiv2023_MultiModal-GPT*) MultiModal-GPT: A Vision and Language Model for Dialogue with Humans.** <br>
*Tao Gong, Chengqi Lyu, Shilong Zhang, Yudong Wang, Miao Zheng, Qian Zhao, Kuikun Liu, Wenwei Zhang, Ping Luo, Kai Chen.*<br>
[[paper]](https://arxiv.org/abs/2305.04790)
[[code]](https://github.com/open-mmlab/multimodal-gpt)

**(*arXiv2023_InternGPT*) InternGPT: Solving Vision-Centric Tasks by Interacting with ChatGPT Beyond Language.** <br>
*Zhaoyang Liu, Yinan He, Wenhai Wang, Weiyun Wang, Yi Wang, Shoufa Chen, Qinglong Zhang, Zeqiang Lai, Yang Yang, Qingyun Li, Jiashuo Yu, Kunchang Li, Zhe Chen, Xue Yang, Xizhou Zhu, Yali Wang, Limin Wang, Ping Luo, Jifeng Dai, Yu Qiao.*<br>
[[paper]](https://arxiv.org/abs/2305.05662)
[[code]](https://github.com/OpenGVLab/InternGPT)

**(*NeurIPS2023_InstructBLIP*) InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning.** <br>
*Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, Steven Hoi.*<br>
[[paper]](https://arxiv.org/abs/2305.06500)
[[code]](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)

**(*EMNLP2023_IdealGPT*) IdealGPT: Iteratively Decomposing Vision and Language Reasoning via Large Language Models.** <br>
*Haoxuan You, Rui Sun, Zhecan Wang, Long Chen, Gengyu Wang, Hammad A. Ayyubi, Kai-Wei Chang, Shih-Fu Chang.*<br>
[[paper]](https://arxiv.org/abs/2305.14985)
[[code]](https://github.com/Hxyou/IdealGPT)

**(*NeurIPS2023_LaVIN*) Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models.** <br>
*Gen Luo, Yiyi Zhou, Tianhe Ren, Shengxin Chen, Xiaoshuai Sun, Rongrong Ji.*<br>
[[paper]](https://arxiv.org/abs/2305.15023)
[[code]](https://luogen1996.github.io/lavin)

**(*arXiv2023_PandaGPT*) PandaGPT: One Model To Instruction-Follow Them All.** <br>
*Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, Deng Cai.*<br>
[[paper]](https://arxiv.org/abs/2305.16355)
[[code]](https://panda-gpt.github.io/)

**(*NeurIPS2023_GILL*) Generating Images with Multimodal Language Models.** <br>
*Jing Yu Koh, Daniel Fried, Ruslan Salakhutdinov.*<br>
[[paper]](https://arxiv.org/abs/2305.17216)
[[code]](http://jykoh.com/gill)

**(*NeurIPS2023_GPT4Tools*) GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction.** <br>
*Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu Li, Ying Shan.*<br>
[[paper]](https://arxiv.org/abs/2305.18752)
[[code]](https://github.com/stevengrove/gpt4tools)

**(*arXiv2023_MIMIC-IT*) MIMIC-IT: Multi-Modal In-Context Instruction Tuning.** <br>
*Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Fanyi Pu, Jingkang Yang, Chunyuan Li, Ziwei Liu.*<br>
[[paper]](https://arxiv.org/abs/2306.05425)
[[code]](https://github.com/Luodian/otter)

**(*AAAI2024_MotionGPT*) MotionGPT: Finetuned LLMs Are General-Purpose Motion Generators.** <br>
*Yaqi Zhang, Di Huang, Bin Liu, Shixiang Tang, Yan Lu, Lu Chen, Lei Bai, Qi Chu, Nenghai Yu, Wanli Ouyang.*<br>
[[paper]](https://arxiv.org/abs/2306.10900)
[[code]](https://qiqiapink.github.io/MotionGPT/)

**(*ICLR2023_Emu*) Generative Pretraining in Multimodality.** <br>
*Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, Xinlong Wang.*<br>
[[paper]](https://arxiv.org/abs/2307.05222)
[[code]](https://github.com/baaivision/Emu)

**(*Blog2023_IDEFICS*) Introducing IDEFICS: An Open Reproduction of State-of-the-Art Visual Language Model.** <br>
*Hugo Laurençon, Daniel van Strien, Stas Bekman, Leo Tronchon, Lucile Saulnier, Thomas Wang, Siddharth Karamcheti, Amanpreet Singh, Giada Pistilli, Yacine Jernite, Victor Sanh.*<br>
[[blog]](https://huggingface.co/blog/idefics)

**(*AAAI2024_BLIVA*) BLIVA: A Simple Multimodal LLM for Better Handling of Text-Rich Visual Questions.** <br>
*Wenbo Hu, Yifan Xu, Yi Li, Weiyue Li, Zeyuan Chen, Zhuowen Tu.*<br>
[[paper]](https://arxiv.org/abs/2308.09936)
[[code]](https://github.com/mlpc-ucsd/BLIVA)

**(*arXiv2023_Qwen-VL*) Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond.** <br>
*Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, Jingren Zhou.*<br>
[[paper]](https://arxiv.org/abs/2308.12966)
[[code]](https://github.com/QwenLM/Qwen-VL)

**(*ICML2024_NExT-GPT*) NExT-GPT: Any-to-Any Multimodal LLM.** <br>
*Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, Tat-Seng Chua.*<br>
[[paper]](https://arxiv.org/abs/2309.05519)
[[code]](https://github.com/NExT-GPT/NExT-GPT)

**(*ACL2024_TextBind*) TextBind: Multi-turn Interleaved Multimodal Instruction-following in the Wild.** <br>
*Huayang Li, Siheng Li, Deng Cai, Longyue Wang, Lemao Liu, Taro Watanabe, Yujiu Yang, Shuming Shi.*<br>
[[paper]](https://arxiv.org/abs/2309.08637)
[[code]](https://github.com/sihengli99/textbind)

**(*arXiv2023_Kosmos-2.5*) Kosmos-2.5: A Multimodal Literate Model.** <br>
*Tengchao Lv, Yupan Huang, Jingye Chen, Lei Cui, Shuming Ma, Yaoyao Chang, Shaohan Huang, Wenhui Wang, Li Dong, Weiyao Luo, Shaoxiang Wu, Guoxin Wang, Cha Zhang, Furu Wei.*<br>
[[paper]](https://arxiv.org/abs/2309.11419)
[[code]](https://github.com/microsoft/unilm/tree/master/kosmos-2.5)

**(*ICLR2024_DreamLLM*) DreamLLM: Synergistic Multimodal Comprehension and Creation.** <br>
*Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, Xiangwen Kong, Xiangyu Zhang, Kaisheng Ma, Li Yi.*<br>
[[paper]](https://arxiv.org/abs/2309.11499)
[[code]](https://dreamllm.github.io/)

**(*arXiv2023_InternLM-XComposer*) InternLM-XComposer: A Vision-Language Large Model for Advanced Text-image Comprehension and Composition.** <br>
*Pan Zhang, Xiaoyi Dong, Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Haodong Duan, Songyang Zhang, Shuangrui Ding, Wenwei Zhang, Hang Yan, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, Jiaqi Wang.*<br>
[[paper]](https://arxiv.org/abs/2309.15112)
[[code]](https://github.com/InternLM/InternLM-XComposer)

**(*CVPR2024_LLaVA1.5*) Improved Baselines with Visual Instruction Tuning.** <br>
*Haotian Liu, Chunyuan Li, Yuheng Li, Yong Jae Lee.*<br>
[[paper]](https://arxiv.org/abs/2310.03744)
[[code]](https://llava-vl.github.io/)

**(*arXiv2023_OpenLEAF*) OpenLEAF: Open-Domain Interleaved Image-Text Generation and Evaluation.** <br>
*Jie An, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Lijuan Wang, Jiebo Luo.*<br>
[[paper]](https://arxiv.org/abs/2310.07749)

**(*arXiv2023_COMM*) From CLIP to DINO: Visual Encoders Shout in Multi-modal Large Language Models.** <br>
*Dongsheng Jiang, Yuchen Liu, Songlin Liu, Jin'e Zhao, Hao Zhang, Zhen Gao, Xiaopeng Zhang, Jin Li, Hongkai Xiong.*<br>
[[paper]](https://arxiv.org/abs/2310.08825)
[[code]](https://github.com/yuchenliu98/comm)

**(*arXiv2023_Open X-Embodiment*) Open X-Embodiment: Robotic Learning Datasets and RT-X Models.** <br>
*Open X-Embodiment Collaboration.*<br>
[[paper]](https://arxiv.org/abs/2310.08864)
[[code]](https://robotics-transformer-x.github.io/)

**(*arXiv2023_MiniGPT-v2*) MiniGPT-v2: Large Language Model as a Unified Interface for Vision-Language Multi-Task Learning.** <br>
*Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, Mohamed Elhoseiny.*<br>
[[paper]](https://arxiv.org/abs/2310.09478)
[[code]](https://minigpt-v2.github.io/)

**(*arXiv2023_Woodpecker*) Woodpecker: Hallucination Correction for Multimodal Large Language Models.** <br>
*Shukang Yin, Chaoyou Fu, Sirui Zhao, Tong Xu, Hao Wang, Dianbo Sui, Yunhang Shen, Ke Li, Xing Sun, Enhong Chen.*<br>
[[paper]](https://arxiv.org/abs/2310.16045)
[[code]](https://github.com/BradyFU/Woodpecker)

**(*CVPR2024_CapsFusion*) CapsFusion: Rethinking Image-Text Data at Scale.** <br>
*Qiying Yu, Quan Sun, Xiaosong Zhang, Yufeng Cui, Fan Zhang, Yue Cao, Xinlong Wang, Jingjing Liu.*<br>
[[paper]](https://arxiv.org/abs/2310.20550)
[[code]](https://github.com/baaivision/CapsFusion)

**(*Blog2023_Fuyu-8B*) Fuyu-8B: A Multimodal Architecture for AI Agents.** <br>
*Rohan Bavishi, Erich Elsen, Curtis Hawthorne, Maxwell Nye, Augustus Odena, Arushi Somani, Sağnak Taşırlar.*<br>
[[blog]](https://www.adept.ai/blog/fuyu-8b)

**(*Blog2024_Fuyu-Heavy*) Fuyu-Heavy: A New Multimodal Model.** <br>
*Adept Team.*<br>
[[blog]](https://www.adept.ai/blog/adept-fuyu-heavy)

**(*arXiv2023_CogVLM*) CogVLM: Visual Expert for Pretrained Language Models.** <br>
*Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, Jie Tang.*<br>
[[paper]](https://arxiv.org/abs/2311.03079)
[[code]](https://github.com/THUDM/CogVLM)

**(*arXiv2023_OtterHD*) OtterHD: A High-Resolution Multi-modality Model.** <br>
*Bo Li, Peiyuan Zhang, Jingkang Yang, Yuanhan Zhang, Fanyi Pu, Ziwei Liu.*<br>
[[paper]](https://arxiv.org/abs/2311.04219)
[[code]](https://github.com/Luodian/Otter)

**(*arXiv2023_mPLUG-Owl2*) mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration.** <br>
*Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, Fei Huang, Jingren Zhou.*<br>
[[paper]](https://arxiv.org/abs/2311.04257)
[[code]](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2)

**(*CVPR2024_Monkey*) Monkey: Image Resolution and Text Label Are Important Things for Large Multi-modal Models.** <br>
*Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, Xiang Bai.*<br>
[[paper]](https://arxiv.org/abs/2311.06607)
[[code]](https://github.com/Yuliang-Liu/Monkey)

**(*arXiv2023_LVIS-Instruct4V*) To See is to Believe: Prompting GPT-4V for Better Visual Instruction Tuning.** <br>
*Junke Wang, Lingchen Meng, Zejia Weng, Bo He, Zuxuan Wu, Yu-Gang Jiang.*<br>
[[paper]](https://arxiv.org/abs/2311.07574)
[[code]](https://github.com/X2FD/LVIS-INSTRUCT4V)

**(*arXiv2023_ShareGPT4V*) ShareGPT4V: Improving Large Multi-Modal Models with Better Captions.** <br>
*Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, Dahua Lin.*<br>
[[paper]](https://arxiv.org/abs/2311.12793)
[[code]](https://sharegpt4v.github.io/)

**(*CVPR2024_Powers-of-Ten*) Generative Powers of Ten.** <br>
*Xiaojuan Wang, Janne Kontkanen, Brian Curless, Steve Seitz, Ira Kemelmacher, Ben Mildenhall, Pratul Srinivasan, Dor Verbin, Aleksander Holynski.*<br>
[[paper]](https://arxiv.org/abs/2312.02149)
[[code]](https://powers-of-10.github.io/)

**(*CVPR2024_OneLLM*) OneLLM: One Framework to Align All Modalities with Language.** <br>
*Jiaming Han, Kaixiong Gong, Yiyuan Zhang, Jiaqi Wang, Kaipeng Zhang, Dahua Lin, Yu Qiao, Peng Gao, Xiangyu Yue.*<br>
[[paper]](https://arxiv.org/abs/2312.03700)
[[code]](https://github.com/csuhan/OneLLM)

**(*CVPR2024_Honeybee*) Honeybee: Locality-enhanced Projector for Multimodal LLM.** <br>
*Junbum Cha, Wooyoung Kang, Jonghwan Mun, Byungseok Roh.*<br>
[[paper]](https://arxiv.org/abs/2312.06742)
[[code]](https://github.com/kakaobrain/honeybee)

**(*arXiv2023_CogAgent*) CogAgent: A Visual Language Model for GUI Agents.** <br>
*Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxuan Zhang, Juanzi Li, Bin Xu, Yuxiao Dong, Ming Ding, Jie Tang.*<br>
[[paper]](https://arxiv.org/abs/2312.08914)
[[code]](https://github.com/THUDM/CogVLM)

**(*CVPR2024_InternVL*) InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks.** <br>
*Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, Jifeng Dai.*<br>
[[paper]](https://arxiv.org/abs/2312.14238)
[[code]](https://github.com/OpenGVLab/InternVL)

**(*arXiv2023_MobileVLM*) MobileVLM : A Fast, Strong and Open Vision Language Assistant for Mobile Devices.** <br>
*Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu, Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang, Xiaolin Wei, Chunhua Shen.*<br>
[[paper]](https://arxiv.org/abs/2312.16886)
[[code]](https://github.com/Meituan-AutoML/MobileVLM)

**(*arXiv2024_InternLM-XComposer2*) InternLM-XComposer2: Mastering Free-form Text-Image Composition and Comprehension in Vision-Language Large Model.** <br>
*Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, Jiaqi Wang.*<br>
[[paper]](https://arxiv.org/abs/2401.16420)
[[code]](https://github.com/InternLM/InternLM-XComposer)

**(*arXiv2024_CoBSAT*) Can MLLMs Perform Text-to-Image In-Context Learning?.** <br>
*Yuchen Zeng, Wonjun Kang, Yicong Chen, Hyung Il Koo, Kangwook Lee.*<br>
[[paper]](https://arxiv.org/abs/2402.01293)
[[code]](https://github.com/UW-Madison-Lee-Lab/CoBSAT)

**(*arXiv2024_MobileVLM-V2*) MobileVLM V2: Faster and Stronger Baseline for Vision Language Model.** <br>
*Xiangxiang Chu, Limeng Qiao, Xinyu Zhang, Shuang Xu, Fei Wei, Yang Yang, Xiaofei Sun, Yiming Hu, Xinyang Lin, Bo Zhang, Chunhua Shen.*<br>
[[paper]](https://arxiv.org/abs/2402.03766)
[[code]](https://github.com/Meituan-AutoML/MobileVLM)

**(*ICML2024_Prismatic-VLMs*) Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models.** <br>
*Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, Dorsa Sadigh.*<br>
[[paper]](https://arxiv.org/abs/2402.07865)
[[code]](https://github.com/TRI-ML/prismatic-vlms)

**(*arXiv2024_Bunny*) Efficient Multimodal Learning from Data-centric Perspective.** <br>
*Muyang He, Yexin Liu, Boya Wu, Jianhao Yuan, Yueze Wang, Tiejun Huang, Bo Zhao.*<br>
[[paper]](https://arxiv.org/abs/2402.11530)
[[code]](https://github.com/BAAI-DCAI/Bunny)

**(*arXiv2024_DeepSeek-VL*) DeepSeek-VL: Towards Real-World Vision-Language Understanding.** <br>
*Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, Yaofeng Sun, Chengqi Deng, Hanwei Xu, Zhenda Xie, Chong Ruan.*<br>
[[paper]](https://arxiv.org/abs/2403.05525)
[[code]](https://github.com/deepseek-ai/DeepSeek-VL)

**(*arXiv2024_LLaVA-UHD*) LLaVA-UHD: an LMM Perceiving Any Aspect Ratio and High-Resolution Images.** <br>
*Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, Maosong Sun, Gao Huang.*<br>
[[paper]](https://arxiv.org/abs/2403.11703v1)
[[code]](https://github.com/thunlp/LLaVA-UHD)

**(*arXiv2024_S2*) When Do We Not Need Larger Vision Models?.** <br>
*Baifeng Shi, Ziyang Wu, Maolin Mao, Xin Wang, Trevor Darrell.*<br>
[[paper]](https://arxiv.org/abs/2403.13043)
[[code]](https://github.com/bfshi/scaling_on_scales)

**(*arXiv2024_LLaVA-PruMerge*) LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models.** <br>
*Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, Yan Yan.*<br>
[[paper]](https://arxiv.org/abs/2403.15388)
[[code]](https://llava-prumerge.github.io/)

**(*arXiv2024_Mini-Gemini*) Mini-Gemini: Mining the Potential of Multi-modality Vision Language Models.** <br>
*Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, Jiaya Jia.*<br>
[[paper]](https://arxiv.org/abs/2403.18814)
[[code]](https://github.com/dvlab-research/MiniGemini)

**(*Blog2024_Idefics2*) Introducing Idefics2: A Powerful 8B Vision-Language Model for the community.** <br>
*Leo Tronchon, Hugo Laurençon, Victor Sanh.*<br>
[[blog]](https://huggingface.co/blog/idefics2)

**(*arXiv2024_InternLM-XComposer2-4KHD*) InternLM-XComposer2-4KHD: A Pioneering Large Vision-Language Model Handling Resolutions from 336 Pixels to 4K HD.** <br>
*Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Zhe Chen, Xinyue Zhang, Wei Li, Jingwen Li, Wenhai Wang, Kai Chen, Conghui He, Xingcheng Zhang, Jifeng Dai, Yu Qiao, Dahua Lin, Jiaqi Wang.*<br>
[[paper]](https://arxiv.org/abs/2404.06512)
[[code]](https://github.com/InternLM/InternLM-XComposer)

**(*arXiv2024_BRAVE*) BRAVE: Broadening the visual encoding of vision-language models.** <br>
*Oğuzhan Fatih Kar, Alessio Tonioni, Petra Poklukar, Achin Kulshrestha, Amir Zamir, Federico Tombari.*<br>
[[paper]](https://arxiv.org/abs/2404.07204)
[[code]](https://brave-vlms.epfl.ch/)

**(*arXiv2024_InternVL1.5*) How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites.** <br>
*Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, Ji Ma, Jiaqi Wang, Xiaoyi Dong, Hang Yan, Hewei Guo, Conghui He, Botian Shi, Zhenjiang Jin, Chao Xu, Bin Wang, Xingjian Wei, Wei Li, Wenjian Zhang, Bo Zhang, Pinlong Cai, Licheng Wen, Xiangchao Yan, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, Wenhai Wang.*<br>
[[paper]](https://arxiv.org/abs/2404.16821)
[[code]](https://github.com/OpenGVLab/InternVL)

**(*arXiv2024_MANTIS*) MANTIS: Interleaved Multi-Image Instruction Tuning.** <br>
*Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, Wenhu Chen.*<br>
[[paper]](https://arxiv.org/abs/2405.01483)
[[code]](https://github.com/TIGER-AI-Lab/Mantis)

**(*arXiv2024_Lumina-T2X*) Lumina-T2X: Transforming Text into Any Modality, Resolution, and Duration via Flow-based Large Diffusion Transformers.** <br>
*Peng Gao, Le Zhuo, Dongyang Liu, Ruoyi Du, Xu Luo, Longtian Qiu, Yuhang Zhang, Chen Lin, Rongjie Huang, Shijie Geng, Renrui Zhang, Junlin Xi, Wenqi Shao, Zhengkai Jiang, Tianshuo Yang, Weicai Ye, He Tong, Jingwen He, Yu Qiao, Hongsheng Li.*<br>
[[paper]](https://arxiv.org/abs/2405.05945)
[[code]](https://github.com/Alpha-VLLM/Lumina-T2X)

**(*arXiv2024_Cambrian-1*) Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs.** <br>
*Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, Saining Xie.*<br>
[[paper]](https://arxiv.org/abs/2406.16860)
[[code]](https://cambrian-mllm.github.io/)

**(*Blog2024_LLaVA-NeXT*) LLaVA-NeXT-series.** <br>
[[blog]](https://llava-vl.github.io/blog/)

**(*Blog2024_LMMS-Eval*) Accelerating the Development of Large Multimodal Models with LMMs-Eval.** <br>
[[blog]](https://lmms-lab.github.io/lmms-eval-blog/lmms-eval-0.1/)

**(*arXiv2024_LlamaGen*) Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation.** <br>
*Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, Zehuan Yuan.*<br>
[[paper]](https://arxiv.org/abs/2406.06525)
[[code]](https://github.com/FoundationVision/LlamaGen)

**(*arXiv2024_MAR*) Autoregressive Image Generation without Vector Quantization.** <br>
*Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, Kaiming He.*<br>
[[paper]](https://arxiv.org/abs/2406.11838)

**(*arXiv2024_EVE*) Unveiling Encoder-Free Vision-Language Models.** <br>
*Haiwen Diao, Yufeng Cui, Xiaotong Li, Yueze Wang, Huchuan Lu, Xinlong Wang.*<br>
[[paper]](https://arxiv.org/abs/2406.11832)
[[code]](https://github.com/baaivision/EVE)


### ``*Large Video Multimodal Model*``

**(*arXiv2022_InternVideo*) InternVideo: General Video Foundation Models via Generative and Discriminative Learning.** <br>
*Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, Sen Xing, Guo Chen, Junting Pan, Jiashuo Yu, Yali Wang, Limin Wang, Yu Qiao.*<br>
[[paper]](https://arxiv.org/abs/2212.03191)
[[code]](https://github.com/OpenGVLab/InternVideo)

**(*arXiv2022_VideoCoCa*) VideoCoCa: Video-Text Modeling with Zero-Shot Transfer from Contrastive Captioners.** <br>
*Shen Yan, Tao Zhu, Zirui Wang, Yuan Cao, Mi Zhang, Soham Ghosh, Yonghui Wu, Jiahui Yu.*<br>
[[paper]](https://arxiv.org/abs/2212.04979v3)

**(*arXiv2023_VideoChat*) VideoChat: Chat-Centric Video Understanding.** <br>
*KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, Yu Qiao.*<br>
[[paper]](https://arxiv.org/abs/2305.06355)
[[code]](https://github.com/opengvlab/ask-anything)

**(*EMNLP2023_Video-LLaMA*) Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding.** <br>
*Hang Zhang, Xin Li, Lidong Bing.*<br>
[[paper]](https://arxiv.org/abs/2306.02858)
[[code]](https://github.com/damo-nlp-sg/video-llama)

**(*arXiv2023_Video-ChatGPT*) Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models.** <br>
*Muhammad Maaz, Hanoona Rasheed, Salman Khan, Fahad Shahbaz Khan.*<br>
[[paper]](https://arxiv.org/abs/2306.05424)
[[code]](https://github.com/mbzuai-oryx/Video-ChatGPT)

**(*arXiv2023_Valley*) Valley: Video Assistant with Large Language model Enhanced abilitY.** <br>
*Ruipu Luo, Ziwang Zhao, Min Yang, Junwei Dong, Da Li, Pengcheng Lu, Tao Wang, Linmei Hu, Minghui Qiu, Zhongyu Wei.*<br>
[[paper]](https://arxiv.org/abs/2306.07207)
[[code]](https://github.com/rupertluo/valley)

**(*CVPR2024_MovieChat*) MovieChat: From Dense Token to Sparse Memory for Long Video Understanding.** <br>
*Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, Yan Lu, Jenq-Neng Hwang, Gaoang Wang.*<br>
[[paper]](https://arxiv.org/abs/2307.16449)
[[code]](https://github.com/rese1f/MovieChat)

**(*EMNLP2023_TESTA*) TESTA: Temporal-Spatial Token Aggregation for Long-form Video-Language Understanding.** <br>
*Shuhuai Ren, Sishuo Chen, Shicheng Li, Xu Sun, Lu Hou.*<br>
[[paper]](https://arxiv.org/abs/2310.19060)
[[code]](https://github.com/RenShuhuai-Andy/TESTA)

**(*CVPR2024_Chat-UniVi*) Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding.** <br>
*Peng Jin, Ryuichi Takanobu, Caiwan Zhang, Xiaochun Cao, Li Yuan.*<br>
[[paper]](https://arxiv.org/abs/2311.08046)
[[code]](https://github.com/PKU-YuanGroup/Chat-UniVi)

**(*arXiv2023_VideoChat2*) MVBench: A Comprehensive Multi-modal Video Understanding Benchmark.** <br>
*Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, Yu Qiao.*<br>
[[paper]](https://arxiv.org/abs/2311.17005)
[[code]](https://github.com/OpenGVLab/Ask-Anything)

**(*arXiv2023_LLaMA-VID*) LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models.** <br>
*Yanwei Li, Chengyao Wang, Jiaya Jia.*<br>
[[paper]](https://arxiv.org/abs/2311.17043)
[[code]](https://github.com/dvlab-research/LLaMA-VID)

**(*arXiv2024_Video-LaVIT*) Video-LaVIT: Unified Video-Language Pre-training with Decoupled Visual-Motional Tokenization.** <br>
*Yang Jin, Zhicheng Sun, Kun Xu, Kun Xu, Liwei Chen, Hao Jiang, Quzhe Huang, Chengru Song, Yuliang Liu, Di Zhang, Yang Song, Kun Gai, Yadong Mu.*<br>
[[paper]](https://arxiv.org/abs/2402.03161)
[[code]](https://video-lavit.github.io/)

**(*arXiv2024_LSTP*) LSTP: Language-guided Spatial-Temporal Prompt Learning for Long-form Video-Text Understanding.** <br>
*Yuxuan Wang, Yueqian Wang, Pengfei Wu, Jianxin Liang, Dongyan Zhao, Zilong Zheng.*<br>
[[paper]](https://arxiv.org/abs/2402.16050)
[[code]](https://github.com/bigai-nlco/LSTP-Chat)

**(*arXiv2024_ShareGPT4Video*) ShareGPT4Video: Improving Video Understanding and Generation with Better Captions.** <br>
*Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, Li Yuan, Yu Qiao, Dahua Lin, Feng Zhao, Jiaqi Wang.*<br>
[[paper]](https://arxiv.org/abs/2406.04325v1)
[[code]](https://sharegpt4video.github.io/)


### ``*Large Model Distillation*``

**(*EMNLP2016_Seq-KD*) Sequence-Level Knowledge Distillation.** <br>
*Yoon Kim, Alexander M. Rush.*<br>
[[paper]](https://arxiv.org/abs/1606.07947)
[[code]](https://github.com/harvardnlp/seq2seq-attn)

**(*arXiv2020_ImitKD*) Autoregressive Knowledge Distillation through Imitation Learning.** <br>
*Alexander Lin, Jeremy Wohlwend, Howard Chen, Tao Lei.*<br>
[[paper]](https://arxiv.org/abs/2009.07253)
[[code]](https://github.com/asappresearch/imitkd)

**(*ICLR2024_MINILLM*) MINILLM: Knowledge Distillation of Large Language Models.** <br>
*Yuxian Gu, Li Dong, Furu Wei, Minlie Huang.*<br>
[[paper]](https://arxiv.org/abs/2306.08543)
[[code]](https://aka.ms/MiniLLM)

**(*ICLR2024_GKD*) On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes.** <br>
*Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, Olivier Bachem.*<br>
[[paper]](https://arxiv.org/abs/2306.13649)
[[code]](https://github.com/microsoft/LMOps/tree/main/minillm)

**(*ACL2023_f-DISTILL*) f-Divergence Minimization for Sequence-Level Knowledge Distillation.** <br>
*Yuqiao Wen, Zichao Li, Wenyu Du, Lili Mou.*<br>
[[paper]](https://arxiv.org/abs/2307.15190)
[[code]](https://github.com/MANGA-UOFA/fdistill)

**(*arXiv2023_DistillSpec*) DistillSpec: Improving Speculative Decoding via Knowledge Distillation.** <br>
*Yongchao Zhou, Kaifeng Lyu, Ankit Singh Rawat, Aditya Krishna Menon, Afshin Rostamizadeh, Sanjiv Kumar, Jean-François Kagy, Rishabh Agarwal.*<br>
[[paper]](https://arxiv.org/abs/2310.08461)

**(*arXiv2023_MiniMA*) Towards the Law of Capacity Gap in Distilling Language Models.** <br>
*Chen Zhang, Dawei Song, Zheyu Ye, Yan Gao.*<br>
[[paper]](https://arxiv.org/abs/2311.07052)
[[code]](https://github.com/GeneZC/MiniMA)

**(*arXiv2024_Self-Rewarding*) Self-Rewarding Language Models.** <br>
*Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, Jason Weston.*<br>
[[paper]](https://arxiv.org/abs/2401.10020)


### ``*Related Survey*``

**(*arXiv2020_Survey*) Efficient Transformers: A Survey.** <br>
*Yi Tay, Mostafa Dehghani, Dara Bahri, Donald Metzler.*<br>
[[paper]](https://arxiv.org/abs/2009.06732)

**(*arXiv2023_Survey*) A Survey of Large Language Models.** <br>
*Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen.*<br>
[[paper]](https://arxiv.org/abs/2303.18223)

**(*arXiv2023_Survey*) Towards AGI in Computer Vision: Lessons Learned from GPT and Large Language Models.** <br>
*Lingxi Xie, Longhui Wei, Xiaopeng Zhang, Kaifeng Bi, Xiaotao Gu, Jianlong Chang, Qi Tian.*<br>
[[paper]](https://arxiv.org/abs/2306.08641)

**(*arXiv2023_Survey*) A Survey on Multimodal Large Language Models.** <br>
*Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, Enhong Chen.*<br>
[[paper]](https://arxiv.org/abs/2306.13549)
[[code]](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)

**(*arXiv2023_Survey*) Multimodal Foundation Models: From Specialists to General-Purpose Assistants.** <br>
*Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, Jianfeng Gao.*<br>
[[paper]](https://arxiv.org/abs/2309.10020)
[[code]](https://vlp-tutorial.github.io/2023/)

**(*arXiv2023_Survey*) A Survey of Chain of Thought Reasoning: Advances, Frontiers and Future.** <br>
*Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Tao He, Haotian Wang, Weihua Peng, Ming Liu, Bing Qin, Ting Liu.*<br>
[[paper]](https://arxiv.org/abs/2309.15402)
[[code]](https://github.com/zchuz/CoT-Reasoning-Survey)

**(*CVPR2023w_Survey*) Recent Advances in Vision Foundation Models.** <br>
*Linjie Li, Zhe Gan, Chunyuan Li, Jianwei Yang, Zhengyuan Yang, Jianfeng Gao, Lijuan Wang.*<br>
[[paper]](https://vlp-tutorial.github.io/2023/)

**(*arXiv2023_Survey*) A Challenger to GPT-4V? Early Explorations of Gemini in Visual Expertise.** <br>
*Chaoyou Fu, Renrui Zhang, Zihan Wang, Yubo Huang, Zhengye Zhang, Longtian Qiu, Gaoxiang Ye, Yunhang Shen, Mengdan Zhang, Peixian Chen, Sirui Zhao, Shaohui Lin, Deqiang Jiang, Di Yin, Peng Gao, Ke Li, Hongsheng Li, Xing Sun.*<br>
[[paper]](https://arxiv.org/abs/2312.12436)
[[code]](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)


### ``*Related Benchmark*``

**(*NeurIPS2023_LAMM*) LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark.** <br>
*Zhenfei Yin, Jiong Wang, Jianjian Cao, Zhelun Shi, Dingning Liu, Mukai Li, Lu Sheng, Lei Bai, Xiaoshui Huang, Zhiyong Wang, Jing Shao, Wanli Ouyang.*<br>
[[paper]](https://arxiv.org/abs/2306.06687)
[[code]](https://openlamm.github.io/)

**(*arXiv2023_MME*) MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models.** <br>
*Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, Rongrong Ji.*<br>
[[paper]](https://arxiv.org/abs/2306.13394)
[[code]](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)

**(*arXiv2023_MMBench*) MMBench: Is Your Multi-modal Model an All-around Player?.** <br>
*Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, Dahua Lin.*<br>
[[paper]](https://arxiv.org/abs/2307.06281)
[[code]](https://opencompass.org.cn/mmbench)

**(*arXiv2023_SEED-Bench*) SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension.** <br>
*Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, Ying Shan.*<br>
[[paper]](https://arxiv.org/abs/2307.16125)
[[code]](https://github.com/AILab-CVC/SEED-Bench)

**(*arXiv2023_MagnifierBench*) OtterHD: A High-Resolution Multi-modality Model.** <br>
*Bo Li, Peiyuan Zhang, Jingkang Yang, Yuanhan Zhang, Fanyi Pu, Ziwei Liu.*<br>
[[paper]](https://arxiv.org/abs/2311.04219)
[[code]](https://huggingface.co/datasets/Otter-AI/MagnifierBench)

**(*arXiv2023_Video-Bench*) Video-Bench: A Comprehensive Benchmark and Toolkit for Evaluating Video-based Large Language Models.** <br>
*Munan Ning, Bin Zhu, Yujia Xie, Bin Lin, Jiaxi Cui, Lu Yuan, Dongdong Chen, Li Yuan.*<br>
[[paper]](https://arxiv.org/abs/2311.16103)
[[code]](https://github.com/PKU-YuanGroup/Video-Bench)

**(*arXiv2023_MVBench*) MVBench: A Comprehensive Multi-modal Video Understanding Benchmark.** <br>
*Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, Yu Qiao.*<br>
[[paper]](https://arxiv.org/abs/2311.17005)
[[code]](https://github.com/OpenGVLab/Ask-Anything)

**(*arXiv2023_SEED-Bench-2*) SEED-Bench-2: Benchmarking Multimodal Large Language Models.** <br>
*Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, Ying Shan.*<br>
[[paper]](https://arxiv.org/abs/2311.17092)
[[code]](https://github.com/AILab-CVC/SEED-Bench)

**(*arXiv2023_VBench*) VBench: Comprehensive Benchmark Suite for Video Generative Models.** <br>
*Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, Ziwei Liu.*<br>
[[paper]](https://arxiv.org/abs/2311.17982)
[[code]](https://github.com/Vchitect/VBench)

**(*arXiv2024_VL-ICL*) VL-ICL Bench: The Devil in the Details of Benchmarking Multimodal In-Context Learning.** <br>
*Yongshuo Zong, Ondrej Bohdal, Timothy Hospedales.*<br>
[[paper]](https://arxiv.org/abs/2403.13164)
[[code]](https://github.com/ys-zong/VL-ICL)

**(*arXiv2024_SEED-Bench-2-Plus*) SEED-Bench-2-Plus: Benchmarking Multimodal Large Language Models with Text-Rich Visual Comprehension.** <br>
*Bohao Li, Yuying Ge, Yi Chen, Yixiao Ge, Ruimao Zhang, Ying Shan.*<br>
[[paper]](https://arxiv.org/abs/2404.16790)
[[code]](https://github.com/AILab-CVC/SEED-Bench)

**(*arXiv2024_Video-MME*) Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis.** <br>
*Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Rongrong Ji, Xing Sun.*<br>
[[paper]](https://arxiv.org/abs/2405.21075)
[[code]](https://video-mme.github.io/)