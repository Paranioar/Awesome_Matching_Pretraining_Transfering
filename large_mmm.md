Methods Summary of Large Multi-Modality Model
==============================

## ``Catalogue ``
* [Large Language Model](#large-language-model)
* [Large Vision Model](#large-vision-model)
* [Large MMM for Perception](#large-mmm-for-perception)
    * [Region Perception](#region-perception)
    * [Image Perception](#image-perception)
    * [Video Perception](#video-perception)
* [Large MMM for Generation](#large-mmm-for-generation)
    * [Class Generation](#class-generation)
    * [Image Generation](#image-generation)
    * [Video Generation](#video-generation)
* [Large MMM for Unification](#large-mmm-for-unification)
* [Large MMM for Manipulation](#large-mmm-for-manipulation)
* [Large Model Distillation](#large-model-distillation)
* [Related Survey](#related-survey)
* [Related Benchmark](#related-benchmark)


### ``*Large Language Model*``

**(*arXiv2017_PPO*) Proximal Policy Optimization Algorithms.** <br>
*John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov.*<br>
[[paper]](https://arxiv.org/abs/1707.06347)

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

**(*arXiv2021_RoPE*) RoFormer: Enhanced Transformer with Rotary Position Embedding.** <br>
*Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu.*<br>
[[paper]](https://arxiv.org/abs/2104.09864)
[[code]](https://huggingface.co/docs/transformers/model_doc/roformer)

**(*arXiv2022_PaLM*) PaLM: Scaling Language Modeling with Pathways.** <br>
*Google Research.*<br>
[[paper]](https://arxiv.org/abs/2204.02311)
[[code]](https://github.com/lucidrains/PaLM-pytorch)

**(*arXiv2022_ReAct*) ReAct: Synergizing Reasoning and Acting in Language Models.** <br>
*Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao.*<br>
[[paper]](https://arxiv.org/abs/2210.03629)
[[code]](https://react-lm.github.io/)

**(*arXiv2023_LLaMA*) LLaMA: Open and Efficient Foundation Language Models.** <br>
*LLaMA Team.*<br>
[[paper]](https://arxiv.org/abs/2302.13971)
[[code]](https://github.com/facebookresearch/llama)

**(*arXiv2023_Reflexion*) Reflexion: Language Agents with Verbal Reinforcement Learning.** <br>
*Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao.*<br>
[[paper]](https://arxiv.org/abs/2303.11366)
[[code]](https://github.com/noahshinn/reflexion)

**(*arXiv2023_ToT*) Tree of Thoughts: Deliberate Problem Solving with Large Language Models.** <br>
*Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan.*<br>
[[paper]](https://arxiv.org/abs/2305.10601)
[[code]](https://github.com/princeton-nlp/tree-of-thought-llm)

**(*arXiv2023_RWKV*) RWKV: Reinventing RNNs for the Transformer Era.** <br>
*RWKV Team.*<br>
[[paper]](https://arxiv.org/abs/2305.13048)
[[code]](https://github.com/BlinkDL/RWKV-LM)

**(*arXiv2023_DPO*) Direct Preference Optimization: Your Language Model is Secretly a Reward Model.** <br>
*Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn.*<br>
[[paper]](https://arxiv.org/abs/2305.18290)

**(*arXiv2023_LLM-Judge*) Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.** <br>
*Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica.*<br>
[[paper]](https://arxiv.org/abs/2306.05685)
[[code]](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)

**(*arXiv2023_RETNET*) Retentive Network: A Successor to Transformer for Large Language Models.** <br>
*Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, Furu Wei.*<br>
[[paper]](https://arxiv.org/abs/2307.08621)
[[code]](https://aka.ms/retnet)

**(*arXiv2023_Llama 2*) Llama 2: Open Foundation and Fine-Tuned Chat Models.** <br>
*LLaMA Team.*<br>
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

**(*arXiv2024_OLMo*) OLMo: Accelerating the Science of Language Models.** <br>
*Allen Institute.AI.*<br>
[[paper]](https://arxiv.org/abs/2402.00838)
[[code]](https://github.com/allenai/olmo)

**(*arXiv2024_Scaling*) Unraveling the Mystery of Scaling Laws: Part I.** <br>
*Hui Su, Zhi Tian, Xiaoyu Shen, Xunliang Cai.*<br>
[[paper]](https://arxiv.org/abs/2403.06563)

**(*arXiv2024_Phi-3*) Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone.** <br>
*Microsoft Team.*<br>
[[paper]](https://arxiv.org/abs/2404.14219)

**(*arXiv2024_Mambav2*) Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.** <br>
*Tri Dao, Albert Gu.*<br>
[[paper]](https://arxiv.org/abs/2405.21060)
[[code]](https://github.com/state-spaces/mamba)

**(*arXiv2024_TTT*) Learning to (Learn at Test Time): RNNs with Expressive Hidden States.** <br>
*Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin.*<br>
[[paper]](https://arxiv.org/abs/2407.04620)
[[code]](https://github.com/test-time-training/ttt-lm-pytorch)

**(*arXiv2024_Qwen2*) Qwen2 Technical Report.** <br>
*Qwen Team.*<br>
[[paper]](https://arxiv.org/abs/2407.10671)
[[code]](https://github.com/QwenLM/Qwen2)

**(*arXiv2024_Llama3*) The Llama 3 Herd of Models.** <br>
*Llama Team.*<br>
[[paper]](https://arxiv.org/abs/2407.21783)
[[model]](https://www.llama.com/)

**(*arXiv2024_Gemma2*) Gemma 2: Improving Open Language Models at a Practical Size.** <br>
*DeepMind Team.*<br>
[[paper]](https://arxiv.org/abs/2408.00118)
[[code]](https://blog.google/technology/developers/google-gemma-2/)

**(*arXiv2024_OLMoE*) OLMoE: Open Mixture-of-Experts Language Models.** <br>
*OLMoE Team.*<br>
[[paper]](https://arxiv.org/abs/2409.02060)
[[code]](https://github.com/allenai/OLMoE)

**(*arXiv2024_DIFF-Transformer*) Differential Transformer.** <br>
*Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang, Furu Wei.*<br>
[[paper]](https://arxiv.org/abs/2410.05258)

**(*arXiv2024_ScPO*) Self-Consistency Preference Optimization.** <br>
*Archiki Prasad, Weizhe Yuan, Richard Yuanzhe Pang, Jing Xu, Maryam Fazel-Zarandi, Mohit Bansal, Sainbayar Sukhbaatar, Jason Weston, Jane Yu.*<br>
[[paper]](https://arxiv.org/abs/2411.04109)

**(*arXiv2024_Coconut*) Training Large Language Models to Reason in a Continuous Latent Space.** <br>
*Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, Yuandong Tian.*<br>
[[paper]](https://arxiv.org/abs/2412.06769)

**(*arXiv2024_LCM*) Large Concept Models: Language Modeling in a Sentence Representation Space.** <br>
*LCM Team.*<br>
[[paper]](https://arxiv.org/abs/2412.08821)
[[code]](https://github.com/facebookresearch/large_concept_model)

**(*arXiv2024_Phi-4*) Phi-4 Technical Report.** <br>
*Microsoft Research.*<br>
[[paper]](https://arxiv.org/abs/2412.08905)

**(*arXiv2024_BLT*) Byte Latent Transformer: Patches Scale Better Than Tokens.** <br>
*Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Chunting Zhou, Lili Yu, Jason Weston, Luke Zettlemoyer, Gargi Ghosh, Mike Lewis, Ari Holtzman, Srinivasan Iyer.*<br>
[[paper]](https://arxiv.org/abs/2412.09871)
[[code]](https://github.com/facebookresearch/blt)

**(*arXiv2024_Qwen2.5*) Qwen2.5 Technical Report.** <br>
*Qwen Team.*<br>
[[paper]](https://arxiv.org/abs/2412.15115)

**(*arXiv2024_DeepSeek-V3*) DeepSeek-V3 Technical Report.** <br>
*DeepSeek-AI Team.*<br>
[[paper]](https://arxiv.org/abs/2412.19437)
[[code]](https://github.com/deepseek-ai/DeepSeek-V3)

**(*arXiv2025_DeepSeek-R1*) DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.** <br>
*DeepSeek-AI Team.*<br>
[[paper]](https://arxiv.org/abs/2501.12948)
[[code]](https://github.com/deepseek-ai/deepseek-r1)

**(*arXiv2025_LLaDA*) Large Language Diffusion Models.** <br>
*Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, Chongxuan Li.*<br>
[[paper]](https://arxiv.org/abs/2502.09992)
[[code]](https://github.com/ML-GSAI/LLaDA)

**(*arXiv2025_LayerNorm-Scaling*) The Curse of Depth in Large Language Models.** <br>
*Wenfang Sun, Xinyuan Song, Pengxiang Li, Lu Yin, Yefeng Zheng, Shiwei Liu.*<br>
[[paper]](https://arxiv.org/abs/2502.05795)
[[code]](https://github.com/lmsdss/LayerNorm-Scaling)

**(*ICLR2025_BD3-LMs*) Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models.** <br>
*Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, Volodymyr Kuleshov.*<br>
[[paper]](https://arxiv.org/abs/2503.09573)
[[code]](https://m-arriola.com/bd3lms/)

**(*arXiv2025_DAPO*) DAPO: An Open-Source LLM Reinforcement Learning System at Scale.** <br>
*ByteDance Seed.*<br>
[[paper]](https://arxiv.org/abs/2503.14476)
[[code]](https://github.com/BytedTsinghua-SIA/DAPO)

**(*arXiv2025_GPG*) GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning.** <br>
*Xiangxiang Chu, Hailang Huang, Xiao Zhang, Fei Wei, Yong Wang.*<br>
[[paper]](https://arxiv.org/abs/2504.02546)

**(*arXiv2025_EMPO*) Right Question is Already Half the Answer: Fully Unsupervised LLM Reasoning Incentivization.** <br>
*Qingyang Zhang, Haitao Wu, Changqing Zhang, Peilin Zhao, Yatao Bian.*<br>
[[paper]](https://arxiv.org/abs/2504.05812)
[[code]](https://github.com/QingyangZhang/EMPO)

**(*arXiv2025_Findings*) Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?.** <br>
*Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, Gao Huang.*<br>
[[paper]](https://arxiv.org/abs/2504.13837)

**(*arXiv2025_TTRL*) TTRL: Test-Time Reinforcement Learning.** <br>
*Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Yuchen Zhang, Xinwei Long, Ermo Hua, Biqing Qi, Youbang Sun, Zhiyuan Ma, Lifan Yuan, Ning Ding, Bowen Zhou.*<br>
[[paper]](https://arxiv.org/abs/2504.16084)
[[code]](https://github.com/PRIME-RL/TTRL)

**(*arXiv2025_One-Shot-RLVR*) Reinforcement Learning for Reasoning in Large Language Models with One Training Example.** <br>
*Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Liyuan Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, Yelong Shen.*<br>
[[paper]](https://arxiv.org/abs/2504.20571)
[[code]](https://github.com/ypwang61/One-Shot-RLVR)

**(*arXiv2025_Qwen3*) Qwen3 Technical Report.** <br>
*Qwen Team.*<br>
[[paper]](https://arxiv.org/pdf/2505.09388)
[[code]](https://github.com/QwenLM/Qwen3)

**(*arXiv2025_ParScale*) Parallel Scaling Law for Language Models.** <br>
*Mouxiang Chen, Binyuan Hui, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Jianling Sun, Junyang Lin, Zhongxin Liu.*<br>
[[paper]](https://arxiv.org/abs/2505.10475)
[[code]](https://github.com/qwenlm/parscale)

**(*Blog2025_Mercury*) Introducing Mercury, The First Commercial-scale Diffusion Large Language Model.** <br>
*Inception Team.*<br>
[[paper]](https://www.inceptionlabs.ai/news)

**(*Blog2025_Dream*) Introducing Dream 7B, Diffusion Reasoning Model.** <br>
*Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong.*<br>
[[paper]](https://hkunlp.github.io/blog/2025/dream/)

**(*Blog2025_Gemini-Diffusion*) Gemini Diffusion.** <br>
*Google DeepMind.*<br>
[[paper]](https://deepmind.google/models/gemini-diffusion/)

**(*arXiv2025_MiMo*) MiMo: Unlocking the Reasoning Potential of Language Model -- From Pretraining to Posttraining.** <br>
*Xiaomi LLM-Core Team.*<br>
[[paper]](https://arxiv.org/abs/2505.07608)
[[code]](https://github.com/xiaomimimo/MiMo)

**(*arXiv2025_SLOT*) SLOT: Sample-specific Language Model Optimization at Test-time.** <br>
*Yang Hu, Xingyu Zhang, Xueji Fang, Zhiyang Chen, Xiao Wang, Huatian Zhang, Guojun Qi.*<br>
[[paper]](https://arxiv.org/abs/2505.12392)
[[code]](https://github.com/maple-research-lab/SLOT)

**(*arXiv2025_EM*) The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning.** <br>
*Shivam Agarwal, Zimin Zhang, Lifan Yuan, Jiawei Han, Hao Peng.*<br>
[[paper]](https://arxiv.org/abs/2505.15134)

**(*arXiv2025_RRMs*) Reward Reasoning Model.** <br>
*Jiaxin Guo, Zewen Chi, Li Dong, Qingxiu Dong, Xun Wu, Shaohan Huang, Furu Wei.*<br>
[[paper]](https://arxiv.org/abs/2505.14674)
[[code]](https://huggingface.co/Reward-Reasoning)

**(*arXiv2025_LLaDA-1.5*) LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models.** <br>
*Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Hu, Jun Zhou, Jianfei Chen, Yankai Lin, Ji-Rong Wen, Chongxuan Li.*<br>
[[paper]](https://arxiv.org/abs/2505.19223)
[[code]](https://ml-gsai.github.io/LLaDA-1.5-Demo/)

**(*arXiv2025_One-Shot-EM*) One-shot Entropy Minimization.** <br>
*Zitian Gao, Lynx Chen, Joey Zhou, Bryan Dai.*<br>
[[paper]](https://arxiv.org/abs/2505.20282)
[[code]](https://github.com/zitian-gao/one-shot-em)

**(*arXiv2025_SRT*) Can Large Reasoning Models Self-Train?.** <br>
*Sheikh Shafayat, Fahim Tajwar, Ruslan Salakhutdinov, Jeff Schneider, Andrea Zanette.*<br>
[[paper]](https://arxiv.org/abs/2505.21444)
[[code]](https://self-rewarding-llm-training.github.io/)

**(*arXiv2025_Fast-dLLM*) Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding.** <br>
*Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, Enze Xie.*<br>
[[paper]](https://arxiv.org/abs/2505.22618)
[[code]](https://github.com/NVlabs/Fast-dLLM)

**(*arXiv2025_R3M*) Reflect, Retry, Reward: Self-Improving LLMs via Reinforcement Learning.** <br>
*Shelly Bensal, Umar Jamil, Christopher Bryant, Melisa Russak, Kiran Kamble, Dmytro Mozolevskyi, Muayad Ali, Waseem AlShikh.*<br>
[[paper]](https://arxiv.org/abs/2505.24726)

**(*arXiv2025_RPT*) Reinforcement Pre-Training.** <br>
*Qingxiu Dong, Li Dong, Yao Tang, Tianzhu Ye, Yutao Sun, Zhifang Sui, Furu Wei.*<br>
[[paper]](https://arxiv.org/abs/2506.08007)
[[code]](https://aka.ms/GeneralAI)

**(*arXiv2025_EMPO*) EMPO: A Clustering-Based On-Policy Algorithm for Offline Reinforcement Learing.** <br>
*Jongeui Park, Myungsik Cho, Youngchul Sung.*<br>
[[paper]](https://openreview.net/pdf?id=Vga4Kz3rEj)

**(*arXiv2025_Spurious-Rewards*) Spurious Rewards: Rethinking Training Signals in RLVR.** <br>
*Rulin Shao, Shuyue Stella Li, Rui Xin, Scott Geng, Yiping Wang, Sewoong Oh, Simon Shaolei Du, Nathan Lambert, Sewon Min, Ranjay Krishna, Yulia Tsvetkov, Hannaneh Hajishirzi, Pang Wei Koh, Luke Zettlemoyer.*<br>
[[paper]](https://arxiv.org/abs/2506.10947)
[[code]](https://github.com/ruixin31/Spurious_Rewards)

**(*arXiv2025_OctoThinker*) OctoThinker: Mid-training Incentivizes Reinforcement Learning Scaling.** <br>
*Zengzhi Wang, Fan Zhou, Xuefeng Li, Pengfei Liu.*<br>
[[paper]](https://arxiv.org/abs/2506.20512)

**(*arXiv2025_Step-3*) Step-3 is Large yet Affordable: Model-system Co-design for Cost-effective Decoding.** <br>
*StepFun Inc.*<br>
[[paper]](https://arxiv.org/abs/2507.19427)

**(*arXiv2025_Kimi-K2*) Kimi K2: Open Agentic Intelligence.** <br>
*MoonshotAI.*<br>
[[paper]]https://kimik2.com/)
[[code]](https://github.com/MoonshotAI/Kimi-K2)

**(*arXiv2025_GLM-4.5*) GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models.** <br>
*Zhipu AI & Tsinghua University.*<br>
[[paper]]https://www.arxiv.org/abs/2508.06471)
[[code]](https://github.com/zai-org/GLM-4.5)


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

**(*CVPR2022_RegionCLIP*) RegionCLIP: Region-based Language-Image Pretraining.** <br>
*Yiwu Zhong, Jianwei Yang, Pengchuan Zhang, Chunyuan Li, Noel Codella, Liunian Harold Li, Luowei Zhou, Xiyang Dai, Lu Yuan, Yin Li, Jianfeng Gao.*<br>
[[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhong_RegionCLIP_Region-Based_Language-Image_Pretraining_CVPR_2022_paper.pdf)
[[code]](https://github.com/microsoft/RegionCLIP)

**(*CVPR2022_Uni-Perceiver*) Uni-Perceiver: Pre-training Unified Architecture for Generic Perception for Zero-shot and Few-shot Tasks.** <br>
*Xizhou Zhu, Jinguo Zhu, Hao Li, Xiaoshi Wu, Xiaogang Wang, Hongsheng Li, Xiaohua Wang, Jifeng Dai.*<br>
[[paper]](https://arxiv.org/abs/2112.01522)
[[code]](https://github.com/fundamentalvision/Uni-Perceiver)

**(*ICLR2022_UniFormer*) UniFormer: Unified Transformer for Efficient Spatiotemporal Representation Learning.** <br>
*Kunchang Li, Yali Wang, Peng Gao, Guanglu Song, Yu Liu, Hongsheng Li, Yu Qiao.*<br>
[[paper]](https://arxiv.org/abs/2201.04676)
[[code]](https://github.com/Sense-X/UniFormer)

**(*ECCV2022_MVP*) MVP: Multimodality-guided Visual Pre-training.** <br>
*Longhui Wei, Lingxi Xie, Wengang Zhou, Houqiang Li, Qi Tian.*<br>
[[paper]](https://arxiv.org/abs/2203.05175)

**(*arXiv2022_Pix2Seq*) A Unified Sequence Interface for Vision Tasks.** <br>
*Ting Chen, Saurabh Saxena, Lala Li, Tsung-Yi Lin, David J. Fleet, Geoffrey Hinton.*<br>
[[paper]](https://arxiv.org/abs/2206.07669)
[[code]](https://github.com/google-research/pix2seq)

**(*arXiv2022_Unified-IO*) Unified-IO: A Unified Model for Vision, Language, and Multi-Modal Tasks.** <br>
*Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, Aniruddha Kembhavi.*<br>
[[paper]](https://arxiv.org/abs/2206.08916)
[[code]](https://unified-io.allenai.org/)

**(*arXiv2022_BEiTv2*) BEiTv2: Masked Image Modeling with Vector-Quantized Visual Tokenizers.** <br>
*Zhiliang Peng, Li Dong, Hangbo Bao, Qixiang Ye, Furu Wei.*<br>
[[paper]](https://arxiv.org/abs/2208.06366)
[[code]](https://github.com/microsoft/unilm/tree/master/beit2)

**(*arXiv2022_Visual-Prompting*) Visual Prompting via Image Inpainting.** <br>
*Amir Bar, Yossi Gandelsman, Trevor Darrell, Amir Globerson, Alexei A. Efros.*<br>
[[paper]](https://arxiv.org/abs/2209.00647)
[[code]](https://github.com/amirbar/visual_prompting)

**(*ICLR2023_CLIP-ViP*) CLIP-ViP: Adapting Pre-trained Image-Text Model to Video-Language Representation Alignment.** <br>
*Hongwei Xue, Yuchong Sun, Bei Liu, Jianlong Fu, Ruihua Song, Houqiang Li, Jiebo Luo.*<br>
[[paper]](https://arxiv.org/abs/2209.06430)
[[code]](https://github.com/microsoft/XPretrain/tree/main/CLIP-ViP)

**(*ICLR2023_PaLI*) PaLI: A Jointly-Scaled Multilingual Language-Image Model.** <br>
*Google Research.*<br>
[[paper]](https://arxiv.org/abs/2209.06794)
[[code]](https://github.com/google-research/big_vision)

**(*CVPR2023_OVSeg*) Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP.** <br>
*Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan Zhao, Hang Zhang, Peizhao Zhang, Peter Vajda, Diana Marculescu.*<br>
[[paper]](https://arxiv.org/abs/2210.04150)
[[code]](https://jeff-liangf.github.io/projects/ovseg)

**(*ICLR2023_ToME*) Token Merging: Your ViT But Faster.** <br>
*Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, Judy Hoffman.*<br>
[[paper]](https://arxiv.org/abs/2210.09461)
[[code]](https://github.com/facebookresearch/tome)

**(*CVPR2023_InternImage*) InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions.** <br>
*Wenhai Wang, Jifeng Dai, Zhe Chen, Zhenhang Huang, Zhiqi Li, Xizhou Zhu, Xiaowei Hu, Tong Lu, Lewei Lu, Hongsheng Li, Xiaogang Wang, Yu Qiao.*<br>
[[paper]](https://arxiv.org/abs/2211.05778)
[[code]](https://github.com/OpenGVLab/InternImage)

**(*CVPR2023_EVA*) EVA: Exploring the Limits of Masked Visual Representation Learning at Scale.** <br>
*Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, Yue Cao.*<br>
[[paper]](https://arxiv.org/abs/2211.07636)
[[code]](https://github.com/baaivision/EVA)

**(*CVPR2023_MAGE*) MAGE: MAsked Generative Encoder to Unify Representation Learning and Image Synthesis.** <br>
*Tianhong Li, Huiwen Chang, Shlok Kumar Mishra, Han Zhang, Dina Katabi, Dilip Krishnan.*<br>
[[paper]](https://arxiv.org/abs/2211.09117)
[[code]](https://github.com/LTH14/mage)

**(*arXiv2023_UniFormerV2*) UniFormerV2: Spatiotemporal Learning by Arming Image ViTs with Video UniFormer.** <br>
*Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Limin Wang, Yu Qiao.*<br>
[[paper]](https://arxiv.org/abs/2211.09552)
[[code]](https://github.com/OpenGVLab/UniFormerV2)

**(*CVPR2023_M3I*) Towards All-in-one Pre-training via Maximizing Multi-modal Mutual Information.** <br>
*Weijie Su, Xizhou Zhu, Chenxin Tao, Lewei Lu, Bin Li, Gao Huang, Yu Qiao, Xiaogang Wang, Jie Zhou, Jifeng Dai.*<br>
[[paper]](https://arxiv.org/abs/2211.09807)
[[code]](https://github.com/OpenGVLab/M3I-Pretraining)

**(*arXiv2022_Uni-Perceiverv2*) Uni-Perceiver v2: A Generalist Model for Large-Scale Vision and Vision-Language Tasks.** <br>
*Hao Li, Jinguo Zhu, Xiaohu Jiang, Xizhou Zhu, Hongsheng Li, Chun Yuan, Xiaohua Wang, Yu Qiao, Xiaogang Wang, Wenhai Wang, Jifeng Dai.*<br>
[[paper]](https://arxiv.org/abs/2211.09808)
[[code]](https://github.com/fundamentalvision/Uni-Perceiver)

**(*CVPR2023_FLIP*) Scaling Language-Image Pre-training via Masking.** <br>
*Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feichtenhofer, Kaiming He.*<br>
[[paper]](https://arxiv.org/abs/2212.00794)
[[code]](https://github.com/facebookresearch/flip)

**(*CVPR2023_Painter*) Images Speak in Images: A Generalist Painter for In-Context Visual Learning.** <br>
*Xinlong Wang, Wen Wang, Yue Cao, Chunhua Shen, Tiejun Huang.*<br>
[[paper]](https://arxiv.org/abs/2212.02499)
[[code]](https://github.com/baaivision/Painter)

**(*CVPR2023_MAGVIT*) MAGVIT: Masked Generative Video Transformer.** <br>
*Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G. Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, Lu Jiang.*<br>
[[paper]](https://arxiv.org/abs/2212.05199)
[[code]](https://magvit.cs.cmu.edu/)

**(*CVPR2023_FlexiViT*) FlexiViT: One Model for All Patch Sizes.** <br>
*Lucas Beyer, Pavel Izmailov, Alexander Kolesnikov, Mathilde Caron, Simon Kornblith, Xiaohua Zhai, Matthias Minderer, Michael Tschannen, Ibrahim Alabdulmohsin, Filip Pavetic.*<br>
[[paper]](https://arxiv.org/abs/2212.08013)
[[code]](https://github.com/google-research/big_vision)

**(*CVPR2023_X-Decoder*) Generalized Decoding for Pixel, Image, and Language.** <br>
*Xueyan Zou, Zi-Yi Dou, Jianwei Yang, Zhe Gan, Linjie Li, Chunyuan Li, Xiyang Dai, Harkirat Behl, Jianfeng Wang, Lu Yuan, Nanyun Peng, Lijuan Wang, Yong Jae Lee, Jianfeng Gao.*<br>
[[paper]](https://arxiv.org/abs/2212.11270)
[[code]](https://x-decoder-vl.github.io/)

**(*ECCV2024_GroundingDINO*) Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection.** <br>
*Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, Lei Zhang.*<br>
[[paper]](https://arxiv.org/abs/2303.05499)
[[code]](https://github.com/IDEA-Research/GroundingDINO)

**(*ICML2023_ViT-22B*) Scaling Vision Transformers to 22 Billion Parameters.** <br>
*Google Research.*<br>
[[paper]](https://arxiv.org/abs/2302.05442)

**(*arXiv2023_EVA-02*) EVA-02: A Visual Representation for Neon Genesis.** <br>
*Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, Yue Cao.*<br>
[[paper]](https://arxiv.org/abs/2303.11331)
[[code]](https://github.com/baaivision/EVA/tree/master/EVA-02)

**(*ICCV2023_SigLIP*) Sigmoid Loss for Language Image Pre-Training.** <br>
*Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer.*<br>
[[paper]](https://arxiv.org/abs/2303.15343)
[[code]](https://github.com/google-research/big_vision)

**(*arXiv2023_EVA-CLIP*) EVA-CLIP: Improved Training Techniques for CLIP at Scale.** <br>
*Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, Yue Cao.*<br>
[[paper]](https://arxiv.org/abs/2303.15389)
[[code]](https://github.com/baaivision/EVA/tree/master/EVA-CLIP)

**(*ICCV2023_UMT*) Unmasked Teacher: Towards Training-Efficient Video Foundation Models.** <br>
*Kunchang Li, Yali Wang, Yizhuo Li, Yi Wang, Yinan He, Limin Wang, Yu Qiao.*<br>
[[paper]](https://arxiv.org/abs/2303.16058)
[[code]](https://github.com/OpenGVLab/unmasked_teacher)

**(*CVPR2023_VideoMAEv2*) VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking.** <br>
*Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, Yu Qiao.*<br>
[[paper]](https://arxiv.org/abs/2303.16727)
[[code]](https://github.com/OpenGVLab/VideoMAEv2)

**(*arXiv2023_SAM*) Segment Anything.** <br>
*Meta FAIR.*<br>
[[paper]](https://arxiv.org/abs/2304.02643)
[[code]](https://github.com/facebookresearch/segment-anything)

**(*ICCV2023_SegGPT*) SegGPT: Segmenting Everything In Context.** <br>
*Xinlong Wang, Xiaosong Zhang, Yue Cao, Wen Wang, Chunhua Shen, Tiejun Huang.*<br>
[[paper]](https://arxiv.org/abs/2304.03284)
[[code]](https://github.com/baaivision/Painter)

**(*arXiv2023_CLIP_Surgery*) A Closer Look at the Explainability of Contrastive Language-Image Pre-training.** <br>
*Yi Li, Hualiang Wang, Yiqun Duan, Jiheng Zhang, Xiaomeng Li.*<br>
[[paper]](https://arxiv.org/abs/2304.05653)
[[code]](https://github.com/xmed-lab/CLIP_Surgery)

**(*CVPRW2023_SAM-not-perfect*) Segment Anything Is Not Always Perfect: An Investigation of SAM on Different Real-world Applications.** <br>
*Wei Ji, Jingjing Li, Qi Bi, Tingwei Liu, Wenbo Li, Li Cheng.*<br>
[[paper]](https://arxiv.org/abs/2304.05750)
[[code]](https://github.com/liutingwed/sam-not-perfect)

**(*NeurIPS2023_SEEM*) Segment Everything Everywhere All at Once.** <br>
*Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li, Jianfeng Wang, Lijuan Wang, Jianfeng Gao, Yong Jae Lee.*<br>
[[paper]](https://arxiv.org/abs/2304.06718)
[[code]](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)

**(*arXiv2023_FastSAM*) Fast Segment Anything.** <br>
*Xu Zhao, Wenchao Ding, Yongqi An, Yinglong Du, Tao Yu, Min Li, Ming Tang, Jinqiao Wang.*<br>
[[paper]](https://arxiv.org/abs/2306.12156)
[[code]](https://github.com/CASIA-IVA-Lab/FastSAM)

**(*arXiv2023_MetaCLIP*) Demystifying CLIP Data.** <br>
*Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, Christoph Feichtenhofer.*<br>
[[paper]](https://arxiv.org/abs/2309.16671)
[[code]](https://github.com/facebookresearch/MetaCLIP)

**(*ICLR2024_MAGVITv2*) Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation.** <br>
*Lijun Yu, José Lezama, Nitesh B. Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G. Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A. Ross, Lu Jiang.*<br>
[[paper]](https://arxiv.org/abs/2310.05737)

**(*CVPR2024_MobileCLIP*) MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training.** <br>
*Pavan Kumar Anasosalu Vasu, Hadi Pouransari, Fartash Faghri, Raviteja Vemulapalli, Oncel Tuzel.*<br>
[[paper]](https://arxiv.org/abs/2311.17049)
[[code]](https://github.com/apple/ml-mobileclip)

**(*CVPR2024_LVM*) Sequential Modeling Enables Scalable Learning for Large Vision Models.** <br>
*Yutong Bai, Xinyang Geng, Karttikeya Mangalam, Amir Bar, Alan Yuille, Trevor Darrell, Jitendra Malik, Alexei A Efros.*<br>
[[paper]](https://arxiv.org/abs/2312.00785)
[[code]](https://github.com/ytongbai/LVM)

**(*NeurIPS2024_FIND*) Interfacing Foundation Models' Embeddings.** <br>
*Xueyan Zou, Linjie Li, Jianfeng Wang, Jianwei Yang, Mingyu Ding, Junyi Wei, Zhengyuan Yang, Feng Li, Hao Zhang, Shilong Liu, Arul Aravinthan, Yong Jae Lee, Lijuan Wang.*<br>
[[paper]](https://arxiv.org/abs/2312.07532)
[[code]](https://github.com/ux-decoder/find)

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

**(*arXiv2024_MM-GEM*) Multi-Modal Generative Embedding Model.** <br>
*Feipeng Ma, Hongwei Xue, Guangting Wang, Yizhou Zhou, Fengyun Rao, Shilin Yan, Yueyi Zhang, Siying Wu, Mike Zheng Shou, Xiaoyan Sun.*<br>
[[paper]](https://arxiv.org/abs/2405.19333)

**(*CVPR2024w_EgoVideo*) EgoVideo: Exploring Egocentric Foundation Model and Downstream Adaptation.** <br>
*Baoqi Pei, Guo Chen, Jilan Xu, Yuping He, Yicheng Liu, Kanghua Pan, Yifei Huang, Yali Wang, Tong Lu, Limin Wang, Yu Qiao.*<br>
[[paper]](https://arxiv.org/abs/2406.18070)
[[code]](https://github.com/OpenGVLab/EgoVideo)

**(*arXiv2024_MambaVision*) MambaVision: A Hybrid Mamba-Transformer Vision Backbone.** <br>
*Ali Hatamizadeh, Jan Kautz.*<br>
[[paper]](https://arxiv.org/abs/2407.08083)
[[code]](https://github.com/NVlabs/MambaVision)

**(*arXiv2024_SAM2*) SAM 2: Segment Anything in Images and Videos.** <br>
*Meta FAIR.*<br>
[[paper]](https://arxiv.org/abs/2408.00714)
[[code]](https://github.com/facebookresearch/sam2)

**(*arXiv2024_AIMv2*) Multimodal Autoregressive Pre-training of Large Vision Encoders.** <br>
*Apple Team.*<br>
[[paper]](https://arxiv.org/abs/2411.14402)
[[code]](https://github.com/apple/ml-aim)

**(*arXiv2024_4DS*) Scaling 4D Representations.** <br>
*Google DeepMind.*<br>
[[paper]](https://arxiv.org/abs/2412.15212)

**(*arXiv2025_SigLIP2*) SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features.** <br>
*Google DeepMind.*<br>
[[paper]](https://arxiv.org/abs/2502.14786)
[[code]](https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/image_text/README_siglip2.md)

**(*arXiv2025_Web-DINO*) Scaling Language-Free Visual Representation Learning.** <br>
*Meta FAIR.*<br>
[[paper]](https://arxiv.org/abs/2504.01017)
[[code]](https://davidfan.io/webssl/)

**(*arXiv2025_PE*) Perception Encoder: The best visual embeddings are not at the output of the network.** <br>
*Meta FAIR.*<br>
[[paper]](https://arxiv.org/abs/2504.13181)
[[code]](https://github.com/facebookresearch/perception_models)

**(*arXiv2025_VPRL*) Visual Planning: Let's Think Only with Images.** <br>
*Yi Xu, Chengzu Li, Han Zhou, Xingchen Wan, Caiqi Zhang, Anna Korhonen, Ivan Vulić.*<br>
[[paper]](https://arxiv.org/abs/2505.11409)
[[code]](https://github.com/yix8/visualplanning)

**(*arXiv2025_HIPS*) Hidden in plain sight: VLMs overlook their visual representations.** <br>
*Stephanie Fu, Tyler Bonnen, Devin Guillory, Trevor Darrell.*<br>
[[paper]](https://arxiv.org/abs/2506.08008)
[[code]](https://hidden-plain-sight.github.io/)

**(*arXiv2025_MetaCLIP-2*) MetaCLIP 2: A Worldwide Scaling Recipe.** <br>
*Yung-Sung Chuang, Yang Li, Dong Wang, Ching-Feng Yeh, Kehan Lyu, Ramya Raghavendra, James Glass, Lifei Huang, Jason Weston, Luke Zettlemoyer, Xinlei Chen, Zhuang Liu, Saining Xie, Wen-tau Yih, Shang-Wen Li, Hu Xu.*<br>
[[paper]](https://arxiv.org/abs/2507.22062)
[[code]](https://github.com/facebookresearch/MetaCLIP)

**(*arXiv2025_DINOv3*) DINOv3.** <br>
*Meta AI Research.*<br>
[[paper]](https://arxiv.org/abs/2508.10104)
[[code]](https://github.com/facebookresearch/dinov3)

**(*arXiv2025_MobileCLIP2*) MobileCLIP2: Improving Multi-Modal Reinforced Training.** <br>
*Fartash Faghri, Pavan Kumar Anasosalu Vasu, Cem Koc, Vaishaal Shankar, Alexander Toshev, Oncel Tuzel, Hadi Pouransari.*<br>
[[paper]](https://arxiv.org/abs/2508.20691)
[[code]](https://github.com/apple/ml-mobileclip-dr)


### ``*Large MMM for Perception*``

#### ``*Region Perception*``

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

**(*arXiv2023_GRILL*) GRILL: Grounded Vision-language Pre-training via Aligning Text and Image Regions.** <br>
*Woojeong Jin, Subhabrata Mukherjee, Yu Cheng, Yelong Shen, Weizhu Chen, Ahmed Hassan Awadallah, Damien Jose, Xiang Ren.*<br>
[[paper]](https://arxiv.org/abs/2305.14676)

**(*arXiv2023_DAC*) Dense and Aligned Captions (DAC) Promote Compositional Reasoning in VL Models.** <br>
*Sivan Doveh, Assaf Arbelle, Sivan Harary, Roei Herzig, Donghyun Kim, Paola Cascante-bonilla, Amit Alfassy, Rameswar Panda, Raja Giryes, Rogerio Feris, Shimon Ullman, Leonid Karlinsky.*<br>
[[paper]](https://arxiv.org/abs/2305.19595)

**(*arXiv2023_Kosmos-2*) Kosmos-2: Grounding Multimodal Large Language Models to the World.** <br>
*Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, Furu Wei.*<br>
[[paper]](https://arxiv.org/abs/2306.14824)
[[code]](https://aka.ms/kosmos-2)

**(*arXiv2023_Shikra*) Shikra: Unleashing Multimodal LLM's Referential Dialogue Magic.** <br>
*Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, Rui Zhao.*<br>
[[paper]](https://arxiv.org/abs/2306.15195)
[[code]](https://github.com/shikras/shikra)

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

**(*ICLR2024_All-Seeing*) The All-Seeing Project: Towards Panoptic Visual Recognition and Understanding of the Open World.** <br>
*Weiyun Wang, Min Shi, Qingyun Li, Wenhai Wang, Zhenhang Huang, Linjie Xing, Zhe Chen, Hao Li, Xizhou Zhu, Zhiguo Cao, Yushi Chen, Tong Lu, Jifeng Dai, Yu Qiao.*<br>
[[paper]](https://arxiv.org/abs/2308.01907)
[[code]](https://github.com/OpenGVLab/All-Seeing)

**(*ICLR2024_Ferret*) Ferret: Refer and Ground Anything Anywhere at Any Granularity.** <br>
*Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, Yinfei Yang.*<br>
[[paper]](https://arxiv.org/abs/2310.07704)
[[code]](https://github.com/apple/ml-ferret)

**(*arXiv2023_MiniGPT-v2*) MiniGPT-v2: Large Language Model as a Unified Interface for Vision-Language Multi-Task Learning.** <br>
*Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, Mohamed Elhoseiny.*<br>
[[paper]](https://arxiv.org/abs/2310.09478)
[[code]](https://minigpt-v2.github.io/)

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

**(*arXiv2023_DINOv*) Visual In-Context Prompting.** <br>
*Feng Li, Qing Jiang, Hao Zhang, Tianhe Ren, Shilong Liu, Xueyan Zou, Huaizhe Xu, Hongyang Li, Chunyuan Li, Jianwei Yang, Lei Zhang, Jianfeng Gao.*<br>
[[paper]](https://arxiv.org/abs/2311.13601)
[[code]](https://github.com/UX-Decoder/DINOv)

**(*arXiv2023_TAP*) Tokenize Anything via Prompting.** <br>
*Ting Pan, Lulu Tang, Xinlong Wang, Shiguang Shan.*<br>
[[paper]](https://arxiv.org/abs/2312.09128)
[[code]](https://github.com/baaivision/tokenize-anything)

**(*arXiv2023_Osprey*) Osprey: Pixel Understanding with Visual Instruction Tuning.** <br>
*Yuqian Yuan, Wentong Li, Jian Liu, Dongqi Tang, Xinjie Luo, Chi Qin, Lei Zhang, Jianke Zhu.*<br>
[[paper]](https://arxiv.org/abs/2312.10032)
[[code]](https://github.com/CircleRadon/Osprey)

**(*arXiv2023_V*\*) V\*: Guided Visual Search as a Core Mechanism in Multimodal LLMs.** <br>
*Penghao Wu, Saining Xie.*<br>
[[paper]](https://arxiv.org/abs/2312.14135)
[[code]](https://github.com/penghao-wu/vstar)

**(*ECCV2024_All-Seeingv2*) The All-Seeing Project V2: Towards General Relation Comprehension of the Open World.** <br>
*Weiyun Wang, Yiming Ren, Haowen Luo, Tiantong Li, Chenxiang Yan, Zhe Chen, Wenhai Wang, Qingyun Li, Lewei Lu, Xizhou Zhu, Yu Qiao, Jifeng Dai.*<br>
[[paper]](https://arxiv.org/abs/2402.19474)
[[code]](https://github.com/OpenGVLab/All-Seeing)

**(*arXiv2023_AnyRef*) Multi-modal Instruction Tuned LLMs with Fine-grained Visual Perception.** <br>
*Junwen He, Yifan Wang, Lijun Wang, Huchuan Lu, Jun-Yan He, Jin-Peng Lan, Bin Luo, Xuansong Xie.*<br>
[[paper]](https://arxiv.org/abs/2403.02969)
[[code]](https://github.com/jwh97nn/AnyRef)

**(*arXiv2023_GiT*) GiT: Towards Generalist Vision Transformer through Universal Language Interface.** <br>
*Haiyang Wang, Hao Tang, Li Jiang, Shaoshuai Shi, Muhammad Ferjad Naeem, Hongsheng Li, Bernt Schiele, Liwei Wang.*<br>
[[paper]](https://arxiv.org/abs/2403.09394)
[[code]](https://github.com/Haiyang-W/GiT)

**(*COLM2024_Ferretv2*) Ferret-v2: An Improved Baseline for Referring and Grounding with Large Language Models.** <br>
*Haotian Zhang, Haoxuan You, Philipp Dufter, Bowen Zhang, Chen Chen, Hong-You Chen, Tsu-Jui Fu, William Yang Wang, Shih-Fu Chang, Zhe Gan, Yinfei Yang.*<br>
[[paper]](https://arxiv.org/abs/2404.07973)
[[code]](https://github.com/apple/ml-ferret)

**(*arXiv2024_VisionLLMv2*) VisionLLM v2: An End-to-End Generalist Multimodal Large Language Model for Hundreds of Vision-Language Tasks.** <br>
*Jiannan Wu, Muyan Zhong, Sen Xing, Zeqiang Lai, Zhaoyang Liu, Wenhai Wang, Zhe Chen, Xizhou Zhu, Lewei Lu, Tong Lu, Ping Luo, Yu Qiao, Jifeng Dai.*<br>
[[paper]](https://arxiv.org/abs/2406.08394)
[[code]](https://github.com/opengvlab/visionllm)

**(*Blog2025_o3/o4-mini*) Thinking with images.** <br>
*OpenAI Team.*<br>
[[paper]](https://openai.com/index/thinking-with-images/)

**(*arXiv2025_DyFo*) DyFo: A Training-Free Dynamic Focus Visual Search for Enhancing LMMs in Fine-Grained Visual Understanding.** <br>
*Geng Li, Jinglin Xu, Yunzhen Zhao, Yuxin Peng.*<br>
[[paper]](https://arxiv.org/abs/2504.14920)
[[code]](https://github.com/PKU-ICST-MIPL/DyFo_CVPR2025)

**(*arXiv2025_UniVG-R1*) UniVG-R1: Reasoning Guided Universal Visual Grounding with Reinforcement Learning.** <br>
*Sule Bai, Mingxing Li, Yong Liu, Jing Tang, Haoji Zhang, Lei Sun, Xiangxiang Chu, Yansong Tang.*<br>
[[paper]](https://arxiv.org/abs/2505.14231)
[[code]](https://amap-ml.github.io/UniVG-R1-page/)

**(*arXiv2025_DeepEyes*) DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning.** <br>
*Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, Xing Yu.*<br>
[[paper]](https://arxiv.org/abs/2505.14362)
[[code]](https://github.com/Visual-Agent/DeepEyes)

**(*arXiv2025_Chain-of-Focus*) Chain-of-Focus: Adaptive Visual Search and Zooming for Multimodal Reasoning via RL.** <br>
*Xintong Zhang, Zhi Gao, Bofei Zhang, Pengxiang Li, Xiaowen Zhang, Yang Liu, Tao Yuan, Yuwei Wu, Yunde Jia, Song-Chun Zhu, Qing Li.*<br>
[[paper]](https://arxiv.org/abs/2505.15436)
[[code]](https://cof-reasoning.github.io/)

**(*arXiv2025_GRIT*) GRIT: Teaching MLLMs to Think with Images.** <br>
*Yue Fan, Xuehai He, Diji Yang, Kaizhi Zheng, Ching-Chen Kuo, Yuting Zheng, Sravana Jyothi Narayanaraju, Xinze Guan, Xin Eric Wang.*<br>
[[paper]](https://arxiv.org/abs/2505.15879)
[[code]](https://grounded-reasoning.github.io/)

**(*arXiv2025_VLM-R3*) VLM-R3: Region Recognition, Reasoning, and Refinement for Enhanced Multimodal Chain-of-Thought.** <br>
*Chaoya Jiang, Yongrui Heng, Wei Ye, Han Yang, Haiyang Xu, Ming Yan, Ji Zhang, Fei Huang, Shikun Zhang.*<br>
[[paper]](https://arxiv.org/abs/2505.16192v1)

**(*arXiv2025_Ground-R1*) Ground-R1: Incentivizing Grounded Visual Reasoning via Reinforcement Learning.** <br>
*Meng Cao, Haoze Zhao, Can Zhang, Xiaojun Chang, Ian Reid, Xiaodan Liang.*<br>
[[paper]](https://arxiv.org/abs/2505.20272)

**(*arXiv2025_VRAG-RL*) VRAG-RL: Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning with Reinforcement Learning.** <br>
*Qiuchen Wang, Ruixue Ding, Yu Zeng, Zehui Chen, Lin Chen, Shihang Wang, Pengjun Xie, Fei Huang, Feng Zhao.*<br>
[[paper]](https://arxiv.org/abs/2505.22019)
[[code]](https://github.com/Alibaba-NLP/VRAG)

**(*arXiv2025_SAM-R1*) SAM-R1: Leveraging SAM for Reward Feedback in Multimodal Segmentation via Reinforcement Learning.** <br>
*Jiaqi Huang, Zunnan Xu, Jun Zhou, Ting Liu, Yicheng Xiao, Mingwen Ou, Bowen Ji, Xiu Li, Kehong Yuan.*<br>
[[paper]](https://arxiv.org/abs/2505.22596)

**(*arXiv2025_ViGoRL*) Grounded Reinforcement Learning for Visual Reasoning.** <br>
*Gabriel Sarch, Snigdha Saha, Naitik Khandelwal, Ayush Jain, Michael J. Tarr, Aviral Kumar, Katerina Fragkiadaki.*<br>
[[paper]](https://arxiv.org/abs/2505.23678v1)
[[code]](https://visually-grounded-rl.github.io/)

**(*arXiv2025_PixelThink*) PixelThink: Towards Efficient Chain-of-Pixel Reasoning.** <br>
*Song Wang, Gongfan Fang, Lingdong Kong, Xiangtai Li, Jianyun Xu, Sheng Yang, Qiang Li, Jianke Zhu, Xinchao Wang.*<br>
[[paper]](https://arxiv.org/abs/2505.23727)
[[code]](https://pixelthink.github.io/)

**(*arXiv2025_DINO-R1*) DINO-R1: Incentivizing Reasoning Capability in Vision Foundation Models.** <br>
*Chenbin Pan, Wenbin He, Zhengzhong Tu, Liu Ren.*<br>
[[paper]](https://arxiv.org/abs/2505.24025)
[[code]](https://christinepan881.github.io/DINO-R1/)

**(*arXiv2025_SpatialLM*) SpatialLM: Training Large Language Models for Structured Indoor Modeling.** <br>
*Yongsen Mao, Junhao Zhong, Chuan Fang, Jia Zheng, Rui Tang, Hao Zhu, Ping Tan, Zihan Zhou.*<br>
[[paper]](https://arxiv.org/abs/2506.07491)
[[code]](https://manycore-research.github.io/SpatialLM/)


#### ``*Image Perception*``

**(*ICML2022_BLIP*) BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation.** <br>
*Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi.*<br>
[[paper]](https://arxiv.org/abs/2201.12086)
[[code]](https://github.com/salesforce/BLIP)

**(*ICML2022_OFA*) OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework.** <br>
*Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, Hongxia Yang.*<br>
[[paper]](https://arxiv.org/abs/2202.03052)
[[code]](https://github.com/OFA-Sys/OFA)

**(*NeurIPS2022_Flamingo*) Flamingo: a Visual Language Model for Few-Shot Learning.** <br>
*DeepMind Team.*<br>
[[paper]](https://arxiv.org/abs/2204.14198)
[[code]](https://github.com/mlfoundations/open_flamingo)

**(*arXiv2022_CoCa*) CoCa: Contrastive Captioners are Image-Text Foundation Models.** <br>
*Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, Yonghui Wu.*<br>
[[paper]](https://arxiv.org/abs/2205.01917)
[[code]](https://github.com/lucidrains/CoCa-pytorch)

**(*CVPR2023_BEiTv3*) Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks.** <br>
*Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, Furu Wei.*<br>
[[paper]](https://arxiv.org/abs/2208.10442)
[[code]](https://github.com/microsoft/unilm/tree/master/beit3)

**(*arXiv2023_BLIP-2*) BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.** <br>
*Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi.*<br>
[[paper]](https://arxiv.org/abs/2301.12597)
[[code]](https://github.com/salesforce/lavis)

**(*ICML2023_mPLUG-2*) mPLUG-2: A Modularized Multi-modal Foundation Model Across Text, Image and Video.** <br>
*Haiyang Xu, Qinghao Ye, Ming Yan, Yaya Shi, Jiabo Ye, Yuanhong Xu, Chenliang Li, Bin Bi, Qi Qian, Wei Wang, Guohai Xu, Ji Zhang, Songfang Huang, Fei Huang, Jingren Zhou.*<br>
[[paper]](https://arxiv.org/abs/2302.00402v1)
[[code]](https://github.com/alibaba/AliceMind)

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

**(*CVPR2023_ImageBind*) ImageBind: One Embedding Space To Bind Them All.** <br>
*Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, Ishan Misra.*<br>
[[paper]](https://arxiv.org/abs/2305.05665)
[[code]](https://github.com/facebookresearch/imagebind)

**(*NeurIPS2023_InstructBLIP*) InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning.** <br>
*Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, Steven Hoi.*<br>
[[paper]](https://arxiv.org/abs/2305.06500)
[[code]](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)

**(*arXiv2023_ONE-PEACE*) ONE-PEACE: Exploring One General Representation Model Toward Unlimited Modalities.** <br>
*Peng Wang, Shijie Wang, Junyang Lin, Shuai Bai, Xiaohuan Zhou, Jingren Zhou, Xinggang Wang, Chang Zhou.*<br>
[[paper]](https://arxiv.org/abs/2305.11172)
[[code]](https://github.com/OFA-Sys/ONE-PEACE)

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

**(*arXiv2023_Meta-Transformer*) Meta-Transformer: A Unified Framework for Multimodal Learning.** <br>
*Yiyuan Zhang, Kaixiong Gong, Kaipeng Zhang, Hongsheng Li, Yu Qiao, Wanli Ouyang, Xiangyu Yue.*<br>
[[paper]](https://arxiv.org/abs/2307.10802)
[[code]](https://github.com/invictus717/MetaTransformer)

**(*Blog2023_IDEFICS*) Introducing IDEFICS: An Open Reproduction of State-of-the-Art Visual Language Model.** <br>
*Hugo Laurençon, Daniel van Strien, Stas Bekman, Leo Tronchon, Lucile Saulnier, Thomas Wang, Siddharth Karamcheti, Amanpreet Singh, Giada Pistilli, Yacine Jernite, Victor Sanh.*<br>
[[blog]](https://huggingface.co/blog/idefics)

**(*AAAI2024_BLIVA*) BLIVA: A Simple Multimodal LLM for Better Handling of Text-Rich Visual Questions.** <br>
*Wenbo Hu, Yifan Xu, Yi Li, Weiyue Li, Zeyuan Chen, Zhuowen Tu.*<br>
[[paper]](https://arxiv.org/abs/2308.09936)
[[code]](https://github.com/mlpc-ucsd/BLIVA)

**(*arXiv2023_Qwen-VL*) Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond.** <br>
*Qwen Team.*<br>
[[paper]](https://arxiv.org/abs/2308.12966)
[[code]](https://github.com/QwenLM/Qwen-VL)

**(*ACL2024_TextBind*) TextBind: Multi-turn Interleaved Multimodal Instruction-following in the Wild.** <br>
*Huayang Li, Siheng Li, Deng Cai, Longyue Wang, Lemao Liu, Taro Watanabe, Yujiu Yang, Shuming Shi.*<br>
[[paper]](https://arxiv.org/abs/2309.08637)
[[code]](https://github.com/sihengli99/textbind)

**(*arXiv2023_Kosmos-2.5*) Kosmos-2.5: A Multimodal Literate Model.** <br>
*Tengchao Lv, Yupan Huang, Jingye Chen, Lei Cui, Shuming Ma, Yaoyao Chang, Shaohan Huang, Wenhui Wang, Li Dong, Weiyao Luo, Shaoxiang Wu, Guoxin Wang, Cha Zhang, Furu Wei.*<br>
[[paper]](https://arxiv.org/abs/2309.11419)
[[code]](https://github.com/microsoft/unilm/tree/master/kosmos-2.5)

**(*arXiv2023_X-Training*) Small-scale proxies for large-scale Transformer training instabilities.** <br>
*Mitchell Wortsman, Peter J. Liu, Lechao Xiao, Katie Everett, Alex Alemi, Ben Adlam, John D. Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, Jeffrey Pennington, Jascha Sohl-dickstein, Kelvin Xu, Jaehoon Lee, Justin Gilmer, Simon Kornblith.*<br>
[[paper]](https://arxiv.org/abs/2309.14322)

**(*arXiv2023_LLaVA-RLHF*) Aligning Large Multimodal Models with Factually Augmented RLHF.** <br>
*Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, Kurt Keutzer, Trevor Darrell.*<br>
[[paper]](https://arxiv.org/abs/2309.14525)
[[code]](https://llava-rlhf.github.io/)

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

**(*CVPR2024_Honeybee*) Honeybee: Locality-enhanced Projector for Multimodal LLM.** <br>
*Junbum Cha, Wooyoung Kang, Jonghwan Mun, Byungseok Roh.*<br>
[[paper]](https://arxiv.org/abs/2312.06742)
[[code]](https://github.com/kakaobrain/honeybee)

**(*CVPR2024_VILA*) VILA: On Pre-training for Visual Language Models.** <br>
*Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, Song Han.*<br>
[[paper]](https://arxiv.org/abs/2312.07533)
[[code]](https://github.com/nvlabs/vila)

**(*arXiv2023_CogAgent*) CogAgent: A Visual Language Model for GUI Agents.** <br>
*Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxuan Zhang, Juanzi Li, Bin Xu, Yuxiao Dong, Ming Ding, Jie Tang.*<br>
[[paper]](https://arxiv.org/abs/2312.08914)
[[code]](https://github.com/THUDM/CogVLM)

**(*CVPR2024_InternVL*) InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks.** <br>
*Shanghai AI Laboratory.*<br>
[[paper]](https://arxiv.org/abs/2312.14238)
[[code]](https://github.com/OpenGVLab/InternVL)

**(*arXiv2023_MobileVLM*) MobileVLM : A Fast, Strong and Open Vision Language Assistant for Mobile Devices.** <br>
*Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu, Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang, Xiaolin Wei, Chunhua Shen.*<br>
[[paper]](https://arxiv.org/abs/2312.16886)
[[code]](https://github.com/Meituan-AutoML/MobileVLM)

**(*CVPR2024_MMVP*) Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs.** <br>
*Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, Saining Xie.*<br>
[[paper]](https://arxiv.org/abs/2401.06209)
[[code]](https://github.com/tsb0601/MMVP)

**(*arXiv2024_InternLM-XComposer2*) InternLM-XComposer2: Mastering Free-form Text-Image Composition and Comprehension in Vision-Language Large Model.** <br>
*Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Xinyue Zhang, Wei Li, Jingwen Li, Kai Chen, Conghui He, Xingcheng Zhang, Yu Qiao, Dahua Lin, Jiaqi Wang.*<br>
[[paper]](https://arxiv.org/abs/2401.16420)
[[code]](https://github.com/InternLM/InternLM-XComposer)

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

**(*ECCV2024_FastV*) An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models.** <br>
*Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, Baobao Chang.*<br>
[[paper]](https://arxiv.org/abs/2403.06764)
[[code]](https://github.com/pkunlp-icler/FastV)

**(*arXiv2024_LLaVA-UHD*) LLaVA-UHD: an LMM Perceiving Any Aspect Ratio and High-Resolution Images.** <br>
*Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, Maosong Sun, Gao Huang.*<br>
[[paper]](https://arxiv.org/abs/2403.11703v1)
[[code]](https://github.com/thunlp/LLaVA-UHD)

**(*arXiv2024_CoS*) Chain-of-Spot: Interactive Reasoning Improves Large Vision-Language Models.** <br>
*Zuyan Liu, Yuhao Dong, Yongming Rao, Jie Zhou, Jiwen Lu.*<br>
[[paper]](https://arxiv.org/abs/2403.12966)
[[code]](https://github.com/dongyh20/chain-of-spot)

**(*arXiv2024_S2*) When Do We Not Need Larger Vision Models?.** <br>
*Baifeng Shi, Ziyang Wu, Maolin Mao, Xin Wang, Trevor Darrell.*<br>
[[paper]](https://arxiv.org/abs/2403.13043)
[[code]](https://github.com/bfshi/scaling_on_scales)

**(*arXiv2024_Cobra*) Cobra: Extending Mamba to Multi-Modal Large Language Model for Efficient Inference.** <br>
*Han Zhao, Min Zhang, Wei Zhao, Pengxiang Ding, Siteng Huang, Donglin Wang.*<br>
[[paper]](https://arxiv.org/abs/2403.14520)
[[code]](https://sites.google.com/view/cobravlm)

**(*arXiv2024_LLaVA-PruMerge*) LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models.** <br>
*Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, Yan Yan.*<br>
[[paper]](https://arxiv.org/abs/2403.15388)
[[code]](https://llava-prumerge.github.io/)

**(*Blog2024_Idefics2*) Introducing Idefics2: A Powerful 8B Vision-Language Model for the community.** <br>
*Leo Tronchon, Hugo Laurençon, Victor Sanh.*<br>
[[blog]](https://huggingface.co/blog/idefics2)

**(*arXiv2024_InternLM-XComposer2-4KHD*) InternLM-XComposer2-4KHD: A Pioneering Large Vision-Language Model Handling Resolutions from 336 Pixels to 4K HD.** <br>
*Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, Hang Yan, Yang Gao, Zhe Chen, Xinyue Zhang, Wei Li, Jingwen Li, Wenhai Wang, Kai Chen, Conghui He, Xingcheng Zhang, Jifeng Dai, Yu Qiao, Dahua Lin, Jiaqi Wang.*<br>
[[paper]](https://arxiv.org/abs/2404.06512)
[[code]](https://github.com/InternLM/InternLM-XComposer)

**(*ECCV2024_BRAVE*) BRAVE: Broadening the visual encoding of vision-language models.** <br>
*Oğuzhan Fatih Kar, Alessio Tonioni, Petra Poklukar, Achin Kulshrestha, Amir Zamir, Federico Tombari.*<br>
[[paper]](https://arxiv.org/abs/2404.07204)
[[code]](https://brave-vlms.epfl.ch/)

**(*arXiv2024_InternVL1.5*) How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites.** <br>
*Shanghai AI Laboratory.*<br>
[[paper]](https://arxiv.org/abs/2404.16821)
[[code]](https://github.com/OpenGVLab/InternVL)

**(*arXiv2024_MANTIS*) MANTIS: Interleaved Multi-Image Instruction Tuning.** <br>
*Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, Wenhu Chen.*<br>
[[paper]](https://arxiv.org/abs/2405.01483)
[[code]](https://github.com/TIGER-AI-Lab/Mantis)

**(*NeurIPS2024_DenseConnector*) Dense Connector for MLLMs.** <br>
*Huanjin Yao, Wenhao Wu, Taojiannan Yang, YuXin Song, Mengxi Zhang, Haocheng Feng, Yifan Sun, Zhiheng Li, Wanli Ouyang, Jingdong Wang.*<br>
[[paper]](https://arxiv.org/abs/2405.13800)
[[code]](https://github.com/HJYao00/DenseConnector)

**(*arXiv2024_Ovis*) Ovis: Structural Embedding Alignment for Multimodal Large Language Model.** <br>
*Shiyin Lu, Yang Li, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, Han-Jia Ye.*<br>
[[paper]](https://arxiv.org/abs/2405.20797)
[[code]](https://github.com/AIDC-AI/Ovis)

**(*arXiv2024_MiCo*) Explore the Limits of Omni-modal Pretraining at Scale.** <br>
*Yiyuan Zhang, Handong Li, Jing Liu, Xiangyu Yue.*<br>
[[paper]](https://arxiv.org/abs/2406.09412)
[[code]](https://github.com/invictus717/MiCo)

**(*arXiv2024_Cambrian-1*) Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs.** <br>
*Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, Saining Xie.*<br>
[[paper]](https://arxiv.org/abs/2406.16860)
[[code]](https://cambrian-mllm.github.io/)

**(*Blog2024_LLaVA-NeXT*) LLaVA-NeXT-series.** <br>
[[blog]](https://llava-vl.github.io/blog/)

**(*Blog2024_InternVL*) InternVL-series.** <br>
[[paper]](https://internvl.github.io/blog/)

**(*NeurIPS2024_EVE*) Unveiling Encoder-Free Vision-Language Models.** <br>
*Haiwen Diao, Yufeng Cui, Xiaotong Li, Yueze Wang, Huchuan Lu, Xinlong Wang.*<br>
[[paper]](https://arxiv.org/abs/2406.11832)
[[code]](https://github.com/baaivision/EVE)

**(*arXiv2024_InternLM-XComposer-2.5*) InternLM-XComposer-2.5: A Versatile Large Vision Language Model Supporting Long-Contextual Input and Output.** <br>
*Pan Zhang, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Rui Qian, Lin Chen, Qipeng Guo, Haodong Duan, Bin Wang, Linke Ouyang, Songyang Zhang, Wenwei Zhang, Yining Li, Yang Gao, Peng Sun, Xinyue Zhang, Wei Li, Jingwen Li, Wenhai Wang, Hang Yan, Conghui He, Xingcheng Zhang, Kai Chen, Jifeng Dai, Yu Qiao, Dahua Lin, Jiaqi Wang.*<br>
[[paper]](https://arxiv.org/abs/2407.03320)
[[code]](https://github.com/InternLM/InternLM-XComposer)

**(*arXiv2024_SOLO*) A Single Transformer for Scalable Vision-Language Modeling.** <br>
*Yangyi Chen, Xingyao Wang, Hao Peng, Heng Ji.*<br>
[[paper]](https://arxiv.org/abs/2407.06438)
[[code]](https://github.com/Yangyi-Chen/SOLO)

**(*arXiv2024_PaliGemma*) PaliGemma: A versatile 3B VLM for transfer.** <br>
*DeepMind Team.*<br>
[[paper]](https://arxiv.org/abs/2407.07726v1)
[[code]](https://github.com/google-research/big_vision)

**(*arXiv2024_LMMs-Eval*) LMMs-Eval: Reality Check on the Evaluation of Large Multimodal Models.** <br>
[[blog]](https://lmms-lab.github.io/lmms-eval-blog/lmms-eval-0.1/)
[[paper]](https://arxiv.org/abs/2407.12772)
[[code]](https://huggingface.co/spaces/lmms-lab/LiveBench)

**(*arXiv2024_EVLM*) EVLM: An Efficient Vision-Language Model for Visual Understanding.** <br>
*Kaibing Chen, Dong Shen, Hanwen Zhong, Huasong Zhong, Kui Xia, Di Xu, Wei Yuan, Yifei Hu, Bin Wen, Tianke Zhang, Changyi Liu, Dewen Fan, Huihui Xiao, Jiahong Wu, Fan Yang, Size Li, Di Zhang.*<br>
[[paper]](https://arxiv.org/abs/2407.14177)

**(*arXiv2024_VILA2*) VILA2: VILA Augmented VILA.** <br>
*Yunhao Fang, Ligeng Zhu, Yao Lu, Yan Wang, Pavlo Molchanov, Jang Hyun Cho, Marco Pavone, Song Han, Hongxu Yin.*<br>
[[paper]](https://arxiv.org/abs/2407.17453)
[[code]](https://github.com/NVlabs/VILA/tree/main)

**(*arXiv2024_DIVA*) Diffusion Feedback Helps CLIP See Better.** <br>
*Wenxuan Wang, Quan Sun, Fan Zhang, Yepeng Tang, Jing Liu, Xinlong Wang.*<br>
[[paper]](https://arxiv.org/abs/2407.20171)
[[code]](https://github.com/baaivision/diva)

**(*arXiv2024_MoMa*) MoMa: Efficient Early-Fusion Pre-training with Mixture of Modality-Aware Experts.** <br>
*Xi Victoria Lin, Akshat Shrivastava, Liang Luo, Srinivasan Iyer, Mike Lewis, Gargi Ghosh, Luke Zettlemoyer, Armen Aghajanyan.*<br>
[[paper]](https://arxiv.org/abs/2407.21770v3)

**(*arXiv2024_MiniCPM-V*) MiniCPM-V: A GPT-4V Level MLLM on Your Phone.** <br>
*MiniCPM-V Team.*<br>
[[paper]](https://arxiv.org/abs/2408.01800)
[[code]](https://github.com/OpenBMB/MiniCPM-V)

**(*arXiv2024_LLaVA-OneVision*) LLaVA-OneVision: Easy Visual Task Transfer.** <br>
*Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, Chunyuan Li.*<br>
[[paper]](https://arxiv.org/abs/2408.03326)
[[code]](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/)

**(*arXiv2024_mPLUG-Owl3*) mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models.** <br>
*Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, Jingren Zhou.*<br>
[[paper]](https://arxiv.org/abs/2408.04840)
[[code]](https://github.com/x-plug/mplug-owl)

**(*arXiv2024_CROME*) CROME: Cross-Modal Adapters for Efficient Multimodal LLM.** <br>
*Sayna Ebrahimi, Sercan O. Arik, Tejas Nama, Tomas Pfister.*<br>
[[paper]](https://arxiv.org/abs/2408.06610)

**(*arXiv2024_Sys2-LLaVA*) Visual Agents as Fast and Slow Thinkers.** <br>
*Guangyan Sun, Mingyu Jin, Zhenting Wang, Cheng-Long Wang, Siqi Ma, Qifan Wang, Tong Geng, Ying Nian Wu, Yongfeng Zhang, Dongfang Liu.*<br>
[[paper]](https://arxiv.org/abs/2408.08862)
[[code]](https://github.com/GuangyanS/Sys2-LLaVA)

**(*arXiv2024_BLIP-3*) xGen-MM (BLIP-3): A Family of Open Large Multimodal Models.** <br>
*Salesforce AI Research.*<br>
[[paper]](https://www.arxiv.org/abs/2408.08872)
[[code]](https://www.salesforceairesearch.com/opensource/xGen-MM/index.html)

**(*arXiv2024_MaVEn*) MaVEn: An Effective Multi-granularity Hybrid Visual Encoding Framework for Multimodal Large Language Model.** <br>
*Chaoya Jiang, Jia Hongrui, Haiyang Xu, Wei Ye, Mengfan Dong, Ming Yan, Ji Zhang, Fei Huang, Shikun Zhang.*<br>
[[paper]](https://arxiv.org/abs/2408.12321)

**(*Blog2024_QwenVL*) QwenVL-series.** <br>
[[blog]](https://qwenlm.github.io/blog/)

**(*arXiv2024_Eagle*) Eagle: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders.** <br>
*Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, Humphrey Shi, Bryan Catanzaro, Andrew Tao, Jan Kautz, Zhiding Yu, Guilin Liu.*<br>
[[paper]](https://arxiv.org/abs/2408.15998)
[[code]](https://github.com/NVlabs/Eagle)

**(*arXiv2024_AC-score*) Law of Vision Representation in MLLMs.** <br>
*Shijia Yang, Bohan Zhai, Quanzeng You, Jianbo Yuan, Hongxia Yang, Chenfeng Xu.*<br>
[[paper]](https://arxiv.org/abs/2408.16357)
[[code]](https://github.com/bronyayang/Law_of_Vision_Representation_in_MLLMs)

**(*arXiv2024_NVLM*) NVLM: Open Frontier-Class Multimodal LLMs.** <br>
*Wenliang Dai, Nayeon Lee, Boxin Wang, Zhuoling Yang, Zihan Liu, Jon Barker, Tuomas Rintamaki, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping.*<br>
[[paper]](https://arxiv.org/abs/2409.11402)

**(*arXiv2024_Qwen2-VL*) Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution.** <br>
*Qwen Team.*<br>
[[paper]](https://arxiv.org/abs/2409.12191)
[[code]](https://github.com/QwenLM/Qwen2-VL)

**(*arXiv2024_Oryx*) Oryx MLLM: On-Demand Spatial-Temporal Understanding at Arbitrary Resolution.** <br>
*Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, Yongming Rao.*<br>
[[paper]](https://arxiv.org/abs/2409.12961)
[[code]](https://github.com/Oryx-mllm/Oryx)

**(*arXiv2024_SSC-DSC*) Revisit Large-Scale Image-Caption Data in Pre-training Multimodal Foundation Models.** <br>
*Zhengfeng Lai, Vasileios Saveris, Chen Chen, Hong-You Chen, Haotian Zhang, Bowen Zhang, Juan Lao Tebar, Wenze Hu, Zhe Gan, Peter Grasch, Meng Cao, Yinfei Yang.*<br>
[[paper]](https://arxiv.org/abs/2410.02740)

**(*arXiv2024_SparseVLM*) SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference.** <br>
*Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, Shanghang Zhang.*<br>
[[paper]](https://arxiv.org/abs/2410.04417)
[[code]](https://github.com/Gumpest/SparseVLMs)

**(*arXiv2024_Aria*) Aria: An Open Multimodal Native Mixture-of-Experts Model.** <br>
*Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao Niu, Guoyin Wang, Bei Chen, Junnan Li.*<br>
[[paper]](https://arxiv.org/abs/2410.05993)
[[code]](https://github.com/rhymes-ai/aria)

**(*arXiv2024_Pixtral-12B*) Pixtral 12B.** <br>
*Mistral.AI.*<br>
[[paper]](https://arxiv.org/pdf/2410.07073)
[[code]](https://mistral.ai/news/pixtral-12b/)

**(*arXiv2024_Mono-InternVL*) Mono-InternVL: Pushing the Boundaries of Monolithic Multimodal Large Language Models with Endogenous Visual Pre-training.** <br>
*Gen Luo, Xue Yang, Wenhan Dou, Zhaokai Wang, Jifeng Dai, Yu Qiao, Xizhou Zhu.*<br>
[[paper]](https://arxiv.org/abs/2410.08202)
[[code]](https://internvl.github.io/blog/2024-10-10-Mono-InternVL/)

**(*arXiv2024_ROSS*) Reconstructive Visual Instruction Tuning.** <br>
*Haochen Wang, Anlin Zheng, Yucheng Zhao, Tiancai Wang, Zheng Ge, Xiangyu Zhang, Zhaoxiang Zhang.*<br>
[[paper]](https://arxiv.org/abs/2410.09575)
[[code]](https://haochen-wang409.github.io/ross)

**(*arXiv2024_Infinity-MM*) Infinity-MM: Scaling Multimodal Performance with Large-Scale and High-Quality Instruction Data.** <br>
*BAAI Group.*<br>
[[paper]](https://arxiv.org/abs/2410.18558)
[[code]](https://huggingface.co/datasets/BAAI/Infinity-MM)

**(*arXiv2024_GPT-4o*) GPT-4o System Card.** <br>
*OpenAI Group.*<br>
[[paper]](https://arxiv.org/abs/2410.21276)

**(*arXiv2024_TaskVector*) Task Vectors are Cross-Modal.** <br>
*Grace Luo, Trevor Darrell, Amir Bar.*<br>
[[paper]](https://arxiv.org/abs/2410.22330)
[[code]](https://task-vectors-are-cross-modal.github.io/)

**(*arXiv2024_MoT*) Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models.** <br>
*Weixin Liang, Lili Yu, Liang Luo, Srinivasan Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, Wen-tau Yih, Luke Zettlemoyer, Xi Victoria Lin.*<br>
[[paper]](https://arxiv.org/abs/2411.04996)

**(*arXiv2024_Insight-V*) Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models.** <br>
*Yuhao Dong, Zuyan Liu, Hai-Long Sun, Jingkang Yang, Winston Hu, Yongming Rao, Ziwei Liu.*<br>
[[paper]](https://arxiv.org/abs/2411.14432)
[[code]](https://github.com/dongyh20/Insight-V)

**(*arXiv2024_SAE*) Large Multi-modal Models Can Interpret Features in Large Multi-modal Models.** <br>
*Kaichen Zhang, Yifei Shen, Bo Li, Ziwei Liu.*<br>
[[paper]](https://arxiv.org/abs/2411.14982)

**(*arXiv2024_InformationFlow*) Cross-modal Information Flow in Multimodal Large Language Models.** <br>
*Zhi Zhang, Srishti Yadav, Fengze Han, Ekaterina Shutova.*<br>
[[paper]](https://arxiv.org/abs/2411.18620)

**(*arXiv2024_PaliGemma2*) PaliGemma 2: A Family of Versatile VLMs for Transfer.** <br>
*DeepMind Team.*<br>
[[paper]](https://arxiv.org/abs/2412.03555)

**(*arXiv2024_Florence-VL*) Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion.** <br>
*Microsoft Research.*<br>
[[paper]](https://arxiv.org/abs/2412.04424)
[[code]](https://github.com/JiuhaiChen/Florence-VL)

**(*arXiv2024_VisionZip*) VisionZip: Longer is Better but Not Necessary in Vision Language Models.** <br>
*Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, Jiaya Jia.*<br>
[[paper]](https://arxiv.org/abs/2412.04467)
[[code]](https://github.com/dvlab-research/VisionZip)

**(*arXiv2024_NVILA*) NVILA: Efficient Frontier Visual Language Models.** <br>
*NVIDIA Team.*<br>
[[paper]](https://arxiv.org/abs/2412.04468)
[[code]](https://github.com/NVlabs/VILA)

**(*arXiv2024_MAmmoTH-VL*) MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale.** <br>
*Jarvis Guo, Tuney Zheng, Yuelin Bai, Bo Li, Yubo Wang, King Zhu, Yizhi Li, Graham Neubig, Wenhu Chen, Xiang Yue.*<br>
[[paper]](https://arxiv.org/abs/2412.05237)
[[code]](https://mammoth-vl.github.io/)

**(*arXiv2024_CompCap*) CompCap: Improving Multimodal Large Language Models with Composite Captions.** <br>
*Xiaohui Chen, Satya Narayan Shukla, Mahmoud Azab, Aashu Singh, Qifan Wang, David Yang, ShengYun Peng, Hanchao Yu, Shen Yan, Xuewen Zhang, Baosheng He.*<br>
[[paper]](https://arxiv.org/abs/2412.05243)

**(*arXiv2024_InternVL2.5*) Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling.** <br>
*Shanghai AI Laboratory.*<br>
[[paper]](https://arxiv.org/abs/2412.05271)
[[code]](https://github.com/opengvlab/internvl)

**(*arXiv2024_MMGiC*) Exploring Multi-Grained Concept Annotations for Multimodal Large Language Models.** <br>
*Xiao Xu, Tianhao Niu, Yuxi Xie, Libo Qin, Wanxiang Che, Min-Yen Kan.*<br>
[[paper]](https://arxiv.org/abs/2412.05939)
[[code]](https://github.com/LooperXX/MMGiC)

**(*arXiv2024_POINTS1.5*) POINTS1.5: Building a Vision-Language Model towards Real World Applications.** <br>
*Yuan Liu, Le Tian, Xiao Zhou, Xinyu Gao, Kavio Yu, Yang Yu, Jie Zhou.*<br>
[[paper]](https://arxiv.org/abs/2412.08443)

**(*arXiv2024_Euclid*) Euclid: Supercharging Multimodal LLMs with Synthetic High-Fidelity Visual Descriptions.** <br>
*Jiarui Zhang, Ollie Liu, Tianyu Yu, Jinyi Hu, Willie Neiswanger.*<br>
[[paper]](https://arxiv.org/abs/2412.08737)

**(*arXiv2024_V2PE*) V2PE: Improving Multimodal Long-Context Capability of Vision-Language Models with Variable Visual Position Encoding.** <br>
*Junqi Ge, Ziyi Chen, Jintao Lin, Jinguo Zhu, Xihui Liu, Jifeng Dai, Xizhou Zhu.*<br>
[[paper]](https://arxiv.org/abs/2412.09616)
[[code]](https://github.com/OpenGVLab/V2PE)

**(*arXiv2024_DeepSeek-VL2*) DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding.** <br>
*DeepSeek-AI Team.*<br>
[[paper]](https://arxiv.org/abs/2412.10302)
[[code]](https://github.com/deepseek-ai/DeepSeek-VL2)

**(*arXiv2024_HoVLE*) HoVLE: Unleashing the Power of Monolithic Vision-Language Models with Holistic Vision-Language Embedding.** <br>
*Chenxin Tao, Shiqian Su, Xizhou Zhu, Chenyu Zhang, Zhe Chen, Jiawen Liu, Wenhai Wang, Lewei Lu, Gao Huang, Yu Qiao, Jifeng Dai.*<br>
[[paper]](https://arxiv.org/abs/2412.16158)
[[code]](https://huggingface.co/OpenGVLab/HoVLE)

**(*arXiv2025_EVEv2*) EVEv2: Improved Baselines for Encoder-Free Vision-Language Models.** <br>
*Haiwen Diao, Xiaotong Li, Yufeng Cui, Yueze Wang, Haoge Deng, Ting Pan, Wenxuan Wang, Huchuan Lu, Xinlong Wang.*<br>
[[paper]](https://arxiv.org/abs/2502.06788)
[[code]](https://github.com/baaivision/EVE)

**(*arXiv2024_Qwen2.5-VL*) Qwen2.5-VL Technical Report.** <br>
*Qwen Team.*<br>
[[paper]](https://arxiv.org/abs/2502.13923)
[[code]](https://github.com/qwenlm/qwen2.5-vl)

**(*arXiv2025_Visual-RFT*) Visual-RFT: Visual Reinforcement Fine-Tuning.** <br>
*Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, Jiaqi Wang.*<br>
[[paper]](https://arxiv.org/abs/2503.01785)
[[code]](https://github.com/liuziyu77/visual-rft)

**(*arXiv2025_UnifiedReward*) Unified Reward Model for Multimodal Understanding and Generation.** <br>
*Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, Jiaqi Wang.*<br>
[[paper]](https://arxiv.org/abs/2503.05236)
[[code]](https://codegoat24.github.io/UnifiedReward/)

**(*arXiv2025_Vision-R1*) Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models.** <br>
*Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, Shaohui Lin.*<br>
[[paper]](https://arxiv.org/abs/2503.06749)
[[code]](https://github.com/Osilly/Vision-R1)

**(*arXiv2025_LMM-R1*) LMM-R1: Empowering 3B LMMs with Strong Reasoning Abilities Through Two-Stage Rule-Based RL.** <br>
*Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, Xu Yang.*<br>
[[paper]](https://arxiv.org/abs/2503.07536)
[[code]](https://github.com/TideDra/lmm-r1)

**(*arXiv2025_R1-Onevision*) R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization.** <br>
*Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, Bo Zhang, Wei Chen.*<br>
[[paper]](https://arxiv.org/abs/2503.10615)
[[code]](https://github.com/Fancy-MLLM/R1-onevision)

**(*arXiv2025_BREEN*) BREEN: Bridge Data-Efficient Encoder-Free Multimodal Learning with Learnable Queries.** <br>
*Tianle Li, Yongming Rao, Winston Hu, Yu Cheng.*<br>
[[paper]](https://arxiv.org/abs/2503.12446)

**(*arXiv2025_HaploVL*) HaploVL: A Single-Transformer Baseline for Multi-Modal Understanding.** <br>
*Rui Yang, Lin Song, Yicheng Xiao, Runhui Huang, Yixiao Ge, Ying Shan, Hengshuang Zhao.*<br>
[[paper]](https://arxiv.org/abs/2503.14694)

**(*arXiv2025_GenHancer*) GenHancer: Imperfect Generative Models are Secretly Strong Vision-Centric Enhancers.** <br>
*Shijie Ma, Yuying Ge, Teng Wang, Yuxin Guo, Yixiao Ge, Ying Shan.*<br>
[[paper]](https://arxiv.org/abs/2503.19480)
[[code]](https://mashijie1028.github.io/GenHancer/)

**(*arXiv2025_VoRA*) Vision as LoRA.** <br>
*Han Wang, Yongjie Ye, Bingru Li, Yuxiang Nie, Jinghui Lu, Jingqun Tang, Yanjie Wang, Can Huang.*<br>
[[paper]](https://arxiv.org/abs/2503.20680)
[[code]](https://github.com/Hon-Wong/VoRA)

**(*arXiv2025_Ross3D*) Ross3D: Reconstructive Visual Instruction Tuning with 3D-Awareness.** <br>
*Haochen Wang, Yucheng Zhao, Tiancai Wang, Haoqiang Fan, Xiangyu Zhang, Zhaoxiang Zhang.*<br>
[[paper]](https://arxiv.org/abs/2504.01901)

**(*arXiv2025_SAIL*) The Scalability of Simplicity: Empirical Analysis of Vision-Language Learning with a Single Transformer.** <br>
*Weixian Lei, Jiacong Wang, Haochen Wang, Xiangtai Li, Jun Hao Liew, Jiashi Feng, Zilong Huang.*<br>
[[paper]](https://arxiv.org/abs/2504.10462v1)
[[paper]](https://github.com/bytedance/SAIL)

**(*arXiv2025_PerceptionLM*) PerceptionLM: Open-Access Data and Models for Detailed Visual Understanding.** <br>
*Meta FAIR.*<br>
[[paper]](https://arxiv.org/abs/2504.13180)
[[code]](https://github.com/facebookresearch/perception_models)

**(*arXiv2025_InternVL3*) InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models.** <br>
*Shanghai AI Laboratory.*<br>
[[paper]](https://arxiv.org/abs/2504.10479)
[[code]](https://github.com/opengvlab/internvl)

**(*arXiv2025_ZipR1*) ZipR1: Reinforcing Token Sparsity in MLLMs.** <br>
*Feng Chen, Yefei He, Lequan Lin, Jing Liu, Bohan Zhuang, Qi Wu.*<br>
[[paper]](https://arxiv.org/abs/2504.18579)

**(*arXiv2025_UnifiedReward-GRPO*) Unified Multimodal Chain-of-Thought Reward Model through Reinforcement Fine-Tuning.** <br>
*Yibin Wang, Zhimin Li, Yuhang Zang, Chunyu Wang, Qinglin Lu, Cheng Jin, Jiaqi Wang.*<br>
[[paper]](https://arxiv.org/abs/2505.03318)
[[code]](https://codegoat24.github.io/UnifiedReward/think)

**(*arXiv2025_Seed1.5-VL*) Seed1.5-VL Technical Report.** <br>
*ByteDance Seed.*<br>
[[paper]](https://arxiv.org/abs/2505.07062)
[[code]](https://www.volcengine.com/)

**(*arXiv2025_MiMo-VL*) MiMo-VL Technical Report.** <br>
*Xiaomi LLM-Core Team.*<br>
[[paper]](https://github.com/XiaomiMiMo/MiMo-VL/blob/main/MiMo-VL-Technical-Report.pdf)
[[code]](https://github.com/XiaomiMiMo/MiMo-VL)

**(*ICML2025_ProxyV*) Streamline Without Sacrifice - Squeeze out Computation Redundancy in LMM.** <br>
*Penghao Wu, Lewei Lu, Ziwei Liu.*<br>
[[paper]](https://arxiv.org/abs/2505.15816)
[[code]](https://github.com/penghao-wu/ProxyV)

**(*arXiv2025_R1-ShareVL*) R1-ShareVL: Incentivizing Reasoning Capability of Multimodal Large Language Models via Share-GRPO.** <br>
*Huanjin Yao, Qixiang Yin, Jingyi Zhang, Min Yang, Yibo Wang, Wenhao Wu, Fei Su, Li Shen, Minghui Qiu, Dacheng Tao, Jiaxing Huang.*<br>
[[paper]](https://arxiv.org/abs/2505.16673)
[[code]](https://github.com/HJYao00/R1-ShareVL)

**(*arXiv2025_LaViDa*) LaViDa: A Large Diffusion Language Model for Multimodal Understanding.** <br>
*Shufan Li, Konstantinos Kallidromitis, Hritik Bansal, Akash Gokul, Yusuke Kato, Kazuki Kozuka, Jason Kuen, Zhe Lin, Kai-Wei Chang, Aditya Grover.*<br>
[[paper]](https://arxiv.org/abs/2505.16839)
[[code]](https://github.com/jacklishufan/lavida)

**(*arXiv2025_LLaDA-V*) LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning.** <br>
*Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, Chongxuan Li.*<br>
[[paper]](https://arxiv.org/abs/2505.16933)
[[code]](https://ml-gsai.github.io/LLaDA-V-demo/)

**(*arXiv2025_Dimple*) Dimple: Discrete Diffusion Multimodal Large Language Model with Parallel Decoding.** <br>
*Runpeng Yu, Xinyin Ma, Xinchao Wang.*<br>
[[paper]](https://arxiv.org/abs/2505.16990)
[[code]](https://github.com/yu-rp/Dimple)

**(*arXiv2025_un2CLIP*) un2CLIP: Improving CLIP's Visual Detail Capturing Ability via Inverting unCLIP.** <br>
*Yinqi Li, Jiahe Zhao, Hong Chang, Ruibing Hou, Shiguang Shan, Xilin Chen.*<br>
[[paper]](https://arxiv.org/abs/2505.24517)
[[code]](https://github.com/LiYinqi/un2CLIP)

**(*arXiv2025_SRPO*) SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning.** <br>
*Zhongwei Wan, Zhihao Dou, Che Liu, Yu Zhang, Dongfei Cui, Qinjian Zhao, Hui Shen, Jing Xiong, Yi Xin, Yifan Jiang, Yangfan He, Mi Zhang, Shen Yan.*<br>
[[paper]](https://arxiv.org/abs/2506.01713)
[[code]](https://github.com/SUSTechBruce/SRPO_MLLMs)

**(*arXiv2025_SynthRL*) SynthRL: Scaling Visual Reasoning with Verifiable Data Synthesis.** <br>
*Zijian Wu, Jinjie Ni, Xiangyan Liu, Zichen Liu, Hang Yan, Michael Qizhe Shieh.*<br>
[[paper]](https://arxiv.org/abs/2506.02096)

**(*arXiv2025_ASVR*) Autoregressive Semantic Visual Reconstruction Helps VLMs Understand Better.** <br>
*Dianyi Wang, Wei Song, Yikun Wang, Siyuan Wang, Kaicheng Yu, Zhongyu Wei, Jiaqi Wang.*<br>
[[paper]](https://arxiv.org/abs/2506.09040)
[[code]](https://github.com/AlenjandroWang/ASVR)

**(*arXiv2025_Gemini-2.5*) Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities.** <br>
*Gemini Team.*<br>
[[paper]](https://arxiv.org/abs/2507.06261)

**(*arXiv2025_MMSearch-R1*) MMSearch-R1: Incentivizing LMMs to Search.** <br>
*Jinming Wu, Zihao Deng, Wei Li, Yiding Liu, Bo You, Bo Li, Zejun Ma, Ziwei Liu.*<br>
[[paper]](https://arxiv.org/abs/2506.20670)
[[code]](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)

**(*arXiv2025_DenseWorld-1M*) DenseWorld-1M: Towards Detailed Dense Grounded Caption in the Real World.** <br>
*Xiangtai Li, Tao Zhang, Yanwei Li, Haobo Yuan, Shihao Chen, Yikang Zhou, Jiahao Meng, Yueyi Sun, Shilin Xu, Lu Qi, Tianheng Cheng, Yi Lin, Zilong Huang, Wenhao Huang, Jiashi Feng, Guang Shi.*<br>
[[paper]](https://arxiv.org/abs/2506.24102)
[[code]](https://github.com/lxtGH/DenseWorld-1M)

**(*arXiv2025_GLM-4.1V-Thinking*) GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning.** <br>
*GLM-V Team.*<br>
[[paper]](https://arxiv.org/abs/2507.01006)
[[code]](https://github.com/THUDM/GLM-4.1V-Thinking)

**(*arXiv2025_Keye-VL*) Kwai Keye-VL Technical Report.** <br>
*Keye Team, Kuaishou Group.*<br>
[[paper]](https://arxiv.org/abs/2507.01949)
[[code]](https://github.com/Kwai-Keye/Keye)

**(*arXiv2025_Analysis*) Scaling Laws for Optimal Data Mixtures.** <br>
*Mustafa Shukor, Louis Bethune, Dan Busbridge, David Grangier, Enrico Fini, Alaaeldin El-Nouby, Pierre Ablin.*<br>
[[paper]](https://arxiv.org/abs/2507.09404)

**(*arXiv2025_Mono-InternVL-1.5*) Mono-InternVL-1.5: Towards Cheaper and Faster Monolithic Multimodal Large Language Models.** <br>
*Gen Luo, Wenhan Dou, Wenhao Li, Zhaokai Wang, Xue Yang, Changyao Tian, Hao Li, Weiyun Wang, Wenhai Wang, Xizhou Zhu, Yu Qiao, Jifeng Dai.*<br>
[[paper]](https://arxiv.org/abs/2507.12566)
[[code]](https://github.com/OpenGVLab/Mono-InternVL)

**(*arXiv2025_Analysis*) VLMs have Tunnel Vision: Evaluating Nonlocal Visual Reasoning in Leading VLMs.** <br>
*Shmuel Berman, Jia Deng.*<br>
[[paper]](https://arxiv.org/abs/2507.13361)

**(*Blog2025_ARC-AGI*) Can lossless information compression by itself produce intelligent behavior?** <br>
*Isaac Liao, Albert Gu.*<br>
[[Blog]](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html)

**(*Blog2025_LFM2*) Introducing LFM2: The Fastest On-Device Foundation Models on the Market.** <br>
*Liquid AI.*<br>
[[Blog]](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models)

**(*arXiv2025_Ovis2.5*) Ovis2.5 Technical Report.** <br>
*Ovis Team, Alibaba Group.*<br>
[[paper]](https://arxiv.org/abs/2508.11737)
[[code]](https://github.com/AIDC-AI/Ovis)



#### ``*Video Perception*``

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

**(*arXiv2023_VideoLLM*) VideoLLM: Modeling Video Sequence with Large Language Models.** <br>
*Guo Chen, Yin-Dong Zheng, Jiahao Wang, Jilan Xu, Yifei Huang, Junting Pan, Yi Wang, Yali Wang, Yu Qiao, Tong Lu, Limin Wang.*<br>
[[paper]](https://arxiv.org/abs/2305.13292)
[[code]](https://github.com/cg1177/VideoLLM)

**(*arXiv2023_VSTAR*) VSTAR: A Video-grounded Dialogue Dataset for Situated Semantic Understanding with Scene and Topic Transitions.** <br>
*Yuxuan Wang, Zilong Zheng, Xueliang Zhao, Jinpeng Li, Yueqian Wang, Dongyan Zhao.*<br>
[[paper]](https://arxiv.org/abs/2305.18756)
[[code]](https://github.com/patrick-tssn/VSTAR)

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

**(*arXiv2023_PG-Video-LLaVA*) PG-Video-LLaVA: Pixel Grounding Large Video-Language Models.** <br>
*Shehan Munasinghe, Rusiru Thushara, Muhammad Maaz, Hanoona Abdul Rasheed, Salman Khan, Mubarak Shah, Fahad Khan.*<br>
[[paper]](https://arxiv.org/abs/2311.13435)
[[code]](https://github.com/mbzuai-oryx/Video-LLaVA)

**(*arXiv2023_VideoChat2*) MVBench: A Comprehensive Multi-modal Video Understanding Benchmark.** <br>
*Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, Yu Qiao.*<br>
[[paper]](https://arxiv.org/abs/2311.17005)
[[code]](https://github.com/OpenGVLab/Ask-Anything)

**(*arXiv2023_LLaMA-VID*) LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models.** <br>
*Yanwei Li, Chengyao Wang, Jiaya Jia.*<br>
[[paper]](https://arxiv.org/abs/2311.17043)
[[code]](https://github.com/dvlab-research/LLaMA-VID)

**(*arXiv2024_LSTP*) LSTP: Language-guided Spatial-Temporal Prompt Learning for Long-form Video-Text Understanding.** <br>
*Yuxuan Wang, Yueqian Wang, Pengfei Wu, Jianxin Liang, Dongyan Zhao, Zilong Zheng.*<br>
[[paper]](https://arxiv.org/abs/2402.16050)
[[code]](https://github.com/bigai-nlco/LSTP-Chat)

**(*arXiv2024_LLaVA-Hound-DPO*) Direct Preference Optimization of Video Large Multimodal Models from Language Model Reward.** <br>
*Ruohong Zhang, Liangke Gui, Zhiqing Sun, Yihao Feng, Keyang Xu, Yuanhan Zhang, Di Fu, Chunyuan Li, Alexander Hauptmann, Yonatan Bisk, Yiming Yang.*<br>
[[paper]](https://arxiv.org/abs/2404.01258)
[[code]](https://github.com/riflezhang/llava-hound-dpo)

**(*arXiv2024_ShareGPT4Video*) ShareGPT4Video: Improving Video Understanding and Generation with Better Captions.** <br>
*Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, Li Yuan, Yu Qiao, Dahua Lin, Feng Zhao, Jiaqi Wang.*<br>
[[paper]](https://arxiv.org/abs/2406.04325v1)
[[code]](https://sharegpt4video.github.io/)

**(*arXiv2024_LongVA*) Long Context Transfer from Language to Vision.** <br>
*Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, Ziwei Liu.*<br>
[[paper]](https://arxiv.org/abs/2406.16852)
[[code]](https://github.com/EvolvingLMMs-Lab/LongVA)

**(*arXiv2024_SF-LLaVA*) SlowFast-LLaVA: A Strong Training-Free Baseline for Video Large Language Models.** <br>
*Mingze Xu, Mingfei Gao, Zhe Gan, Hong-You Chen, Zhengfeng Lai, Haiming Gang, Kai Kang, Afshin Dehghan.*<br>
[[paper]](https://arxiv.org/abs/2407.15841)
[[code]](https://github.com/apple/ml-slowfast-llava)

**(*arXiv2024_VITA*) VITA: Towards Open-Source Interactive Omni Multimodal LLM.** <br>
*Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Meng Zhao, Yifan Zhang, Xiong Wang, Di Yin, Long Ma, Xiawu Zheng, Ran He, Rongrong Ji, Yunsheng Wu, Caifeng Shan, Xing Sun.*<br>
[[paper]](https://arxiv.org/abs/2408.05211)
[[code]](https://vita-home.github.io/)

**(*arXiv2024_LongVILA*) LongVILA: Scaling Long-Context Visual Language Models for Long Videos.** <br>
*Fuzhao Xue, Yukang Chen, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, Ethan He, Hongxu Yin, Pavlo Molchanov, Jan Kautz, Linxi Fan, Yuke Zhu, Yao Lu, Song Han.*<br>
[[paper]](https://arxiv.org/abs/2408.10188)
[[code]](https://github.com/NVlabs/VILA/blob/main/LongVILA.md)

**(*arXiv2024_Video-CCAM*) Video-CCAM: Enhancing Video-Language Understanding with Causal Cross-Attention Masks for Short and Long Videos.** <br>
*Jiajun Fei, Dian Li, Zhidong Deng, Zekun Wang, Gang Liu, Hui Wang.*<br>
[[paper]](https://arxiv.org/abs/2408.14023)
[[code]](https://github.com/qq-mm/video-ccam)

**(*arXiv2024_LongLLaVA*) LongLLaVA: Scaling Multi-modal LLMs to 1000 Images Efficiently via a Hybrid Architecture.** <br>
*Xidong Wang, Dingjie Song, Shunian Chen, Chen Zhang, Benyou Wang.*<br>
[[paper]](https://arxiv.org/abs/2409.02889)
[[code]](https://github.com/freedomintelligence/longllava)

**(*arXiv2024_Video-XL*) Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding.** <br>
*Yan Shu, Peitian Zhang, Zheng Liu, Minghao Qin, Junjie Zhou, Tiejun Huang, Bo Zhao.*<br>
[[paper]](https://arxiv.org/abs/2409.14485)

**(*arXiv2024_LLaVA-Video*) Video Instruction Tuning With Synthetic Data.** <br>
*Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, Chunyuan Li.*<br>
[[paper]](https://arxiv.org/abs/2410.02713)
[[code]](https://llava-vl.github.io/blog/2024-09-30-llava-video/)

**(*arXiv2024_AuroraCap*) AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark.** <br>
*Wenhao Chai, Enxin Song, Yilun Du, Chenlin Meng, Vashisht Madhavan, Omer Bar-Tal, Jeng-Neng Hwang, Saining Xie, Christopher D. Manning.*<br>
[[paper]](https://arxiv.org/abs/2410.03051)
[[code]](https://rese1f.github.io/aurora-web/)

**(*arXiv2024_BLIP-3-Video*) xGen-MM-Vid (BLIP-3-Video): You Only Need 32 Tokens to Represent a Video Even in VLMs.** <br>
*Michael S. Ryoo, Honglu Zhou, Shrikant Kendre, Can Qin, Le Xue, Manli Shu, Silvio Savarese, Ran Xu, Caiming Xiong, Juan Carlos Niebles.*<br>
[[paper]](https://arxiv.org/abs/2410.16267)
[[code]](https://www.salesforceairesearch.com/opensource/xGen-MM-Vid/index.html)

**(*arXiv2024_LongVU*) LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding.** <br>
*Meta AI.*<br>
[[paper]](https://arxiv.org/abs/2410.17434)
[[code]](https://github.com/Vision-CAIR/LongVU)

**(*arXiv2024_VISTA*) VISTA: Enhancing Long-Duration and High-Resolution Video Understanding by Video Spatiotemporal Augmentation.** <br>
*Weiming Ren, Huan Yang, Jie Min, Cong Wei, Wenhu Chen.*<br>
[[paper]](https://arxiv.org/abs/2412.00927)
[[code]](https://tiger-ai-lab.github.io/VISTA/)

**(*arXiv2024_InternLM-XComposer2.5-OmniLive*) InternLM-XComposer2.5-OmniLive: A Comprehensive Multimodal System for Long-term Streaming Video and Audio Interactions.** <br>
*Shanghai AI Laboratory.*<br>
[[paper]](https://arxiv.org/abs/2412.09596)
[[code]](https://github.com/InternLM/InternLM-XComposer/tree/main/InternLM-XComposer-2.5-OmniLive)

**(*arXiv2024_FastVLM*) FastVLM: Efficient Vision Encoding for Vision Language Models.** <br>
*Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokul Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, Hadi Pouransari.*<br>
[[paper]](https://arxiv.org/abs/2412.13303)

**(*arXiv2024_Video-Panda*) Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models.** <br>
*Jinhui Yi, Syed Talal Wasim, Yanan Luo, Muzammal Naseer, Juergen Gall.*<br>
[[paper]](https://arxiv.org/abs/2412.18609)
[[code]](https://github.com/jh-yi/video-panda)

**(*arXiv2025_VITA-1.5*) VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction.** <br>
*Chaoyou Fu, Haojia Lin, Xiong Wang, Yi-Fan Zhang, Yunhang Shen, Xiaoyu Liu, Haoyu Cao, Zuwei Long, Heting Gao, Ke Li, Long Ma, Xiawu Zheng, Rongrong Ji, Xing Sun, Caifeng Shan, Ran He.*<br>
[[paper]](https://arxiv.org/abs/2501.01957)
[[code]](https://github.com/VITA-MLLM/VITA)

**(*arXiv2025_VideoLLaMA-3*) VideoLLaMA 3: Frontier Multimodal Foundation Models for Image and Video Understanding.** <br>
*Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, Peng Jin, Wenqi Zhang, Fan Wang, Lidong Bing, Deli Zhao.*<br>
[[paper]](https://arxiv.org/abs/2501.13106)
[[code]](https://github.com/DAMO-NLP-SG/VideoLLaMA3)

**(*arXiv2025_Ola*) Ola: Pushing the Frontiers of Omni-Modal Language Model with Progressive Modality Alignment.** <br>
*Zuyan Liu, Yuhao Dong, Jiahui Wang, Ziwei Liu, Winston Hu, Jiwen Lu, Yongming Rao.*<br>
[[paper]](https://arxiv.org/abs/2502.04328)
[[code]](https://github.com/Ola-Omni/Ola)

**(*arXiv2025_VideoRoPE*) VideoRoPE: What Makes for Good Video Rotary Position Embedding?.** <br>
*Xilin Wei, Xiaoran Liu, Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Jian Tong, Haodong Duan, Qipeng Guo, Jiaqi Wang, Xipeng Qiu, Dahua Lin.*<br>
[[paper]](https://arxiv.org/abs/2502.05173)
[[code]](https://github.com/Wiselnn570/VideoRoPE)

**(*arXiv2025_Pixel-Reasoner*) Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning.** <br>
*Alex Su, Haozhe Wang, Weiming Ren, Fangzhen Lin, Wenhu Chen.*<br>
[[paper]](https://arxiv.org/abs/2505.15966)

**(*arXiv2025_SynPO*) SynPO: Synergizing Descriptiveness and Preference Optimization for Video Detailed Captioning.** <br>
*Jisheng Dang, Yizhou Zhang, Hao Ye, Teng Wang, Siming Chen, Huicheng Zheng, Yulan Guo, Jianhuang Lai, Bin Hu.*<br>
[[paper]](https://arxiv.org/abs/2506.00835)

**(*arXiv2025_VideoCap-R1*) VideoCap-R1: Enhancing MLLMs for Video Captioning via Structured Thinking.** <br>
*Desen Meng, Rui Huang, Zhilin Dai, Xinhao Li, Yifan Xu, Jun Zhang, Zhenpeng Huang, Meng Zhang, Lingshu Zhang, Yi Liu, Limin Wang.*<br>
[[paper]](https://arxiv.org/abs/2506.01725)

**(*arXiv2025_VGR*) VGR: Visual Grounded Reasoning.** <br>
*Jiacong Wang, Zijian Kang, Haochen Wang, Haiyong Jiang, Jiawen Li, Bohong Wu, Ya Wang, Jiao Ran, Xiao Liang, Chao Feng, Jun Xiao.*<br>
[[paper]](https://arxiv.org/abs/2506.11991)

**(*arXiv2025_ARC-Hunyuan-Video-7B*) ARC-Hunyuan-Video-7B: Structured Video Comprehension of Real-World Shorts.** <br>
*Yuying Ge, Yixiao Ge, Chen Li, Teng Wang, Junfu Pu, Yizhuo Li, Lu Qiu, Jin Ma, Lisheng Duan, Xinyu Zuo, Jinwen Luo, Weibo Gu, Zexuan Li, Xiaojing Zhang, Yangyu Tao, Han Hu, Di Wang, Ying Shan.*<br>
[[paper]](https://arxiv.org/abs/2507.20939)
[[code]](https://tencentarc.github.io/posts/arc-video-announcement/)



### ``*Large MMM for Generation*``

#### ``*Class Generation*``

**(*ICML2020_iGPT*) Generative Pretraining From Pixels.** <br>
*Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, Ilya Sutskever.*<br>
[[paper]](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)

**(*NeurIPS2020_DDPM*) Denoising Diffusion Probabilistic Models.** <br>
*Jonathan Ho, Ajay Jain, Pieter Abbeel.*<br>
[[paper]](https://arxiv.org/abs/2006.11239)
[[code]](https://github.com/hojonathanho/diffusion)

**(*ICLR2021_DDIM*) Denoising Diffusion Implicit Models.** <br>
*Jiaming Song, Chenlin Meng, Stefano Ermon.*<br>
[[paper]](https://arxiv.org/abs/2010.02502)
[[code]](https://github.com/ermongroup/ddim)

**(*CVPR2022_MaskGIT*) MaskGIT: Masked Generative Image Transformer.** <br>
*Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, William T. Freeman.*<br>
[[paper]](https://arxiv.org/abs/2202.04200)
[[code]](https://github.com/google-research/maskgit)

**(*ICLR2023_Rectified-Flow*) Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.** <br>
*Xingchao Liu, Chengyue Gong, Qiang Liu.*<br>
[[paper]](https://arxiv.org/abs/2209.03003)
[[code]](https://github.com/gnobitab/RectifiedFlow)

**(*ICCV2023_DiT*) Scalable Diffusion Models with Transformers.** <br>
*William Peebles, Saining Xie.*<br>
[[paper]](https://arxiv.org/abs/2212.09748)
[[code]](https://www.wpeebles.com/DiT)

**(*arXiv2023_GIVT*) GIVT: Generative Infinite-Vocabulary Transformers.** <br>
*Michael Tschannen, Cian Eastwood, Fabian Mentzer.*<br>
[[paper]](https://arxiv.org/abs/2312.02116)
[[code]](https://github.com/google-research/big_vision)

**(*ECCV2024_SiT*) SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers.** <br>
*Nanye Ma, Mark Goldstein, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, Saining Xie.*<br>
[[paper]](https://arxiv.org/abs/2401.08740)
[[code]](https://github.com/willisma/SiT)

**(*arXiv2024_FiT*) FiT: Flexible Vision Transformer for Diffusion Model.** <br>
*Zeyu Lu, Zidong Wang, Di Huang, Chengyue Wu, Xihui Liu, Wanli Ouyang, Lei Bai.*<br>
[[paper]](https://arxiv.org/abs/2402.12376)
[[code]](https://github.com/whlzy/FiT)

**(*CVPR2024_V2T-Tokenizer*) Beyond Text: Frozen Large Language Models in Visual Signal Comprehension.** <br>
*Lei Zhu, Fangyun Wei, Yanye Lu.*<br>
[[paper]](https://arxiv.org/abs/2403.07874)
[[code]](https://github.com/zh460045050/V2L-Tokenizer)

**(*arXiv2024_VAR*) Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction.** <br>
*Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, Liwei Wang.*<br>
[[paper]](https://arxiv.org/abs/2404.02905)
[[code]](https://github.com/FoundationVision/VAR)

**(*arXiv2024_LlamaGen*) Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation.** <br>
*Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, Zehuan Yuan.*<br>
[[paper]](https://arxiv.org/abs/2406.06525)
[[code]](https://github.com/FoundationVision/LlamaGen)

**(*NeurIPS2024_LI-DiT*) Exploring the Role of Large Language Models in Prompt Encoding for Diffusion Models.** <br>
*Bingqi Ma, Zhuofan Zong, Guanglu Song, Hongsheng Li, Yu Liu.*<br>
[[paper]](https://arxiv.org/abs/2406.11831)

**(*NeurIPS2024_MAR*) Autoregressive Image Generation without Vector Quantization.** <br>
*Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, Kaiming He.*<br>
[[paper]](https://arxiv.org/abs/2406.11838)
[[code]](https://github.com/LTH14/mar)

**(*ICLR2025_REPA*) Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think.** <br>
*Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, Saining Xie.*<br>
[[paper]](https://arxiv.org/abs/2410.06940)
[[code]](https://sihyun.me/REPA)

**(*ICLR2025_DC-AE*) Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models.** <br>
*Junyu Chen, Han Cai, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, Yao Lu, Song Han.*<br>
[[paper]](https://arxiv.org/abs/2410.10733)
[[code]](https://github.com/mit-han-lab/efficientvit)

**(*arXiv2024_RAR*) Randomized Autoregressive Visual Generation.** <br>
*Qihang Yu, Ju He, Xueqing Deng, Xiaohui Shen, Liang-Chieh Chen.*<br>
[[paper]](https://arxiv.org/abs/2411.00776)
[[code]](https://github.com/bytedance/1d-tokenizer)

**(*arXiv2024_RandAR*) RandAR: Decoder-only Autoregressive Visual Generation in Random Orders.** <br>
*Ziqi Pang, Tianyuan Zhang, Fujun Luan, Yunze Man, Hao Tan, Kai Zhang, William T. Freeman, Yu-Xiong Wang.*<br>
[[paper]](https://arxiv.org/abs/2412.01827)
[[code]](https://rand-ar.github.io/)

**(*arXiv2024_ACDiT*) ACDiT: Interpolating Autoregressive Conditional Modeling and Diffusion Transformer.** <br>
*Jinyi Hu, Shengding Hu, Yuxuan Song, Yufei Huang, Mingxuan Wang, Hao Zhou, Zhiyuan Liu, Wei-Ying Ma, Maosong Sun.*<br>
[[paper]](https://arxiv.org/abs/2412.07720)
[[code]](https://github.com/thunlp/acdit)

**(*arXiv2024_LatentLM*) Multimodal Latent Language Modeling with Next-Token Diffusion.** <br>
*Yutao Sun, Hangbo Bao, Wenhui Wang, Zhiliang Peng, Li Dong, Shaohan Huang, Jianyong Wang, Furu Wei.*<br>
[[paper]](https://arxiv.org/abs/2412.08635)

**(*Blog2024_DMFM*) Diffusion Meets Flow Matching: Two Sides of the Same Coin.** <br>
*DeepMind Team.*<br>
[[blog]](https://diffusionflow.github.io/)
[[wechat]](https://mp.weixin.qq.com/s/BUaE2_VJwJi1VtNI3x-aIA)

**(*arXiv2024_PAR*) Parallelized Autoregressive Visual Generation.** <br>
*Yuqing Wang, Shuhuai Ren, Zhijie Lin, Yujin Han, Haoyuan Guo, Zhenheng Yang, Difan Zou, Jiashi Feng, Xihui Liu.*<br>
[[paper]](https://arxiv.org/abs/2412.15119)
[[code]](https://epiphqny.github.io/PAR-project)

**(*arXiv2024_FlowAR*) FlowAR: Scale-wise Autoregressive Image Generation Meets Flow Matching.** <br>
*Sucheng Ren, Qihang Yu, Ju He, Xiaohui Shen, Alan Yuille, Liang-Chieh Chen.*<br>
[[paper]](https://arxiv.org/abs/2412.15205)
[[code]](https://github.com/OliverRensu/FlowAR)

**(*arXiv2024_NPP*) Next Patch Prediction for Autoregressive Visual Generation.** <br>
*Yatian Pang, Peng Jin, Shuo Yang, Bin Lin, Bin Zhu, Zhenyu Tang, Liuhan Chen, Francis E. H. Tay, Ser-Nam Lim, Harry Yang, Li Yuan.*<br>
[[paper]](https://arxiv.org/abs/2412.15321)
[[code]](https://github.com/PKU-YuanGroup/Next-Patch-Prediction)

**(*arXiv2025_REPA-E*) REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers.** <br>
*Xingjian Leng, Jaskirat Singh, Yunzhong Hou, Zhenchang Xing, Saining Xie, Liang Zheng.*<br>
[[paper]](https://arxiv.org/abs/2504.10483)
[[code]](https://end2end-diffusion.github.io/)

**(*arXiv2025_VFMTok*) Vision Foundation Models as Effective Visual Tokenizers for Autoregressive Image Generation.** <br>
*Anlin Zheng, Xin Wen, Xuanyang Zhang, Chuofan Ma, Tiancai Wang, Gang Yu, Xiangyu Zhang, Xiaojuan Qi.*<br>
[[paper]](https://arxiv.org/abs/2507.08441)

**(*ICCV2025_DC-AE-1.5*) DC-AE 1.5: Accelerating Diffusion Model Convergence with Structured Latent Space.** <br>
*Junyu Chen, Dongyun Zou, Wenkun He, Junsong Chen, Enze Xie, Song Han, Han Cai.*<br>
[[paper]](https://arxiv.org/abs/2508.00413)
[[code]](https://github.com/dc-ai-projects/DC-Gen)

**(*ICCV2025_OneVAE*) OneVAE: Joint Discrete and Continuous Optimization Helps Discrete Video VAE Train Better.** <br>
*Yupeng Zhou, Zhen Li, Ziheng Ouyang, Yuming Chen, Ruoyi Du, Daquan Zhou, Bin Fu, Yihao Liu, Peng Gao, Ming-Ming Cheng, Qibin Hou.*<br>
[[paper]](https://arxiv.org/abs/2508.09857)
[[code]](https://github.com/HVision-NKU/OneVAE)


#### ``*Image Generation*``

**(*CVPR2022_LDM*) High-Resolution Image Synthesis with Latent Diffusion Models.** <br>
*Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer.*<br>
[[paper]](https://arxiv.org/abs/2112.10752)
[[code]](https://github.com/CompVis/latent-diffusion)

**(*ICCV2023_ControlNet*) Adding Conditional Control to Text-to-Image Diffusion Models.** <br>
*Lvmin Zhang, Anyi Rao, Maneesh Agrawala.*<br>
[[paper]](https://arxiv.org/abs/2302.05543)
[[code]](https://github.com/lllyasviel/ControlNet)

**(*CVPR2023_GigaGAN*) Scaling up GANs for Text-to-Image Synthesis.** <br>
*Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, Taesung Park.*<br>
[[paper]](https://arxiv.org/abs/2303.05511)
[[code]](https://mingukkang.github.io/GigaGAN/)

**(*NeurIPS2023_GILL*) Generating Images with Multimodal Language Models.** <br>
*Jing Yu Koh, Daniel Fried, Ruslan Salakhutdinov.*<br>
[[paper]](https://arxiv.org/abs/2305.17216)
[[code]](http://jykoh.com/gill)

**(*arXiv2024_Emu*) Emu: Enhancing Image Generation Models Using Photogenic Needles in a Haystack.** <br>
*GenAI Team.*<br>
[[paper]](https://arxiv.org/abs/2309.15807)

**(*ICLR2024_PixArt-α*) PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis.** <br>
*Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, Zhenguo Li.*<br>
[[paper]](https://arxiv.org/abs/2310.00426)
[[code]](https://pixart-alpha.github.io/)

**(*CVPR2024_Powers-of-Ten*) Generative Powers of Ten.** <br>
*Xiaojuan Wang, Janne Kontkanen, Brian Curless, Steve Seitz, Ira Kemelmacher, Ben Mildenhall, Pratul Srinivasan, Dor Verbin, Aleksander Holynski.*<br>
[[paper]](https://arxiv.org/abs/2312.02149)
[[code]](https://powers-of-10.github.io/)

**(*ICMLw2024_PIXART-δ*) PIXART-δ: Fast and Controllable Image Generation with Latent Consistency Models.** <br>
*Junsong Chen, Yue Wu, Simian Luo, Enze Xie, Sayak Paul, Ping Luo, Hang Zhao, Zhenguo Li.*<br>
[[paper]](https://arxiv.org/abs/2401.05252)
[[code]](https://github.com/PixArt-alpha/PixArt-alpha)

**(*arXiv2024_CoBSAT*) Can MLLMs Perform Text-to-Image In-Context Learning?.** <br>
*Yuchen Zeng, Wonjun Kang, Yicong Chen, Hyung Il Koo, Kangwook Lee.*<br>
[[paper]](https://arxiv.org/abs/2402.01293)
[[code]](https://github.com/UW-Madison-Lee-Lab/CoBSAT)

**(*arXiv2024_SD3*) Scaling Rectified Flow Transformers for High-Resolution Image Synthesis.** <br>
*Stability AI.*<br>
[[paper]](https://arxiv.org/abs/2403.03206)
[[code]](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)

**(*ECCV2024_PixArt-Σ*) PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation.** <br>
*Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, Zhenguo Li.*<br>
[[paper]](https://arxiv.org/abs/2403.04692)
[[code]](https://pixart-alpha.github.io/PixArt-sigma-project/)

**(*ECCV2024_ZigMa*) ZigMa: A DiT-style Zigzag Mamba Diffusion Model.** <br>
*Vincent Tao Hu, Stefan Andreas Baumann, Ming Gui, Olga Grebenkova, Pingchuan Ma, Johannes Fischer, Björn Ommer.*<br>
[[paper]](https://arxiv.org/abs/2403.13802)
[[code]](https://github.com/CompVis/zigma)

**(*arXiv2024_Ctrl-Adapter*) Ctrl-Adapter: An Efficient and Versatile Framework for Adapting Diverse Controls to Any Diffusion Model.** <br>
*Han Lin, Jaemin Cho, Abhay Zala, Mohit Bansal.*<br>
[[paper]](https://arxiv.org/abs/2404.09967)
[[code]](https://ctrl-adapter.github.io/)

**(*arXiv2024_Lumina-T2X*) Lumina-T2X: Transforming Text into Any Modality, Resolution, and Duration via Flow-based Large Diffusion Transformers.** <br>
*Peng Gao, Le Zhuo, Dongyang Liu, Ruoyi Du, Xu Luo, Longtian Qiu, Yuhang Zhang, Chen Lin, Rongjie Huang, Shijie Geng, Renrui Zhang, Junlin Xi, Wenqi Shao, Zhengkai Jiang, Tianshuo Yang, Weicai Ye, He Tong, Jingwen He, Yu Qiao, Hongsheng Li.*<br>
[[paper]](https://arxiv.org/abs/2405.05945)
[[code]](https://github.com/Alpha-VLLM/Lumina-T2X)

**(*arXiv2024_CDF*) Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion.** <br>
*Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, Vincent Sitzmann.*<br>
[[paper]](https://arxiv.org/abs/2407.01392)
[[code]](https://github.com/buoyancy99/diffusion-forcing)

**(*arXiv2024_MARS*) MARS: Mixture of Auto-Regressive Models for Fine-grained Text-to-image Synthesis.** <br>
*Wanggui He, Siming Fu, Mushui Liu, Xierui Wang, Wenyi Xiao, Fangxun Shu, Yi Wang, Lei Zhang, Zhelun Yu, Haoyuan Li, Ziwei Huang, LeiLei Gan, Hao Jiang.*<br>
[[paper]](https://arxiv.org/abs/2407.07614)
[[code]](https://github.com/fusiming3/mars)

**(*arXiv2024_MELLE*) Autoregressive Speech Synthesis without Vector Quantization.** <br>
*Lingwei Meng, Long Zhou, Shujie Liu, Sanyuan Chen, Bing Han, Shujie Hu, Yanqing Liu, Jinyu Li, Sheng Zhao, Xixin Wu, Helen Meng, Furu Wei.*<br>
[[paper]](https://arxiv.org/abs/2407.08551)
[[code]](https://aka.ms/melle)

**(*arXiv2024_PGv3*) Playground v3: Improving Text-to-Image Alignment with Deep-Fusion Large Language Models.** <br>
*Playground Research.*<br>
[[paper]](https://arxiv.org/abs/2409.10695)

**(*arXiv2024_OmniGen*) OmniGen: Unified Image Generation.** <br>
*Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, Zheng Liu.*<br>
[[paper]](https://arxiv.org/abs/2409.11340)
[[code]](https://github.com/VectorSpaceLab/OmniGen)

**(*arXiv2024_ControlAR*) ControlAR: Controllable Image Generation with Autoregressive Models.** <br>
*Zongming Li, Tianheng Cheng, Shoufa Chen, Peize Sun, Haocheng Shen, Longjin Ran, Xiaoxin Chen, Wenyu Liu, Xinggang Wang.*<br>
[[paper]](https://arxiv.org/abs/2410.02705)
[[code]](https://github.com/hustvl/ControlAR)

**(*arXiv2024_Meissonic*) Meissonic: Revitalizing Masked Generative Transformers for Efficient High-Resolution Text-to-Image Synthesis.** <br>
*Jinbin Bai, Tian Ye, Wei Chow, Enxin Song, Qing-Guo Chen, Xiangtai Li, Zhen Dong, Lei Zhu, Shuicheng Yan.*<br>
[[paper]](https://arxiv.org/abs/2410.08261)
[[code]](https://github.com/viiika/Meissonic)

**(*ICLR2025_SANA*) SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers.** <br>
*Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, Song Han.*<br>
[[paper]](https://arxiv.org/abs/2410.10629)
[[code]](https://github.com/NVlabs/Sana)

**(*arXiv2024_ZipAR*) ZipAR: Accelerating Autoregressive Image Generation through Spatial Locality.** <br>
*Yefei He, Feng Chen, Yuanyu He, Shaoxuan He, Hong Zhou, Kaipeng Zhang, Bohan Zhuang.*<br>
[[paper]](https://arxiv.org/abs/2412.04062)
[[code]](https://github.com/ThisisBillhe/ZipAR)

**(*arXiv2024_Infinity*) Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis.** <br>
*Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, Xiaobing Liu.*<br>
[[paper]](https://arxiv.org/abs/2412.04431)
[[code]](https://github.com/FoundationVision/Infinity)

**(*arXiv2024_UniReal*) UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics.** <br>
*Xi Chen, Zhifei Zhang, He Zhang, Yuqian Zhou, Soo Ye Kim, Qing Liu, Yijun Li, Jianming Zhang, Nanxuan Zhao, Yilin Wang, Hui Ding, Zhe Lin, Hengshuang Zhao.*<br>
[[paper]](https://arxiv.org/abs/2412.07774)
[[code]](https://xavierchen34.github.io/UniReal-Page/)

**(*arXiv2025_IMAGINE-E*) IMAGINE-E: Image Generation Intelligence Evaluation of State-of-the-art Text-to-Image Models.** <br>
*Jiayi Lei, Renrui Zhang, Xiangfei Hu, Weifeng Lin, Zhen Li, Wenjian Sun, Ruoyi Du, Le Zhuo, Zhongyu Li, Xinyue Li, Shitian Zhao, Ziyu Guo, Yiting Lu, Peng Gao, Hongsheng Li.*<br>
[[paper]](https://arxiv.org/abs/2501.13920)
[[code]](https://github.com/jylei16/Imagine-e)

**(*CVPR2025_PARM*) Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step.** <br>
*Ziyu Guo, Renrui Zhang, Chengzhuo Tong, Zhizheng Zhao, Rui Huang, Haoquan Zhang, Manyuan Zhang, Jiaming Liu, Shanghang Zhang, Peng Gao, Hongsheng Li, Pheng-Ann Heng.*<br>
[[paper]](https://arxiv.org/abs/2501.13926)
[[code]](https://github.com/ZiyuGuo99/Image-Generation-CoT)

**(*ICML2025_SANA-1.5*) SANA 1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer.** <br>
*Enze Xie, Junsong Chen, Yuyang Zhao, Jincheng Yu, Ligeng Zhu, Chengyue Wu, Yujun Lin, Zhekai Zhang, Muyang Li, Junyu Chen, Han Cai, Bingchen Liu, Daquan Zhou, Song Han.*<br>
[[paper]](https://arxiv.org/abs/2501.18427)
[[code]](https://github.com/NVlabs/Sana)

**(*arXiv2025_SANA-Sprint*) SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation.** <br>
*Junsong Chen, Shuchen Xue, Yuyang Zhao, Jincheng Yu, Sayak Paul, Junyu Chen, Han Cai, Song Han, Enze Xie.*<br>
[[paper]](https://arxiv.org/abs/2503.09641)
[[code]](https://github.com/NVlabs/Sana)

**(*arXiv2025_LeX-Art*) LeX-Art: Rethinking Text Generation via Scalable High-Quality Data Synthesis.** <br>
*Shitian Zhao, Qilong Wu, Xinyue Li, Bo Zhang, Ming Li, Qi Qin, Dongyang Liu, Kaipeng Zhang, Hongsheng Li, Yu Qiao, Peng Gao, Bin Fu, Zhen Li.*<br>
[[paper]](https://arxiv.org/abs/2503.21749)
[[code]](https://zhaoshitian.github.io/lexart/)

**(*arXiv2025_Lumina-Image-2.0*) Lumina-Image 2.0: A Unified and Efficient Image Generative Framework.** <br>
*Qi Qin, Le Zhuo, Yi Xin, Ruoyi Du, Zhen Li, Bin Fu, Yiting Lu, Jiakang Yuan, Xinyue Li, Dongyang Liu, Xiangyang Zhu, Manyuan Zhang, Will Beddow, Erwann Millon, Victor Perez, Wenhai Wang, Conghui He, Bo Zhang, Xiaohong Liu, Hongsheng Li, Yu Qiao, Chang Xu, Peng Gao.*<br>
[[paper]](https://arxiv.org/abs/2503.21758)
[[code]](https://github.com/Alpha-VLLM/Lumina-Image-2.0)

**(*arXiv2025_PosterMaker*) PosterMaker: Towards High-Quality Product Poster Generation with Accurate Text Rendering.** <br>
*Yifan Gao, Zihang Lin, Chuanbin Liu, Min Zhou, Tiezheng Ge, Bo Zheng, Hongtao Xie.*<br>
[[paper]](https://arxiv.org/abs/2504.06632)
[[code]](https://poster-maker.github.io/)

**(*arXiv2025_DDT-LLaMA*) Generative Multimodal Pretraining with Discrete Diffusion Timestep Tokens.** <br>
*Kaihang Pan, Wang Lin, Zhongqi Yue, Tenglong Ao, Liyu Jia, Wei Zhao, Juncheng Li, Siliang Tang, Hanwang Zhang.*<br>
[[paper]](https://arxiv.org/abs/2504.14666)
[[code]](https://ddt-llama.github.io/)

**(*arXiv2025_Flow-GRPO*) Flow-GRPO: Training Flow Matching Models via Online RL.** <br>
*Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, Wanli Ouyang.*<br>
[[paper]](https://arxiv.org/abs/2505.05470)
[[code]](https://github.com/yifan123/flow_grpo)

**(*arXiv2025_Selftok*) Selftok: Discrete Visual Tokens of Autoregression, by Diffusion, and for Reasoning.** <br>
*Bohan Wang, Zhongqi Yue, Fengda Zhang, Shuo Chen, Li'an Bi, Junzhe Zhang, Xue Song, Kennard Yanting Chan, Jiachun Pan, Weijia Wu, Mingze Zhou, Wang Lin, Kaihang Pan, Saining Zhang, Liyu Jia, Wentao Hu, Wei Zhao, Hanwang Zhang.*<br>
[[paper]](https://arxiv.org/abs/2505.07538)
[[code]](https://selftok-team.github.io/report/)

**(*Blog2025_FLUX.1-Kontext*) Introducing FLUX.1 Kontext and the BFL Playground.** <br>
*Black Forest Labs.*<br>
[[blog]](https://bfl.ai/announcements/flux-1-kontext)
[[demo]](https://playground.bfl.ai/image/generate)

**(*arXiv2025_TwGI*) Thinking with Generated Images.** <br>
*Ethan Chern, Zhulin Hu, Steffi Chern, Siqi Kou, Jiadi Su, Yan Ma, Zhijie Deng, Pengfei Liu.*<br>
[[paper]](https://arxiv.org/abs/2505.22525)
[[code]](https://github.com/GAIR-NLP/thinking-with-generated-images)

**(*arXiv2025_Hita*) Hita: Holistic Tokenizer for Autoregressive Image Generation.** <br>
*Anlin Zheng, Haochen Wang, Yucheng Zhao, Weipeng Deng, Tiancai Wang, Xiangyu Zhang, Xiaojuan Qi.*<br>
[[paper]](https://arxiv.org/abs/2507.02358)
[[code]](https://github.com/CVMI-Lab/Hita)

**(*arXiv2025_Qwen-Image*) Qwen-Image Technical Report.** <br>
*Qwen Team.*<br>
[[paper]](https://arxiv.org/abs/2508.02324)
[[code]](https://github.com/QwenLM/Qwen-Image)

**(*arXiv2025_NextStep-1*) NextStep-1: Toward Autoregressive Image Generation with Continuous Tokens at Scale.** <br>
*NextStep-Team, StepFun.*<br>
[[paper]](https://arxiv.org/abs/2508.10711)
[[code]](https://github.com/stepfun-ai/NextStep-1)


#### ``*Video Generation*``

**(*arXiv2023_GAIA-1*) GAIA-1: A Generative World Model for Autonomous Driving.** <br>
*Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, Gianluca Corrado.*<br>
[[paper]](https://arxiv.org/abs/2309.17080)

**(*arXiv2024_Panda-70M*) Panda-70M: Captioning 70M Videos with Multiple Cross-Modality Teachers.** <br>
*Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, Sergey Tulyakov.*<br>
[[paper]](https://arxiv.org/abs/2402.19479)
[[code]](https://snap-research.github.io/Panda-70M)

**(*arXiv2024_Pyramid-Flow*) Pyramidal Flow Matching for Efficient Video Generative Modeling.** <br>
*Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, Zhouchen Lin.*<br>
[[paper]](https://arxiv.org/abs/2410.05954)
[[code]](https://github.com/jy0205/Pyramid-Flow)

**(*arXiv2024_Koala-36M*) Koala-36M: A Large-scale Video Dataset Improving Consistency between Fine-grained Conditions and Video Content.** <br>
*Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, Fei Yang, Pengfei Wan, Di Zhang.*<br>
[[paper]](https://arxiv.org/abs/2410.08260)
[[code]](https://koala36m.github.io/)

**(*arXiv2024_Movie-Gen*) Movie Gen: A Cast of Media Foundation Models.** <br>
*Movie Gen team.*<br>
[[paper]](https://arxiv.org/abs/2410.13720)
[[code]](https://go.fb.me/MovieGenResearchVideos)

**(*arXiv2024_HunyuanVideo*) HunyuanVideo: A Systematic Framework For Large Video Generative Models.** <br>
*Hunyuan Foundation Model Team.*<br>
[[paper]](https://arxiv.org/abs/2412.03603)
[[code]](https://github.com/Tencent/HunyuanVideo)

**(*arXiv2024_DiCoDe*) DiCoDe: Diffusion-Compressed Deep Tokens for Autoregressive Video Generation with Language Models.** <br>
*Yizhuo Li, Yuying Ge, Yixiao Ge, Ping Luo, Ying Shan.*<br>
[[paper]](https://arxiv.org/abs/2412.04446)
[[code]](https://liyz15.github.io/DiCoDe)

**(*arXiv2024_NOVA*) Autoregressive Video Generation without Vector Quantization.** <br>
*Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, Xinlong Wang.*<br>
[[paper]](https://arxiv.org/abs/2412.14169)
[[code]](https://github.com/baaivision/NOVA)

**(*arXiv2025_Lumina-Video*) Lumina-Video: Efficient and Flexible Video Generation with Multi-scale Next-DiT.** <br>
*Dongyang Liu, Shicheng Li, Yutong Liu, Zhen Li, Kai Wang, Xinyue Li, Qi Qin, Yufei Liu, Yi Xin, Zhongyu Li, Bin Fu, Chenyang Si, Yuewen Cao, Conghui He, Ziwei Liu, Yu Qiao, Qibin Hou, Hongsheng Li, Peng Gao.*<br>
[[paper]](https://arxiv.org/abs/2502.06782)
[[code]](https://www.github.com/Alpha-VLLM/Lumina-Video)

**(*arXiv2025_VACE*) VACE: All-in-One Video Creation and Editing.** <br>
*Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, Yu Liu.*<br>
[[paper]](https://arxiv.org/abs/2503.07598)
[[code]](https://ali-vilab.github.io/VACE-Page/)

**(*arXiv2025_FAR*) Long-Context Autoregressive Video Modeling with Next-Frame Prediction.** <br>
*Yuchao Gu, Weijia Mao, Mike Zheng Shou.*<br>
[[paper]](https://arxiv.org/abs/2503.19325)
[[code]](https://farlongctx.github.io/)

**(*arXiv2025_GAIA-2*) GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving.** <br>
*Lloyd Russell, Anthony Hu, Lorenzo Bertoni, George Fedoseev, Jamie Shotton, Elahe Arani, Gianluca Corrado.*<br>
[[paper]](https://arxiv.org/abs/2503.20523)

**(*arXiv2025_WORLDMEM*) WORLDMEM: Long-term Consistent World Simulation with Memory.** <br>
*Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, Xingang Pan.*<br>
[[paper]](https://arxiv.org/abs/2504.12369)
[[code]](https://xizaoqu.github.io/worldmem/)

**(*arXiv2025_MAGI-1*) MAGI-1: Autoregressive Video Generation at Scale.** <br>
*Sand AI.*<br>
[[paper]](https://arxiv.org/abs/2505.13211)
[[code]](https://github.com/SandAI-org/MAGI-1)

**(*arXiv2025_Frame-In-N-Out*) Frame In-N-Out: Unbounded Controllable Image-to-Video Generation.** <br>
*Boyang Wang, Xuweiyi Chen, Matheus Gadelha, Zezhou Cheng.*<br>
[[paper]](https://arxiv.org/abs/2505.21491)
[[code]](https://uva-computer-vision-lab.github.io/Frame-In-N-Out/)

**(*arXiv2025_NFD*) Playing with Transformer at 30+ FPS via Next-Frame Diffusion.** <br>
*Xinle Cheng, Tianyu He, Jiayi Xu, Junliang Guo, Di He, Jiang Bian.*<br>
[[paper]](https://arxiv.org/abs/2506.01380)

**(*Blog2025_Seedance-1.0*) Seedance 1.0: Exploring the Boundaries of Video Generation Models.** <br>
*ByteDance Seed.*<br>
[[paper]](https://arxiv.org/abs/2506.09113)
[[demo]](https://console.volcengine.com/ark/region:ark+cn-beijing/experience/vision?type=GenVideo)

**(*arXiv2025_CI-VID*) CI-VID: A Coherent Interleaved Text-Video Dataset.** <br>
*Yiming Ju, Jijin Hu, Zhengxiong Luo, Haoge Deng, hanyu Zhao, Li Du, Chengwei Wu, Donglin Hao, Xinlong Wang, Tengfei Pan.*<br>
[[paper]](https://arxiv.org/abs/2507.01938)
[[code]](https://github.com/ymju-BAAI/CI-VID)

**(*arXiv2025_Lumos-1*) Lumos-1: On Autoregressive Video Generation from a Unified Model Perspective.** <br>
*Hangjie Yuan, Weihua Chen, Jun Cen, Hu Yu, Jingyun Liang, Shuning Chang, Zhihui Lin, Tao Feng, Pengwei Liu, Jiazheng Xing, Hao Luo, Jiasheng Tang, Fan Wang, Yi Yang.*<br>
[[paper]](https://arxiv.org/abs/2507.08801)
[[code]](https://github.com/alibaba-damo-academy/Lumos)


### ``*Large MMM for Unification*``

**(*NeurIPS2023_CoDi*) Any-to-Any Generation via Composable Diffusion.** <br>
*Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, Mohit Bansal.*<br>
[[paper]](https://arxiv.org/abs/2305.11846)
[[code]](https://codi-gen.github.io/)

**(*ICLR2023_Emu*) Generative Pretraining in Multimodality.** <br>
*BAAI Team.*<br>
[[paper]](https://arxiv.org/abs/2307.05222)
[[code]](https://github.com/baaivision/Emu)

**(*ICLR2024_LaVIT*) Unified Language-Vision Pretraining in LLM with Dynamic Discrete Visual Tokenization.** <br>
*Yang Jin, Kun Xu, Kun Xu, Liwei Chen, Chao Liao, Jianchao Tan, Quzhe Huang, Bin Chen, Chenyi Lei, An Liu, Chengru Song, Xiaoqiang Lei, Di Zhang, Wenwu Ou, Kun Gai, Yadong Mu.*<br>
[[paper]](https://arxiv.org/abs/2309.04669)
[[code]](https://github.com/jy0205/LaVIT)

**(*ICML2024_NExT-GPT*) NExT-GPT: Any-to-Any Multimodal Large Language Model.** <br>
*Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, Tat-Seng Chua.*<br>
[[paper]](https://arxiv.org/abs/2309.05519)
[[code]](https://github.com/NExT-GPT/NExT-GPT)

**(*ICLR2024_DreamLLM*) DreamLLM: Synergistic Multimodal Comprehension and Creation.** <br>
*Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, Xiangwen Kong, Xiangyu Zhang, Kaisheng Ma, Li Yi.*<br>
[[paper]](https://arxiv.org/abs/2309.11499)
[[code]](https://dreamllm.github.io/)

**(*ICLR2024_SEED*) Making LLaMA SEE and Draw with SEED Tokenizer.** <br>
*Yuying Ge, Sijie Zhao, Ziyun Zeng, Yixiao Ge, Chen Li, Xintao Wang, Ying Shan.*<br>
[[paper]](https://arxiv.org/abs/2310.01218)
[[code]](https://github.com/ailab-cvc/seed)

**(*CVPR2024_OneLLM*) OneLLM: One Framework to Align All Modalities with Language.** <br>
*Jiaming Han, Kaixiong Gong, Yiyuan Zhang, Jiaqi Wang, Kaipeng Zhang, Dahua Lin, Yu Qiao, Peng Gao, Xiangyu Yue.*<br>
[[paper]](https://arxiv.org/abs/2312.03700)
[[code]](https://github.com/csuhan/OneLLM)

**(*arXiv2023_VL-GPT*) VL-GPT: A Generative Pre-trained Transformer for Vision and Language Understanding and Generation.** <br>
*Jinguo Zhu, Xiaohan Ding, Yixiao Ge, Yuying Ge, Sijie Zhao, Hengshuang Zhao, Xiaohua Wang, Ying Shan.*<br>
[[paper]](https://arxiv.org/abs/2312.09251)
[[code]](https://github.com/ailab-cvc/vl-gpt)

**(*arXiv2023_Gemini*) Gemini: A Family of Highly Capable Multimodal Models.** <br>
*Gemini Team.*<br>
[[paper]](https://arxiv.org/abs/2312.11805)

**(*CVPR2024_Emu2*) Generative Multimodal Models are In-Context Learners.** <br>
*BAAI Team.*<br>
[[paper]](https://arxiv.org/abs/2312.13286)
[[code]](https://baaivision.github.io/emu2)

**(*arXiv2023_Unified-IO-2*) Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action.** <br>
*Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, Aniruddha Kembhavi.*<br>
[[paper]](https://arxiv.org/abs/2312.17172)
[[code]](https://github.com/allenai/unified-io-2)

**(*arXiv2024_MM-Interleaved*) MM-Interleaved: Interleaved Image-Text Generative Modeling via Multi-modal Feature Synchronizer.** <br>
*Changyao Tian, Xizhou Zhu, Yuwen Xiong, Weiyun Wang, Zhe Chen, Wenhai Wang, Yuntao Chen, Lewei Lu, Tong Lu, Jie Zhou, Hongsheng Li, Yu Qiao, Jifeng Dai.*<br>
[[paper]](https://arxiv.org/abs/2401.10208)
[[code]](https://github.com/OpenGVLab/MM-Interleaved)

**(*arXiv2024_Video-LaVIT*) Video-LaVIT: Unified Video-Language Pre-training with Decoupled Visual-Motional Tokenization.** <br>
*Yang Jin, Zhicheng Sun, Kun Xu, Kun Xu, Liwei Chen, Hao Jiang, Quzhe Huang, Chengru Song, Yuliang Liu, Di Zhang, Yang Song, Kun Gai, Yadong Mu.*<br>
[[paper]](https://arxiv.org/abs/2402.03161)
[[code]](https://video-lavit.github.io/)

**(*arXiv2024_LWM*) World Model on Million-Length Video And Language With Blockwise RingAttention.** <br>
*Hao Liu, Wilson Yan, Matei Zaharia, Pieter Abbeel.*<br>
[[paper]](https://arxiv.org/abs/2402.08268)
[[code]](https://github.com/LargeWorldModel/LWM)

**(*CVPR2024_AnyGPT*) AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling.** <br>
*Jun Zhan, Junqi Dai, Jiasheng Ye, Yunhua Zhou, Dong Zhang, Zhigeng Liu, Xin Zhang, Ruibin Yuan, Ge Zhang, Linyang Li, Hang Yan, Jie Fu, Tao Gui, Tianxiang Sun, Yugang Jiang, Xipeng Qiu.*<br>
[[paper]](https://arxiv.org/abs/2402.12226)
[[code]](https://github.com/OpenMOSS/AnyGPT)

**(*arXiv2024_Mini-Gemini*) Mini-Gemini: Mining the Potential of Multi-modality Vision Language Models.** <br>
*Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, Jiaya Jia.*<br>
[[paper]](https://arxiv.org/abs/2403.18814)
[[code]](https://github.com/dvlab-research/MiniGemini)

**(*arXiv2024_SEED-X*) SEED-X: Multimodal Models with Unified Multi-granularity Comprehension and Generation.** <br>
*Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, Ying Shan.*<br>
[[paper]](https://arxiv.org/abs/2404.14396)
[[code]](https://github.com/AILab-CVC/SEED-X)

**(*arXiv2024_Chameleon*) Chameleon: Mixed-Modal Early-Fusion Foundation Models.** <br>
*Chameleon Team.*<br>
[[paper]](https://arxiv.org/abs/2405.09818)

**(*arXiv2024_X-VILA*) X-VILA: Cross-Modality Alignment for Large Language Model.** <br>
*Hanrong Ye, De-An Huang, Yao Lu, Zhiding Yu, Wei Ping, Andrew Tao, Jan Kautz, Song Han, Dan Xu, Pavlo Molchanov, Hongxu Yin.*<br>
[[paper]](https://arxiv.org/abs/2405.19335)

**(*NeurIPS2024_Vitron*) Vitron: A Unified Pixel-level Vision LLM for Understanding, Generating, Segmenting, Editing.** <br>
*Hao Fei, Shengqiong Wu, Hanwang Zhang, Tat-Seng Chua, Shuicheng Yan.*<br>
[[paper]](https://haofei.vip/downloads/papers/Skywork_Vitron_2024.pdf)
[[code]](https://github.com/SkyworkAI/Vitron)

**(*arXiv2024_ANOLE*) ANOLE: An Open, Autoregressive, Native Large Multimodal Models for Interleaved Image-Text Generation.** <br>
*Ethan Chern, Jiadi Su, Yan Ma, Pengfei Liu.*<br>
[[paper]](https://arxiv.org/abs/2407.06135)
[[code]](https://github.com/gair-nlp/anole)

**(*arXiv2024_SEED-Story*) SEED-Story: Multimodal Long Story Generation with Large Language Model.** <br>
*Shuai Yang, Yuying Ge, Yang Li, Yukang Chen, Yixiao Ge, Ying Shan, Yingcong Chen.*<br>
[[paper]](https://arxiv.org/abs/2407.08683)
[[code]](https://github.com/TencentARC/SEED-Story)

**(*arXiv2024_Transfusion*) Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model.** <br>
*Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, Omer Levy.*<br>
[[paper]](https://arxiv.org/abs/2408.11039v1)

**(*arXiv2024_Show-o*) Show-o: One Single Transformer to Unify Multimodal Understanding and Generation.** <br>
*Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, Mike Zheng Shou.*<br>
[[paper]](https://arxiv.org/abs/2408.12528)
[[code]](https://github.com/showlab/Show-o)

**(*arXiv2024_VILA-U*) VILA-U: a Unified Foundation Model Integrating Visual Understanding and Generation.** <br>
*Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, Song Han, Yao Lu.*<br>
[[paper]](https://arxiv.org/abs/2409.04429)

**(*arXiv2024_MonoFormer*) MonoFormer: One Transformer for Both Diffusion and Autoregression.** <br>
*Chuyang Zhao, Yuxing Song, Wenhao Wang, Haocheng Feng, Errui Ding, Yifan Sun, Xinyan Xiao, Jingdong Wang.*<br>
[[paper]](https://arxiv.org/abs/2409.16280)
[[code]](https://monoformer.github.io/)

**(*arXiv2024_MIO*) MIO: A Foundation Model on Multimodal Tokens.** <br>
*Zekun Wang, King Zhu, Chunpu Xu, Wangchunshu Zhou, Jiaheng Liu, Yibo Zhang, Jiashuo Wang, Ning Shi, Siyu Li, Yizhi Li, Haoran Que, Zhaoxiang Zhang, Yuanxing Zhang, Ge Zhang, Ke Xu, Jie Fu, Wenhao Huang.*<br>
[[paper]](https://arxiv.org/abs/2409.17692)

**(*arXiv2024_Emu3*) Emu3: Next-Token Prediction is All You Need.** <br>
*BAAI Team.*<br>
[[paper]](https://arxiv.org/abs/2409.18869)
[[code]](https://emu.baai.ac.cn/)

**(*arXiv2024_MMAR*) MMAR: Towards Lossless Multi-Modal Auto-Regressive Probabilistic Modeling.** <br>
*Jian Yang, Dacheng Yin, Yizhou Zhou, Fengyun Rao, Wei Zhai, Yang Cao, Zheng-Jun Zha.*<br>
[[paper]](https://arxiv.org/abs/2410.10798)
[[code]](https://github.com/ydcUstc/MMAR)

**(*arXiv2024_Janus*) Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation.** <br>
*DeepSeek-AI Team.*<br>
[[paper]](https://arxiv.org/abs/2410.13848)
[[code]](https://github.com/deepseek-ai/janus)

**(*arXiv2024_MotionGPT-2*) MotionGPT-2: A General-Purpose Motion-Language Model for Motion Generation and Understanding.** <br>
*Yuan Wang, Di Huang, Yaqi Zhang, Wanli Ouyang, Jile Jiao, Xuetao Feng, Yan Zhou, Pengfei Wan, Shixiang Tang, Dan Xu.*<br>
[[paper]](https://arxiv.org/abs/2410.21747)

**(*arXiv2024_JanusFlow*) JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation.** <br>
*DeepSeek-AI Team.*<br>
[[paper]](https://arxiv.org/abs/2411.07975)
[[code]](https://github.com/deepseek-ai/janus)

**(*arXiv2024_Spider*) Spider: Any-to-Many Multimodal LLM.** <br>
*Jinxiang Lai, Jie Zhang, Jun Liu, Jian Li, Xiaocheng Lu, Song Guo.*<br>
[[paper]](https://arxiv.org/abs/2411.09439)

**(*arXiv2024_MUSE-VL*) MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding.** <br>
*Rongchang Xie, Chen Du, Ping Song, Chang Liu.*<br>
[[paper]](https://arxiv.org/abs/2411.17762)

**(*arXiv2024_JetFormer*) JetFormer: An Autoregressive Generative Model of Raw Images and Text.** <br>
*Michael Tschannen, André Susano Pinto, Alexander Kolesnikov.*<br>
[[paper]](https://arxiv.org/abs/2411.19722)

**(*arXiv2024_Orthus*) Orthus: Autoregressive Interleaved Image-Text Generation with Modality-Specific Heads.** <br>
*Siqi Kou, Jiachun Jin, Chang Liu, Ye Ma, Jian Jia, Quan Chen, Peng Jiang, Zhijie Deng.*<br>
[[paper]](https://arxiv.org/abs/2412.00127)

**(*arXiv2024_OmniFlow*) OmniFlow: Any-to-Any Generation with Multi-Modal Rectified Flows.** <br>
*Shufan Li, Konstantinos Kallidromitis, Akash Gokul, Zichun Liao, Yusuke Kato, Kazuki Kozuka, Aditya Grover.*<br>
[[paper]](https://arxiv.org/abs/2412.01169)
[[code]](https://github.com/jacklishufan/OmniFlows)

**(*arXiv2024_TokenFlow*) TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation.** <br>
*Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K. Du, Zehuan Yuan, Xinglong Wu.*<br>
[[paper]](https://arxiv.org/abs/2412.03069)
[[code]](https://byteflow-ai.github.io/TokenFlow/)

**(*arXiv2024_Liquid*) Liquid: Language Models are Scalable and Unified Multi-modal Generators.** <br>
*Junfeng Wu, Yi Jiang, Chuofan Ma, Yuliang Liu, Hengshuang Zhao, Zehuan Yuan, Song Bai, Xiang Bai.*<br>
[[paper]](https://arxiv.org/abs/2412.04332)
[[code]](https://github.com/FoundationVision/Liquid)

**(*arXiv2024_Divot*) Divot: Diffusion Powers Video Tokenizer for Comprehension and Generation.** <br>
*Yuying Ge, Yizhuo Li, Yixiao Ge, Ying Shan.*<br>
[[paper]](https://arxiv.org/abs/2412.04432)
[[code]](https://github.com/TencentARC/Divot)

**(*arXiv2024_SynerGen-VL*) SynerGen-VL: Towards Synergistic Image Understanding and Generation with Vision Experts and Token Folding.** <br>
*Hao Li, Changyao Tian, Jie Shao, Xizhou Zhu, Zhaokai Wang, Jinguo Zhu, Wenhan Dou, Xiaogang Wang, Hongsheng Li, Lewei Lu, Jifeng Dai.*<br>
[[paper]](https://arxiv.org/abs/2412.09604)

**(*arXiv2024_MetaMorph*) MetaMorph: Multimodal Understanding and Generation via Instruction Tuning.** <br>
*Shengbang Tong, David Fan, Jiachen Zhu, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, Zhuang Liu.*<br>
[[paper]](https://arxiv.org/abs/2412.14164)

**(*arXiv2024_LlamaFusion*) LlamaFusion: Adapting Pretrained Language Models for Multimodal Generation.** <br>
*Weijia Shi, Xiaochuang Han, Chunting Zhou, Weixin Liang, Xi Victoria Lin, Luke Zettlemoyer, Lili Yu.*<br>
[[paper]](https://arxiv.org/abs/2412.15188)

**(*arXiv2025_D-DiT*) Dual Diffusion for Unified Image Generation and Understanding.** <br>
*Zijie Li, Henry Li, Yichun Shi, Amir Barati Farimani, Yuval Kluger, Linjie Yang, Peng Wang.*<br>
[[paper]](https://arxiv.org/abs/2501.00289)

**(*arXiv2025_QLIP*) QLIP: Text-Aligned Visual Tokenization Unifies Auto-Regressive Multimodal Understanding and Generation.** <br>
*Yue Zhao, Fuzhao Xue, Scott Reed, Linxi Fan, Yuke Zhu, Jan Kautz, Zhiding Yu, Philipp Krähenbühl, De-An Huang.*<br>
[[paper]](https://arxiv.org/abs/2502.05178)
[[code]](https://nvlabs.github.io/QLIP/)

**(*arXiv2025_USP*) USP: Unified Self-Supervised Pretraining for Image Generation and Understanding.** <br>
*Xiangxiang Chu, Renda Li, Yong Wang.*<br>
[[paper]](https://arxiv.org/abs/2503.06132)
[[code]](https://github.com/AMAP-ML/USP)

**(*arXiv2025_OmniMamba*) OmniMamba: Efficient and Unified Multimodal Understanding and Generation via State Space Models.** <br>
*Jialv Zou, Bencheng Liao, Qian Zhang, Wenyu Liu, Xinggang Wang.*<br>
[[paper]](https://arxiv.org/abs/2503.08686)
[[code]](https://github.com/hustvl/OmniMamba)

**(*arXiv2025_UniFluid*) Unified Autoregressive Visual Generation and Understanding with Continuous Tokens.** <br>
*Lijie Fan, Luming Tang, Siyang Qin, Tianhong Li, Xuan Yang, Siyuan Qiao, Andreas Steiner, Chen Sun, Yuanzhen Li, Tao Zhu, Michael Rubinstein, Michalis Raptis, Deqing Sun, Radu Soricut.*<br>
[[paper]](https://arxiv.org/abs/2503.13436)

**(*arXiv2025_UniDisc*) Unified Multimodal Discrete Diffusion.** <br>
*Alexander Swerdlow, Mihir Prabhudesai, Siddharth Gandhi, Deepak Pathak, Katerina Fragkiadaki.*<br>
[[paper]](https://arxiv.org/abs/2503.20853)
[[code]](https://unidisc.github.io/)

**(*arXiv2025_UGen*) UGen: Unified Autoregressive Multimodal Model with Progressive Vocabulary Learning.** <br>
*Hongxuan Tang, Hao Liu, Xinyan Xiao.*<br>
[[paper]](https://arxiv.org/abs/2503.21193v1)

**(*arXiv2025_Harmon*) Harmonizing Visual Representations for Unified Multimodal Understanding and Generation.** <br>
*Size Wu, Wenwei Zhang, Lumin Xu, Sheng Jin, Zhonghua Wu, Qingyi Tao, Wentao Liu, Wei Li, Chen Change Loy.*<br>
[[paper]](https://arxiv.org/abs/2503.21979v1)
[[code]](https://github.com/wusize/Harmon)

**(*arXiv2025_ILLUME+*) ILLUME+: Illuminating Unified MLLM with Dual Visual Tokenization and Diffusion Refinement.** <br>
*Runhui Huang, Chunwei Wang, Junwei Yang, Guansong Lu, Yunlong Yuan, Jianhua Han, Lu Hou, Wei Zhang, Lanqing Hong, Hengshuang Zhao, Hang Xu.*<br>
[[paper]](https://arxiv.org/abs/2504.01934v1)
[[code]](https://illume-unified-mllm.github.io/)

**(*arXiv2025_MetaQuery*) Transfer between Modalities with MetaQueries.** <br>
*Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, Ji Hou, Saining Xie.*<br>
[[paper]](https://arxiv.org/abs/2504.06256)
[[code]](https://xichenpan.com/metaquery)

**(*arXiv2025_Mogao*) Mogao: An Omni Foundation Model for Interleaved Multi-Modal Generation.** <br>
*Chao Liao, Liyang Liu, Xun Wang, Zhengxiong Luo, Xinyu Zhang, Wenliang Zhao, Jie Wu, Liang Li, Zhi Tian, Weilin Huang.*<br>
[[paper]](https://arxiv.org/abs/2505.05472)

**(*arXiv2025_BLIP3-o*) BLIP3-o: A Family of Fully Open Unified Multimodal Models-Architecture, Training and Dataset.** <br>
*Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, Le Xue, Caiming Xiong, Ran Xu.*<br>
[[paper]](https://arxiv.org/abs/2505.09568)
[[code]](https://github.com/jiuhaichen/blip3o)

**(*arXiv2025_ETT*) End-to-End Vision Tokenizer Tuning.** <br>
*Wenxuan Wang, Fan Zhang, Yufeng Cui, Haiwen Diao, Zhuoyan Luo, Huchuan Lu, Jing Liu, Xinlong Wang.*<br>
[[paper]](https://arxiv.org/abs/2505.10562)

**(*arXiv2025_UniCTokens*) UniCTokens: Boosting Personalized Understanding and Generation via Unified Concept Tokens.** <br>
*Ruichuan An, Sihan Yang, Renrui Zhang, Zijun Shen, Ming Lu, Gaole Dai, Hao Liang, Ziyu Guo, Shilin Yan, Yulin Luo, Bocheng Zou, Chaoqun Yang, Wentao Zhang.*<br>
[[paper]](https://arxiv.org/abs/2505.14671v1)
[[code]](https://github.com/arctanxarc/UniCTokens)

**(*arXiv2025_BAGEL*) Emerging Properties in Unified Multimodal Pretraining.** <br>
*Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, Haoqi Fan.*<br>
[[paper]](https://arxiv.org/abs/2505.14683)
[[code]](https://bagel-ai.org/)

**(*arXiv2025_MMaDA*) MMaDA: Multimodal Large Diffusion Language Models.** <br>
*Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, Mengdi Wang.*<br>
[[paper]](https://arxiv.org/abs/2505.15809)
[[code]](https://github.com/Gen-Verse/MMaDA)

**(*arXiv2025_FUDOKI*) FUDOKI: Discrete Flow-based Unified Understanding and Generation via Kinetic-Optimal Velocities.** <br>
*Jin Wang, Yao Lai, Aoxue Li, Shifeng Zhang, Jiacheng Sun, Ning Kang, Chengyue Wu, Zhenguo Li, Ping Luo.*<br>
[[paper]](https://arxiv.org/abs/2505.20147)
[[code]](https://fudoki-hku.github.io/)

**(*arXiv2025_HaploOmni*) HaploOmni: Unified Single Transformer for Multimodal Video Understanding and Generation.** <br>
*Yicheng Xiao, Lin Song, Rui Yang, Cheng Cheng, Zunnan Xu, Zhaoyang Zhang, Yixiao Ge, Xiu Li, Ying Shan.*<br>
[[paper]](https://arxiv.org/abs/2506.02975v1)
[[code]](https://github.com/Tencent/HaploVLM)

**(*arXiv2025_UniWorld-V1*) UniWorld-V1: High-Resolution Semantic Encoders for Unified Visual Understanding and Generation.** <br>
*Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, Yatian Pang, Li Yuan.*<br>
[[paper]](https://arxiv.org/abs/2506.03147)
[[code]](https://github.com/PKU-YuanGroup/UniWorld-V1)

**(*arXiv2025_Ming-Omni*) Ming-Omni: A Unified Multimodal Model for Perception and Generation.** <br>
*Inclusion AI, Ant Group.*<br>
[[paper]](https://arxiv.org/abs/2506.09344)
[[code]](https://github.com/inclusionAI/Ming/tree/main)

**(*arXiv2025_Show-o2*) Show-o2: Improved Native Unified Multimodal Models.** <br>
*Jinheng Xie, Zhenheng Yang, Mike Zheng Shou.*<br>
[[paper]](https://arxiv.org/abs/2506.15564)
[[code]](https://github.com/showlab/Show-o)

**(*arXiv2025_UniFork*) UniFork: Exploring Modality Alignment for Unified Multimodal Understanding and Generation.** <br>
*Teng Li, Quanfeng Lu, Lirui Zhao, Hao Li, Xizhou Zhu, Yu Qiao, Jun Zhang, Wenqi Shao.*<br>
[[paper]](https://arxiv.org/abs/2506.17202)
[[code]](https://github.com/tliby/UniFork)

**(*arXiv2025_OmniGen2*) OmniGen2: Exploration to Advanced Multimodal Generation.** <br>
*Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, Zheng Liu.*<br>
[[paper]](https://arxiv.org/abs/2506.18871)
[[code]](https://github.com/VectorSpaceLab/OmniGen2)

**(*arXiv2025_Omni-Video*) Omni-Video: Democratizing Unified Video Understanding and Generation.** <br>
*Zhiyu Tan, Hao Yang, Luozheng Qin, Jia Gong, Mengping Yang, Hao Li.*<br>
[[paper]](https://arxiv.org/abs/2507.06119v2)
[[code]](https://howellyoung-s.github.io/OmniVideo_project/)

**(*arXiv2025_X-Omni*) X-Omni: Reinforcement Learning Makes Discrete Autoregressive Image Generative Models Great Again.** <br>
*Zigang Geng, Yibing Wang, Yeyao Ma, Chen Li, Yongming Rao, Shuyang Gu, Zhao Zhong, Qinglin Lu, Han Hu, Xiaosong Zhang, Linus, Di Wang, Jie Jiang.*<br>
[[paper]](https://arxiv.org/abs/2507.22058)

**(*arXiv2025_RecA*) Reconstruction Alignment Improves Unified Multimodal Models.** <br>
*Ji Xie, Trevor Darrell, Luke Zettlemoyer, XuDong Wang.*<br>
[[paper]](https://arxiv.org/abs/2509.07295)
[[code]](https://reconstruction-alignment.github.io/)

**(*arXiv2025_Lumina-DiMOO*) Lumina-DiMOO: An Omni Diffusion Large Language Model for Multi-Modal Generation and Understanding.** <br>
*xx.*<br>
[[paper]](xxx)
[[code]](xxx)


### ``*Large MMM for Manipulation*``

**(*arXiv2025_Magma*) Magma: A Foundation Model for Multimodal AI Agents.** <br>
*Jianwei Yang, Reuben Tan, Qianhui Wu, Ruijie Zheng, Baolin Peng, Yongyuan Liang, Yu Gu, Mu Cai, Seonghyeon Ye, Joel Jang, Yuquan Deng, Lars Liden, Jianfeng Gao.*<br>
[[paper]](https://arxiv.org/abs/2502.13130)
[[code]](https://microsoft.github.io/Magma)

**(*arXiv2025_Cosmos-Reason1*) Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning.** <br>
*NVIDIA Team.*<br>
[[paper]](https://arxiv.org/abs/2503.15558)
[[code]](https://github.com/nvidia-cosmos/cosmos-reason1)

**(*arXiv2025_ViSA-Flow*) ViSA-Flow: Accelerating Robot Skill Learning via Large-Scale Video Semantic Action Flow.** <br>
*Changhe Chen, Quantao Yang, Xiaohao Xu, Nima Fazeli, Olov Andersson.*<br>
[[paper]](https://arxiv.org/abs/2505.01288)
[[code]](https://visaflow-web.github.io/ViSAFLOW)

**(*arXiv2025_Sekai*) Sekai: A Video Dataset towards World Exploration.** <br>
*Zhen Li, Chuanhao Li, Xiaofeng Mao, Shaoheng Lin, Ming Li, Shitian Zhao, Zhaopan Xu, Xinyue Li, Yukang Feng, Jianwen Sun, Zizhen Li, Fanrui Zhang, Jiaxin Ai, Zhixiang Wang, Yuwei Wu, Tong He, Jiangmiao Pang, Yu Qiao, Yunde Jia, Kaipeng Zhang.*<br>
[[paper]](https://arxiv.org/abs/2506.15675)
[[code]](https://lixsp11.github.io/sekai-project/)

**(*arXiv2025_GR-3*) GR-3 Technical Report.** <br>
*ByteDance Seed.*<br>
[[paper]](https://arxiv.org/abs/2507.15493)
[[code]](https://seed.bytedance.com/GR3/)

**(*arXiv2025_HunyuanWorld-1.0*) HunyuanWorld 1.0: Generating Immersive, Explorable, and Interactive 3D Worlds from Words or Pixels.** <br>
*Tencent Hunyuan.*<br>
[[paper]](https://arxiv.org/abs/2507.21809)
[[code]](https://3d-models.hunyuan.tencent.com/world/)

**(*arXiv2025_M3-Agent*) Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory.** <br>
*Lin Long, Yichen He, Wentao Ye, Yiyuan Pan, Yuan Lin, Hang Li, Junbo Zhao, Wei Li.*<br>
[[paper]](https://arxiv.org/abs/2508.09736v1)
[[code]](https://github.com/bytedance-seed/m3-agent)

**(*arXiv2025_Matrix-Game 2.0*) Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model.** <br>
*Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, Baixin Xu, Hao-Xiang Guo, Kaixiong Gong, Cyrus Wu, Wei Li, Xuchen Song, Yang Liu, Eric Li, Yahui Zhou.*<br>
[[paper]](https://arxiv.org/abs/2508.13009)
[[code]](https://matrix-game-v2.github.io/)



### ``*Large Model Distillation*``

**(*EMNLP2016_Seq-KD*) Sequence-Level Knowledge Distillation.** <br>
*Yoon Kim, Alexander M. Rush.*<br>
[[paper]](https://arxiv.org/abs/1606.07947)
[[code]](https://github.com/harvardnlp/seq2seq-attn)

**(*EMNLP2020_ImitKD*) Autoregressive Knowledge Distillation through Imitation Learning.** <br>
*Alexander Lin, Jeremy Wohlwend, Howard Chen, Tao Lei.*<br>
[[paper]](https://arxiv.org/abs/2009.07253)
[[code]](https://github.com/asappresearch/imitkd)

**(*ICLR2013_Ensemble*) Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning.** <br>
*Zeyuan Allen-Zhu, Yuanzhi Li.*<br>
[[paper]](https://arxiv.org/abs/2012.09816)

**(*arXiv2022_Unified-KD*) Knowledge Distillation of Transformer-based Language Models Revisited.** <br>
*Chengqiang Lu, Jianwei Zhang, Yunfei Chu, Zhengyu Chen, Jingren Zhou, Fei Wu, Haiqing Chen, Hongxia Yang.*<br>
[[paper]](https://arxiv.org/abs/2206.14366)

**(*arXiv2022_MT-CoT*) Explanations from Large Language Models Make Small Reasoners Better.** <br>
*Shiyang Li, Jianshu Chen, Yelong Shen, Zhiyu Chen, Xinlu Zhang, Zekun Li, Hong Wang, Jing Qian, Baolin Peng, Yi Mao, Wenhu Chen, Xifeng Yan.*<br>
[[paper]](https://arxiv.org/abs/2210.06726)

**(*ACL2023_SOCRATIC-CoT*) Distilling Reasoning Capabilities into Smaller Language Models.** <br>
*Kumar Shridhar, Alessandro Stolfo, Mrinmaya Sachan.*<br>
[[paper]](https://arxiv.org/abs/2212.00193)

**(*ACL2023_FT-CoT*) Large Language Models Are Reasoning Teachers.** <br>
*Namgyu Ho, Laura Schmid, Se-Young Yun.*<br>
[[paper]](https://arxiv.org/abs/2212.10071)

**(*ACL2023_DISCO*) DISCO: Distilling Counterfactuals with Large Language Models.** <br>
*Zeming Chen, Qiyue Gao, Antoine Bosselut, Ashish Sabharwal, Kyle Richardson.*<br>
[[paper]](https://arxiv.org/abs/2212.10534)

**(*arXiv2022_ICT-D*) In-context Learning Distillation: Transferring Few-shot Learning Ability of Pre-trained Language Models.** <br>
*Yukun Huang, Yanda Chen, Zhou Yu, Kathleen McKeown.*<br>
[[paper]](https://arxiv.org/abs/2212.10670)

**(*ICML2023_ModelSpecializing*) Specializing Smaller Language Models towards Multi-Step Reasoning.** <br>
*Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, Tushar Khot.*<br>
[[paper]](https://arxiv.org/abs/2301.12726)

**(*ACL2023_SCOTT*) SCOTT: Self-Consistent Chain-of-Thought Distillation.** <br>
*Peifeng Wang, Zhengyang Wang, Zheng Li, Yifan Gao, Bing Yin, Xiang Ren.*<br>
[[paper]](https://arxiv.org/abs/2305.01879)
[[code]](https://github.com/wangpf3/consistent-cot-distillation)

**(*ACL2023_Distilling-step-by-step*) Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes.** <br>
*Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alexander Ratner, Ranjay Krishna, Chen-Yu Lee, Tomas Pfister.*<br>
[[paper]](https://arxiv.org/abs/2305.02301)

**(*EMNLP2023_Lion*) Lion: Adversarial Distillation of Proprietary Large Language Models.** <br>
*Yuxin Jiang, Chunkit Chan, Mingyang Chen, Wei Wang.*<br>
[[paper]](https://arxiv.org/abs/2305.12870)
[[code]](https://github.com/YJiangcm/Lion)

**(*ICLR2024_MINILLM*) MINILLM: Knowledge Distillation of Large Language Models.** <br>
*Yuxian Gu, Li Dong, Furu Wei, Minlie Huang.*<br>
[[paper]](https://arxiv.org/abs/2306.08543)
[[code]](https://aka.ms/MiniLLM)

**(*NeurIPS2023_KD*) Propagating Knowledge Updates to LMs Through Distillation.** <br>
*Shankar Padmanabhan, Yasumasa Onoe, Michael J.Q. Zhang, Greg Durrett, Eunsol Choi.*<br>
[[paper]](https://arxiv.org/abs/2306.09306)
[[code]](https://github.com/shankarp8/knowledge_distillation)

**(*ICLR2024_GKD*) On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes.** <br>
*Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, Olivier Bachem.*<br>
[[paper]](https://arxiv.org/abs/2306.13649)
[[code]](https://github.com/microsoft/LMOps/tree/main/minillm)

**(*ACL2023_f-DISTILL*) f-Divergence Minimization for Sequence-Level Knowledge Distillation.** <br>
*Yuqiao Wen, Zichao Li, Wenyu Du, Lili Mou.*<br>
[[paper]](https://arxiv.org/abs/2307.15190)
[[code]](https://github.com/MANGA-UOFA/fdistill)

**(*arXiv2023_BabyLlama*) Baby Llama: knowledge distillation from an ensemble of teachers trained on a small dataset with no performance penalty.** <br>
*Inar Timiryasov, Jean-Loup Tastet.*<br>
[[paper]](https://arxiv.org/abs/2308.02019)
[[code]](https://github.com/timinar/babyllama)

**(*ICLR2024_DistillSpec*) DistillSpec: Improving Speculative Decoding via Knowledge Distillation.** <br>
*Yongchao Zhou, Kaifeng Lyu, Ankit Singh Rawat, Aditya Krishna Menon, Afshin Rostamizadeh, Sanjiv Kumar, Jean-François Kagy, Rishabh Agarwal.*<br>
[[paper]](https://arxiv.org/abs/2310.08461)

**(*arXiv2023_MiniMA*) Towards the Law of Capacity Gap in Distilling Language Models.** <br>
*Chen Zhang, Dawei Song, Zheyu Ye, Yan Gao.*<br>
[[paper]](https://arxiv.org/abs/2311.07052)
[[code]](https://github.com/GeneZC/MiniMA)

**(*ICML2024_Self-Rewarding*) Self-Rewarding Language Models.** <br>
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

**(*arXiv2023_Survey*) A Survey on Model Compression for Large Language Models.** <br>
*Xunyu Zhu, Jian Li, Yong Liu, Can Ma, Weiping Wang.*<br>
[[paper]](https://arxiv.org/abs/2308.07633)

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

**(*arXiv2023_Survey*) Efficient Large Language Models: A Survey.** <br>
*Zhongwei Wan, Xin Wang, Che Liu, Samiul Alam, Yu Zheng, Jiachen Liu, Zhongnan Qu, Shen Yan, Yi Zhu, Quanlu Zhang, Mosharaf Chowdhury, Mi Zhang.*<br>
[[paper]](https://arxiv.org/abs/2312.03863)
[[code]](https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey)

**(*arXiv2023_Survey*) A Challenger to GPT-4V? Early Explorations of Gemini in Visual Expertise.** <br>
*Chaoyou Fu, Renrui Zhang, Zihan Wang, Yubo Huang, Zhengye Zhang, Longtian Qiu, Gaoxiang Ye, Yunhang Shen, Mengdan Zhang, Peixian Chen, Sirui Zhao, Shaohui Lin, Deqiang Jiang, Di Yin, Peng Gao, Ke Li, Hongsheng Li, Xing Sun.*<br>
[[paper]](https://arxiv.org/abs/2312.12436)
[[code]](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)

**(*arXiv2024_Survey*) From GPT-4 to Gemini and Beyond: Assessing the Landscape of MLLMs on Generalizability, Trustworthiness and Causality through Four Modalities.** <br>
*Chaochao Lu, Chen Qian, Guodong Zheng, Hongxing Fan, Hongzhi Gao, Jie Zhang, Jing Shao, Jingyi Deng, Jinlan Fu, Kexin Huang, Kunchang Li, Lijun Li, Limin Wang, Lu Sheng, Meiqi Chen, Ming Zhang, Qibing Ren, Sirui Chen, Tao Gui, Wanli Ouyang, Yali Wang, Yan Teng, Yaru Wang, Yi Wang, Yinan He, Yingchun Wang, Yixu Wang, Yongting Zhang, Yu Qiao, Yujiong Shen, Yurong Mou, Yuxi Chen, Zaibin Zhang, Zhelun Shi, Zhenfei Yin, Zhipin Wang.*<br>
[[paper]](https://arxiv.org/abs/2401.15071)

**(*arXiv2024_Survey*) Vision Mamba: A Comprehensive Survey and Taxonomy.** <br>
*Xiao Liu, Chenxu Zhang, Lei Zhang.*<br>
[[paper]](https://arxiv.org/abs/2405.04404)
[[code]](https://github.com/lx6c78/Vision-Mamba-A-Comprehensive-Survey-and-Taxonomy)

**(*arXiv2024_Survey*) Autoregressive Models in Vision: A Survey.** <br>
*Jing Xiong, Gongye Liu, Lun Huang, Chengyue Wu, Taiqiang Wu, Yao Mu, Yuan Yao, Hui Shen, Zhongwei Wan, Jinfa Huang, Chaofan Tao, Shen Yan, Huaxiu Yao, Lingpeng Kong, Hongxia Yang, Mi Zhang, Guillermo Sapiro, Jiebo Luo, Ping Luo, Ngai Wong.*<br>
[[paper]](https://arxiv.org/abs/2411.05902)
[[code]](https://github.com/ChaofanTao/Autoregressive-Models-in-Vision-Survey)

**(*arXiv2025_Survey*) On Path to Multimodal Generalist: General-Level and General-Bench.** <br>
*Hao Fei, Yuan Zhou, Juncheng Li, Xiangtai Li, Qingshan Xu, Bobo Li, Shengqiong Wu, Yaoting Wang, Junbao Zhou, Jiahao Meng, Qingyu Shi, Zhiyuan Zhou, Liangtao Shi, Minghe Gao, Daoan Zhang, Zhiqi Ge, Weiming Wu, Siliang Tang, Kaihang Pan, Yaobo Ye, Haobo Yuan, Tao Zhang, Tianjie Ju, Zixiang Meng, Shilin Xu, Liyu Jia, Wentao Hu, Meng Luo, Jiebo Luo, Tat-Seng Chua, Shuicheng Yan, Hanwang Zhang.*<br>
[[paper]](https://arxiv.org/abs/2505.04620)
[[code]](https://generalist.top/)

**(*arXiv2025_Survey*) Shifting AI Efficiency From Model-Centric to Data-Centric Compression.** <br>
*Xuyang Liu, Zichen Wen, Shaobo Wang, Junjie Chen, Zhishan Tao, Yubo Wang, Xiangqi Jin, Chang Zou, Yiyu Wang, Chenfei Liao, Xu Zheng, Honggang Chen, Weijia Li, Xuming Hu, Conghui He, Linfeng Zhang.*<br>
[[paper]](https://arxiv.org/abs/2505.19147)
[[code]](https://github.com/xuyang-liu16/awesome-token-level-model-compression)


### ``*Related Benchmark*``

**(*NeurIPS2023_LAMM*) LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark.** <br>
*Zhenfei Yin, Jiong Wang, Jianjian Cao, Zhelun Shi, Dingning Liu, Mukai Li, Lu Sheng, Lei Bai, Xiaoshui Huang, Zhiyong Wang, Jing Shao, Wanli Ouyang.*<br>
[[paper]](https://arxiv.org/abs/2306.06687)
[[code]](https://openlamm.github.io/)

**(*arXiv2023_MME*) MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models.** <br>
*Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, Rongrong Ji.*<br>
[[paper]](https://arxiv.org/abs/2306.13394)
[[code]](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)

**(*ECCV2024_MMBench*) MMBench: Is Your Multi-modal Model an All-around Player?.** <br>
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

**(*arXiv2024_ConvBench*) ConvBench: A Multi-Turn Conversation Evaluation Benchmark with Hierarchical Capability for Large Vision-Language Models.** <br>
*Shuo Liu, Kaining Ying, Hao Zhang, Yue Yang, Yuqi Lin, Tianle Zhang, Chuanhao Li, Yu Qiao, Ping Luo, Wenqi Shao, Kaipeng Zhang.*<br>
[[paper]](https://arxiv.org/abs/2403.20194)

**(*ECCV2024_BLINK*) BLINK: Multimodal Large Language Models Can See but Not Perceive.** <br>
*Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A. Smith, Wei-Chiu Ma, Ranjay Krishna.*<br>
[[paper]](https://arxiv.org/abs/2404.12390)
[[code]](https://zeyofu.github.io/blink/)

**(*arXiv2024_MMT-Bench*) MMT-Bench: A Comprehensive Multimodal Benchmark for Evaluating Large Vision-Language Models Towards Multitask AGI.** <br>
*Kaining Ying, Fanqing Meng, Jin Wang, Zhiqian Li, Han Lin, Yue Yang, Hao Zhang, Wenbo Zhang, Yuqi Lin, Shuo Liu, Jiayi Lei, Quanfeng Lu, Runjian Chen, Peng Xu, Renrui Zhang, Haozhe Zhang, Peng Gao, Yali Wang, Yu Qiao, Ping Luo, Kaipeng Zhang, Wenqi Shao.*<br>
[[paper]](https://arxiv.org/abs/2404.16006)

**(*arXiv2024_SEED-Bench-2-Plus*) SEED-Bench-2-Plus: Benchmarking Multimodal Large Language Models with Text-Rich Visual Comprehension.** <br>
*Bohao Li, Yuying Ge, Yi Chen, Yixiao Ge, Ruimao Zhang, Ying Shan.*<br>
[[paper]](https://arxiv.org/abs/2404.16790)
[[code]](https://github.com/AILab-CVC/SEED-Bench)

**(*arXiv2024_Video-MME*) Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis.** <br>
*Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Rongrong Ji, Xing Sun.*<br>
[[paper]](https://arxiv.org/abs/2405.21075)
[[code]](https://video-mme.github.io/)

**(*arXiv2024_VSI-Bench*) Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces.** <br>
*Jihan Yang, Shusheng Yang, Anjali W. Gupta, Rilyn Han, Li Fei-Fei, Saining Xie.*<br>
[[paper]](https://arxiv.org/abs/2412.14171)
[[code]](https://github.com/vision-x-nyu/thinking-in-space)

**(*arXiv2025_WISE*) WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation.** <br>
*Yuwei Niu, Munan Ning, Mengren Zheng, Bin Lin, Peng Jin, Jiaqi Liao, Kunpeng Ning, Bin Zhu, Li Yuan.*<br>
[[paper]](https://arxiv.org/abs/2503.07265v1)
[[code]](https://github.com/PKU-YuanGroup/WISE)

**(*arXiv2025_GPT-ImgEval*) GPT-ImgEval: A Comprehensive Benchmark for Diagnosing GPT4o in Image Generation.** <br>
*Zhiyuan Yan, Junyan Ye, Weijia Li, Zilong Huang, Shenghai Yuan, Xiangyang He, Kaiqing Lin, Jun He, Conghui He, Li Yuan.*<br>
[[paper]](https://arxiv.org/abs/2504.02782)
[[code]](https://github.com/PicoTrex/GPT-ImgEval)

**(*arXiv2025_PhyX*) PhyX: Does Your Model Have the "Wits" for Physical Reasoning?.** <br>
*Hui Shen, Taiqiang Wu, Qi Han, Yunta Hsieh, Jizhou Wang, Yuyue Zhang, Yuxin Cheng, Zijian Hao, Yuansheng Ni, Xin Wang, Zhongwei Wan, Kai Zhang, Wendong Xu, Jing Xiong, Ping Luo, Wenhu Chen, Chaofan Tao, Zhuoqing Mao, Ngai Wong.*<br>
[[paper]](https://arxiv.org/abs/2505.15929)
[[code]](https://phyx-bench.github.io/)

**(*arXiv2025_MMMG*) MMMG: a Comprehensive and Reliable Evaluation Suite for Multitask Multimodal Generation.** <br>
*Jihan Yao, Yushi Hu, Yujie Yi, Bin Han, Shangbin Feng, Guang Yang, Bingbing Wen, Ranjay Krishna, Lucy Lu Wang, Yulia Tsvetkov, Noah A. Smith, Banghua Zhu.*<br>
[[paper]](https://arxiv.org/abs/2505.17613)
[[code]](https://github.com/yaojh18/MMMG)

**(*arXiv2025_VS-Bench*) VS-Bench: Evaluating VLMs for Strategic Reasoning and Decision-Making in Multi-Agent Environments.** <br>
*Zelai Xu, Zhexuan Xu, Xiangmin Yi, Huining Yuan, Xinlei Chen, Yi Wu, Chao Yu, Yu Wang.*<br>
[[paper]](https://arxiv.org/abs/2506.02387)
[[code]](https://vs-bench.github.io/)

**(*arXiv2025_PhysUniBench*) PhysUniBench: An Undergraduate-Level Physics Reasoning Benchmark for Multimodal Models.** <br>
*Lintao Wang, Encheng Su, Jiaqi Liu, Pengze Li, Peng Xia, Jiabei Xiao, Wenlong Zhang, Xinnan Dai, Xi Chen, Yuan Meng, Mingyu Ding, Lei Bai, Wanli Ouyang, Shixiang Tang, Aoran Wang, Xinzhu Ma.*<br>
[[paper]](https://arxiv.org/abs/2506.17667)
[[code]](https://prismax-team.github.io/PhysUniBenchmark/)
