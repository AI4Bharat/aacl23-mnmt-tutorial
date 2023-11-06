# aacl23-mnmt-tutorial

[📘 slides](https://docs.google.com/presentation/d/1BW9N9Fi8X9QQYB_DmyjHm2w-0idKlfuydH3eswgpIqs) | [▶️ Recording]()

## Reading List

### Fundamental concepts

#### Architecture
* Sequence to Sequence Learning with Neural Networks \
[Paper](https://arxiv.org/abs/1409.3215)

* Neural Machine Translation by Jointly Learning to Align and Translate \
[Paper](https://arxiv.org/abs/1409.0473)

* Attention Is All You Need \
[Paper](https://arxiv.org/abs/1706.03762)

* BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding \
[Paper](https://arxiv.org/abs/1706.03762)

* BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension\
[Paper](https://aclanthology.org/2020.acl-main.703/) [Code](https://github.com/facebookresearch/fairseq/tree/main/examples/bart)

#### Vocabulary
* Neural Machine Translation of Rare Words with Subword Units \
[Paper](https://aclanthology.org/P16-1162/) [Code](https://github.com/rsennrich/subword-nmt)

* Neural Machine Translation with Byte-Level Subwords \
[Paper](https://arxiv.org/abs/1909.03341)

* Neural Machine Translation with Byte-Level Subwords \
[Paper](https://arxiv.org/abs/1909.03341)

* SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing \
[Paper](https://aclanthology.org/D18-2012/) [Code](https://github.com/google/sentencepiece)

### Prominent Massively Multilingual NMT systems
* Google's Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation \
[Paper](https://aclanthology.org/Q17-1024/)

* Massively Multilingual Neural Machine Translation \
[Paper](https://aclanthology.org/N19-1388/)

* Massively Multilingual Neural Machine Translation in the Wild: Findings and Challenges \
[Paper](https://arxiv.org/abs/1907.05019)

* Beyond English-Centric Multilingual Machine Translation (M2M-100)\
[Paper](https://www.jmlr.org/papers/volume22/20-1307/20-1307.pdf) [Code](https://github.com/facebookresearch/fairseq/tree/main/examples/m2m_100)

* Multilingual Denoising Pre-training for Neural Machine Translation (MBART-25) \
[Paper](https://aclanthology.org/2020.tacl-1.47/) [Code](https://github.com/facebookresearch/fairseq/tree/main/examples/mbart)

* Multilingual Translation from Denoising Pre-Training (MBART-50) \
[Paper](https://aclanthology.org/2021.findings-acl.304/) [Code](https://github.com/facebookresearch/fairseq/tree/main/examples/multilingual)

* DeltaLM: Encoder-Decoder Pre-training for Language Generation and Translation by Augmenting Pretrained Multilingual Encoders \
[Paper](https://arxiv.org/abs/2106.13736) [Code](https://github.com/microsoft/unilm/tree/master/deltalm)

* No Language Left Behind: Scaling Human-Centered Machine Translation (NLLB-200) \
[Paper](https://arxiv.org/abs/2207.04672) [Code](https://github.com/facebookresearch/fairseq/tree/nllb)

* MADLAD-400: A Multilingual And Document-Level Large Audited Dataset \
[Paper](https://arxiv.org/abs/2309.04662) [Model](https://github.com/google-research/google-research/tree/master/madlad_400)

* Towards the Next 1000 Languages in Multilingual Machine Translation: Exploring the Synergy Between Supervised and Self-Supervised Learning \
[Paper](https://arxiv.org/abs/2201.03110)

### Models for related languages.
#### African
* MMTAfrica: Multilingual Machine Translation for African Languages \
[Paper](https://aclanthology.org/2021.wmt-1.48/) [Code](https://github.com/edaiofficial/mmtafrica)

* AfroMT: Pretraining Strategies and Reproducible Benchmarks for Translation of 8 African Languages \
[Paper](https://aclanthology.org/2021.emnlp-main.99/) [Code](https://github.com/machelreid/afromt)

* ANVITA-African: A Multilingual Neural Machine Translation System for African Languages \
[Paper](https://aclanthology.org/2022.wmt-1.106/)

* AfroLM: A Self-Active Learning-based Multilingual Pretrained Language Model for 23 African Languages \
[Paper](https://arxiv.org/abs/2211.03263) [Code](https://github.com/bonaventuredossou/MLM_AL)

#### Middle-East / North-African

* AraBERT: Transformer-based Model for Arabic Language Understanding \
[Paper](https://aclanthology.org/2020.osact-1.2/) [Code](https://github.com/aub-mind/araBERT)

* The Interplay of Variant, Size, and Task Type
in Arabic Pre-trained Language Models (CAMeLBERT) \
[Paper](https://arxiv.org/abs/2103.06678) [Code](https://github.com/CAMeL-Lab/CAMeLBERT)

#### South-East Asian

* SG Translate Together - Uplifting Singapore’s translation standards with the community through technology \
[Paper](https://aclanthology.org/2022.amta-upg.28/)

* IndoNLG: Benchmark and Resources for Evaluating Indonesian Natural Language Generation (IndoBART, IndoGPT) \
[Paper](https://aclanthology.org/2021.emnlp-main.699/) [Code](https://github.com/IndoNLP/indonlg)

* WangchanBERTa: Pretraining transformer-based Thai Language Models \
[Paper](https://arxiv.org/abs/2101.09635) [Code](https://github.com/vistec-AI/thai2transformers)

* WangchanBERTa: Pretraining transformer-based Thai Language Models \
[Paper](https://arxiv.org/abs/2101.09635) [Code](https://github.com/vistec-AI/thai2transformers)

#### European languages

* OPUS-MT – Building open translation services for the World \
[Paper](https://aclanthology.org/2020.eamt-1.61/) [Code](https://github.com/Helsinki-NLP/Opus-MT)

#### Indigenous languages of America
* IndT5: A Text-to-Text Transformer for 10 Indigenous Languages \
[Paper](https://aclanthology.org/2021.americasnlp-1.30/) [Code](https://github.com/UBC-NLP/IndT5)

* Enhancing Translation for Indigenous Languages: Experiments with Multilingual Models \
[Paper](https://aclanthology.org/2023.americasnlp-1.22/)

#### Indian subcontinent

* Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages (IndicTrans1) \
[Paper](https://arxiv.org/abs/2104.05596) [Code](https://github.com/AI4Bharat/IndicTrans)

* IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages \
[Paper](https://arxiv.org/abs/2305.16307) [Code](https://github.com/AI4Bharat/IndicTrans2)

* IndicBART: A Pre-trained Model for Indic Natural Language Generation \
[Paper](https://aclanthology.org/2022.findings-acl.145/) [Code](https://github.com/AI4Bharat/indic-bart/)

#### China
* ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information \
[Paper](https://aclanthology.org/2021.acl-long.161/) [Code](https://github.com/ShannonAI/ChineseBert)

* CPT: A Pre-Trained Unbalanced Transformer for Both Chinese Language Understanding and Generation \
[Paper](https://arxiv.org/abs/2109.05729) [Code](https://github.com/fastnlp/CPT)

#### Creoles
* KreolMorisienMT: A Dataset for Mauritian Creole Machine Translation \
[Paper](https://aclanthology.org/2022.findings-aacl.3/) [Code](https://github.com/prajdabre/KreolMorisienNLG)

* CreoleVal: Multilingual Multitask Benchmarks for Creoles \
[Paper](https://arxiv.org/abs/2310.19567) [Code](https://github.com/hclent/CreoleVal)


### Dataset Curation

#### Monolingual Data Curation - Large scale

* Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (Data:C4, Model:T5) \
[Paper](https://arxiv.org/abs/1910.10683) [Code](https://github.com/google-research/text-to-text-transfer-transformer)

* mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer (Data:mC4, Model:mT5) \
[Paper](https://aclanthology.org/2021.naacl-main.41/) [Code](https://github.com/google-research/multilingual-t5)

* The Pile: An 800GB Dataset of Diverse Text for Language Modeling \
[Paper](https://arxiv.org/abs/2101.00027) [Data](https://github.com/EleutherAI/the-pile)

* The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only \
[Paper](https://arxiv.org/abs/2306.01116)


#### Monolingual Data Curation - Language-family specific

---
**NOTE**

We refer the reader to the papers on language-family specific models, as these include monolingual data creation, bitext mining and model training.

Additional papers other than those mentioned above are included in this subsection.

---

* IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages \
[Paper](https://aclanthology.org/2020.findings-emnlp.445/) [Code](https://github.com/AI4Bharat/Indic-BERT-v1)

* Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages \
[Paper](https://aclanthology.org/2023.acl-long.693/) [Code](https://github.com/ai4bharat/indicbert)

* Varta: A Large-Scale Headline-Generation Dataset for Indic Languages \
[Paper](https://aclanthology.org/2023.findings-acl.215/) [Code](https://github.com/rahular/varta)

* WebCrawl African : A Multilingual Parallel Corpora for African Languages \
[Paper](https://aclanthology.org/2022.wmt-1.105/) [Code](https://github.com/pavanpankaj/Web-Crawl-African)

#### Parallel Corpora Creation

* CCAligned: A Massive Collection of Cross-Lingual Web-Document Pairs \
[Paper](https://aclanthology.org/2020.emnlp-main.480/) [Data](https://www.statmt.org/cc-aligned/)

* Billion-scale similarity search with GPUs (FAISS) \
[Paper](https://arxiv.org/abs/1702.08734) [Code](https://github.com/facebookresearch/faiss)

* CCMatrix: Mining Billions of High-Quality Parallel Sentences on the Web \
[Paper](https://aclanthology.org/2021.acl-long.507/) [Code](https://github.com/facebookresearch/LASER/tree/main/tasks/CCMatrix)

* xSIM++: An Improved Proxy to Bitext Mining Performance for Low-Resource Languages \
[Paper](https://aclanthology.org/2023.acl-short.10/) [Data](https://github.com/facebookresearch/LASER/tree/main/tasks/xsimplusplus)

#### Sentence Embedding Models

* Language-agnostic BERT Sentence Embedding \
[Paper](https://aclanthology.org/2020.emnlp-main.480/) [Code](https://github.com/UKPLab/sentence-transformers)

* Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond \
(LASER1) \
[Paper](https://aclanthology.org/Q19-1038/) [Code](https://github.com/facebookresearch/LASER)

* Bitext Mining Using Distilled Sentence Representations for Low-Resource Languages \
(LASER3) \
[Paper](https://aclanthology.org/2022.findings-emnlp.154/) [Code](https://github.com/facebookresearch/LASER/blob/main/nllb/README.md)

* Multilingual Representation Distillation with Contrastive Learning (LASER3-CO) \
[Paper](https://aclanthology.org/2023.eacl-main.108/)

* Learning Multilingual Sentence Representations with Cross-lingual Consistency Regularization (MuSR) \
[Paper](https://arxiv.org/abs/2306.06919) [Code](https://github.com/gpengzhi/crossconst-sr)

* SONAR: Sentence-Level Multimodal and Language-Agnostic Representations \
[Paper](https://ai.meta.com/research/publications/sonar-sentence-level-multimodal-and-language-agnostic-representations/) [Code](https://github.com/facebookresearch/SONAR)

#### Data Quality v/s Scale

* Data and Parameter Scaling Laws for Neural Machine Translation \
[Paper](https://aclanthology.org/2021.emnlp-main.478/) [Code](https://github.com/mitchellgordon95/mt-scaling)

* Data Scaling Laws in NMT: The Effect of Noise and Architecture \
[Paper](https://proceedings.mlr.press/v162/bansal22b/bansal22b.pdf) 

* “A Little is Enough”: Few-Shot Quality Estimation based Corpus Filtering improves Machine Translation \
[Paper](https://aclanthology.org/2023.findings-acl.892/) 

#### Human-annotated Seed Corpora

* The TDIL Program and the Indian Language Corpora Intitiative (ILCI) \
[Paper](https://aclanthology.org/L10-1602/)

* Small Data, Big Impact: Leveraging Minimal Data for Effective Machine Translation \
[Paper](https://aclanthology.org/2023.acl-long.154/) [Data](https://github.com/facebookresearch/flores/tree/main/nllb_seed) 

* MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages \
[Paper](https://aclanthology.org/2023.acl-long.235/) [Data](https://github.com/alexa/massive) 

### Benchmarks

* The FLORES Evaluation Datasets for Low-Resource Machine Translation: Nepali–English and Sinhala–English \
[Paper](https://aclanthology.org/D19-1632/)

* The Flores-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation \
[Paper](https://aclanthology.org/2023.acl-long.235/) [Data](https://github.com/facebookresearch/flores) 

* NTREX-128 – News Test References for MT Evaluation of 128 Languages \
[Paper](https://aclanthology.org/2022.sumeval-1.4/) [Data](https://github.com/MicrosoftTranslator/NTREX) 

### Modeling
---
**NOTE**

We refer the reader to the papers on massively multilingual models, as these include some aspects of modeling.

Additional papers other than those mentioned above are included in this subsection.

---

#### Vocabulary

* How Robust is Neural Machine Translation to Language Imbalance in Multilingual Tokenizer Training? \
[Paper](https://aclanthology.org/2022.amta-research.8/)

* Out-of-the-box Universal Romanization Tool uroman \
[Paper](https://aclanthology.org/P18-4003/)

* The IndicNLP Library \
[Paper](https://github.com/anoopkunchukuttan/indic_nlp_library/blob/master/docs/indicnlp.pdf)
[Code](https://github.com/anoopkunchukuttan/indic_nlp_library/tree/master)

* Pre-training via Leveraging Assisting Languages
for Neural Machine Translation \
[Paper](https://aclanthology.org/2020.acl-srw.37/) [Code](https://aclanthology.org/attachments/2020.acl-srw.37.Software.zip)

* BPE-Dropout: Simple and Effective Subword Regularization \
[Paper](https://aclanthology.org/2020.acl-main.170/)

* Efficient Neural Machine Translation for Low-Resource Languages via Exploiting Related Languages \
[Paper](https://aclanthology.org/2020.acl-srw.22/)
[Code](https://aclanthology.org/attachments/2020.acl-srw.22.Software.zip)

* Exploiting Language Relatedness for Low Web-Resource Language Model Adaptation: An Indic Languages Study \
[Paper](https://aclanthology.org/2021.acl-long.105/)
[Code](https://github.com/yashkhem1/RelateLM)

* Language Relatedness and Lexical Closeness can help Improve Multilingual NMT: IITBombay@MultiIndicNMT WAT2021 \
[Paper](https://aclanthology.org/2021.wat-1.26)

* Auxiliary Subword Segmentations as Related Languages for Low Resource Multilingual Translation \
[Paper](https://aclanthology.org/2022.eamt-1.16/)

* Overlap-based Vocabulary Generation Improves Cross-lingual Transfer Among Related Languages \
[Paper](https://aclanthology.org/2022.acl-long.18/) [Code](https://github.com/vaidehi99/obpe)

* Transfer Learning in Multilingual Neural Machine Translation with Dynamic Vocabulary  
[Paper](https://aclanthology.org/2018.iwslt-1.8/)

#### Leveraging Ordering Information

* Addressing word-order Divergence in Multilingual Neural Machine Translation for extremely Low Resource Languages
[Paper](https://aclanthology.org/N19-1387/)
[Code](https://github.com/anoopkunchukuttan/cfilt_preorder)

* Language Related Issues for Machine Translation between Closely Related South Slavic Languages \
[Paper](https://aclanthology.org/W16-4806/)

* A Massively Multilingual Analysis of Cross-linguality in Shared Embedding Space \
[Paper](https://aclanthology.org/2021.emnlp-main.471/)
[Code](https://github.com/alexjonesnlp/xlanalysis5k)

* Towards a Common Understanding of Contributing Factors for Cross-Lingual Transfer in Multilingual Language Models: A Review \
[Paper](https://arxiv.org/abs/2305.16768)

* Decomposed Prompting for Machine Translation Between Related Languages using Large Language Models \
[Paper](https://arxiv.org/abs/2305.13085)
[Code](https://github.com/ratishsp/decomt)

### Training 

#### Joint Training / Language-Relatedness

* Multi-Way, Multilingual Neural Machine Translation with a Shared Attention Mechanism  
[Paper](https://aclanthology.org/N16-1101/) 

* Multi-Task Learning for Multiple Language Translation  
[Paper](https://aclanthology.org/P15-1166/)

* Multi-Task Learning for Multiple Language Translation  
[Paper](https://aclanthology.org/P15-1166/)

* Contact Relatedness can help improve multilingual NMT: Microsoft STCI-MT @ WMT20  
[Paper](https://aclanthology.org/2020.wmt-1.19/)

* Investigating Multilingual NMT Representations at Scale  
[Paper](https://aclanthology.org/D19-1167/)

* Enabling Multi-Source Neural Machine Translation By Concatenating Source Sentences In Multiple Languages  
[Paper](https://aclanthology.org/2017.mtsummit-papers.8/)

* Multilingual Neural Machine Translation with Language Clustering  
[Paper](https://aclanthology.org/D19-1089/)

* Bridging Linguistic Typology and Multilingual Machine Translation with Multi-View Language Representations  
[Paper](https://aclanthology.org/2020.emnlp-main.187/)
[Code](https://github.com/aoncevay/multiview-langrep)

* Delexicalized Cross-lingual Dependency Parsing for Xibe  
[Paper](https://aclanthology.org/2021.ranlp-1.182/)

* An Empirical Study of Language Relatedness for Transfer Learning in Neural Machine Translation  
[Paper](https://aclanthology.org/Y17-1038/)

* Efficient Unsupervised NMT for Related Languages with Cross-Lingual Language Models and Fidelity Objectives  
[Paper](https://aclanthology.org/2021.vardial-1.6/)

* Adapting High-resource NMT Models to Translate Low-resource Related Languages without Parallel Data  
[Paper](https://aclanthology.org/2021.acl-long.66/)

#### Data Curriculum / Multi-stage training

* Instance Weighting for Neural Machine Translation Domain Adaptation  
[Paper](https://aclanthology.org/D17-1155/) [Code](https://github.com/wangruinlp/nmt_instance_weighting)

* Exploiting Multilingualism through Multistage Fine-Tuning for Low-Resource Neural Machine Translation  
[Paper](https://aclanthology.org/D19-1146/)

* Data Selection Curriculum for Neural Machine Translation  
[Paper](https://aclanthology.org/2022.findings-emnlp.113/)

### Modeling

#### Mixture of Experts

* GShard: Scaling Giant Models with Conditional
Computation and Automatic Sharding  
[Paper](https://arxiv.org/abs/2006.16668)

* ST-MoE: Designing Stable and Transferable Sparse Expert Models    
[Paper](https://arxiv.org/abs/2202.08906) [Code](https://github.com/lucidrains/st-moe-pytorch)

* Towards Understanding Mixture of Experts in Deep Learning  
[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html) [Code](https://github.com/uclaml/MoE)

* Beyond Distillation: Task-level Mixture-of-Experts for Efficient Inference  
[Paper](https://aclanthology.org/2021.findings-emnlp.304/)

* Universal Neural Machine Translation for
Extremely Low Resource Languages  
[Paper](https://aclanthology.org/N18-1032/) [Code](https://github.com/MultiPath/NA-NMT/tree/universal_translation)

* Transfer Learning across Low-Resource, Related Languages for Neural Machine Translation  
[Paper](https://aclanthology.org/I17-2050/) [Code](https://aclanthology.org/I17-2050/)

#### Decoder-only MT models

* Examining Scaling and Transfer of Language Model Architectures for Machine Translation (LM4MT)  
[Paper](https://arxiv.org/abs/2202.00528)

* ALMA: Advanced Language Model-based translator  
[Paper](https://arxiv.org/abs/2309.11674) [Code](https://github.com/fe1ixxu/ALMA)

#### Zero-shot transfer-learning / Adaptation to new languages. 

* Rapid Adaptation of Neural Machine Translation to New Languages \
[Paper](https://aclanthology.org/D18-1103/) [Code](https://github.com/neubig/rapid-adaptation)

* Improving Zero-Shot Cross-lingual Transfer Between Closely Related Languages by Injecting Character-Level Noise \
[Paper](https://aclanthology.org/2022.findings-acl.321/)

* Utilizing Lexical Similarity to Enable Zero-Shot Machine Translation for Extremely Low-resource Languages \
[Paper](https://arxiv.org/abs/2305.05214)

* Improving Zero-Shot Translation by Disentangling Positional Information  
[Paper](https://aclanthology.org/2021.acl-long.101/) [Code](https://github.com/nlp-dke/NMTGMinor/tree/master/recipes/zero-shot)

* Simple, Scalable Adaptation for Neural Machine Translation
[Paper](https://aclanthology.org/D19-1165/)

* T-Modules: Translation Modules for Zero-Shot Cross-Modal Machine Translation  
[Paper](https://aclanthology.org/2022.emnlp-main.391/)

* Parameter Sharing Methods for Multilingual Self-Attentional Translation Models  
[Paper](https://aclanthology.org/W18-6327/) [Code](https://github.com/DevSinghSachan/multilingual_nmt)

* From Bilingual to Multilingual Neural Machine Translation by Incremental Training  
[Paper](https://aclanthology.org/P19-2033/)

* Language-Family Adapters for Low-Resource Multilingual Neural Machine Translation  
[Paper](https://aclanthology.org/2023.loresmt-1.5/)

* Improving Neural Machine Translation of Indigenous Languages with Multilingual Transfer Learning  
[Paper](https://aclanthology.org/2023.loresmt-1.6/)

### Model Compression
* Sequence-Level Knowledge Distillation  
[Paper](https://aclanthology.org/D16-1139/) [Code](https://github.com/harvardnlp/seq2seq-attn)

* Learning both Weights and Connections for Efficient Neural Networks  
[Paper](https://arxiv.org/abs/1506.02626)

* Memory-efficient NLLB-200: Language-specific Expert Pruning of a Massively Multilingual Machine Translation Model  
[Paper](https://aclanthology.org/2023.acl-long.198/) [Code](https://github.com/naver/nllb-pruning)

* LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale  
[Paper](https://arxiv.org/abs/2208.07339) [Code](https://github.com/TimDettmers/bitsandbytes)

* The case for 4-bit precision: k-bit Inference Scaling Laws [Paper](https://arxiv.org/abs/2212.09720)  

* An Empirical Study of Leveraging Knowledge Distillation for Compressing Multilingual Neural Machine Translation Models  
[Paper](https://aclanthology.org/2023.eamt-1.11/) [Code](https://github.com/VarunGumma/fairseq)

* Multilingual Neural Machine Translation with Language Clustering  
[Paper](https://aclanthology.org/D19-1089/)

### Evaluation

#### Automatic Evaluation

* Bleu: a Method for Automatic Evaluation of Machine Translation  
[Paper](https://aclanthology.org/P02-1040/)  

* chrF: character n-gram F-score for automatic MT evaluation  
[Paper](https://aclanthology.org/W15-3049/)  

* chrF++: words helping character n-grams  
[Paper](https://aclanthology.org/W17-4770/)  

* A Call for Clarity in Reporting BLEU Scores  
[Paper](https://www.aclweb.org/anthology/W18-6319) [Code](https://github.com/mjpost/sacrebleu)

* BLEURT: Learning Robust Metrics for Text Generation  
[Paper](https://arxiv.org/abs/2004.04696)
[Code](https://github.com/google-research/bleurt)

* Learning Compact Metrics for MT  
[Paper](https://arxiv.org/abs/2110.06341)

* IndicMT Eval: A Dataset to Meta-Evaluate Machine Translation Metrics for Indian Languages  
[Paper](https://aclanthology.org/2023.acl-long.795/) [Code](https://github.com/AI4Bharat/IndicMT-Eval)

* COMET: A Neural Framework for MT Evaluation  
[Paper](https://aclanthology.org/2020.emnlp-main.213/) [Code](https://github.com/Unbabel/COMET)

* Identifying Weaknesses in Machine Translation Metrics Through Minimum Bayes Risk Decoding: A Case Study for COMET  
[Paper](https://arxiv.org/abs/2202.05148)
[Code](https://github.com/ZurichNLP/mbr-sensitivity)

* Extrinsic Evaluation of Machine Translation Metrics  
[Paper](https://aclanthology.org/2023.acl-long.730/)

* Large Language Models Are State-of-the-Art
Evaluators of Translation Quality  
[Paper](https://arxiv.org/abs/2302.14520) [Code](https://github.com/MicrosoftTranslator/GEMBA)

* The Devil is in the Errors: Leveraging Large Language Models for Fine-grained Machine Translation Evaluation
[Paper](https://arxiv.org/abs/2308.07286)

##### Human Evaluation

* Continuous Measurement Scales in Human Evaluation of Machine Translation  
[Paper](https://aclanthology.org/W13-2305/)

* Is Machine Translation Getting Better over Time?  
[Paper](https://aclanthology.org/E14-1047/)

* Multidimensional quality metrics: a flexible system for assessing translation quality  
[Paper](https://aclanthology.org/2013.tc-1.6/)

* Experts, Errors, and Context: A Large-Scale Study of Human Evaluation for Machine Translation  
[Paper](https://aclanthology.org/2021.tacl-1.87/) [Code](https://github.com/google/wmt-mqm-human-evaluation)

* SemEval-2016 Task 1: Semantic Textual Similarity, Monolingual and Cross-Lingual Evaluation  
[Paper](https://aclanthology.org/S16-1081/)

* Consistent Human Evaluation of Machine Translation across Language Pairs  
[Paper](https://aclanthology.org/2022.amta-research.24/)


### Toolkits
* [FairseqV1](https://github.com/facebookresearch/fairseq)
* [FairseqV2](https://github.com/facebookresearch/fairseq2)
* [Transformers](https://github.com/huggingface/transformers/)
* [tensor2tensor](https://github.com/tensorflow/tensor2tensor)
* [trax](https://github.com/google/trax)
* [OpenNMT](https://github.com/OpenNMT/OpenNMT-py)
* [MarianNMT](https://github.com/marian-nmt/marian)
* [JoeyNMT](https://github.com/joeynmt/joeynmt)
* [Sockeye](https://github.com/awslabs/sockeye)
* [YANMTT](https://github.com/prajdabre/yanmtt)

## Citation
```bash
@InProceedings{gala-chitale-dabre:2023:ijcnlp,
  author    = {Gala, Jay  and  Chitale, Pranjal A.  and  Dabre, Raj},
  title     = {Developing State-Of-The-Art Massively Multilingual Machine Translation Systems for Related Languages},
  booktitle      = {Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics},
  month          = {November},
  year           = {2023},
  address        = {Nusa Dua, Bali},
  publisher      = {Association for Computational Linguistics},
  pages     = {35--42},
  url       = {https://aclanthology.org/2023.ijcnlp-tutorials.6}
}
```
