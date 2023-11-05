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

(Under construction)

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
