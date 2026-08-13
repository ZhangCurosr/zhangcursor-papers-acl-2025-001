# Second Language (Arabic) Acquisition of LLMs via Progressive Vocabulary Expansion

Jianqing Zhu†<sup>1</sup>, Huang Huang†<sup>2</sup>, Zhihang Lin†<sup>4</sup>, Juhao Liang†<sup>3,4</sup>, Zhengyang Tang†<sup>3,4</sup>   
Khalid Almubarak<sup>6</sup>, Abdulmohsen Alharthik<sup>1</sup>, Bang An<sup>1</sup>, Juncai He<sup>1</sup>, Xiangbo Wu<sup>3</sup>,   
Fei Yu<sup>3,4</sup>, Junying Chen<sup>3,4</sup>, Zhuoheng Ma<sup>4</sup>, Yuhao Du<sup>4</sup>, He Zhang<sup>4</sup>, Saied Alshahrani<sup>7</sup>, Emad A. Alghamdi<sup>5</sup>, Lian Zhang<sup>2</sup>, Ruoyu Sun<sup>3,4</sup>, Haizhou Li<sup>3,4</sup>, Benyou Wang\*<sup>3,4</sup>, Jinchao Xu<sup>1</sup>

<sup>1</sup> King Abdullah University of Science and Technology <sup>2</sup> Shenzhen International Center for Industrial and Applied Mathematics, Shenzhen Research Institute of Big Data <sup>3</sup> Shenzhen Research Institute of Big Data <sup>4</sup> The Chinese University of Hong Kong <sup>5</sup> King Abdulaziz University <sup>6</sup> Prince Sattam bin Abdulaziz University <sup>7</sup> University of Bisha

## Abstract

This paper addresses the critical need for democratizing large language models (LLM) in the Arab world, a region that has seen slower progress in developing models comparable to state-of-the-art offerings like GPT-4 or GPT-3.5, due to a predominant focus on mainstream languages (e.g., English and Chinese). One practical objective for Arabic LLMs is to utilize Arabic-specific vocabulary in the tokenizer to accelerate decoding. However, using a different vocabulary often leads to degradation of the model’s learned knowledge, since many words become out-of-vocabulary (OOV) at the beginning of training. Inspired by the vocabulary learning during Second Language (Arabic) Acquisition for humans, the released AraLLaMA employs progressive vocabulary expansion, which is implemented by a modified BPE algorithm that progressively extends the Arabic subwords in its dynamic vocabulary during training, thereby balancing the OOV ratio at every stage. The ablation study demonstrated the effectiveness of Progressive Vocabulary Expansion. Moreover, AraLLaMA achieves decent performance comparable to the best Arabic LLMs across a variety of Arabic benchmarks. Our model weights are available at: https://github.com/ FreedomIntelligence/AraLLaMa.

## 1 Introduction

In the evolving landscape of large language models (LLMs), the predominant focus has been on English and Chinese. This focus has left other linguistic communities, notably the Arab world, with slower progress in developing comparable models. Within the Arab world <sup>1</sup>, the development of models such as Jais (Sengupta et al., 2023) and AceGPT (Huang et al., 2024) marks a significant step forward, yet these models do not rival the capabilities of state-of-the-art models like GPT-4 (Achiam et al., 2023) or even GPT-3.5. In line with democratization (Touvron et al., 2023a,b), our development of Arabic LLMs focuses on language adaptation settings that utilize existing standard LLM architectures (like LLaMA (Touvron et al., 2023b)) and well-trained weights, thus saving computing resources and ensuring compatibility.

A primary challenge in adapting English-centric LLMs to other languages lies in vocabulary expansion (Touvron et al., 2023b; Cui et al., 2023; Huang et al., 2024; Zhao et al., 2024). For instance, AceGPT exhibits slower decoding speeds when processing Arabic, which may be attributed to limitations in its vocabulary adaptation mechanisms. It decodes Arabic words into sequences of alphabetical letters rather than at a more efficient granularity, such as Arabic subwords. This inefficiency significantly limits its broader applicability, despite its performance being nearly on par with GPT-3.5 in some benchmarks. A key concern related to vocabulary expansion is the risk that abrupt increases may result in a high incidence of out-of-vocabulary (OOV) tokens—units absent from the model’s established vocabulary. Such a surge in OOV words can compromise the linguistic knowledge embedded within the core models. Addressing this issue requires a considerable volume of pre-training data to restore and maintain the model’s linguistic capabilities effectively.

The core philosophy behind AraLLaMA is inspired by the process of vocabulary learning in human Second Language Acquisition, emphasizing that individuals typically expand their vocabulary gradually through incremental learning, rather than through instantaneous acquisition. AraLLaMA progressively extends the Arabic subwords in its vocabulary during pre-training, effectively reducing the ratio of OOV words at every stage. AraLLaMA, based on the initialization of LLaMA2 (Touvron et al., 2023b), not only retains the foundational knowledge of LLaMA2, but also enables effective cross-lingual transfer from English to Arabic. Ablation on TinyLLaMA (Zhang et al., 2024) demonstrated the effectiveness of the proposed progressive vocabulary expansion, see Section 6.1.

Followed by extensive instruction tuning, AraL-LaMA achieves decent performance comparable to the best Arabic LLMs across various Arabic benchmarks. The contributions of this work are three-fold: 1) We introduce progressive vocabulary expansion, utilizing a modified byte pair encoding (BPE) algorithm inspired by human Second Language Acquisition, and demonstrate its effectiveness. 2) We present AraLLaMA, a pioneering open-source Arabic Large Language Model that decodes Arabic texts three times faster than its predecessor (Huang et al., 2024) while delivering superior performance. 3) We provide the community with access to the complete data processing pipeline, pre-training/fine-tuning data, and model weights. AraLLaMA is compatible with the most popular LLM architecture (i.e., LLaMA) and can be seamlessly integrated into most LLM applications.

## 2 Motivation: Second Language Acquisition for Humans and LLMs

## 2.1 Cognitively-inspired Motivation: Second Language Acquisition for Humans

Definition 1. Second Language Acquisition (SLA) refers to the process by which people learn a language other than their native language (Krashen, 1981). SLA can occur through formal instruction in an educational setting or informally through social interaction and exposure to the language in natural settings.

In learning a second language (L2), learners pass through several developmental stages as they gain proficiency in L2, including the acquisition of phonetics, vocabulary, grammar, and pragmatics. Of these language skills, vocabulary acquisition is crucial for language learning. Several studies have posited that L2 learners mostly learn new words incidentally (Ramos and Dario, 2015; Nation, 2001). This suggests that an individual might gradually master a word or a set of words in an unconscious manner. This leads to a phenomenon:

Phenomenon 1. In Second Language Acquisition, human individuals typically expand their vocabulary gradually, in a fashion of incremental learning rather than an instantaneous acquisition.

A formal description of levels of language development is laid out in the Common European Framework of Reference for Languages (CEFR) <sup>2</sup>. Table 8 (show in Appendix B) showcases the required number of vocabulary size for different CEFR levels. The CEFR provides detailed descriptions of the skills language learners must achieve to effectively communicate. This can be taken as evidence of the progressive nature of vocabulary acquisition.

## 2.2 Problem Definition: Second Language Acquisition for LLMs

Language adaptation The focus on developing large-scale open-source language models for high-resource languages like English and Chinese has unintentionally marginalized low-resource languages, despite there being about 7,000 languages in use globally. The lack of data and computational resources makes it challenging to develop effective models for these languages. A common practice is to enhance existing models by adding specialized data for these underrepresented languages (Cui et al., 2023; Huang et al., 2024; Zhao et al., 2024), a.k.a, language adaptation.

Vocabulary expansion in language adaptation As a preliminary study, we identified Arabic tokens from LLaMA2 vocabulary using regular expressions. It was observed that LLaMA2 vocabulary only includes the basic characters of the Arabic language, resulting in relatively slow encoding and decoding speeds compared to English. During domain adaptation, it is crucial for vocabulary expansion for the second language, since it could significantly speed up decoding speeds as the number of decoded tokens is reduced due to the adapted vocabulary. Furthermore, although augmenting the existing vocabulary with tokens from additional languages, followed by training on corresponding language corpora, appears to be a logical strategy, empirical evidence suggests that the gains from this method are modest. This insight underscores the complexity of enhancing support for low-resource languages within the framework of current largescale language models.

![](images/85092e9b99dc3d5baf555d8fa18ae1b55a46fa88387ed135c55c89764124debe.jpg)  
Figure 1: Second Language Acquisition for human, an English-speaking child’s journey to Arabic fluency, from basic vocabulary to cultural roficiency

Research question Therefore, inspired by the humans’ Second Language Acquisition, we argue for

Is it beneficial to adopt progressive vocabulary learning in language adaptation ofLLMs?

## 3 Methodology: Progressive Vocabulary Expansion for Language Adaptation

Conventional Byte Pair Encoding (BPE) algorithms first create a complete vocabulary by iteratively merging the most frequent character pairs from a corpus, and then commence model training with this static vocabulary. This approach, while effective for monolingual models, presents challenges when adapting to new languages as it offers no mechanism for vocabulary evolution during training. To address this limitation, we propose Progressive Vocabulary Expansion.

## 3.1 Incremental Byte Pair Encoding Algorithm

In contrast to standard BPE algorithms (Sennrich et al., 2015) that use a static vocabulary, we introduce Incremental Byte Pair Encoding (I-BPE) that dynamically augments the vocabulary during training. This approach mirrors human language acquisition, where vocabulary growth occurs simultaneously with deepening language comprehension. Algorithm 1 outlines our method.

The key innovation of I-BPE is its staged approach to vocabulary expansion. At each stage, we expand the vocabulary to a predetermined size, then train the model while gradually increasing the proportion of data corresponding to newly added tokens. This approach significantly reduces out-ofvocabulary (OOV) tokens at each training phase, enabling the model to incorporate new linguistic elements while preserving previously acquired knowledge.

Algorithm 1 Incremental Byte Pair Encoding (I-  
BPE) Algorithm   
1: Input: (1) Initial vocabulary V; (2) Vocabu  
lary size at each stage: $s _ { 0 } , s _ { 1 } , \ldots , s _ { n } ; ( 3 )$ Pro  
portion of training corpus for newly added to  
kens at each stage: $r _ { 0 } , r _ { 1 } , \ldots , r _ { n } ;$   
2: Output: Final vocabulary V for model train  
ing and application   
3: for i = 0 to n do   
4: while $| V | < s _ { i }$ do   
5: Compute frequency of all adjacent to  
ken pairs in V   
6: Identify the most frequent token pair   
$P _ { f r e q }$   
7: Merge $P _ { f r e q }$ into a new token $T _ { n e w }$   
8: Add $T _ { n e w }$ to vocabulary V   
9: end while   
10: Adjust corpus proportion for newly added   
tokens to $r _ { i }$   
11: Train model with the updated vocabulary   
V until convergence   
12: end for   
13: Return Finalized vocabulary V

## 3.2 Expansion Strategies Comparison

For implementing vocabulary expansion, we in-كvestigated two principled strategies (illustrated in <sup>ه</sup>Figure 2):

Uniform Expansion: Adds a fixed number K of tokens at each stage, resulting in $( T - 1 ) \times K$ total additions over $T$ stages.

<sub>تل</sub>Exponential Expansion: Doubles the number of new tokens at each stage following the sequence $\{ 0 , 1 , 2 , \cdots , 2 ^ { T - 2 } \}$ , mimicking human language

acquisition patterns.

![](images/3e3deceb449807822a98ca141992d1bf34b7ba2d89b155a304f6742c664a07c4.jpg)  
Figure 2: Compression ratio comparison between uniform and exponential vocabulary expansion strategies.

Our comparative analysis using an identical Arabic corpus through 16 progressive stages revealed crucial differences between the two approaches. As shown in Figure 2 and detailed in Table 11, uniform expansion causes abrupt improvements in compression ratio during early stages followed by diminishing returns. This sudden introduction of many tokens creates training instabilities and risks catastrophic forgetting as the model’s representation space must rapidly accommodate numerous new tokens simultaneously.

Exponential expansion, however, offers critical advantages through its graduated approach: it provides superior training stability as the gradual introduction of tokens allows smooth adaptation of the model’s representation space; it maintains consistent OOV ratios throughout training, preventing sudden vocabulary distribution shifts; and it achieves significant computational efficiency with a 3 times reduction in sequence length compared to the original LLaMA tokenizer. Based on these findings, we implemented exponential expansion with 12,800 Arabic subwords across 16 stages $( \log _ { 2 } ^ { 1 2 8 0 0 } )$ ), representing the optimal saturation point for compression ratio improvement.

## 3.3 Compression Ratios and Tokenizer Evaluation

To rigorously assess the effectiveness of our vocabulary expansion approach, we conducted a comprehensive comparative evaluation of tokenization performance across multiple leading models. Using an identical Arabic corpus of 39 million words, we analyzed how different tokenizers processed Arabic text, with particular attention to efficiency metrics that impact both performance and computational requirements.

The results in Table 1 reveal notable differences in how these models handle Arabic text. Our model achieved a token compression ratio of 0.3174 (ratio of tokens to original text size), representing a 68% improvement over LLaMA2’s baseline, which directly enhances inference speed and reduces memory requirements. We evaluated several key metrics established in recent tokenizer evaluation literature:

1. Subword Fertility (Rust et al., 2021; Moosa et al., 2023): This metric measures the average number of tokens per word. Our model achieves the most optimal fertility (1.7063), approximately 3 times more efficient than LLaMA2 (5.3844) and Mistral (5.2833), while also outperforming multilingual models like Bloomz (Muennighoff et al., 2022) (2.0668) and Jais (1.9260) that were specifically designed with Arabic support.

2. Word Integrity (Moosa et al., 2023): For Arabic’s rich morphology, preserving words as single tokens is vital. Our model achieves 63.23% word integrity, far exceeding LLaMA2 (1.8%) and outperforming Arabicoptimized models like Jais (38.95%) and Bloomz (31.76%).

3. Total Tokens: For the identical test corpus, our model requires only 66.55 million tokens, compared to LLaMA2’s 210.03 million,a reduction of approximately 68% that translates directly to memory savings and computational efficiency in both training and inference phases.

4. Rényi Efficiency (Zouhar et al., 2023): This information-theoretic measure (higher values indicate better vocabulary utilization) shows our model (0.7491) achieves comparable efficiency to LLaMA2 (0.7731) despite its much lower token count, indicating efficient use of vocabulary space while maintaining high word integrity.

The comparative analysis indicates that our model achieves an optimal equilibrium between morphological preservation and computational efficiency. Although models such as LLaMA2 and Mistral exhibit marginally superior Rényi Efficiency coefficients, this advantage is offset by substantial deficiencies in word integrity preservation and significantly elevated token densities. When compared with models specifically optimized for Arabic processing, such as Jais and Bloomz, our model consistently demonstrates superior performance across the majority of evaluation metrics, validating the efficacy of the progressive vocabulary expansion methodology for non-Latin script languages.

<table><tr><td>Tokenizer</td><td>Total Words</td><td>Total Tokens</td><td>Subword Fertility</td><td>Ratio of Words Unbroken</td><td>Rényi Efficiency</td></tr><tr><td>LLaMA2(AceGPT)</td><td>39,006,442</td><td>210,027,671</td><td>5.3844</td><td>0.0183</td><td>0.7731</td></tr><tr><td>Bloomz</td><td>39,006,442</td><td>80,617,499</td><td>2.0668</td><td>0.3176</td><td>0.7709</td></tr><tr><td>Mistral</td><td>39,006,442</td><td>206,082,344</td><td>5.2833</td><td>0.0185</td><td>0.7928</td></tr><tr><td>Jais</td><td>39,006,442</td><td>75,126,494</td><td>1.9260</td><td>0.3895</td><td>0.7343</td></tr><tr><td>Our model</td><td>39,006,442</td><td>66,554,771</td><td>1.7063</td><td>0.6323</td><td>0.7491</td></tr></table>

Table 1: Comprehensive tokenizer evaluation using standard metrics across different models.

## 4 Training Methodology

Based on the Progressive Vocabulary Expansion methodology described above, we develop AraL-LaMA, an Arabic Large Language Model that implements our proposed I-BPE algorithm. In this section, we detail the AraLLaMA training process, including data engineering and training specifics.

## 4.1 Data Engineering

Pre-training Corpora Our pre-training dataset comprises both Arabic and English corpora. We employ an array of Arabic corpora encompassing multiple categories as delineated in Table 9 (shown in Appendix D). These include filtered versions of Common Crawl, WebText, and Wikipedia1 sourced from Joud and BAAI, all of which were subjected to additional cleaning processes. Moreover, we gather and purify additional corpora, namely Wikipedia2, Books, and Newspapers. The English corpus is sourced from SlimPajama (Soboleva et al., 2023) and Proof-Pile-2 (Azerbayev et al., 2023).

Data for Instruction Tuning After pre-training, we aim to elicit the knowledge out of AraLLaMA via instruction tuning. Inspired by GLAN (Li et al., 2024), we introduce ALAN (Arabic Instruction Tuning for Language Models). This method utilizes specific topics targeting Arabic knowledge to generate a vast amount of synthetic instruction data.

Specifically, we identified 127 critical topics within Arabic culture, science, and engineering as our focus. ALAN decomposes these topics into a structured hierarchy of fields, sub-fields, and individual disciplines. For each discipline, ALAN compiles a comprehensive list of subjects and designs a syllabus with specific knowledge points for each one. Using GPT-4-0613, ALAN has generated 11,430 subjects and 244,812 detailed knowledge points. We provide more concrete examples in Appendix H.

Armed with this extensive collection of subjects and knowledge points, we direct the LLM to create questions and answers related to these knowledge concepts. The syllabus consists of several lectures, each with 2 to 5 knowledge points. To diversify the knowledge base, we combine knowledge points from both the same and different lectures to produce diverse instructions and answers. Additionally, to vary the instruction types, the LLM generates three kinds of questions at random: multiplechoice, open-ended, and coding questions. In total, we’ve generated 733,419 instruction tuning data pieces.

We also incorporated instruction tuning data from previous AceGPT projects. These include Quora-Arabic, Alpaca-Arabic (Taori et al., 2023), Code-Alpaca-Arabic (Chaudhary, 2023), Evol-Instruct-Arabic (Xu et al., 2023), and ShareGPT data.

## 4.2 Training details

The refined methodology for LLaMA2 model’s vocabulary expansion incorporated 12,800 Arabic subwords derived through the I-BPE algorithm. The initialization procedure for each new token employed decomposition into constituent subwords from the original LLaMA2 vocabulary, with embedding initialization achieved via averaging the embeddings of these component tokens. This initialization strategy preserves semantic relationships between new and existing tokens, thereby enhancing training stability and facilitating vocabulary integration.

The training procedure was structured into 16 distinct stages <sup>3</sup>, each processing 30B tokens, culminating in a total corpus of 480B tokens. A cosine annealing schedule governed the proportion of Arabic to English content, with Arabic representation increasing systematically from 30% to 90% across stages. This progressive exposure enables gradual adaptation to Arabic linguistic structures while preserving cross-lingual transfer capabilities via continued English exposure. Mathematical and programming content was maintained at a consistent 5% throughout all stages to ensure robust inference capabilities in these domains (see Appendix E). The final training distribution comprised approximately 251.4B Arabic tokens and 204.6B English tokens.

The pre-training framework consisted of two principal epochs: an initial epoch employing vocabulary annealing for data distribution optimization, followed by a secondary epoch utilizing the fully refined vocabulary. An additional 20B tokens of training data were processed subsequent to vocabulary expansion to further enhance model performance. Each training phase implemented a discrete cosine learning rate schedule with warm-up period, producing a vocabulary-specific model at its conclusion, thereby rendering each phase functionally independent.

This stage-wise approach facilitates systematic integration of new tokens, enabling the model to adapt to evolving data representations while developing comprehensive understanding of linguistic patterns. The graduated modulation of language distribution—progressively increasing Arabic content while decreasing English representation—optimizes the model’s capacity to process Arabic while maintaining cross-lingual capabilities.

The implementation utilized LLaMA2 architecture in 7B and 13B parameter configurations, trained on 2,368 Ascend 910A processors. Training durations were 7 and 11 days for the 7B and 13B models, respectively. The computational configuration employed model parallelism of degree 2 and pipeline parallelism of degree 4. Optimization was conducted using AdamW with 4,096-token context length. Each training stage utilized a cosine learning rate scheduler initialized at 1e-5 and decaying to 2e-6, with a 15% warm-up interval. Gradient accumulation factor 8 yielded an effective batch size of 4,736, enabling processing of approximately 0.019B tokens per batch.

## 5 Experiments

## 5.1 Experimental settings

Benchmarking Datasets To assess world knowledge, we employ four widely used benchmarks. MMLU (Measuring Massive Multitask Language Understanding) (Hendrycks et al., 2021a) evaluates knowledge acquired during pretraining across a broad range of subjects; we utilize both the original English version and the Arabic version introduced in (Huang et al., 2024) to ensure multilingual coverage. RACE (Reading Comprehension from Examinations) serves as a large-scale English reading comprehension benchmark that focuses on educational knowledge. EXAMS (Multi-subject High School Examinations Dataset for Cross-lingual and Multilingual Question Answering) further expands coverage by including subject-diverse exam questions drawn from multiple languages and curricula. ArabicMMLU complements these by providing an Arabic-specific variant of MMLU, tailored to reflect regional knowledge across various Arab countries and subjects. Beyond general knowledge evaluation, we also examine cultural and value alignment using ACVA-all and ACVA-clean, which focus on Arabic cultural relevance and localization. To comprehensively evaluate inference and reasoning abilities, we translate two commonsense reasoning benchmarks of varying difficulty—BoolQ and ARC-Challenge (ARC-C)—into Arabic, allowing for consistent cross-lingual assessment.

To ensure a fair comparison of candidate models, we adhere to the settings established for each benchmark separately. Furthermore, for translated benchmarks, we utilize the generation approach evaluation method as outlined in (Huang et al., 2024). Specifically, we employed GPT-3.5-Turbo-1106 to translate datasets from English to Arabic for benchmarks that were not originally in Arabic.

Baselines To compare LLMs trained or available in Arabic, we have selected several prominent Arabic LLMs or multilingual LLMs as baselines for comparison: (1) AceGPT-[7B,13B]: This set includes fully fine-tuned generative text models based on LLaMA2, specifically customized for the Arabic domain. (2) Mistral-7B-Instructv0.2 (Jiang et al., 2023): The fine-tuned model achieves a balance between performance and efficiency. (3) Jais-[13B,30B] (Sengupta et al., 2023): A pre-trained bilingual large language model designed for both Arabic and English. (4) Bloom-[7B]: A multilingual language model extensively trained on diverse textual data, allowing it to produce fluent text in 46 languages and 13 programming languages. (5) LLaMA2-[7B,13B]: A popular and competitive baseline model in the general domain. (6) OpenAI GPT: This includes GPT4 and ChatGPT, closed-source LLMs also strong at multilingual tasks.

## 5.2 Evaluation Results

Evaluation on Base Models In our study, the performance of base models was assessed on two Arabic-specific MMLU datasets: Arabic MMLU translate (Huang et al., 2024) and ArabicMMLU (Koto et al., 2024). The left side of Table 2 details the models’ accuracies on the Arabic MMLU translate dataset within a few-shot setting. It is evident from the data that the AraLLaMA-7B-base and AraLLaMA-13B-base models exhibit superior accuracy rates compared to models of similar scale. Notably, the AraLLaMA-13B-base model outperforms the Jais-30B model, which has a significantly larger parameter count.

Additionally, the right side of Table 2 presents the accuracy results of models in a zero-shot learning scenario. Here again, the AraLLaMA models stand out for their exceptional performance, even when compared to models with similar parameter sizes. In particular, the AraLLaMA-13B-base model demonstrates a marked advantage over the Jais-30B-base model, notwithstanding the latter’s larger size in terms of parameters.

These findings affirm the effectiveness of the AraLLaMA models, developed through an annealing algorithm to expand the vocabulary, highlighting our methodology as a productive strategy for enhancing large models’ adaptability to less prevalent languages. This contribution significantly advances the field of language model adaptation, offering a novel avenue for enriching language technology’s inclusivity and depth.

Evaluation on Chat Models Table 3 presents the comprehensive evaluation results across various benchmarks for the candidate models, spanning from Arabic to English. Overall, AraL-LaMA outperforms all baseline models in the Arabic tasks. Particularly noteworthy is its proficiency in knowledge-related evaluations such as Arabic-translated MMLU and EXAMS, surpassing other models by at least 1.3%. This highlights the model’s expertise in addressing Arabic knowledge-related questions. Additionally, AraL-LaMA demonstrates strong performance in tasks related to Arabic culture and value alignment. In terms of commonsense reasoning, AraLLaMA exhibits notable skills in tasks such as the translated versions of BoolQ and ARC-Challenge, showcasing its reasoning capabilities in Arabic. Beyond Arabic benchmarks, we also investigated the English proficiency of the models to determine whether specialization in one language affects performance in the other. The results indicate that the model maintains its English proficiency and displays robustness in multilingual assessments. It is noteworthy that the lower accuracy of the Jais is attributed to its refusal to answer for unknown reasons.

In a comprehensive evaluation of the ACVA dataset aimed at gauging the understanding of Arabic cultural nuances under a zero-shot setting, our AraLLaMA models showcased unparalleled performance. The AraLLaMA-13B-chat, in particular, stood out with exceptional Average F1 scores of 76.37% and 76.90% in “all set" and "clean Set" categories, respectively, even outperforming the renowned GPT-3.5 Turbo in the "All set" category. This performance not only highlights the AraL-LaMA models’ superior grasp of Arabic culture but also establishes them as leading figures among open-source models in this nuanced domain. Compared to other top-tier open-source contenders, including the Jais-30B-chat variants, the AraLLaMA-13B-chat model’s superior results. The instructionfollowing tests can be found in Appendix I.

## 6 More Analysis

## 6.1 Ablation Study on Progressive Vocabulary Expansion

To further demonstrate the effectiveness of progressive vocabulary expansion in downstream task adaptation, we conduct continuous pre-training on a 1B-parameter TinyLLaMA model (Zhang et al., 2024), followed by supervised fine-tuning. More details on the experimental setup can be found in Appendix J.

A comprehensive analysis is conducted by applying the same Supervised Fine-Tuning (SFT) protocol across three pre-training configurations: the baseline TinyLLaMA model, TinyLLaMA with Progressive Vocabulary Expansion (PVE), and TinyLLaMA with Vocabulary Expansion all at once (VE). The performance of these models is evaluated on the Arabic MMLU (see Table 4) and Arabic Vicuna-80 (see Table 5) benchmarks. Experiment results demonstrate that vocabulary expansion significantly enhances model performance, with the PVE approach yielding superior results across various categories in the Arabic MMLU benchmark, achieving an average score of 40.7 compared to 38.5 for VE and 36.5 for the baseline model. Similarly, in the Arabic Vicuna-80 comparison, the PVE method led to the highest accuracy of 29.18%, outperforming VE (22.61%) and the baseline model (21.3%). These results underscore the effectiveness of progressive vocabulary expansion in enhancing language model performance, particularly in complex language tasks.

<table><tr><td rowspan="2">Models</td><td colspan="5">Arabic-trans MMLU (Huang et al., 2024)</td><td rowspan="2"></td><td colspan="4">ArabicMMLU (Koto et al., 2024)</td><td rowspan="2">Avg.</td><td rowspan="2">Total Avg.</td></tr><tr><td>STEM</td><td>Human- ities</td><td>Social Sciences</td><td>Others</td><td>Avg. STEM</td><td>Social Sciences</td><td>Human- ities</td><td>Arabic Language</td><td>Other</td></tr><tr><td>Bloomz-7B-base</td><td>33.35</td><td>29.29</td><td>37.58</td><td>34.53</td><td>33.69</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLaMA2-7B-base</td><td>30.30</td><td>29.33</td><td>27.46</td><td>30.78</td><td>29.47</td><td>33.7</td><td>32.8</td><td>33.5</td><td>28.4</td><td>36.7</td><td>33.4</td><td>31.43</td></tr><tr><td>AceGPT-7B-base</td><td>29.73</td><td>30.95</td><td>33.45</td><td>34.42</td><td>32.14</td><td>35.4</td><td>35.9</td><td>36.2</td><td>31.1</td><td>41.7</td><td>36.3</td><td>34.22</td></tr><tr><td>AraLLaMA-7B-base</td><td>33.03</td><td>32.08</td><td>35.39</td><td>35.59</td><td>34.03</td><td>36.7</td><td>36.5</td><td>34.1</td><td>30.0</td><td>41.2</td><td>37.0</td><td>35.52</td></tr><tr><td>LLaMA2-13B-base</td><td>32.94</td><td>32.30</td><td>33.42</td><td>37.27</td><td>33.76</td><td>32.9</td><td>35.0</td><td>37.8</td><td>35.8</td><td>39.3</td><td>36.1</td><td>34.93</td></tr><tr><td>Jais-13B-base</td><td>30.51</td><td>31.25</td><td>33.74</td><td>33.43</td><td>33.76</td><td>30.3</td><td>31.4</td><td>33.6</td><td>28.1</td><td>36.3</td><td>32.2</td><td>32.98</td></tr><tr><td>AceGPT-13B-base</td><td>36.60</td><td>38.74</td><td>43.76</td><td>42.72</td><td>40.45</td><td>42.7</td><td>45.5</td><td>48.3</td><td>42.4</td><td>50.7</td><td>46.1</td><td>43.28</td></tr><tr><td>AraLLaMA-13B-base</td><td>36.13</td><td>40.07</td><td>45.43</td><td>42.17</td><td>40.95</td><td>42.4</td><td>45.7</td><td>48.4</td><td>46.3</td><td>52.5</td><td>47.6</td><td>44.28</td></tr><tr><td>Jais-30B-v1-base</td><td>32.67</td><td>30.67</td><td>42.13</td><td>39.60</td><td>36.27</td><td>39.5</td><td>45.6</td><td>50.5</td><td>34.6</td><td>49.1</td><td>44.8</td><td>40.54</td></tr><tr><td>GPT-3.5 Turbo</td><td>43.38</td><td>44.12</td><td>55.57</td><td>53.21</td><td>49.07</td><td>53.8</td><td>57.0</td><td>57.5</td><td>57.6</td><td>63.8</td><td>57.7</td><td>53.39</td></tr></table>

Table 2: Evaluation of base models. We adopt a few-shot setting on Arabic-translated MMLU (Huang et al., 2024) and a zero-shot setting with option logit probability in ArabicMMLU (Koto et al., 2024). Numbers with the best performance are in bold in 7B and 13B groups.
<table><tr><td rowspan="2">Models</td><td colspan="8">Arabic</td><td rowspan="2">English</td><td colspan="2"></td><td rowspan="2">|Total Avg.</td></tr><tr><td>MMLU (trans)</td><td>MMLU (Koto et al., 2024)</td><td>EXAMS</td><td>clean</td><td>all</td><td>(trans)</td><td>ACVA ACVA BoolQ ARC-C| (trans)</td><td>Avg.</td><td>BoolQ RACE</td><td>Avg.</td></tr><tr><td>LLaMA2-7B-chat</td><td>13.78</td><td>33.40</td><td>13.05</td><td>20.99</td><td>21.80</td><td>34.92</td><td>23.72</td><td>|21.09</td><td>71.31</td><td>50.49</td><td></td><td>|31.49</td></tr><tr><td>Phoenix-7b</td><td>29.72</td><td>44.74</td><td>31.93</td><td>43.80</td><td>41.86</td><td>66.70</td><td>33.53</td><td>41.75</td><td>62.23</td><td>60.97</td><td>|60.90 61.60</td><td>46.16</td></tr><tr><td>AceGPT-7B-chat</td><td>30.69</td><td>36.31</td><td>33.73</td><td>53.87</td><td>53.07</td><td>60.70</td><td>38.05</td><td>43.77</td><td>54.74</td><td>53.97</td><td>54.36</td><td>46.12</td></tr><tr><td>Mistral-7B-Instruct-v0.2</td><td>27.93</td><td>41.44</td><td>21.56</td><td>64.56</td><td>63.47</td><td>60.18</td><td>35.67</td><td>44.97</td><td>84.53</td><td>73.17</td><td>78.85</td><td>52.50</td></tr><tr><td>AraLLaMA-7B-chat</td><td>45.77</td><td>56.62</td><td>43.69</td><td>69.46</td><td>70.86</td><td>72.45</td><td>60.49</td><td>59.90</td><td>75.78</td><td>72.13</td><td>73.96</td><td>63.02</td></tr><tr><td>Jais-13B-chat</td><td>19.52</td><td>54.83</td><td>19.71</td><td>66.75</td><td>61.41</td><td>41.25</td><td>11.95</td><td>39.34</td><td>28.13</td><td>20.08</td><td>24.10</td><td>35.96</td></tr><tr><td>LLaMA2-13B-chat</td><td>8.92</td><td>36.12</td><td>16.11</td><td>35.12</td><td>35.71</td><td>54.13</td><td>27.47</td><td>30.51</td><td>62.87</td><td>48.28</td><td>55.58</td><td>36.08</td></tr><tr><td>AceGPT-13B-chat</td><td>35.59</td><td>52.61</td><td>38.72</td><td>70.82</td><td>70.21</td><td>66.85</td><td>44.20</td><td>54.14</td><td>60.55</td><td>45.22</td><td>52.88</td><td>53.86</td></tr><tr><td>AraLLaMA-13B-chat</td><td>47.33</td><td>61.70</td><td>48.37</td><td>76.90</td><td>76.37</td><td>69.33</td><td>63.99</td><td>63.42</td><td>83.67</td><td>80.82</td><td>82.24</td><td>67.61</td></tr><tr><td>Jais-30B-chat-v1</td><td>38.12</td><td>59.33</td><td>40.45</td><td>74.46</td><td>72.41</td><td>73.76</td><td>50.94</td><td>58.49</td><td>65.05</td><td>75.26</td><td>70.16</td><td>61.09</td></tr><tr><td>Jais-30B-chat-v3</td><td>35.68</td><td>62.36</td><td>32.24</td><td>73.63</td><td>73.66</td><td>76.30</td><td>51.02</td><td>57.84</td><td>79.54</td><td>85.23</td><td>82.43</td><td>63.29</td></tr><tr><td>GPT-3.5 Turbo</td><td>46.07</td><td>57.72</td><td>45.63</td><td>74.45</td><td>76.88</td><td>76.12</td><td>60.24</td><td>62.44</td><td>85.32</td><td>84.65</td><td>84.99</td><td>67.45</td></tr></table>

Table 3: Chat Models Evaluation in zero-shot setting. Numbers with best performance are in bold in 7B and 13B groups.

## 6.2 Benchmarking in English dataset

We evaluated the accuracy of both base and chat models on the English MMLU dataset. As illustrated in Table 2 (shown in Appendix G), in the base model category, AraLLaMA’s accuracy is slightly lower than that of the original LLaMA model but notably higher than the AceGPT model, which is also trained on the LLaMA architecture. This indicates that expanding Arabic capabilities via an annealing algorithm does not compromise the model’s inherent English proficiency. This offers a viable solution for language transfer in large models. After undergoing SFT, AraLLaMA achieves the highest accuracy among models of similar size and surpasses the Jais-30B model, which has a greater number of parameters.

## 6.3 Decoding Efficiency Analysis

We conducted a systematic evaluation of generation efficiency between LLaMA2 and AraLLaMA 7B chat models on Arabic text generation tasks. Each model was tested on standardized Arabic prompts with a maximum output length of 100 tokens. To ensure statistical reliability, we performed five independent trials and analyzed only Arabic language outputs, excluding any non-Arabic tokens from the performance calculations.

Table 6 shows that while both models achieve similar token processing speeds ( 30 tokens/second, $\begin{array} { r l r } { p } & { { } > } & { 0 . 0 5 ) } \end{array}$ , AraLLaMA generates words 4.5× faster. This efficiency gain (from $4 . 5 5 \pm 0 . 5 0$ to $2 0 . 3 7 \pm 0 . 0 4$ words/second) demonstrates the effectiveness of our vocabulary expansion approach. The improved word-level performance while maintaining similar token-level speeds indicates that our language-specific tokenization strategy successfully optimizes text generation for Arabic’s morphological complexity.

<table><tr><td>Model</td><td>STEM</td><td>Social Sciences</td><td>Humanities</td><td>Arabic Language</td><td>Other</td><td>Avg.</td></tr><tr><td>TinyLLaMA chat</td><td>35.1</td><td>36.9</td><td>38.5</td><td>28.6</td><td>39.8</td><td>36.5</td></tr><tr><td>TinyLLaMA (VE) chat</td><td>35.3</td><td>39.7</td><td>40.1</td><td>33.8</td><td>41.6</td><td>38.5</td></tr><tr><td>TinyLLaMA (PVE) chat</td><td>36.3</td><td>40.7</td><td>44.2</td><td>33.5</td><td>45.7</td><td>40.7</td></tr></table>

Table 4: Performance comparison on ArabicMMLU (Koto et al., 2024) across different domains.

<table><tr><td>Model</td><td>Accuracy (%)</td></tr><tr><td>TinyLLaMA chat</td><td>21.30 (baseline)</td></tr><tr><td>TinyLLaMA (VE) chat</td><td> $2 2 . 6 1 \ : ( + 1 . 3 1 )$ </td></tr><tr><td>TinyLLaMA (PVE) chat</td><td> ${ \bf 2 9 . 1 8 \left( + 7 . 8 8 \right) }$ </td></tr></table>

Table 5: Performance Comparison on Arabic Vicuna-80 Benchmark

<table><tr><td>Model</td><td>Tokens/Second</td><td>Words/Second</td></tr><tr><td>LLaMA2</td><td> $2 9 . 6 8 \pm 0 . 0 4$ </td><td> $4 . 5 5 \pm 0 . 5 0$ </td></tr><tr><td> $_ \mathrm { A r a L L a M A }$ </td><td> $3 0 . 1 2 \pm 0 . 0 6$ </td><td> $2 0 . 3 7 \pm 0 . 0 4$ </td></tr></table>

Table 6: Comparative analysis of generation speed between LLaMA2 and AraLLaMA on Arabic text.

## 7 Conclusion

Adapting large-scale models to less commonly spoken languages is fraught with challenges, notably the hurdles of knowledge transfer and the prevalence of OOV terms. We developed a novel annealing training algorithm to address these issues specifically for Arabic. This strategy methodically expands the vocabulary and employs a phased training process, leading to the development of the AraLLaMA 7B and 13B models. Subsequent evaluations of both the base and chat configurations across diverse datasets have unequivocally established AraLLaMA’s superior accuracy compared to peers within the same parameter range. Remarkably, the AraLLaMA also exhibits robust performance advantages over models with significantly more parameters. The proven efficacy of our algorithm is supported by robust empirical evidence. Moving forward, we aim to further democratize access to advanced model technology by making our models, along with their code and datasets, openly available, thus making a meaningful contribution to the progress of the field.

## Limitation

This paper exhibits several limitations. Due to constraints in resources and budget, the models has not undergone evaluation by native Arabic speakers, which could affect its practicality and adoption. Consequently, its use is currently confined to academic research rather than online deployment. Additionally, the writing of this paper was supported by AI tools for grammar correction and refinement.

## Acknowledgements

This work was conducted under the platform of the KAUST-SRIBD Joint Lab on Scientific Computing and Machine Learning. We would like to acknowledge the support of Hetao Shenzhen-Hong Kong Science and Technology Innovation Cooperation Zone Project (No. HZQSWS-KCCYB-2024016), the Shenzhen Science and Technology Program (JCYJ20220818103001002), Shenzhen Doctoral Startup Funding (RCBS20221008093330065), Tianyuan Fund for Mathematics of National Natural Science Foundation of China (NSFC) (12326608), Shenzhen Key Laboratory of Cross-Modal Cognitive Computing (grant number ZDSYS20230626091302006), Shenzhen Stability Science Program 2023, Shenzhen Key Lab of Multi-Modal Cognitive Computing, and KAUST Baseline Research Fund.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q. Jiang, Jia Deng, Stella Biderman, and Sean Welleck. 2023. Llemma: An open language model for mathematics. Preprint, arXiv:2310.10631.

Kaj Bostrom and Greg Durrett. 2020. Byte pair encoding is suboptimal for language model pretraining. arXiv preprint arXiv:2004.03720.

Sahil Chaudhary. 2023. Code alpaca: An instruction-following llama model for code generation. https://github.com/sahil280114/ codealpaca.

Zhihong Chen, Feng Jiang, Junying Chen, Tiannan Wang, Fei Yu, Guiming Chen, Hongbo Zhang, Juhao Liang, Chen Zhang, Zhiyi Zhang, et al. 2023. Phoenix: Democratizing chatgpt across languages. arXiv preprint arXiv:2304.10453.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. 2023. Vicuna: An opensource chatbot impressing gpt-4 with 90%\* chatgpt quality.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. Boolq: Exploring the surprising difficulty of natural yes/no questions. In NAACL.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv:1803.05457v1.

James Coady. 1996. L2 vocabulary acquisition through extensive reading. In Second language vocabulary acquisition: A rationale for pedagogy, pages 225– 237. Cambridge University Press.

Mike Conover, Matt Hayes, Ankit Mathur, Jianwei Xie, Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell, Matei Zaharia, and Reynold Xin. 2023. Free dolly: Introducing the world’s first truly open instructiontuned llm.

Yiming Cui, Ziqing Yang, and Xin Yao. 2023. Efficient and effective text encoding for chinese llama and alpaca. arXiv preprint arXiv:2304.08177.

Momchil Hardalov, Todor Mihaylov, Dimitrina Zlatkova, Yoan Dinkov, Ivan Koychev, and Preslav Nakov. 2020. EXAMS: A multi-subject high school examinations dataset for cross-lingual and multilingual question answering. In Proceedings ofthe 2020

Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 5427–5444. Association for Computational Linguistics.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021a. Measuring massive multitask language understanding. Preprint, arXiv:2009.03300.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021b. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

Huang Huang, Fei Yu, Jianqing Zhu, Xuening Sun, Hao Cheng, Song Dingjie, Zhihong Chen, Mosen Alharthi, Bang An, Juncai He, et al. 2024. Acegpt, localizing large language models in arabic. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8139–8163.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. Preprint, arXiv:2310.06825.

Fajri Koto, Haonan Li, Sara Shatnawi, Jad Doughman, Abdelrahman Boda Sadallah, Aisha Alraeesi, Khalid Almubarak, Zaid Alyafeai, Neha Sengupta, Shady Shehata, et al. 2024. Arabicmmlu: Assessing massive multitask language understanding in arabic. arXiv preprint arXiv:2402.12840.

Stephen Krashen. 1981. Second language acquisition. Second Language Learning, 3(7):19–39.

Stephen Krashen. 1982. Principles and practice in second language acquisition. Pergamon Press.

Taku Kudo. 2018. Subword regularization: Improving neural network translation models with multiple subword candidates. arXiv preprint arXiv:1804.10959.

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. 2017. RACE: Large-scale ReAding comprehension dataset from examinations. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 785– 794, Copenhagen, Denmark. Association for Computational Linguistics.

Héctor Daniel León Romero et al. 2016. Spaced retrieval practice applied to vocabulary learning in secondary education.

Haoran Li, Qingxiu Dong, Zhengyang Tang, Chaojun Wang, Xingxing Zhang, Haoyang Huang, Shaohan Huang, Xiaolong Huang, Zeqiang Huang, Dongdong Zhang, et al. 2024. Synthetic data (almost) from scratch: Generalized instruction tuning for language models. arXiv preprint arXiv:2402.13064.

Ibraheem Muhammad Moosa, Mahmud Elahi Akhter, and Ashfia Binte Habib. 2023. Does transliteration help multilingual language modeling? In Findings of the Association for Computational Linguistics: EACL 2023, pages 670–685, Dubrovnik, Croatia. Association for Computational Linguistics.

Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng-Xin Yong, Hailey Schoelkopf, et al. 2022. Crosslingual generalization through multitask finetuning. arXiv preprint arXiv:2211.01786.

Tatsuya Nakata. 2015. Effects of expanding and equal spacing on second language vocabulary learning: Does gradually increasing spacing increase vocabulary learning? Studies in Second Language Acquisition, 37(4):677–711.

I. S. P. Nation. 2001. Learning Vocabulary in Another Language. Cambridge Applied Linguistics. Cambridge University Press.

Ian SP Nation and I. S. P. Nation. 2001. Learning vocabulary in another language, volume 10. Cambridge University Press.

Xuan-Phi Nguyen, Wenxuan Zhang, Xin Li, Mahani Aljunied, Qingyu Tan, Liying Cheng, Guanzheng Chen, Yue Deng, Sen Yang, Chaoqun Liu, et al. 2023. Seallms–large language models for southeast asia. arXiv preprint arXiv:2312.00738.

Restrepo Ramos and Falcon Dario. 2015. Incidental vocabulary learning in second language acquisition: A literature review. Profile Issues in TeachersProfessional Development, 17(1):157–166.

Phillip Rust, Jonas Pfeiffer, Ivan Vulic, Sebastian Ruder,´ and Iryna Gurevych. 2021. How good is your tokenizer? on the monolingual performance of multilingual language models. In Proceedings of the 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3118–3135. Association for Computational Linguistics.

Elizabeth Salesky, Andrew Runge, Alex Coda, Jan Niehues, and Graham Neubig. 2020. Optimizing segmentation granularity for neural machine translation. Machine Translation, 34(1):41–59.

Neha Sengupta, Sunil Kumar Sahu, Bokang Jia, Satheesh Katipomu, Haonan Li, Fajri Koto, Osama Mohammed Afzal, Samta Kamboj, Onkar Pandit, Rahul Pal, et al. 2023. Jais and jais-chat: Arabic-centric foundation and instruction-tuned open

generative large language models. arXiv preprint arXiv:2308.16149.

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2015. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909.

Daria Soboleva, Al-Khateeb Faisal, Myers Robert Steeves Jacob R, Hestness Joel, and Dey Nolan. 2023. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/ stanford\_alpaca.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023a. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Ahmet Üstün, Viraat Aryabumi, Zheng-Xin Yong, Wei-Yin Ko, Daniel D’souza, Gbemileke Onilude, Neel Bhandari, Shivalika Singh, Hui-Lee Ooi, Amr Kayid, et al. 2024. Aya model: An instruction finetuned open-access multilingual language model. arXiv preprint arXiv:2402.07827.

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. 2023. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244.

Jingjing Xu, Hao Zhou, Chun Gan, Zaixiang Zheng, and Lei Li. 2020. Vocabulary learning via optimal transport for neural machine translation. arXiv preprint arXiv:2012.15671.

Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. 2024. Tinyllama: An open-source small language model. arXiv preprint arXiv:2401.02385.

Songshan Zhang, Hai Xu, and Xian Zhang. 2021. The effects of dictionary use on second language vocabulary acquisition: A meta-analysis. International Journal of Lexicography, 34(1):1–38.

Jun Zhao, Zhihao Zhang, Qi Zhang, Tao Gui, and Xuanjing Huang. 2024. Llama beyond english: An empirical study on language capability transfer. arXiv preprint arXiv:2401.01055.

Vilém Zouhar, Clara Meister, Juan Gastaldi, Li Du, Mrinmaya Sachan, and Ryan Cotterell. 2023. Tokenization and the noiseless channel. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5184–5207, Toronto, Canada. Association for Computational Linguistics.

## A Related Work

Our work primarily focuses on two key areas: lowresource language models and vocabulary expansion.

Low-resource language models Recent efforts have centered on developing open-source LLMs as alternatives to proprietary models like GPT-3.5 Turbo and GPT-4 (Taori et al., 2023; Chiang et al., 2023; Conover et al., 2023; Chen et al., 2023; Sengupta et al., 2023). These initiatives have expanded beyond English, addressing languages with fewer available resources and creating models specifically tailored to diverse linguistic landscapes (Chen et al., 2023; Üstün et al., 2024). SeaLLMs (Nguyen et al., 2023) are adapted from English-centric models by extending vocabulary and fine-tuning to better capture regional language complexities. Jais (Sengupta et al., 2023) introduces a model trained from scratch based on GPT architecture, while AceGPT Huang et al., 2024 offers a model designed to adapt to local Arabic culture, specifically tailored to regional nuances. This trend highlights the growing need for multilingual LLMs that perform well in low-resource environments while maintaining competitive performance against more established models.

Vocabulary expansion Vocabulary expansion for large language models (LLMs) has become a crucial area of research, particularly for improving performance in low-resource languages. Traditional methods like Byte Pair Encoding (BPE), while effective at handling out-of-vocabulary (OOV) words, are suboptimal for pretraining larger models, as discussed by Tay et al. (Bostrom and Durrett, 2020), who propose alternative tokenization methods to better capture linguistic nuances. Pham et al. (Xu et al., 2020) advance this by introducing optimal transport-based vocabulary learning, which optimizes the distribution of subword units, enhancing translation tasks, particularly in multilingual and low-resource settings.

Kudo et al. (Kudo, 2018) propose subword regularization and offer another avenue for improvement by allowing models to learn from multiple subword segmentation rather than a fixed one, increasing robustness and flexibility. In contexts with limited data, Liu et al. (Salesky et al., 2020) have demonstrated that combining subword-based methods with additional pretraining steps significantly improves model performance. These works show that moving beyond traditional vocabulary methods allows for more dynamic and context-aware modeling, enhancing LLMs’ scalability and adaptability across diverse linguistic landscapes.

## B CEFR Language Proficiency Levels

Table 8 illustrates the vocabulary size that learners are expected to acquire at various stages of second language acquisition. The vocabulary size is gradually expanding when humans acquire a second language, as one cannot achieve proficiency in all second-language words at once, as it takes time to digest these words.

## C Cognitive Mechanisms of Vocabulary Acquisition

We reviewed relevant literature to confirm the phenomenon of exponential vocabulary expansion in second language acquisition and the cognitive theories that support it. Studies indicate that learners typically begin by mastering a small set of highfrequency vocabulary in the early stages of language learning. As they progress, their vocabulary size grows rapidly. This process can be explained through the following two aspects:

Cognitive Mechanisms of Incremental Learning In the initial stages, learners build their understanding by repeatedly encountering and using simple foundational words. Research by (Krashen, 1982) and (Nation and Nation, 2001) shows that mastering high-frequency vocabulary is crucial for understanding more complex linguistic structures. These foundational words provide a stable cognitive base, allowing learners to gradually expand their vocabulary (Zhang et al., 2021; Nakata, 2015).

Exponential Vocabulary Growth Once learners acquire foundational vocabulary, the rate of vocabulary expansion accelerates. Through extensive reading and structured learning strategies such as spaced retrieval practice (León Romero et al., 2016), learners are able to acquire complex vocabulary in a relatively short period. (Coady, 1996) emphasize that extensive reading provides a large amount of language input, enabling learners to incrementally encounter and absorb more advanced vocabulary.

## D Arabic data distribution

Table 9 show the Arabic dataset primarily draws from several key sources, with the largest contribution coming from the Common Crawl (filtered) dataset, which accounts for 55.5% of the total data. Other significant sources include WebText, which contributes 26.7%, and Books+Newspapers, providing 8.9% with 2.5 billion tokens. Additionally, Wikipedia is divided into two parts, contributing 3.76% and 5.14%. These diverse sources collectively form the foundation for training the Arabic model.

## E Data mixture

Table 10 shows the data distribution across the pretraining stages is carefully adjusted, with the proportions of Arabic and English data determined using a cosine annealing schedule. Initially, the Arabic data constitutes 30% of the total, while English data makes up 65% and math & coding data consistently accounts for 5%. As the training progresses and new subwords are added, the proportion of Arabic data increases steadily, reaching 90% by the final stage. Concurrently, the English data proportion decreases to 5%, while the math & coding data remains constant at 5% throughout all stages. This dynamic adjustment ensures that the model effectively balances the learning of Arabic and English content, with a strong emphasis on Arabic in the later stages.

## F Comparison of compression ratio and OOV changes at different stages between exponential and uniform expansion

Table 11 illustrates the trends in compression ratio and OOV (Out-Of-Vocabulary) ratio as vocabulary size is incrementally expanded using both Exponential and Uniform methods. In the case of \*\*Exponential Vocabulary Expansion\*\*, both the compression ratio and OOV ratio change gradually, ensuring a more balanced progression as new subwords are added. This gradual change is beneficial for maintaining stability during model training, as it allows the system to adjust incrementally to the growing vocabulary.

## G Evaluation of models in English MMLU dataset

In the evaluation of English MMLU performance, AraLLaMA models, both 7B and 13B, consistently outperform their counterparts across most categories in both few-shot and zero-shot settings (shown in Table 2). Particularly, AraLLaMA-13B achieves the highest average score of 62.89 in zeroshot tasks, demonstrating its superior capability in generalization and task adaptability.

<table><tr><td>Aspect</td><td>Benchmark</td><td>Language (+ translation)</td><td>Size</td><td>Evaluation Types</td><td>Metrics</td></tr><tr><td rowspan="4">Knowledge Ability</td><td>RACE (Lai et al., 2017)</td><td>EN</td><td>4.9K</td><td>Multiple-choice Questions</td><td>Accuracy</td></tr><tr><td>MMLU (Hendrycks et al., 2021b)</td><td>EN (+AR)</td><td>14K</td><td>Multiple-choice Questions</td><td>Accuracy</td></tr><tr><td>ArabicMMLU (Koto et al., 2024)</td><td>AR</td><td>14.5K</td><td>Multiple-choice Questions</td><td>Accuracy</td></tr><tr><td>EXAMS (Hardalov et al., 2020)</td><td>AR</td><td>0.56K</td><td>Multiple-choice Questions</td><td>Accuracy</td></tr><tr><td rowspan="2">Arabic Cultural and Value Alignment</td><td>ACVA-all (Huang et al., 2024)</td><td>AR</td><td>9K</td><td>Yes/No binary Questions</td><td>F1-score</td></tr><tr><td>ACVA-clean</td><td>AR</td><td>2.48K</td><td>Yes/No binary Questions</td><td>F1-score</td></tr><tr><td rowspan="2">Commonsense Reasoning</td><td>BoolQ (Clark et al., 2019)</td><td>EN (+AR)</td><td>3.27K</td><td>Yes/No binary Questions</td><td>Accuracy</td></tr><tr><td>ARC-Challenge (Clark et al., 2018)</td><td>(+AR)</td><td>1.17K</td><td>Multiple-choice Questions</td><td>Accuracy</td></tr></table>

Table 7: Overview of Evaluation benchmarks
<table><tr><td>CEFR Level</td><td>Description</td><td></td><td>Learning Hours Vocabulary Size</td></tr><tr><td rowspan="2">Basic User</td><td>A1</td><td>Beginner Level</td><td>110-130 2000 words</td></tr><tr><td>A2</td><td>Elementary Level</td><td>150-180 3000 words</td></tr><tr><td rowspan="2">Independent User</td><td>B1</td><td>Intermediate Level</td><td>200-230 5000 words</td></tr><tr><td>B2</td><td>Upper Intermediate Level</td><td>200-230 8000 words</td></tr><tr><td rowspan="2">Proficient User</td><td>C1</td><td>Advanced Level</td><td>150-200 10000 words</td></tr><tr><td>C2</td><td>Mastery Level</td><td>250-300 30000 words</td></tr></table>

Table 8: CEFR Language Proficiency Levels.

## H ALAN examples

We provide concrete examples of ALAN below. Note that we translate examples into English using GPT-3.5-Turbo. In practice, our data is in Arabic.

## H.1 Topics

A set of 30 topics, randomly chosen, is listed below:

"Arabic Language and Literature" "Mathematics" "Islamic Studies" "Middle Eastern History and Politics" "Computer science" "Economics" "Healthcare industry" "Social work" "Business" "Geography" "Mining" "Chemical Engineering" "Languages and Literature" "Materials Science and Engineering" "Transport industry" "Chemistry" "Food industry" "Systems science" "Astronomy" "Cultural industry" "Energy industry" "Radiology" "Pediatrics" "Dentistry" "Civil Engineering" "Aerospace industry" "Public administration" "Infectious disease" "Public policy" "Environmental studies and forestry"

## H.2 Subjects

A set of 30 subjects, randomly chosen, is listed below:

"Hypersonic and High-Speed Flows" "Mental Health Nursing" "Mechanical Systems and Energy Efficiency" "Obstetrics and Gynecological Nursing" "Immunology" "Interdisciplinary Geriatric Care" "Signal Processing" "Geography research methods and techniques" "Public Administration and Management" "An introduction to space exploration" "Environmental and Safety Management" "Social and Ethical Aspects of Agriculture" "Folk and Cultural Dance" "Power System Protection and Control" "Collage and Mixed Media" "Advanced Game Theory" "Pediatric Critical Care" "Transport Modeling and Forecasting" "Foundations of Mathematics" "Carbon Capture, Storage, and Utilization" "Customer Service and Relationship Management" "Introduction to Probability" "Virtual Reality and Augmented Reality" "Reservoir Management and Enhanced Oil Recovery" "Safety and Standards in Industrial Robotics" "Social Work with LGBTQ+ populations" "Nutritional Science" "Advanced Gynaecology Courses" "Bioinformatics and Computational Chemistry" "Reusable Launch Vehicle Technology"

## H.3 A syllabus with specific knowledge points We provide an example syllabus with specific knowledge points as below.

Subject title: Hypersonic and High-Speed Flows Lecture title: Introduction to Hypersonic Flows Knowledge points:

\- Definition of hypersonic flows

<table><tr><td>Dataset</td><td># tokens</td><td>Weight in training mix</td></tr><tr><td>Common Crawl (filtered)</td><td>101.3 billion</td><td>55.5%</td></tr><tr><td>WebText</td><td>10.62 billion</td><td>26.7%</td></tr><tr><td>Books+Newspapers</td><td>2.5 billion</td><td>8.9%</td></tr><tr><td>Wikipedia1</td><td>0.36 billion</td><td>3.76%</td></tr><tr><td>Wikipedia2</td><td>0.51 billion</td><td>5.14%</td></tr></table>

Table 9: Arabic data distribution and elapsed epochs
<table><tr><td>Stage</td><td>New subwords added</td><td>Arabic data</td><td>English data</td><td>math &amp; coding data</td></tr><tr><td>1</td><td>0</td><td>30.00%</td><td>65.00%</td><td>5.00%</td></tr><tr><td>2</td><td>1</td><td>30.33%</td><td>64.47%</td><td>5.00%</td></tr><tr><td>3</td><td>2</td><td>31.31%</td><td>63.69%</td><td>5.00%</td></tr><tr><td>4</td><td>4</td><td>32.94%</td><td>62.06%</td><td>5.00%</td></tr><tr><td>5</td><td>8</td><td>35.19%</td><td>59.81%</td><td>5.00%</td></tr><tr><td>6</td><td>16</td><td>38.04%</td><td>56.96%</td><td>5.00%</td></tr><tr><td>7</td><td>32</td><td>41.46%</td><td>53.54%</td><td>5.00%</td></tr><tr><td>8</td><td>64</td><td>45.41%</td><td>49.59%</td><td>5.00%</td></tr><tr><td>9</td><td>128</td><td>49.85%</td><td>45.15%</td><td>5.00%</td></tr><tr><td>10</td><td>256</td><td>54.73%</td><td>40.27%</td><td>5.00%</td></tr><tr><td>11</td><td>512</td><td>60.00%</td><td>35.00%</td><td>5.00%</td></tr><tr><td>12</td><td>1024</td><td>65.60%</td><td>29.40%</td><td>5.00%</td></tr><tr><td>13</td><td>2048</td><td>71.46%</td><td>23.54%</td><td>5.00%</td></tr><tr><td>14</td><td>4196</td><td>77.53%</td><td>17.47%</td><td>5.00%</td></tr><tr><td>15</td><td>8192</td><td>83.73%</td><td>11.27%</td><td>5.00%</td></tr><tr><td>16</td><td>12800</td><td>90.00%</td><td>5.00%</td><td>5.00%</td></tr></table>

Table 10: Detailed distribution of Arabic, English and math & coding data across each pre-training stage.

- Mach number - Vehicle configurations   
- Key characteristics of hypersonic flows - Advantages and limitations of each configuration   
Lecture title: Fundamentals of Shock Waves Lecture title: Aerothermodynamics of Hypersonic   
Knowledge points: Flows   
- Definition of shock waves Knowledge points:   
- Formation of shock waves - Definition of aerothermodynamics   
- Types of shock waves - Aerothermodynamics in hypersonic flows   
Lecture title: High-Temperature Gas Dynamics - Heat transfer in hypersonic flows   
Knowledge points: Lecture title: Hypersonic Flow Control   
- Definition of high-temperature gas dynamics Knowledge points:   
- Behavior of high-temperature gases - Importance of flow control   
- Effects of high-temperature gases on materials - Methods of hypersonic flow control   
Lecture title: Principles of Rarefied Gas Dynamics - Challenges in hypersonic flow control   
Knowledge points: Lecture title: Hypersonic Propulsion Systems   
- Definition of rarefied gas dynamics Knowledge points:   
- The continuum hypothesis - Types of hypersonic propulsion systems   
- Governing equations - Working principles   
Lecture title: High-Speed Flow Over Bodies - Advantages and disadvantages   
Knowledge points: Lecture title: Future Trends in Hypersonic and   
- High-speed flow characteristics High-Speed Flows   
- Impact on the body Knowledge points:   
- Aerodynamic heating - Current research in the field   
Lecture title: Hypersonic Vehicle Configurations - Potential future trends   
Knowledge points: - Challenges and opportunities   
- Types of hypersonic vehicles

<table><tr><td>Add Subword Size</td><td>Compress Ratio (Exponential)</td><td>OOV Ratio (Exponential)</td><td>Add Subword Size</td><td>Compress Ratio (Uniform)</td><td>OOV Ratio (Uniform)</td></tr><tr><td>0</td><td>0.90</td><td>0.000</td><td>0</td><td>0.90</td><td>0.000</td></tr><tr><td>1</td><td>0.88</td><td>0.017</td><td>853</td><td>0.45</td><td>0.669</td></tr><tr><td>2</td><td>0.87</td><td>0.018</td><td>1736</td><td>0.40</td><td>0.116</td></tr><tr><td>4</td><td>0.85</td><td>0.022</td><td>2559</td><td>0.37</td><td>0.068</td></tr><tr><td>8</td><td>0.82</td><td>0.038</td><td>3412</td><td>0.35</td><td>0.049</td></tr><tr><td>16</td><td>0.77</td><td>0.061</td><td>4265</td><td>0.34</td><td>0.039</td></tr><tr><td>32</td><td>0.72</td><td>0.076</td><td>5118</td><td>0.33</td><td>0.031</td></tr><tr><td>64</td><td>0.65</td><td>0.094</td><td>5971</td><td>0.32</td><td>0.026</td></tr><tr><td>128</td><td>0.60</td><td>0.093</td><td>6824</td><td>0.31</td><td>0.021</td></tr><tr><td>256</td><td>0.54</td><td>0.105</td><td>7677</td><td>0.31</td><td>0.019</td></tr><tr><td>512</td><td>0.48</td><td>0.116</td><td>8530</td><td>0.30</td><td>0.017</td></tr><tr><td>1024</td><td>0.43</td><td>0.110</td><td>9383</td><td>0.30</td><td>0.015</td></tr><tr><td>2048</td><td>0.39</td><td>0.118</td><td>10236</td><td>0.30</td><td>0.013</td></tr><tr><td>4096</td><td>0.34</td><td>0.120</td><td>11089</td><td>0.29</td><td>0.012</td></tr><tr><td>8192 12800</td><td>0.31</td><td>0.116</td><td>11942</td><td>0.29</td><td>0.011</td></tr><tr><td></td><td>0.28</td><td>0.070</td><td>12800</td><td>0.28</td><td>0.010</td></tr></table>

Table 11: Comparison of Exponential and Uniform Vocabulary Expansion Methods

## H.4 Synthetic QA data

We provide a synthetic QA example using knowledge points generated by GPT-3.5-Turbo.

Subject title:   
Computer Vision for Industrial Robotics   
Lecture title:   
Stereo Vision and 3D Reconstruction   
Knowledge points:   
- Principles of stereo vision   
- Stereo camera calibration   
- Depth estimation and 3D reconstruction   
- Point cloud processing   
Synthetic question:   
In stereo vision, the process of determining the   
depth of objects in a scene is known as:   
A. Image rectification   
B. Disparity mapping   
C. Camera calibration   
D. Point cloud processing   
Synthetic solution to the question:   
B

The correct answer is B. Disparity mapping. In stereo vision, the depth of objects in a scene is determined by calculating the disparity between corresponding points in the left and right images. Disparity mapping involves finding the pixel-level differences between the two images to estimate the depth information.

## I Instruction-following test

We evaluated the models’ instruction-following capabilities using the Arabic versions of Vicuna-80 (Chiang et al., 2023), translated by GPT-4 and refined by native speakers. Following the methodology in (Chiang et al., 2023), GPT-4 was used as the evaluator, assigning scores to each model’s performance compared to GPT-3.5 Turbo, with a temperature setting of 0.2. For each question, GPT-4 independently scored the responses from both the evaluated model and GPT-3.5 Turbo. The average performance ratio of the evaluated model was calculated by dividing its overall score by that of GPT-3.5 Turbo. Results in Table 13 indicate that AraLLaMA models outperform their counterparts in Arabic Vicuna-80. Notably, AraLLaMA-7B exceeds Jais-13B by approximately 17%, despite having a smaller model size.

<table><tr><td>Model</td><td>Ratio of GPT-3.5</td></tr><tr><td>Jais-13B</td><td>75.40%</td></tr><tr><td>Llama-7B</td><td>78.99%</td></tr><tr><td>AraLLaMA-7B</td><td>92.71%</td></tr></table>

Table 13: Performance ratio of GPT-3.5 Turbo in Arabic Vicuna-80.

## J Details of Ablation Study

## J.1 Experiment Settings:

We undertook continuous pre-training on a 1Bparameter TinyLLaMA model (Zhang et al., 2024), which is derived from the LLaMA architecture and was initially trained on an English corpus comprising 3 trillion tokens. The pre-training regimen was segmented into five distinct stages, during which

![](images/5885f7ff07977861e36dba1cdaa8d13e81214ca3eb470412d928cfe77b2dd968.jpg)  
Figure 3: Loss curve of TinyLLaMa with sliding window average

0, 16, 64, 256, and 1024 Arabic subwords were progressively added to the vocabulary. Each stage allocated a different volume of data, totaling 80 billion tokens, with the proportion of Arabic to English data gradually shifting from 0:10 to 9:1. In a parallel experiment, we introduced 1024 subwords to the vocabulary in a single step, maintaining the same total token count and data distribution as in the phased approach. Both experiments adhered to an identical learning rate strategy, reinstating a cosine learning rate scheduler at the onset of each stage, starting with an initial rate of 1e-5 and tapering to 2e-6, with the initial 5 billion tokens of each stage designated for warm-up. Utilizing 192 GPUs, the experiments were conducted with a batch size of 3072.

## J.2 Progressive Vocabulary Expansion Pre-training

The results shown in Figure 3 demonstrate that the strategy of progressively expanding the vocabulary, which applies a sliding window average technique, yields a reduced final loss. Furthermore, as evidenced in Table 14, within the ArabicMMLU dataset, the approach of incrementally introducing new vocabulary items consistently outperforms the method of a one-time vocabulary expansion. This pattern underscores the effectiveness of gradual vocabulary enhancement in optimizing model performance.

<table><tr><td>Model</td><td>STEM</td><td>Social Sciences</td><td>Humanities</td><td>Arabic Other Language</td><td> $\operatorname { A v g }$ </td></tr><tr><td>Expand vocab at once</td><td>28.6</td><td>26.7</td><td>28.1 24.4</td><td>30.1</td><td>27.0</td></tr><tr><td>Gradually expand vocab (ours)</td><td>29.8</td><td>27.1</td><td>27.2 24.6</td><td>31.4</td><td>27.3</td></tr></table>

Table 14: Zero-shot evaluation for TinyLLaMA in ArabicMMLU (Koto et al., 2024) with option logit probabiltiy