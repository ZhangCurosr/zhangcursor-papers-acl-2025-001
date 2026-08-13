# Steering into New Embedding Spaces: Analyzing Cross-Lingual Alignment Induced by Model Interventions in Multilingual Language Models

Anirudh Sundar<sup>1†</sup> <sup>‡</sup> Sinead Williamson <sup>2</sup> Katherine Metcalf <sup>2</sup>

Barry-John Theobald<sup>2</sup> Skyler Seto <sup>2</sup> Masha Fedzechkina <sup>2</sup> <sup>‡</sup>

<sup>1</sup>AI Virtual Assistant Lab, Georgia Institute of Technology

<sup>2</sup>Apple

asundar34@gatech.edu, {sa\_williamson, kmetcalf, bjtheobald, sseto, mfedzechkina}@apple.com

## Abstract

Aligned representations across languages is a desired property in multilingual large language models (mLLMs), as alignment can improve performance in cross-lingual tasks. Typically alignment requires fine-tuning a model, which is computationally expensive, and sizable language data, which often may not be available. A data-efficient alternative to finetuning is model interventions — a method for manipulating model activations to steer generation into the desired direction. We analyze the effect of a popular intervention (finding experts) on the alignment of cross-lingual rep resentations in mLLMs. We identify the neurons to manipulate for a given language and introspect the embedding space of mLLMs preand post-manipulation. We show that modifying the mLLM’s activations changes its embedding space such that cross-lingual alignment is enhanced. Further, we show that the changes to the embedding space translate into improved downstream performance on retrieval tasks, with up to 2x improvements in top-1 accuracy on cross-lingual retrieval.

## 1 Introduction

Large language models (LLMs) exhibit impressive performance on a variety of tasks from text summarization to zero-shot common-sense reasoning (Raffel et al., 2020; Liu and Lapata, 2019; Bosselut et al., 2019; Richardson and Heck, 2023) and are increasingly deployed in a variety of fields ranging from health to entertainment (Singhal et al., 2025; Wu et al., 2023; Zhong et al., 2024). Despite these capabilities, to ensure that deployed LLMs align with human values, are non-toxic, and do not hallucinate, they often must be adapted post-training (Wei et al., 2024; Rodriguez et al., 2024; Ouyang et al., 2022).

Model interventions have emerged as dataand compute-efficient tools for model adaptation, whereby targeted updates are applied to model activations after pre-training (Rodriguez et al., 2024; Li et al., 2023; Rimsky et al., 2024). One such method isfinding experts (Suau et al., 2022, 2024) which manipulates the activations of expert neurons responsible for encoding a broadly defined concept (e.g., a word or style of text) to steer model generations into a desired direction. This approach has been successfully used in a variety of domains, ranging from achieving gender parity (Suau et al., 2022), to reducing toxicity (Suau et al., 2024), studying geopolitical biases (Faisal and Anastasopoulos, 2023) and multilingual capabilities (Kojima et al., 2024) in mLLMs.

While model interventions successfully control model generations, prior work does not fully detail their effects on model performance. Two observations are relevant. First, model intervention methods increase perplexity on a fixed dataset postintervention (Suau et al., 2024) meaning that the intervention introduces changes in how the model represents language. Second, prior work (Kojima et al., 2024) has shown that intervening on experts for a given language not only increases the probability of the mLLM generating text in that language but also leads to an improvement in prompt-based translation performance, suggesting that the intervention may increase the alignment between representations of different languages.

In this work, we focus on representational changes in mLLMs, with an emphasis on crosslingual alignment, for two reasons. First, gains in mLLM performance are largely attributed to better alignment of multilingual representations (Wu et al., 2024; Lample et al., 2018). This has generated a lot of interest in improving multilingual alignment (Chaudhary et al., 2020; Efimov et al., 2023; Lample and Conneau, 2019; Liu et al., 2025). Second, datasets with the same text in multiple languages are available for a variety of tasks, which enables us to study the impact of the intervention in a controlled way across multiple languages.

![](images/f6323c72a471821c3e3ab8c654731ada2643b1b28ef7d680bf2faed685040bd5.jpg)  
Figure 1: Following the intervention on expert neurons for Spanish, the LLM embeddings for text from different languages cluster more closely together (left, see Section 4). As a result, this intervened model is better than the unintervened model at cross-lingual retrieval where the task is to retrieve the correct translation of a sentence in a query language (right, see Section 5).

Specifically, we examine changes in the embedding space of mLLMs introduced by the finding experts intervention and link these changes to downstream task performance (see Fig. 1). We hypothesize that this intervention increases cross-lingual alignment in mLLMs and present results supporting this hypothesis. Specifically, we find that the intervention projects all languages into a new representation space within the mLLM that is characterized by new properties, some of which are desirable and some are not. On the downside, perplexity increases post-intervention, indicating a degradation in some aspects of the model’s language modeling capability. However, the intervention also leads to more aligned cross-lingual representations, as evidenced by reduced distances between language embeddings (Section 4). The increased alignment translates into a performance gain on cross-lingual retrieval with up to 2x improvement in top-1 accuracy (Section 5), while preserving within-language similarity (Section 6).

## 2 Related Work

Model interventions. Model interventions are a family of approaches that manipulate model activations to control generations (Li et al., 2023; Turner et al., 2024; Rodriguez et al., 2024). Suau et al. (2022) propose a method to identify neurons in pre-trained transformer models that are most predictive of a particular concept (expert neurons) and show that setting the activations of these experts to their mean value can induce the presence of the target concept in model generations. Suau et al. (2024) find the expert neurons for toxic language and steer the LLM to generate less toxic text by dampening these neurons, while Turner et al. (2024) achieve detoxification by using a contrastive prompt. Rimsky et al. (2024) propose a method to control generations by leveraging the differences in residual stream activations between pairs of positive and negative examples. In mLLMs, Kojima et al. (2024) use this approach to produce more target language tokens in open-ended generation. However, prior work does not analyze the changes these interventions introduce in the representational space of mLLMs nor does it explore the impact of the interventions on cross-lingual alignment.

Aligning multilingual representations in mLLMs. Research on LLM representation alignment falls into two broad categories: 1) Improving model performance on downstream tasks via post-training methods such as prompt-based techniques (Huang et al., 2023; Tanwar et al., 2023), fine-tuning, or continuous pre-training (Zhang et al., 2023; Li et al., 2024). 2) Understanding where and how representation alignment is achieved in mLLMs. For example, Wendler et al. (2024) show that English-dominated mLLMs like Llama-2 use English as a pivot language and Zhao et al. 2024 systematically evaluate factors contributing to successful cross-lingual transfer in such models.

## 3 Methods

We seek to understand the impact of model interventions on the representational space of mLLMs with a focus on cross-lingual alignment. We consider three open-source mLLMs: Aya-8b (instruction fine-tuned) (Aryabumi et al., 2024), PolyLM-13b (chat version) (Wei et al., 2023), and Bloom-7b (base) (Scao et al., 2022). Since our aim is to draw conclusions about cross-lingual alignment, we want to make sure that we know what languages were seen in pre-training and include mLLMs for which a detailed description of pre-training datasets is available, excluding LLMs such as Mistral (Jiang et al., 2023), Llama (Touvron et al., 2023), and Gemma (Team et al., 2024). We begin by identifying and intervening on the language experts in the mLLMs and then study cross-lingual alignment in the embedding space and downstream task performance pre- and post-intervention.

## 3.1 Probing dataset construction

Following Kojima et al. (2024), we use the Flores200 dataset (NLLB Team, 2022) to find the expert neurons for a particular target language (i.e., the language specifically targeted by the intervention). Flores200 is a machine translation dataset containing short paragraphs sampled from Wikimedia<sup>1</sup> and subsequently translated into 204 languages by skilled human translators. We limit our investigations to the intervention on five target languages — English, German, French, Spanish, and Japanese. These languages are well represented in pre-training data of the models we are considering, ensuring the existence of expert neurons.

## 3.2 Identifying expert neurons

Expert neurons for a given language are identified following Suau et al. (2024), see Fig. 2. In this approach, a concept of interest c (in our case, a particular language) is defined by a set of example sentences $N = N _ { c } ^ { + } + N _ { c } ^ { - }$ , where $N _ { c } ^ { + }$ is a set of sentences that contain c and $N _ { c } ^ { - }$ is a set of sentences that do not contain c. The activations $\big \{ \{ z _ { m , i } ^ { c } \} _ { i = 1 } ^ { N } \big \} _ { m = 1 } ^ { M }$ of every neuron m in the MLP layers are obtained for inputs from both sets of sentences. The activations $z _ { m , i } ^ { c }$ are used to predict $b ^ { c } ~ = ~ \{ b _ { i } ^ { c } \} _ { i = 1 } ^ { N }$ , where $b _ { i } ^ { c } = \mathbf { 1 } _ { i \in N _ { c } ^ { + } }$ The expertise of neuron m is then defined as the area under the receiver operating curve (AUROC) of this binary classification task, indicating the extent to which the activation of m correctly predicts the presence of c. The requirement for a neuron to be considered an expert for a given language is that its performance as a classifier for the language is above chance (AUROC>0.5). In practice, however, we select only the top experts with the k highest AUROC values (mean AUROC across all intervention targets for Aya-8b is 0.97, Bloom-7b is 0.99, PolyLM-13b is 0.83). We define intervention on each expert m as setting the activations of this expert to its average activation $\mathbb { E } _ { i } \{ z _ { m , i } ^ { c } \} _ { i = 1 } ^ { N _ { c } ^ { + } }$ for the probing set $N _ { c } ^ { + }$ , see Section 3.3.

![](images/8808cfc573e9903217b0ede1e17d11c2011e33a056d45ea8f711f6ba9fe1a2b8.jpg)  
Figure 2: Illustration of the finding experts intervention. First, the activations of all MLP neurons in response to the positive and negative language examples are collected. Next, these activations are used to predict the target language label. The neurons with the highest AU-ROC on this task are considered experts. Intervening on the top k experts increases the probability of target language generation in response to a neutral prompt.

For each of the five languages under consideration, we use the Flores200 dev split for the target language as the positive set $( N _ { c } ^ { + } )$ , and the dev splits for the other four languages plus Chinese as the negative set $\left( N _ { c } ^ { - } \right)$ . We include Chinese to increase variety in the character systems in the negative set but we do not consider it for the positive set $( N _ { c } ^ { + } )$

## 3.3 Intervening on expert neurons

For the intervention, we select the k neurons with the highest expertise (i.e., highest AUROC). We select the value for k that balances generating text in the target language with a low perplexity on the language-specific Wikipedia text. Specifically, for each of the five languages, we sweep over expert set sizes ranging from 100 to 5000. For each setting of language and number of experts, we run freeform generation to generate 256 sentences over eight random seeds (for a total of 2048 sentences) using the beginning of sentence (<BOS>) token as the prompt. We perform generation with temperature=1 and top $\scriptstyle - { \mathrm { p } } = 0 . 9 ^ { 2 }$ . We then use lang-id (Lui and Baldwin, 2012) to measure the probability of the text generated in the target language.

To calculate the perplexity of Wikipedia text in the target language for the original and intervened models, we use the Wikimedia dump from 2023- 11-01<sup>3</sup>. Paragraphs of text shorter than 100 characters are removed and the remaining paragraphs are concatenated together. Finally, a corpus of 10 million tokens is selected from the concatenated paragraphs. The context length is set to the model’s maximum input size (in tokens) and a stride (i.e., the context sliding window) of 512 tokens is used to speed up the perplexity measurement. The activation of the k neurons is set to their respective mean value calculated over the positive sentences (Suau et al., 2022).

For almost all target languages, the probability of generating that language increases postintervention (Fig. 3, top), suggesting that the intervention is successful. The only exception is English in the Aya-8b model, where the intervention reduces the likelihood of generating English. We believe that the intervention steers the model away from the default configuration, and English is the default language for that model. Interestingly, despite Bloom-7b’s training set containing neither German nor Japanese, the intervention results in generating both languages with high probability. Our hypothesis is that the Bloom-7b pretraining data contains some amount of German and Japanese data that is large enough to enable expert discovery and controlled generation.

Example generations for all models are provided in App. B. Overall, all models tend to generate target language tokens. For cases where the probability of target language generation is below 1, we observe that some of the generated sentences are in a language different from the target (typically English) or contain non-language symbols such as code. We do not see generations where the tokens from different languages are mixed in the same sentence, with an exception of some Aya-8b generations where the model starts out with an English word or phrase and then continues with the target language.

While we are successfully able to increase the accuracy of target language generation through the intervention, consistent with prior work (Suau et al., 2024), we observe an increase in perplexity postintervention as the number of activated neurons increases, see Fig. 3 (bottom), suggesting that activating experts introduces changes into the model representation. Thus, the choice of the number of experts is a trade-off between inducing the desired behavior and degrading the model. As a result, for our analyses, we set k to 100 experts for Bloom-7b and 2000 for PolyLM-13b and Aya-8b.

For brevity, we present the results for the intervention on Spanish (randomly chosen) in the main text. The results for the other languages are in the respective appendices.

## 4 The intervention shifts the embedding space increasing cross-lingual alignment

We begin our investigation by quantifying the differences induced by the intervention into the embedding space. For this analysis, we intervene on each of the five target languages discussed in Section 3.3 and examine the effect of the intervention on the representations of 22 languages (the union of all languages present in the pre-training across the three language models). We exclude Arabic and Chinese from consideration due to the lack of conformity in the scripts used<sup>4</sup>. Note that not all of these languages are part of the pre-training for every model under consideration; however we present them for consistency (clearly indicating in all figures if the languages were seen by the model during the pre-training).

For each of the 22 languages, we embed the Flores200 test set (1012 sentences per language) with the original and intervened models’ last layer. To characterize the changes in the embedding space, we calculate two types of distances: (1) the pairwise cosine distance between the embeddings of the 22 languages for the intervened and unintervened spaces and (2) the cosine distance between the mean of the embedding for each of the 22 languages and the medoid of each space (pre- and post-intervention) (Table 1).

Our findings are as follows. The intervention pulls the embeddings of all languages into a new space rather than moving them closer to the embeddings of the target language in the unintervened space (see App. C for sample visualizations of the embedding space changes using UMAP (McInnes et al., 2020) projections). The increase in perplexity post-intervention discussed in Section 3.3 also supports this finding.

The post-intervention embeddings for the different languages are closer to each other compared to the pre-intervention embeddings, as indicated by the reduced pairwise cosine distances between the languages. Specifically, the distances are reduced because the post-intervention embeddings are pulled closer to the medoid of the embedding space. As a result of the shift, all languages are closer to the target post-intervention. We notice that all distances under consideration are reduced less post-intervention for PolyLM-13b compared to the other models. We hypothesize that this relates to the specific data distribution and training procedure used for PolyLM-13b. Unfortunately, since we do not have access to the data the three models under consideration were trained on, we cannot test this hypothesis in this work. We return to this point in Section 9. Taken together, these findings suggest that the intervention projects language embeddings into a new space where they are more aligned. In the following sections, we explore if this change translates into downstream task performance.

![](images/419227f175ea1325cc1faa6c0a0ea8c4e3c787e81625b4e976f4d35e69bd92d6.jpg)  
Figure 3: Language ID accuracy and Log perplexity for the intervention on five target languages. The x-axis shows the number of activated experts (0 indicates the original model). Note that German and Japanese were not in training data for Bloom-7b.

## 5 Cross-lingual retrieval performance improves post-intervention

We now ask if the increased alignment postintervention translates to downstream task performance. We use cross-lingual retrieval as our downstream task: Given a sentence (query) in one language (query language), and a set of sentences (candidates) in a different language (candidate language), which of the candidates is a translation (match)? Our main experiments are carried out on the Flores200 test split (NLLB Team, 2022) as it allows us to test cross-lingual retrieval across multiple combinations of query and candidate languages. As the dev split of the Flores200 dataset was used to identify language experts, we also present results on the validation split of Tatoeba (Tiedemann, 2012) and the test split of BUCC-18 (Hu et al., 2020) for an independent validation of our findings (see App. H).

For each sentence, we compute pre- and postintervention embeddings by averaging over the last hidden state of the mLLM, producing vectors with dimensions matching the model’s hidden size. To identify the closest matching sentence, we compute cosine similarity between the query (e.g., in Spanish), and all candidates (e.g., in French). We select the candidate with the highest cosine similarity as the match, and then measure top-1 accuracy.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Language</td><td colspan="2">Distance (all languages)</td><td colspan="2">Distance to Medoid</td><td colspan="2">Distance to Target</td></tr><tr><td>Pre</td><td>Post</td><td> $\mathrm { P r e }$ </td><td>Post</td><td>Pre</td><td>Post</td></tr><tr><td rowspan="2">Aya-8b</td><td>Target</td><td></td><td></td><td> $0 . 6 2 _ { \pm 0 . 0 3 }$ </td><td> $0 . 1 4 { \scriptstyle \pm 0 . 0 3 }$ </td><td></td><td></td></tr><tr><td>Non-Target</td><td> $0 . 7 2 _ { \pm 0 . 0 0 }$ </td><td> $0 . 1 9 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $0 . 5 8 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $0 . 1 2 _ { \pm 0 . 0 1 }$ </td><td> $0 . 7 7 _ { \pm 0 . 0 1 }$ </td><td> $0 . 2 { \scriptstyle \pm 0 . 0 1 }$ </td></tr><tr><td rowspan="2">Bloom-7b</td><td>Target</td><td></td><td></td><td> $0 . 6 0 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $0 . 0 4 { \scriptstyle \pm 0 . 0 1 }$ </td><td></td><td></td></tr><tr><td>Non-Target</td><td> $0 . 7 2 _ { \pm 0 . 0 0 }$ </td><td> $0 . 1 7 _ { \pm 0 . 0 6 }$ </td><td> $0 . 5 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $0 . 1 1 _ { \pm 0 . 0 1 }$ </td><td> $0 . 7 8 _ { \pm 0 . 0 3 }$ </td><td> $0 . 1 3 { \scriptstyle \pm 0 . 0 1 }$ </td></tr><tr><td rowspan="2">PolyLM-13b</td><td>Target</td><td></td><td></td><td> $0 . 7 2 _ { \pm 0 . 0 4 }$ </td><td> $0 . 4 3 { \scriptstyle \pm 0 . 0 9 }$ </td><td></td><td></td></tr><tr><td>Non-Target</td><td> $0 . 8 5 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $0 . 5 6 _ { \pm 0 . 0 9 }$ </td><td> $0 . 7 2 _ { \pm 0 . 0 1 }$ </td><td> $0 . 4 3 _ { \pm 0 . 0 2 }$ </td><td> $0 . 8 6 _ { \pm 0 . 0 1 }$ </td><td> $0 . 5 4 _ { \pm 0 . 0 2 }$ </td></tr></table>

Table 1: Cosine distances between 22 languages under consideration, mean distance to the target of the intervention, and the distance to the medoid of the embedding space are reduced post-intervention. Distance (all languages) refers to pairwise cosine distance between the embeddings of 22 languages; distance to target refers to the distance between the intervention target and the remaining 21 languages. Pre refers to pre-intervention and post to post-intervention. Distances are means and standard errors of the mean over the five intervention targets.

![](images/185fe735d421fc1676962f63440cda7fb3742c83d2ee05484bfda2e1d12661b6.jpg)  
(a) Bloom-7b

![](images/3cce2c2f56138053b43dec3f1217d4e97cbac213899ccb0fae39da2379a566df.jpg)  
(b) Aya-8b

![](images/2d9f237f9019ba16fa00e0b0982a3928789c0e4d3505807845cadb08a86ee510.jpg)  
(c) PolyLM-13b  
Figure 4: Top-1 retrieval accuracy for the intervention on Spanish for 22 languages in the Bloom-7b model (left), Aya-8b (middle), and PolyLM-13b (right). The languages that are not in the training set for a given model are marked in red.

Top-1 retrieval accuracy improves postintervention for retrieval with the target language. We first examine if the increased proximity to the target language in the post-intervention embedding space translates into top-1 retrieval accuracy improvement when the target is used as the retrieval query for the 22 candidate languages under consideration.

We find that top-1 retrieval accuracy improves post-intervention when using the target as the query language (see Fig. 4 for the Spanish intervention and App. E for the remaining four languages). This finding is consistent across most target languages and models. Candidate languages present in the pre-training data generally demonstrate larger gains post-intervention. The pattern of improvement differs based on the model. Specifically, for Aya-

8b a successful intervention results in consistent improvements in top-1 accuracy for the majority of candidate languages (median=32%; max=74%). For Bloom-7b, top-1 accuracy gains are large (up to 89%) for a small number of candidate languages, with moderate improvements for other languages (median=14%). For PolyLM-13b, the improvements are small (median=0.5%; max=12%).

We note that accuracy gains post-intervention steadily increase with increasing k activated experts from 100 to 2000 for Aya-8b and Bloom-7b, after which the performance becomes languagedependent — showing gains, drops, or no change depending on the language, suggesting diminishing returns or destabilization. We did not observe a noticeable trend in PolyLM-13b across values of k over multiple languages.

To better understand how the increased alignment in the embedding space influences crosslingual retrieval, we look at the mean pairwise cosine distances between the query and candidate languages and explore how this correlates with

<table><tr><td>Model</td><td>Query language</td><td> $r ( \mathrm { a c c } _ { \mathrm { p o s t } } , d _ { \mathrm { p o s t } } )$ </td><td> $r ( \mathrm { a c c } _ { \mathrm { p r e } , d _ { \mathrm { p r e } } } )$ </td><td> $r ( d _ { \mathrm { p o s t } } , d _ { \mathrm { p r e } } )$ </td><td> $r ( \Delta \mathrm { a c c } , d _ { \mathrm { p r e } } - d _ { \mathrm { p o s t } } )$ </td></tr><tr><td rowspan="5">Aya-8b</td><td>es</td><td>-0.51 [-0.88 -0.18]</td><td>-0.89 [-0.98 -0.55]</td><td>0.48 [0.32 0.78]</td><td>0.86 [0.49 0.96]</td></tr><tr><td>fr</td><td>-0.64 [-0.91 -0.45]</td><td>-0.86 [-0.97 -0.57]</td><td>0.51 [0.27 0.89]</td><td>0.89 [0.84 0.97]</td></tr><tr><td>en</td><td>-0.94 [-0.97 -0.85]</td><td>-0.80 [-0.96 -0.34]</td><td>0.65 [0.44 0.92]</td><td>0.10 [-0.69 0.60]</td></tr><tr><td>de</td><td>-0.89 [-0.96 -0.75]</td><td>-0.87 [-0.98 -0.44]</td><td>0.33 [-0.74 0.76]</td><td>0.52 [0.12 0.95]</td></tr><tr><td>jp</td><td>-0.02 [-0.62 0.34]</td><td>-0.96 [-0.99 0.34]</td><td>0.27 [-0.18 0.62]</td><td>0.89 [0.30 0.99]</td></tr><tr><td rowspan="5">Bloom-7b</td><td>es</td><td>-0.97 [-0.99 -0.95]</td><td>-0.83 [-0.99 -0.54]</td><td>0.79 [0.71 0.98]</td><td>0.1 [-0.98 0.88]</td></tr><tr><td>fr</td><td>-0.98 [-0.99 -0.94]</td><td>-0.89 [-0.99 -0.38]</td><td>0.75 [0.62 0.99]</td><td>0.23 [-0.98 0.83]</td></tr><tr><td>en</td><td>-0.89 [-0.99 -0.60]</td><td>-0.89 [-0.99 -0.44]</td><td>0.97 [0.96 0.99]</td><td>0.23 [-0.90 0.86]</td></tr><tr><td>de</td><td>-0.90 [-0.99 -0.74]</td><td>-0.50 [-0.96 0.34]</td><td>0.95 [0.86 0.99]</td><td>-0.72 [-0.97 0.22]</td></tr><tr><td>jp</td><td>-0.90 [-0.99 -0.80]</td><td>NA⁵</td><td>-0.48 [-0.90 0.97]</td><td>0.64 [-0.70 0.93]</td></tr><tr><td rowspan="5">PolyLM-13b</td><td>es</td><td>-0.44 [-0.91 -0.38]</td><td>-0.84 [-0.96 -0.65]</td><td>0.70 [0.44 0.91]</td><td>0.10 [-0.31 0.57]</td></tr><tr><td>fr</td><td>-0.44 [-0.82 -0.35]</td><td>-0.90 [-0.99 -0.62]</td><td>0.66 [0.20 0.93]</td><td>0.30 [-0.18 0.82]</td></tr><tr><td>en</td><td>-0.86 [-0.98 -0.53]</td><td>-0.84 [-0.98 -0.52]</td><td>0.99 [0.96 0.99]</td><td>0.28 [-0.33 0.62]</td></tr><tr><td>de</td><td>-0.01 [-0.51 0.81]</td><td>-0.95 [-0.99 -0.57]</td><td>0.15 [-0.56 0.57]</td><td>-0.04 [-0.71 0.34]</td></tr><tr><td>jp</td><td>-0.52 [-0.91 0.92]</td><td>-0.96 [-0.99 0.00]</td><td>0.73 [0.56 0.96]</td><td>0.25 [-0.25 0.57]</td></tr></table>

Table 2: Pearson correlations (r) between top-1 retrieval accuracy (acc) and mean pairwise cosine distance in the embedding space d. Subscripts indicate the space from which embeddings are sampled: pre = original model; post = intervened model. Numbers in brackets represent bootstrapped 95% confidence intervals. Correlations that are not statistically significant (p-values >0.05) are shown in gray.

reduction in distances.

retrieval accuracy. Table 2 shows average correlations between post-intervention top-1 retrieval accuracy $( \mathrm { a c c } _ { \mathrm { p o s t } } )$ and mean query-candidate language distance both pre- and post-intervention $( d _ { \mathrm { p r e } } , \ d _ { \mathrm { p o s t } } )$ , average correlations between $d _ { \mathrm { p r e } }$ and $d _ { \mathrm { p o s t } }$ , and average correlations between improvement in accuracy $( \Delta \mathrm { a c c } = \mathrm { a c c } _ { \mathrm { p o s t } } - \mathrm { a c c } _ { \mathrm { p r e } } )$ and change in distance between pre- and postintervention embeddings $( d _ { \mathrm { p r e } } - d _ { \mathrm { p o s t } } )$ . When calculating averages, we only include candidate languages seen in pre-training for each model; we note that the general pattern stays the same but the correlations are somewhat weaker if all 22 languages are considered for all models. We find that in this setting, when the query and interventiontarget language are the same, the distance between query/target and match language is predictive of top-1 cross-lingual retrieval accuracy in both preand post-intervention spaces.

As discussed in Section 4, all language embeddings move closer to the target’s embeddings postintervention, which explains the gains in crosslingual retrieval accuracy. The distances in the unintervened and intervened space are positively correlated—language embeddings that are closer to the target pre-intervention are also closer to the target post-intervention. However, the magnitude of the performance gain in the intervened space does not correlate with the reduction in distance between the match and target languages across the two spaces, suggesting that the increased alignment post-intervention cannot be simply explained by a

Top-1 retrieval accuracy improves postintervention for retrieval with the non-target languages. In Section 4, we found that the distances between almost all languages decrease post-intervention—not just the distances to the intervention target. We next examine if these reduced cosine distances between languages other than the intervention target translate into improved top-1 retrieval accuracy when using these languages as the query language. For example, we study if intervening on Spanish experts improves Dutch-English retrieval (in this case, neither the query nor candidate language is the intervention-target language).

We find that, perhaps surprisingly, improvements observed when the query language is the intervention target (see Fig. 4 and Table 2) carry over to query languages other than the intervention-target language (see Fig. 5 for the Spanish intervention and App. F for the remaining four languages). For example, the intervention on Spanish expert neurons for Bloom-7b results in retrieval improvement when English, French, and Portuguese are the query language. The same intervention improves retrieval when querying Hebrew with Persian or when querying Czech with Greek in the Aya-8b model and when querying Russian with Portuguese in PolyLM-13b. These are examples of larger improvements, but many other languages follow the same pattern with smaller gains. Generally, the patterns in improvement are consistent with those seen when Spanish is the query language. Languages that are in the pre-training set have larger accuracy gains. Bloom-7b has large improvements for a small number of languages and no drops in performance. Aya-8b has relatively large improvements for a majority of languages but also has a drop in performance for some. As noted previously, PolyLM-13b performance is uneven—the improvement varies by language with languages in the pre-training set generally having larger improvements.

![](images/78b2dd0524f7e6091450567ccbd7a3eea7be5e7816d1c389713c8e0e479753dd.jpg)  
(a) Bloom-7b

![](images/a6deba5febd1409fac59156b5406dfbaac98f3b96a2034618d12d7c458790e6e.jpg)  
(b) Aya-8b

![](images/7950b1b6cba6f2a5c9dc5b28b92d17b4dec258c2a773c3e665cec0fd34f71b2f.jpg)  
(c) PolyLM-13b  
<sup>Figure</sup> <sup>5:</sup> <sup>(Top-1</sup> <sup>accuracy</sup>post-intervention − <sup>Top-1</sup> <sup>accuracy</sup>pre-intervention<sup>)</sup> <sup>for</sup> <sup>22</sup> <sup>languages</sup> <sup>after</sup> <sup>intervening</sup> <sup>on</sup> <sup>Spanish</sup> expert neurons in the Bloom-7b model (left), Aya-8b (middle), and PolyLM-13b (right). The languages that are not in the training set for a given model are marked in red.

## 6 Within-language similarity is preserved post-intervention

As observed in Section 4, all languages move toward the medoid of the embedding space postintervention, which raises the question of whether language-specific similarities are preserved in the new space. To answer this question, we evaluate model’s performance on a paraphrase retrieval task which tests whether a sentence in the intervened space can be matched with its paraphrase in the intervened space. We utilize the PAWS-X dataset (Hu et al., 2020), which provides paired sentences across seven languages, including all five of our intervention targets. From the test split, we retain only the paraphrase pairs, excluding non-paraphrases and sentences from other languages. This transforms our evaluation into a within-language sentence retrieval task, where the goal is to match each sentence with its paraphrase from a pool of candidates for that language.

The paraphrase retrieval task reveals two key findings about embedding spaces before and after the intervention. First, the top-1 paraphrase retrieval accuracy remains largely unchanged after the intervention (see Table 3), indicating that the new embedding space preserves within-language similarity. Second, when attempting retrieval between intervened and unintervened embeddings of the same language — i.e., using the embeddings from the unintervened model as the query and the embeddings from the intervened model as candidate matches — accuracy drops significantly. This decline supports the observation that the intervention projects embeddings into a distinctly different space from their original unintervened representations discussed in Section 4. This finding also aligns with the increase in perplexity observed postintervention – the intervened space of a given language is not the same space as the original space of this language.

<table><tr><td>Model</td><td>Top-1 Accuracy (Pre) (Post) (Mixed)</td></tr><tr><td>Bloom-7b</td><td>0.80 0.80 0.33</td></tr><tr><td>Aya-8b</td><td>0.85 0.86 0.64</td></tr><tr><td>PolyLM-13b</td><td>0.52 0.56 0.41</td></tr></table>

Table 3: Top-1 accuracy results for the paraphrase retrieval task following the intervention on Spanish. The results for other languages can be found in App. D. Pre = both the query and candidate embeddings are from the original model; Post = both the query and the candidate embeddings are from the intervened model; Mixed = query is from the original model and candidates are from the intervened model.

## 7 Intervention on random neurons does not provide an improvement on downstream tasks

In our analyses so far, we have attributed the changes in the embedding space to the intervention on expert neurons. Before we conclude, we consider an alternative possibility — that the expert neurons do not play a role in increasing alignment post-intervention, but instead alignment is achieved by fixing the activations of a number of neurons in a network. To address this, we assign the activation levels of the language expert neurons used in prior sections to the same number of neurons chosen randomly in the network and repeat our analyses on these models.

We find that intervening on random neurons produces markedly different results compared to activating language experts (see App. G). The embedding space after the intervention on random neurons does not have the same properties as described in Section 4, which translates into the performance on downstream tasks for all models. Specifically, for the Aya-8b and PolyLM-13b top-1 cross-lingual retrieval accuracy drops for all languages postintervention on random neurons compared to preintervention. Interestingly, for Bloom-7b, there is mostly no change for all target languages except French, which surprisingly improves postintervention on random neurons. However, the gains are significantly smaller compared to those after intervening on French experts. Similar to the intervention on language experts, within-language paraphrase retrieval shows only small changes postintervention. When they occur, these changes tend to be negative (i.e., the performance drops) after intervening on random neurons and positive after intervening on the actual language experts.

## 8 Conclusions

We present a novel analysis of the impact of the finding-experts intervention on cross-lingual alignment in mLLMs. We find that intervening on language experts projects model embeddings into a new space where languages are more aligned than in the original space but still preserve withinlanguage similarity. These findings provide an explanation for the increase in perplexity observed post-intervention in prior work (Suau et al., 2022).

We also demonstrate that cross-lingual alignment in mLLMs can be improved through the finding-experts intervention. Applying the intervention to a single language boosts alignment across most languages seen during training and results in up to 2x improvement in top-1 retrieval accuracy. Additionally, we show that the correlation between cross-lingual alignment and cross-lingual retrieval is high and statistically significant. We recommend selecting one such language, the interventiontarget, to enhance overall alignment. Our results show that this approach is most effective when the model has been trained on both the interventiontarget language and the languages we aim to align.

We find that the three models we study show markedly different patterns both in the changes to the embedding space and downstream tasks. We leave it to future to work to determine the causes of these differences, though we hypothesize that they are due to the pre-training differences.

## 9 Limitations

There are several limitations that need to be considered when interpreting our results.

We do not have access to training data or procedure. The major limitation is that we are working with pre-trained models and we have only limited information on training data and procedure. Specifically, for Bloom-7b and PolyLM-13b, we have the information on the proportion of each language in the pre-training set. For Aya-8b, only information on which languages were seen in the pre-training (but no proportions) is available.

PolyLM-13b is an outlier in our analyses. PolyLM-13b emerges as an outlier in all of our analyses. We have ruled out the nature of the discovered experts as the primary factor. For example, we find that while PolyLM-13b’s experts are on average lower quality (lower AUROC) than the experts in the other models, this does not fully account for its performance since we find multiple instances where PolyLM-13b and another model are matched in expert quality but PolyLM-13b still underperforms the other model. We have also rejected the hypothesis that the top experts in PolyLM-13b overlap across languages, rendering the intervention less successful — we find essentially no overlap among the top 2000 experts across five languages in any model.

Our leading hypothesis is that the discrepancy between PolyLM-13b and the other models arises from differences in training data or procedures. However, without access to training details—e.g., pretraining objectives, language distribution, data volume, or curriculum—further analysis is limited. Future work should explore the effect of intervention on alignment in a more controlled setting where the models are trained from scratch on a publicly available dataset manipulating language proportions in the training data to better understand what is driving the difference.

We study the impact of only one intervention on alignment. We have studied only one approach out of a family of approaches to controllable generations (Rimsky et al., 2024; Suau et al., 2024; Rodriguez et al., 2024). Each approach in the family comes with its differences – in the way the neurons targeted by the intervention are discovered, how the changes are introduced to the activations, how many neurons are intervened on, etc. We do not fully understand how these design decisions impact the representation space. For example, it is possible that some of these approaches are more beneficial for alignment while others introduce changes that are more beneficial for other tasks (or not at all). The comparison of approaches is beyond the scope of current work and we leave it for future investigations.

## References

Viraat Aryabumi, John Dang, Dwarak Talupuru, Saurabh Dash, David Cairuz, Hangyu Lin, Bharat Venkitesh, Madeline Smith, Kelly Marchisio, Sebastian Ruder, Acyr Locatelli, Julia Kreutzer, Nick Frosst, Phil Blunsom, Marzieh Fadaee, Ahmet Üstün, and Sara Hooker. 2024. Aya 23: Open weight releases to further multilingual progress. Preprint, arXiv:2405.15032.

Verena Blaschke, Masha Fedzechkina, and Maartje ter Hoeve. 2025. Analyzing the effect of linguistic similarity on cross-lingual transfer: Tasks and experimental setups matter. arXiv preprint arXiv:2501.14491.

Antoine Bosselut, Hannah Rashkin, Maarten Sap, Chaitanya Malaviya, Asli Celikyilmaz, and Yejin Choi. 2019. COMET: Commonsense transformers for automatic knowledge graph construction. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4762–4779, Florence, Italy. Association for Computational Linguistics.

Aditi Chaudhary, Karthik Raman, Krishna Srinivasan, and Jiecao Chen. 2020. Dict-mlm: Improved multilingual pre-training using bilingual dictionaries. ArXiv, abs/2010.12566.

Pavel Efimov, Leonid Boytsov, Elena Arslanova, and Pavel Braslavski. 2023. The impact of crosslingual adjustment of contextual word representations on zero-shot transfer. In Advances in Information Retrieval, pages 51–67, Cham. Springer Nature Switzerland.

Fahim Faisal and Antonios Anastasopoulos. 2023. Geographic and geopolitical biases of language models. In Proc. ofthe 3rd Workshop on Multi-lingual Representation Learning (MRL).

Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin Johnson. 2020. Xtreme: A massively multilingual multi-task benchmark for evaluating cross-lingual generalization. CoRR, abs/2003.11080.

Haoyang Huang, Tianyi Tang, Dongdong Zhang, Xin Zhao, Ting Song, Yan Xia, and Furu Wei. 2023. Not all languages are created equal in LLMs: Improving multilingual capability by cross-lingual-thought prompting. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 12365– 12394, Singapore. Association for Computational Linguistics.

AQ Jiang, A Sablayrolles, A Mensch, C Bamford, DS Chaplot, D de las Casas, F Bressand, G Lengyel, G Lample, L Saulnier, et al. 2023. Mistral 7b (2023). arXiv preprint arXiv:2310.06825.

Takeshi Kojima, Itsuki Okimura, Yusuke Iwasawa, Hitomi Yanaka, and Yutaka Matsuo. 2024. On the multilingual ability of decoder-based pre-trained language models: Finding and controlling language-specific neurons. NAACL.

Guillaume Lample and Alexis Conneau. 2019. Crosslingual language model pretraining. NeurIPS, arXiv:1901.07291.

Guillaume Lample, Alexis Conneau, Ludovic Denoyer, and Marc’Aurelio Ranzato. 2018. Unsupervised machine translation using monolingual corpora only.

Chong Li, Shaonan Wang, Jiajun Zhang, and Chengqing Zong. 2024. Improving in-context learning of multilingual generative language models with crosslingual alignment. NAACL.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023. Inference-time intervention: Eliciting truthful answers from a language model.

Yang Liu and Mirella Lapata. 2019. Text summarization with pretrained encoders. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3730–3740, Hong Kong, China. Association for Computational Linguistics.

Yihong Liu, Mingyang Wang, Amir Hossein Kargaran, Ayyoob ImaniGooghari, Orgest Xhelili, Haotian Ye, Chunlan Ma, François Yvon, and Hinrich Schütze. 2025. How transliterations improve crosslingual alignment. In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 2417–2433, Abu Dhabi, UAE. Association for Computational Linguistics.

Marco Lui and Timothy Baldwin. 2012. langid.py: An off-the-shelf language identification tool. In Proceedings of the ACL 2012 System Demonstrations, pages 25–30, Jeju Island, Korea. Association for Computational Linguistics.

Leland McInnes, John Healy, and James Melville. 2020. Umap: Uniform manifold approximation and projection for dimension reduction. Preprint, arXiv:1802.03426.

et al NLLB Team, Marta R. Costa-jussà. 2022. No language left behind: Scaling human-centered machine translation. EMNLP.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofmachine learning research, 21(140):1–67.

Christopher Richardson and Larry Heck. 2023. Commonsense reasoning for conversational ai: A survey of the state of the art. ArXiv, abs/2302.07926.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. 2024. Steering llama 2 via contrastive activation addition. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15504–15522, Bangkok, Thailand. Association for Computational Linguistics.

Pau Rodriguez, Arno Blaas, Michal Klein, Luca Zappella, Nicholas Apostoloff, Marco Cuturi, and Xavier Suau. 2024. Controlling language and diffusion models by transporting activations. arXiv preprint arXiv:2410.23054.

Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel Hesslow, Roman´ Castagné, Alexandra Sasha Luccioni, François Yvon, et al. 2022. Bloom: A 176b-parameter openaccess multilingual language model. arXiv preprint arXiv:2211.05100.

Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Mohamed Amin, Le Hou, Kevin Clark, Stephen R Pfohl, Heather Cole-Lewis, et al. 2025. Toward expert-level medical question answering with large language models. Nature Medicine, pages 1–8.

Xavier Suau, Pieter Delobelle, Katherine Metcalf, Armand Joulin, Nicholas Apostoloff, Luca Zappella, and Pau Rodriguez. 2024. Whispering experts: Neural interventions for toxicity mitigation in language models. In Forty-first International Conference on Machine Learning.

Xavier Suau, Luca Zappella, and Nicholas Apostoloff. 2022. Self-conditioning pre-trained language models. In Proceedings ofthe 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 4455–4473. PMLR.

Eshaan Tanwar, Subhabrata Dutta, Manish Borthakur, and Tanmoy Chakraborty. 2023. Multilingual LLMs are better cross-lingual in-context learners with alignment. In Proceedings ofthe 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 6292–6307, Toronto, Canada. Association for Computational Linguistics.

Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. 2024. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295.

Joerg Tiedemann. 2012. Parallel data, tools and interfaces in opus. LREC.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models. Preprint, arXiv:2302.13971.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. 2024. Steering language models with activation engineering. Preprint, arXiv:2308.10248.

Jiaheng Wei, Yuanshun Yao, Jean-Francois Ton, Hongyi Guo, Andrew Estornell, and Yang Liu. 2024. Measuring and reducing llm hallucination without goldstandard answers. arXiv preprint arXiv:2402.10412.

Xiangpeng Wei, Haoran Wei, Huan Lin, Tianhao Li, Pei Zhang, Xingzhang Ren, Mei Li, Yu Wan, Zhiwei Cao, Binbin Xie, Tianxiang Hu, Shangjie Li, Binyuan Hui, Bowen Yu, Dayiheng Liu, Baosong Yang, Fei Huang, and Jun Xie. 2023. Polylm: An open source polyglot large language model. Preprint, arXiv:2307.06018.

Chris Wendler, Veniamin Veselovsky, Giovanni Monea, and Robert West. 2024. Do llamas work in english? on the latent language of multilingual transformers. Preprint, arXiv:2402.10588.

Di Wu, Yibin Lei, Andrew Yates, and Christof Monz. 2024. Representational isomorphism and alignment of multilingual large language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 14074–14085, Miami, Florida, USA. Association for Computational Linguistics.

Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann. 2023.

Bloomberggpt: A large language model for finance. arXiv preprint arXiv:2303.17564.

Shaolei Zhang, Qingkai Fang, Zhuocheng Zhang, Zhengrui Ma, Yan Zhou, Langlin Huang, Mengyu Bu, Shangtong Gui, Yunji Chen, Xilin Chen, and Yang Feng. 2023. Bayling: Bridging cross-lingual alignment and instruction following through interactive translation for large language models. Preprint, arXiv:2306.10968.

Jun Zhao, Zhihao Zhang, Luhui Gao, Qi Zhang, Tao Gui, and Xuanjing Huang. 2024. Llama beyond english: An empirical study on language capability transfer. arXiv preprint arXiv:2401.01055.

Shanshan Zhong, Zhongzhan Huang, Shanghua Gao, Wushao Wen, Liang Lin, Marinka Zitnick, and Pan Zhou. 2024. Let’s think outside the box: Exploring leap-of-thought in large language models with multimodal humor generation. CVPR.

## A List of languages studied

The following languages are considered in this work:
<table><tr><td>#</td><td>Language</td><td>ISO 639-1</td><td>ISO 639-3</td></tr><tr><td>1</td><td>Thai</td><td>th</td><td>tha</td></tr><tr><td>2</td><td>Czech</td><td>cs</td><td>ces</td></tr><tr><td>3</td><td>German</td><td>de</td><td>deu</td></tr><tr><td>4</td><td>Greek</td><td>el</td><td>ell</td></tr><tr><td>5</td><td>English</td><td>en</td><td>eng</td></tr><tr><td>6</td><td>French</td><td>fr</td><td>fra</td></tr><tr><td>7</td><td>Hebrew</td><td>he</td><td>heb</td></tr><tr><td>8</td><td>Hindi</td><td>hi</td><td>hin</td></tr><tr><td>9</td><td>Indonesian</td><td>id</td><td>ind</td></tr><tr><td>10</td><td>Italian</td><td>it</td><td>ita</td></tr><tr><td>11</td><td>Japanese</td><td>ja</td><td>jpn</td></tr><tr><td>12</td><td>Korean</td><td>ko</td><td>kor</td></tr><tr><td>13</td><td>Dutch</td><td>nl</td><td>nld</td></tr><tr><td>14</td><td>Persian</td><td>fa</td><td>pes</td></tr><tr><td>15</td><td>Polish</td><td>pl</td><td>pol</td></tr><tr><td>16</td><td>Portuguese</td><td>pt</td><td>por</td></tr><tr><td>17</td><td>Romanian</td><td>ro</td><td>ron</td></tr><tr><td>18</td><td>Russian</td><td>ru</td><td>rus</td></tr><tr><td>19</td><td>Spanish</td><td>es</td><td>spa</td></tr><tr><td>20</td><td>Turkish</td><td>tr</td><td>tur</td></tr><tr><td>21</td><td>Ukrainian</td><td>uk</td><td>ukr</td></tr><tr><td>22</td><td>Vietnamese</td><td>vi</td><td>vie</td></tr></table>

Table 4: ISO 639-1 and ISO 639-3 Language Codes

## B Sample generations

Listing 1: Sample generations by the three models under consideration. 0 experts indicates an unintervened model.   
Aya−8b w i t h 0 e x p e r t s a c t i v a t e d :   
Sample 1 :   
The 49− year −old , who came back t o England fo r a second s p e l l with   
t h e c l u b i n 2018 , h a s been i n c h a r g e o f t h e f i r s t team t h i s   
s e a s o n fo l l o w i n g t h e d e p a r t u r e o f Ga rry Rowett . \ n \ n \ " I t ’ s been an   
a b s o l u t e honour and p l e a s u r e fo r t h e l a s t two y e a r s . But I ’m back   
home and I ’m l o v i n g i t , \ " he t o l d BBC WM. \ n \ n \ " The b i g g e s t t h i n g   
fo r me and t h e c l u b i s t h a t we ’ ve done a g r e a t j o b . I t was a   
g r e a t j o b a t Watford and a f a n t a s t i c j o b a t S t o k e .   
Sample 2 :   
[ a r ] \ n \ u0627 \ u0644 \ u0631 \ u0626 \ u064a \ u0633 :   
\ u0627 \ u0644 \ u0633 \ u064a \ u062f \ u0643 \ u0648 \ u0631 \ u062a \ u064a   
( \ u0643 \ u0631 \ u0648 \ u0627 \ u062a \ u064a \ u0627 ) \ n [ zh ]   
\ n \ u4e3b \ u5e2d \ u ff1 a   
\ u79d1 \ u9c81 \ u7279 \ u5148 \ u751f ( \ u514b \ u7f57 \ u5730   
\ u4e9a ) \ n [ en ] \ n P r e s i d e n t : Mr . K u r t o v i \ u0107 \ n [ e s ] \ n P r e s i d e n t e :   
Sr . K u r t o v i \ u0107 [ b o s n i o ] [ c r o a t a ] \ n [ ru ] \ n \ u041f \ u0440   
\ u0435 \ u0434 \ u0441 \ u0435 \ u0434 \ u0430 \ u0442 \ u0435 \ u043b \ u044c :   
\ u0433 −\ u043d \ u041a \ u0443 \ u0440 \ u0442 \ u043e \ u0432 \ u0438 \ u0447   
[ \ u0411 \ u043e \ u0441 \ u043d \ u0438 \ u044f \ u0438   
\ u0413 \ u0435 \ u0440 \ u0446 \ u0435 \ u0433 \ u043e \ u0432 \ u0438   
\ u043d \ u0430 ] \ n [ f r ] \ nPr   
Sample 3 :   
i m p o r t R e a c t from \ " r e a c t \ " ; \ n i m p o r t { Link } from   
\ " g a t s b y \ " ; \ n i m p o r t Image from \ " . / image \ " ; \ n i m p o r t S o c i a l M e d i a   
fr o m \ " . / s o c i a l − m e d i a \ " ; \ n i m p o r t { u s e T r a n s l a t i o n } fr o m   
\ " n e x t − i 1 8 n e x t \ " ; \ n i m p o r t { G e t S t a t i c P r o p s } from   
\ " n e x t \ " ; \ n \ n c o n s t P a g e : R e a c t . FC = ( ) => { \ n c o n s t { t } =   
u s e T r a n s l a t i o n ( ) ; \ n \ n r e t u r n ( \ n < d i v   
className = \ " a bout −page − l a y o u t page − l a y o u t \ " > \ n < d i v   
className = \ " page − t i t l e \ " > \ n <h1 >{ t ( \ " a b o u t . t i t l e \ " ) } </ h1 > \ n   
<h2 >{ t ( \ " a b o u t . s u b t i t l e \ " ) } </ h2 >\ n </ div >\ n < d i v   
className = \ " c o n t e n t − w r a p p e r \ " > \ n < d i v   
className = \ " a bout − c o n t e n t \ " > \ n < d i v   
className = \ " a bout − s e c t i o n \ " > \ n < d i v className = \ " ab ou t − t e x t \ " > \ n   
<h2 >{ t ( \ " a b o u t . a b o u t \ " ) } </ h2 >\ n   
<p>{ t ( \ " a b o u t . a b o u t C o n t e n t \ " ) } </p >\ n </ div >\ n < d i v   
className = \ " a bout − s e c t i o n −image − c o n t a i n e r \ " > \ n <Image \ n   
imageClassName = \ " abo ut − s e c t i o n −image \ " \ n   
imageName = \ " a bout −image \ " \ n w i d t h = \ " 4 0 0 \ " \ n h e i g h t = \ " 3 0 0 \ " \ n   
Aya−8b w i t h 2000 S p a n i s h e x p e r t s a c t i v a t e d :

```csv
S u c c e s s fu l g e n e r a t i o n :
A l a o c i o s y a l o s d e s o c u p a d o s no s e l e s oprimen l a s l e y e s , n i s e
l e s a b r u m a r c o n i m p u e s t o s y r e c o r t a r s u s b e n e f i c i o s , q u e e s un
d e r e c h o de to do s e r humano y un d e b e r d e l que l o t e n g a que
e j e r c e r , p u e s c o n e l l o s e c o n t r i b u y e a l p r o g r e s o d e l a h u m a n i d a d .
No hay que i n t e n t a r p o r n i n g u n a p r e p o s i c i n h a c e r e l b i e n
h a c i e n d o d a o a l o s d e m s .
Code − s w i t c h g e n e r a t i o n :
I f a n e l e l c a s o d e l p r o y e c t o d e Ley d e F i n a n c i a m i e n t o , l o q u e hemos
e s c u c h a d o d e l g o b i e r n o e s que l o h a r n a e l n i v e l de 2005 o
i n c l u s o 2004 , que s e r a una d e s a t e n d e r de e l p ro b l e m a " , d ij o e l
s e n a d o r .
A n i v e l d e l Senado , l o s 75 miembros de e l P a r l a m e n t o , son 200 y s e
r e q u i e r e n 2 / 3 d e l Senado p a r a l o g r a r e l q u r u m , l o que son 40 de
l o s que v o t e n a fa v o r de l a Ley de F i n a n c i a m i e n t o .
U n s u c c e s s fu l g e n e r a t i o n :
h i s i s t h e two a n o l e s (N e l e l en l a anop l de P o l e l e l en l a n i
1 y en l a n \ n P u b l i c a d o e l 23 de j u l i o de 2010 17 de j u l i o de 2011
p o r \ nEn l a s e c c i \ u 0 0 f3 n d e e l l i b r o d e l a gu \ u 0 0 e d a \ u 2 0 1 c P o p a
l i c . 1 ( 1 2 2 2 2 2 2 2 1 1 1 2 2 2 2
Bloom −7 b w i t h 0 e x p e r t s a c t i v a t e d
Sample 1 :
+ \ " ) \ " ; \ n } \ n \ n Q S t r i n g L i s t Q g s P r o j e c t : : e n t r y L i s t ( c o n s t Q S t r i n g
&scope , c o n s t Q S t r i n g &p a t h ) c o n s t \ n { \ n Q S t r i n g L i s t e n t r i e s ; \ n \ n
c o n s t QMap< Q S t r i n g , Q g s P r oj e c t S e c t i o n > : : c o n s t _ i t e r a t o r i t =
m P r o j e c t S e c t i o n s . c o n s t F i n d ( s c o p e ) ; \ n i f ( i t ! =
m P r o j e c t S e c t i o n s . c o n s t E n d ( ) && i t path == p a t h ) \ n { \ n c o n s t
Q S t r i n g L i s t e n t r i e s F o r S c o p e = i t children . ke y s ( ) ; \ n fo r ( c o n s t
Q S t r i n g &e n t r y : e n t r i e s F o r S c o p e ) \ n { \ n e n t r i e s << e n t r y ; \ n } \ n
} \ n e l s e \ n { \ n c o n s t QMap< QString ,
Q g s P r oj e c t L a y e r G r o u p > : : c o n s t _ i t e r a t o r g r o u p I t =
m P r o j e c t L a y e r G r o u p s . c o n s t F i n d ( s c o p e ) ; \ n i f ( g r o u p I t ! =
m Proj e c tL a ye rG ro up s . constEnd ( ) && g r o u p I t path == p a t h ) \ n
Sample 2 :
a n d < / s w i t h t h e r e s t o f t h e s e n t e n c e . F o r e x a m p l e , t h e
s t r i n g & l t ;& l t ; b&g t ; Foo& l t ; / b&g t ;& l t /& g t ;& l t ; / & g t ;& l t ;
/& g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ;
/& g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; /
&g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ;
/& g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ;
/& g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ;
/& g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ;
/& g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ;
/& g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; /
&g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; /
&g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ;
```

/& g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; / & g t ;& l t ; /   
Sample 3 :   
The same a s my own p e r s o n a l , ’ ’ and t o ’ ’ The same a s t h e o r i g i n a l ’ ’   
fo r o t h e r s o u r c e s . I n o t h e r words , t h e same a s my own p e r s o n a l ,   
my own p e r s o n a l , o r t h e o r i g i n a l have a l l been c o n s i d e r e d   
e q u i v a l e n t r e s p o n s e s ( T a b l e 3 ) . \ n A l th o u g h o n l y 7.6% o f t h e t o t a l   
r e s p o n s e s t o q u e s t i o n 6 were r a t e d a s ’ ’ o t h e r ’ ’ , t h i s t y p e o f   
r e s p o n s e , which we w i l l t e r m ’ ’ o t h e r ’ ’ , i s o f t e n i n f a c t a   
r e s p o n s e a b o u t t h e s o u r c e o f t h e i n fo r m a t i o n o r t h e a u t h o r ( s ) o f   
a b o o k ( e . g . , ’ ’ T h i s b o o k was w r i t t e n by . . . ’ ’ o r ’ ’ T h i s was   
from a book by . . . ’ ’ ) .   
Bloom −7b wi th 2000 S p a n i s h e x p e r t s a c t i v a t e d   
S u c c e s s fu l g e n e r a t i o n :   
que t i e n e que v e r con e l tema de l a s a l u d m e n t a l . Por \ u 0 0 fa l t i m o ,   
s e p r e g u n t a s i l a p e r s o n a que nos a t i e n d e t i e n e s u f i c i e n t e   
e x p e r i e n c i a y s i ha s e g u i d o una fo r m a c i \ u00f3n e s p e c \ u 0 0 e d fi c a en   
e s e t i p o de problem \ u 0 0 e 1 t i c a s . Al no t e n e r una r e s p u e s t a c l a r a a   
e s t a s p r e g u n t a s , tendemos a r e c h a z a r e s e t i p o de c e n t r o s o no   
u t i l i z a r l o s , p o r ejemplo , en c a s o s de c r i s i s o a n s i e d a d   
e x t r e m a . \ nPor t o d o e l l o , en e s t e c u r s o s e p r e t e n d e o f r e c e r una   
a p r o x i m a c i \ u00f3n g e n e r a l a l e s t a d o de l a c u e s t i \ u00f3n de l a   
i n t e r v e n c i \ u00f3n p s i c o l \ u 0 0 f3 g i c a en c r i s i s , t a n t o s i e s t a s e   
l l e v a a cabo d e n t r o de una i n t e r v e n c i \ u00f3n p s i c o l \ u 0 0 f3 g i c a   
como s i e s de t i p o a s i s t e n c i a l   
U n s u c c e s s fu l g e n e r a t i o n :   
no e x i s t e l a c a d e n a de comando . \ n P e r o e s t o s i no me d e j a e j e c u t a r   
nada . \ n E s t o y t r a b a j a n d o con e l Windows 10 y cuando me pongo a   
e j e c u t a r l o me d i c e : \ n \ " p r o c e s a no s e ej e c u t a \ " \ nEl p ro b l e m a e s   
que cuando voy a e j e c u t a r e l ej e c u t a b l e . exe me da un e r r o r como   
q u e no e x i s t e e l a r c h i v o . \ nLo q u e o c u r r e e s q u e no p u e d o e j e c u t a r   
nada . No q u i e r o h a c e r nada y no s e que   
h a c e r . \ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n−   
\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n−   
\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n−   
\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n −\ n−   
\ n −\ n −\ n −\ n −\ n −\ n −\ n−   
PolyLM −13b w i t h 0 e x p e r t s a c t i v a t e d :   
Sample 1 :   
\ u0437 \ u0432 \ u0456 \ u0434 \ u0442 \ u0438 , \ u0432 \ u0443   
\ u043b \ u0438 \ u043a \ u0438 , \ u0440 \ u0456 \ u0437 \ u0430   
\ u043d \ u0456 \ u0432 \ u0430 \ u043d \ u043d \ u0438 , \ u0456   
\ u0441 \ u0442 \ u043e \ u0432 \ u0431 \ u0443 \ u0440 \ u0438   
\ u0434 \ u0435 \ u0440 \ u0435 \ u0432 . \ u0426 \ u0456 \ u0434   
\ u0435 \ u0440 \ u0435 \ u0432 \ u0430 \ u043f \ u043e \ u0432   
\ u0438 \ u043d \ u043d \ u0456 \ u0431 \ u0443 \ u0442 \ u0438   
\ u0440 \ u0456 \ u0432 \ u043d \ u043e \ u043c \ u0456 \ u0440

\ u043d \ u043e \ u0440 \ u043e \ u0437 \ u0440 \ u0456 \ u0437   
\ u0430 \ u043d \ u0438 \ u043c \ u0438 \ u0442 \ u0430 \ u0437   
\ u0430 \ u0433 \ u043b \ u0438 \ u0431 \ u043b \ u0435 \ u043d   
\ u0438 \ u043c \ u0438 , \ u0456 \ u0437 \ u0437 \ u0430 \ u043a   
\ u0440 \ u0456 \ u043f \ u043b \ u0435 \ u043d \ u043d \ u044f   
\ u043c \ u0432 \ u0441 \ u0435 \ u0440 \ u0435 \ u0434 \ u0438 \ u043d \ u0456   
Sample 2 :   
\ u0441 256 −\ u043c \ u044f \ u041c \ u0431 \ u0430 \ u0439   
\ u0442 \ u0430 \ u043c \ u0438 \ u043f \ u0430 \ u043c \ u044f   
\ u0442 \ u0438 \ u04381 \ u0413 \ u0431 \ u043e \ u043f \ u0435   
\ u0440 \ u0430 \ u0442 \ u0438 \ u0432 \ u043d \ u043e \ u0439   
\ u043f \ u0430 \ u043c \ u044f \ u0442 \ u0438 . \ u0414 \ u043b   
\ u044f \ u0432 \ u0432 \ u043e \ u0434 \ u0430 / \ u0432 \ u044b   
\ u0432 \ u043e \ u0434 \ u0430 \ u0431 \ u044b \ u043b \ u0438   
\ u0441 \ u043f \ u043e \ u043b \ u044c \ u0437 \ u043e \ u0432   
\ u0430 \ u043d \ u0438 \ u043d \ u0442 \ u0435 \ u0440 \ u0444   
\ u0435 \ u0439 \ u0441 USB 2 . 0 . \ n \ n <p>The s t u d y by r e s e a r c h e r s a t   
t h e U n i v e r s i t y o f Tokyo found t h a t a s i n g l e d o s e o f p r o b i o t i c s   
c o u l d s i g n i f i c a n t l y r e d u c e t h e i n fl a m m a t i o n a s s o c i a t e d w i t h   
s t r e s s i n t h e b r a i n . < / p >\ n \ n <p> T h i s h o t e l h a s a t o t a l o f 429   
g u e s t rooms and 12 j u n i o r s u i t e s and 2 j u n i o r s u i t e s   
PolyLM −13b w ith 2000 S p a n i s h e x p e r t s a c t i v a t e d :   
S u c c e s s fu l g e n e r a t i o n :   
no l o conozco , p e r o p o r l o que he a c o r r i d o en l a t e l e e s muy   
v i l l a n o y yo c r e o que e s muy v a l i e n t e \ n p e r o e l que ha c o r r i d o e l   
que ha c o r r i d o en l a t e l e no t i e n e nada que v e r con l o n u e s t r o \ n   
p ero yo c r e o que \ u 0 0 e 9 l ha hecho un gra n c os a por ejemplo e l que   
ha c o r r i d o que ha c o r r i d o p o r l a e s t e p a e s una p a r t e de a r g e n t i n a   
donde e s t a e l r i o de l a p l a t a \ n que va de e s t e p a a a r g e n t i n a \ n   
a un g r u p o de amigos que e s t a b a n t r a b a j a n d o en l a ca mento y   
e l l o s e s t a b a n p a s a n d o p o r e l de e s t e p a a l a e s t e p a \ n y e l l o s s e   
p e r c a t que l a que e s t a b a s p a s a n d o p o r e l l o s e r a un d e l r i o s a l a   
de e s t e p a \ n y a e l l o s d e c i d i e r o n d a r l a v u e l t a y i r a nado   
U n s u c c e s s fu l g e n e r a t i o n :   
2 0 0 7 . \ n \ n < < 2005 >\ n < < < > >\ n \ n \ n < < <   
2019\ u5e7412 \ u670813 \ u65e5 \ u ff0 c \ u7531 \ u81f3 \ u3002 , . . \ n \ n \ n   
< < < > > < <\ n \ n . . \ n \ n \ n < < < > > \ n . \ n \ n < < < > >\ n   
\_ ( Nombre de l a c a n c i \ u00f3n ) \ n \ n E s t e e s un ej e m p l o de   
c \ u 0 0 f3 d i g o que s e p o d r \ u00eda u t i l i z a r en J a v a S c r i p t , p a r a c r e a r   
un b o t \ u00f3n con un s o n i d o . \ n ‘ ‘ ‘ j a v a s c r i p t \ n fu n c t i o n   
p l a y B t n ( nombre ) { \ n v a r s o n i d o ; \ n v a r cadena ; \ n sound = new   
Audio ( ) ; \ n c a d e n a = s o n i d o . P l a y ( ) ; \ n sound . c u r r e n t T i m e = 0 ; \ n   
s o n i d o . l oop = t r u e ; \ n s o n i d o . s r c = cadena ; \ n s o n i d o . volume = 1 ; \ n   
s o n i d o . a u t o p l a y = t r u e ; \ n s o n i d o . onended = ( ) => { \ n   
sound . c u r r e n t T i m e = 1 ; \ n } \ n } \

## C Sample visualizations of the embedding space changes

We project the multidimensional embeddings of the 22 languages under consideration into a twodimensional space using UMAP (McInnes et al., 2020) to visualize how the embedding space changes. Sample visualizations for the changes to the embedding space pre- and post-intervention on Spanish are shown in Fig. 6.

![](images/2452eab2c5a3a5262fe29d9b940f391819ca703b2ddb5a45a6d2a20433a7b07b.jpg)  
(a) Bloom-7b

![](images/a6d2a014abddb3b8bee2ad8d054e346f5bc9f26b01b649c8d3841c5d4261d5da.jpg)  
(b) Aya-8b

![](images/e2c1505a427690a8b48692af037e410ad25328e966139d8bbe9d21a97565710e.jpg)  
(c) PolyLM-13b  
Figure 6: UMAP embeddings for 22 languages in the Bloom-7b model (left), Aya-8b (middle), and PolyLM-13b (right). The embeddings post-intervention are marked with ‘\*’ for each language. The dots represent individual sentences in the pre-intervention space; the crosses represent individual sentences in the post-intervention space. The languages that are not in the training set for a given model are marked in red. The colors of the point clouds identify individual languages and do not carry meaning.

## D Paraphrase retrieval accuracy for four intervention-target languages

<table><tr><td rowspan="2">Model</td><td rowspan="2">Language</td><td colspan="2">Top-1 Accuracy</td></tr><tr><td>(Pre)</td><td>(Post) (Mixed)</td></tr><tr><td rowspan="4">Bloom-7b</td><td>en</td><td>0.80</td><td>0.80 0.71</td></tr><tr><td>fr</td><td>0.80</td><td>0.80 0.26</td></tr><tr><td>de</td><td>0.72</td><td>0.75 0.22 0.59</td></tr><tr><td>ja</td><td>0.47</td><td>0.07</td></tr><tr><td rowspan="4">Aya-8b</td><td>en</td><td>0.87</td><td>0.87 0.56</td></tr><tr><td>fr</td><td>0.83</td><td>0.83 0.75</td></tr><tr><td>de</td><td>0.82</td><td>0.82 0.62</td></tr><tr><td>ja</td><td>0.70</td><td>0.76 0.55</td></tr><tr><td rowspan="4">PolyLM-13b</td><td>en</td><td>0.55</td><td>0.53 0.48</td></tr><tr><td>fr</td><td>0.52</td><td>0.50 0.44</td></tr><tr><td>de</td><td>0.50</td><td>0.55 0.39</td></tr><tr><td>ja</td><td>0.57</td><td>0.57 0.32</td></tr></table>

Table 5: Top-1 accuracy results for the paraphrase retrieval task for four intervention languages. Pre= both the query and the candidate embeddings are from the original unintervened model; Post= both the query and the candidate embeddings are from the intervened model; Mixed = query embedding is from the original model and the candidates are from the intervened model.

## E Top-1 cross-lingual retrieval accuracy for four intervention-target languages (query language is the same as the intervention target)

![](images/3f8d658d1c2f0681db3777cf5e4206df98f032561e3ed43ae8a9f4ca108c2d45.jpg)

![](images/3d8f7cfd2212568a3352870f2d7117c8c25afb099f03dc62e7259e533dbcf091.jpg)

![](images/8ac953cb74bcb3569bb62b60640d0c2a238b65b62bdc939250b2a3908c6f84cb.jpg)

![](images/a5bd7ed315a445e971b805349708db0a296a06fbce30d73447c05069f9f069f7.jpg)  
Figure 7: Top-1 retrieval accuracy for 22 languages in the Bloom-7b model. The language of the intervention is provided in the caption to each subfigure. The languages that are not in the training set for a given model are marked in red.

![](images/f49221e1a578d08a10cc3bbb8203b393bd96bb954f7e22231ef0c8d40bc8e13e.jpg)

![](images/b2c6b911bcf1d88b969522dad3c9a1c33db34e409bf9427e092732c1a0c4b86d.jpg)

![](images/b5f1610cf8654b05c5917a721c63fb6d10383be320e1393b57a524501ba1c658.jpg)

![](images/80c5875f61127132c7bd6a57128d37a70e3870026b7d472f13553e0a07f942d7.jpg)  
Figure 8: Top-1 retrieval accuracy for 22 languages in the Aya-8b model. The language of the intervention is provided in the caption to each subfigure. The languages that are not in the training set for a given model are marked in red.

![](images/a3a5d9bfc8c64d37046ee49889fd3fd262d0dd297d91e89cb8e1e865663b67da.jpg)

![](images/320580d2183c09d93dd6e7d7c04699652288d2036ffd9add3a4fd5c81e304890.jpg)

![](images/c6fee81344b32aa63ce5d4f2175dba629365cdf402b5c00b8b2cbe3fd0ab1a21.jpg)

![](images/df1e6470cecb45d0776e4f4e12d366bbc5369796a9006b52e8e65e2a5a6e11fc.jpg)  
Figure 9: Top-1 retrieval accuracy for 22 languages in the PolyLM-13b model. The language of the intervention is provided in the caption to each subfigure. The languages that are not in the training set for a given model are marked in red.

## F Top-1 cross-lingual retrieval accuracy for four intervention-target languages (query language is different from the intervention target)

![](images/48fbdd1f6b26867329f22ea8a2c55b10025db447db51907ad197acf9e5a12558.jpg)

![](images/8979066874a95462938d682655eef0d88124818fa1e81eefa1534a07ede89449.jpg)

![](images/5b51db9f663eb06c4b7d7a889792df1c6ac45d9e8c47a5b037657051db97f2e4.jpg)

![](images/6693b102d25b945c14408bf2c83d107faf9f70c1ec5be65d582e5fb14278b7a0.jpg)  
<sup>Figure</sup> <sup>10:</sup> <sup>(Top-1</sup> <sup>accuracy</sup>post-intervention − <sup>Top-1</sup> <sup>accuracy</sup>pre-intervention<sup>)</sup> <sup>for</sup> <sup>Bloom-7b.</sup> <sup>The</sup> <sup>language</sup> <sup>of</sup> <sup>the</sup> <sup>interven-</sup> tion is provided in the caption to each subfigure. The languages that are not in the training set are marked in red.

![](images/f5f4897842f4b240d7701d210f6f0562d067865832dea1aa1ba6032cd48179ce.jpg)

![](images/2bccccfc8eb4d028921ae13152a382ae23dc33358921b6ec4fdf91a5db2f155c.jpg)

![](images/4fbfefe494951285bc9735c20a79d0b357262b4ec3f4b2e283342f5282c5edbb.jpg)

![](images/15e82a3fa6e71256609fa7e79ed9c49df0b350260074e0f77cb558931d2c7f4f.jpg)  
<sup>Figure</sup> <sup>11:</sup> <sup>(Top-1</sup> <sup>accuracy</sup>post-intervention − <sup>Top-1</sup> <sup>accuracy</sup>pre-intervention<sup>)</sup> <sup>for</sup> <sup>Aya-8b.</sup> <sup>The</sup> <sup>language</sup> <sup>of</sup> <sup>the</sup> <sup>intervention</sup> is provided in the caption to each subfigure. The languages that are not in the training set are marked in red.

![](images/656683822d6996179647559b79dfe4910182bd497d277963767927ee0134b0dd.jpg)

![](images/86d5afb37174129c31ae9804aa92aa3171a7c93519f14baff7bd03e739de826d.jpg)

![](images/097c9c9408e25b1d6e7608b0aeb8c290b13df8962aea05b0a83f0764de675bc4.jpg)

![](images/e47c054cef1099e483ddacc863bbbda1d36bf237b82e716d50519121d33dc4ad.jpg)  
<sup>Figure</sup> <sup>12:</sup> <sup>(Top-1</sup> <sup>accuracy</sup>post-intervention − <sup>Top-1</sup> <sup>accuracy</sup>pre-intervention<sup>)</sup> <sup>for</sup> <sup>PolyLM-13b.</sup> <sup>The</sup> <sup>language</sup> <sup>of</sup> <sup>the</sup> intervention is provided in the caption to each subfigure. The languages that are not in the training set are marked in red.

## G Results for the interventions on random neurons

## G.1 Top-1 paraphrase retrieval accuracy after the intervention on random neurons

<table><tr><td rowspan="2">Model</td><td rowspan="2">Language</td><td colspan="3">Accuracy</td></tr><tr><td>Pre</td><td>Post</td><td>mixed</td></tr><tr><td rowspan="5">Bloom-7b</td><td>en</td><td>0.80</td><td>0.80</td><td>0.78</td></tr><tr><td>es</td><td>0.80</td><td>0.79</td><td>0.62</td></tr><tr><td>fr</td><td>0.80</td><td>0.71</td><td>0.00</td></tr><tr><td>de</td><td>0.72</td><td>0.72</td><td>0.62</td></tr><tr><td>ja</td><td>0.47</td><td>0.45</td><td>0.35</td></tr><tr><td rowspan="5">PolyLM-13b</td><td>en</td><td>0.55</td><td>0.55</td><td>0.51</td></tr><tr><td>es</td><td>0.53</td><td>0.53</td><td>0.48</td></tr><tr><td>fr</td><td>0.53</td><td>0.54</td><td>0.50</td></tr><tr><td>de</td><td>0.50</td><td>0.54</td><td>0.45</td></tr><tr><td>ja</td><td>0.60</td><td>0.58</td><td>0.23</td></tr><tr><td rowspan="5">Aya-8b</td><td>en</td><td>0.87</td><td>0.81</td><td>0.00</td></tr><tr><td>es</td><td>0.85</td><td>0.73</td><td>0.01</td></tr><tr><td>fr</td><td>0.83</td><td>0.70</td><td>0.01</td></tr><tr><td>de</td><td>0.82</td><td>0.70</td><td>0.00</td></tr><tr><td>ja</td><td>0.70</td><td>0.44</td><td>0.00</td></tr></table>

Table 6: Top-1 accuracy results for the paraphrase retrieval task for five intervention languages for the intervention on random neurons. Pre= both the query and the candidate embeddings are from the original unintervened model; Post= both the query and the candidate embeddings are from the intervened model; Mixed = query embedding is from the original model and the candidates are from the intervened model.

## G.2 Top-1 Retrieval accuracy for interventions on random neurons

![](images/b1865166134cbec4da3cebefd89a9b6cc8eab24a3c117507cb9a29c20ce339c1.jpg)

![](images/4521555c2153dfce0d3522769cb5948e0b4c67f8681b1f18f62c1e4abcb65041.jpg)

![](images/9ce2fd25c60147dcbc4dfb4f286cb13ddeabb4a40912f4491a30a7336603d33a.jpg)

![](images/867a138817fc8b8b2751cea2bf2a979a9faa9e3de2f394e11676af3387de3f04.jpg)  
Figure 13: Top-1 retrieval accuracy for 22 languages in the Bloom-7B model with the intervention on 100 random neurons. The language of the intervention is provided in the caption to each subfigure. The languages that are not in the training set are marked in red.

![](images/f982c9c9cbcd93cff18fe8dcc8d67155c1e8c5739362d0acf8dac0e936e4b9de.jpg)

![](images/fe3829a7d9248cd7338f334dd316ad8903accea98f8560f7449f33d29a4d6944.jpg)

![](images/0d63f062a78917dfc73d21df7e99b1e8c3add5dbbe18fd4146f2c35ad39aa920.jpg)

![](images/e82c9db766c4485260dce573f8dec69f554a1a4b158157e7335f61fa1c20e487.jpg)  
Figure 14: Top-1 retrieval accuracy for 22 languages in the Aya-8B model with the intervention on 2000 random neurons. The language of the intervention is provided in the caption to each subfigure. The languages that are not in the training set are marked in red.

![](images/314dd9001c7483e99e085a301e612b317251f93761c5e982cbad9be97ebfd09c.jpg)

![](images/ffa9c672374cbf62bc25638a0cd8a55587239b2ebabaacf6055376cfb0f0fa22.jpg)

![](images/17c3e3cbcfb0d18910a45fc87e9ad371c2100e513026b2b5779695398f6aece7.jpg)

![](images/de653df3c9ea7cbb26a6b8e38eab7a11737f151d9eeb4a7c22c4a1e23d2c69e0.jpg)  
Figure 15: Top-1 retrieval accuracy for 22 languages in the PolyLM-13b-chat model with the intervention on 2000 random neurons. The language of the intervention is provided in the caption to each subfigure. The languages that are not in the training are marked in red.

## H Cross-lingual retrieval results on BUCC-18 and Tatoeba

<table><tr><td>Model</td><td>Language</td><td>Top-1 Accuracy Pre</td><td>Post</td></tr><tr><td colspan="3">Tatoeba</td></tr><tr><td rowspan="4">Aya-8B</td><td>es</td><td>0.114</td><td>0.415</td></tr><tr><td>fr</td><td>0.087</td><td>0.251</td></tr><tr><td>de</td><td>0.119</td><td>0.444</td></tr><tr><td>jp</td><td>0.034</td><td>0.307</td></tr><tr><td rowspan="4">Bloom-7B</td><td>es</td><td>0.008</td><td>0.551</td></tr><tr><td>fr</td><td>0.011</td><td>0.434</td></tr><tr><td>de</td><td>0.006</td><td>0.032</td></tr><tr><td>jp</td><td>0.002</td><td>0.043</td></tr><tr><td rowspan="4">PolyLM-13B</td><td>es</td><td>0.082</td><td>0.178</td></tr><tr><td>fr</td><td>0.067</td><td>0.130</td></tr><tr><td>de</td><td>0.029</td><td>0.171</td></tr><tr><td>jp</td><td>0.000</td><td>0.040</td></tr><tr><td colspan="3">BUCC-18</td><td></td></tr><tr><td rowspan="2">Aya-8B</td><td>fr</td><td>0.012</td><td>0.073</td></tr><tr><td>de</td><td>0.017</td><td>0.332</td></tr><tr><td rowspan="2">Bloom-7B</td><td>fr</td><td>0.000</td><td>0.287</td></tr><tr><td>de</td><td>0.000</td><td>0.02</td></tr><tr><td rowspan="2">PolyLM-13B</td><td>fr</td><td>0.006</td><td>0.286</td></tr><tr><td>de</td><td>0.008</td><td>0.281</td></tr></table>

Table 7: Top-1 retrieval for the intervetion on five target languages for Tatoeba and BUCC-18. Pre= original model; Post= intervened model.

## I Computational budget

All experiments were run on 8 A100(80GB) GPUs. The total approximate running time for 90 GPU/hours Aya-8B, 120 GPU/hours for PolyLM-13B, and 110 GPU/hours for Bloom-7B.

## J License and Attribution

All datasets used in this work are supported by public licenses. PAWS-X, Tatoeba, BUCC are part of the XTREME benchmark licensed under Apache; Flores200 is licensed under Creative Commons. The pre-trained models used in this work are also supported by public licenses Bloom-7B (RAIL 1.0), Aya-8B (Creative Commons), and PolyLM-chat-13B (Apache).