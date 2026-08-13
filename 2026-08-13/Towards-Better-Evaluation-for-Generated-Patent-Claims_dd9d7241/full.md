# Towards Better Evaluation for Generated Patent Claims

Lekang Jiang†, Pascal A Scherz⋄, Stephan Goetz† †University of Cambridge, ⋄PSPB Patent Law {lj408, smg84}@cam.ac.uk, post@pspb.eu

## Abstract

Patent claims define the scope of protection and establish the legal boundaries of an invention. Drafting these claims is a complex and time-consuming process that usually requires the expertise of skilled patent attorneys, which can form a large access barrier for many small enterprises. To solve these challenges, researchers have investigated large language models (LLMs) for automating patent claim generation. However, existing studies highlight inconsistencies between automated evaluation metrics and human expert assessments. To bridge this gap, we introduce Patent-CE, the first comprehensive benchmark for evaluating patent claims. Patent-CE includes comparative claim evaluations annotated by patent experts, focusing on five key criteria: feature completeness, conceptual clarity, terminology consistency, logical linkage, and overall quality. Additionally, we propose PatClaimEval, a novel multi-dimensional evaluation method specifically designed for patent claims. Our experiments demonstrate that PatClaimEval achieves the highest correlation with human expert evaluations across all assessment criteria among all tested metrics. This research provides the groundwork for more accurate evaluations of automated patent claim generation systems.<sup>1</sup>

## 1 Introduction

The patent literature serves as a critical documentation of technological innovation (Mossoff, 2000). Patents are legal documents that grant exclusive rights to inventors in exchange for public disclosure of their inventions (Frumkin, 1947). The patent claims are the most legally significant section of a patent document, as they define the scope of protection and delineate the boundaries of an invention from known techniques to ensure legal enforceability (European Patent Office, 2000). Drafting precise and effective patent claims is a challenging task, which requires not only technical expertise but also an understanding of legal language and jurisdiction-specific regulations (Faber, 1990). Unlike general-purpose texts, patent claims must be both broad enough to encompass potential variations of an invention and specific enough to withstand legal scrutiny. This complexity often necessitates the involvement of skilled patent attorneys, which often renders the process both time-intensive and costly (LLP, 2023).

In response to these challenges, research has explored automated methods for patent claim generation to support inventors and attorneys. Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of general and patent-related tasks (Zhao et al., 2023; Jiang and Goetz, 2025). For instance, Jiang et al. (2025c) examined whether LLMs could generate high-quality patent claims based on patent descriptions, while another work investigated whether LLMs could revise patent claims to improve quality (Jiang et al., 2025b). These studies aim to accelerate the claim drafting process and reduce associated costs.

Despite these advances, a reliable automated evaluation for the quality of generated patent claims remains an unresolved challenge. Previous studies have relied on human expert evaluations, which are both time-consuming and costly, and they revealed inconsistencies between existing automated metrics and human assessments (Zuo et al., 2024; Jiang et al., 2025c,b). Table 1 highlights the fundamental differences between patent claim evaluation criteria and existing text evaluation methods. While current evaluations often rely on sequence overlap, semantic similarity, or multi-dimensional criteria such as coherence and fluency, patent claims have unique language requirements. Such requirements, such as the consistent use of terminology, technical formality, and high-level precision, are not common in other types of texts. Therefore, these differences underscore the limitations of existing metrics and highlight a significant gap in the reliability and validity of automated evaluation methods for patent claims.

<table><tr><td>Reference claims: 1. A mobile communications device comprising: a communication subsystem for communicating with a network;</td></tr><tr><td>a local common address database accessible to a plurality of applications on the mobile communications device; and . . 2. The mobile communications device of claim 1 wherein .. Candidate claims: 1. A mobile communications device comprising: a communication subsystem configured to facilitate communication with a network; a processor operably connected to the communication subsystem and to a memory, the memory containing a local common address database and instructions that ... 2. The mobile communications device of claim 1, wherein ..</td></tr><tr><td></td></tr><tr><td>N-gram-based evaluations (measuring sequence overlaps): BLEU ROUGE METEOR</td></tr><tr><td>Embedding-based evaluations (measuring semantic similarities): BERTScore BARTScore MoverScore SimCSE</td></tr><tr><td></td></tr><tr><td>Multi-dimensional evaluations: UniEval (Coherence, Consistency, Fluency, Relevance) AlignScore (Factual consistency) Human evaluations for patent claims:</td></tr></table>

Table 1: Comparison between current automatic text evaluation metrics and patent claim evaluation criteria. Patent claims have specific requirements different from other texts, which makes the evaluation difficult.

In this paper, we present the first benchmark for patent claim evaluation and propose a novel evaluation method tailored to the unique requirements of patent claims. The main contributions of this work are as follows:

1. We present Patent-CE, the first comprehensive benchmark for patent claim evaluation. Patent-CE includes 1,228 data points, which consist of a reference claim, two candidate claims, and comparative evaluations annotated by patent experts in five dimensions: feature completeness, conceptual clarity, terminology consistency, logical linkage, and overall quality.

2. We propose a novel multi-dimensional evaluation method for patent claims, named Pat-ClaimEval. PatClaimEval leverages Longformer (Beltagy et al., 2020) as its backbone and is trained on our dataset using a variation of contrastive learning (Gao et al., 2021).

3. We demonstrate the effectiveness of Pat-ClaimEval through extensive experiments. Our results show that PatClaimEval achieves the highest correlation with human expert evaluations across all assessment criteria compared to existing metrics, including 6 N-gram-based methods, 4 embeddingbased approaches, 2 multi-dimensional evaluators, and 1 LLM-as-a-judge method.

By tackling the evaluation problem, this research paves the way for more reliable assessments of automated patent claim generation systems, ultimately contributing to advancements in this emerging field.<sup>2</sup>

## 2 Related Works

## 2.1 Patent Claim Generation

Some studies have explored LLMs for automatically generating patent claims. An early investigation by Lee and Hsiang (2020) served as a preliminary effort and focused on fine-tuning GPT-2 (Radford et al., 2019) to generate patent-like texts. The authors found that minimal training was sufficient to produce patent-like outputs but did not assess the quality of the generated text. Building on this, Lee (2020) trained GPT-2 to transform one section of a patent application into another, such as the generation of abstracts from titles or claims from abstracts. However, since abstracts are often generic and imprecise, the generation of claims from abstracts may not be a well-conditioned task.

Therefore, Jiang et al. (2025c) introduced a description-based claim generation task and evaluated the performance of various LLMs on this domain-specific challenge. Their human evaluation by patent professionals highlighted the limitations of various LLMs in generating high-quality patent claims and revealed inconsistencies between automated and human evaluation metrics. While previous studies tested the models on U.S. patents, Jiang et al. (2025a) further investigated the claim generation task on European patents. Additionally, Jiang et al. (2025b) extended the research to claim revision, investigating whether LLMs could further enhance claim quality.

<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=3>Task           Domain          Evaluation Criteria</td></tr><tr><td rowspan=1 colspan=1>QAGS (Wang et al., 2020)</td><td rowspan=1 colspan=3>Summarization        News   Factual consistency</td></tr><tr><td rowspan=1 colspan=1>SummEval (Fabbri et al., 2021)</td><td rowspan=1 colspan=1>Summarization</td><td rowspan=1 colspan=1>News</td><td rowspan=1 colspan=1>Fluency, coherence, consistency, rele-vance</td></tr><tr><td rowspan=1 colspan=1>Persona-Chat (Zhang et al., 2018)</td><td rowspan=1 colspan=1>Dialogue generation</td><td rowspan=1 colspan=1>General</td><td rowspan=1 colspan=1>Fluency, engagingness, consistency,personalization</td></tr><tr><td rowspan=1 colspan=1>Topical-Chat (Mehri and Eskenazi, 2020)</td><td rowspan=1 colspan=3>Naturalness, coherence, engagingness,Dialogue generation    Generalgroundedness, understandability</td></tr><tr><td rowspan=1 colspan=1>ToTTo (Parikh et al., 2020)</td><td rowspan=1 colspan=1>Table-to-text generation</td><td rowspan=1 colspan=2>General  Fluency, faithfulness, coverage</td></tr><tr><td rowspan=1 colspan=1>Patent-CE (Ours)</td><td rowspan=1 colspan=1>Patent claim generation</td><td rowspan=1 colspan=1>Patent</td><td rowspan=1 colspan=1>Feature completeness, conceptual clar-ity, terminology consistency, logicallinkage, overall quality</td></tr></table>

Table 2: Comparison of commonly used benchmarks for text generation evaluation. Patent claims have unique language requirements different from other type of texts.

Another study proposed the task of next-claim generation (Zuo et al., 2024), which involves generating the second and/or third claims based on the first claim. Likewise, this research also demonstrated a weak correlation between automated and human evaluation results for the next-claim generation task.

## 2.2 Benchmarks for Text Generation Evaluation

Accurately and efficiently evaluating the quality of generated texts is important for developing textgeneration LLMs. Researchers have introduced several text evaluation datasets across various domains, each with distinct evaluation criteria. Table 2 summarizes some commonly used benchmarks for text generation evaluations.

Datasets, such as QAGS (Wang et al., 2020) and SummEval (Fabbri et al., 2021), are designed to evaluate summarization tasks within the news domain. These benchmarks focus on criteria including factual consistency, fluency, coherence, and relevance to ensure the generated summaries accurately represent the source text while maintaining readability. In addition, dialogue systems have benefited from benchmarks such as Persona-Chat (Zhang et al., 2018) and Topical-Chat (Mehri and Eskenazi, 2020), which target open-domain conversational tasks. Persona-Chat emphasizes personalization, fluency, and engagingness, while Topical-Chat introduces evaluation metrics for naturalness, coherence, and groundedness to advance the development of more realistic and context-aware conversational AI. Furthermore, the ToTTo dataset (Parikh et al., 2020) supports the task of converting structured tables into natural text. It evaluates fluency, faithfulness, and coverage to ensure the generated text aligns accurately with tabular inputs and effectively conveys the intended information.

Our Patent-CE dataset is specifically designed for the task of patent claim generation. Unlike other benchmarks, Patent-CE emphasizes feature completeness, clarity, terminology consistency, and logical linkage. All these critical aspects are for the legal and technical precision required in patent documentation. This dataset fills an essential gap by providing a benchmark tailored to the patent domain, presenting unique challenges not encountered in general-domain tasks.

## 3 Dataset

## 3.1 Human Annotation

Experienced patent experts were provided with reference and candidate patent claims to evaluate. Their evaluation was based on five aspects, adhering to established examination criteria (Jiang et al., 2025c): feature completeness, conceptual clarity, terminology consistency, logical linkage, and overall quality. These evaluation aspects are consistent with previous research and defined as follows.

(1) Feature Completeness: The extent to which the generated claims encapsulate all critical aspects of the invention. (2) Conceptual Clarity: The clarity and unambiguity of the language used in the claims. (3) Terminology Consistency: The uniformity in the use of terms throughout the claims. (4) Correctness of Logical Linkages: The accuracy with which features are interconnected and related. (5) Overall Quality: An aggregate measure that combines all the above criteria. Detailed evaluation instructions can be found in Appendix A.

## 3.2 Construction

To create a comprehensive dataset and mitigate potential biases, we collected data from three different sources.

First, we used the dataset from Jiang et al. (2025c), in which LLMs were used to generate patent claims based on descriptions from the United States Patent and Trademark Office (USPTO). This dataset also includes human evaluations which compare the performance of different models. Second, we incorporated data from another study that investigated patent claim revision using data from the European Patent Office (EPO) and also included human evaluations (Jiang et al., 2025b). Both studies rated claims based on feature completeness, conceptual clarity, terminology consistency, logical linkage, and overall quality. We integrated these data to construct a comprehensive evaluation benchmark. Additionally, to further increase the dataset size and enhance robustness, we conducted new annotations by consulting patent attorneys. These additional annotations were applied to claims obtained from the aforementioned studies.

We recognize that the absolute quality scores from different sources may vary due to differences in expert interpretation. However, the relative ranking of the same claim sets should remain rather consistent across evaluations. Therefore, similar to prior work by Zuo et al. (2024), our dataset focuses on comparative evaluations. Each data point consists of a quadruplet $( A , B , C , y )$ , where A represents the reference claims, B and C are two generated claims, and the label y indicates whether B or C is better, or if they are of equal quality.

## 3.3 Statistics

The dataset consists of a total of 1,228 data points evaluated in five aspects. As shown in Table 3, the data distribution is relatively balanced for each aspect, with similar proportions in each category. Appendix B introduces more dataset statistics. We randomly selected 184 examples (about 15%) as the test set and used the remaining for training.

<table><tr><td>Dimension</td><td># (B &gt; C)</td><td># (B = C)</td><td># (B &lt; C)</td></tr><tr><td>Completeness</td><td>426</td><td>375</td><td>427</td></tr><tr><td>Clarity</td><td>424</td><td>378</td><td>426</td></tr><tr><td>Consistency</td><td>420</td><td>386</td><td>422</td></tr><tr><td>Linkage</td><td>430</td><td>366</td><td>432</td></tr><tr><td>Quality</td><td>422</td><td>382</td><td>424</td></tr></table>

Table 3: Data distribution of the Patent-CE dataset. The data distribution is relatively balanced in each evaluation dimension.

Our benchmark offers two major advantages: Comprehensiveness: It incorporates patent data from multiple patent offices, which makes it more representative and robust than any previous work. Larger scale: Although some data builds on previous work, we manually refine and annotate more data to substantially expand the dataset size and broaden coverage.

## 4 Method

We propose PatClaimEval, a new automated evaluation method to assess the quality of generated patent claims compared to gold claims. Given a reference claim set P and a candidate claim set Q, the model predicts a quality score of Q, denoted as s(Q P). We train five models to evaluate patent claims from different aspects, including feature completeness, conceptual clarity, terminology consistency, logical linkage, and overall quality. We do not jointly train one model for all five aspects, such as using multi-task learning (Zhang and Yang, 2021), because of the conflicting optimization objectives for different tasks. For example, feature completeness and clarity are not inherently related—the claim could include all essential features, but the expression is ambiguous.

## 4.1 Model Architecture

We leverage Longformer<sup>3</sup> (Beltagy et al., 2020) as the backbone to handle long input sequences efficiently. We have not used patent-specific LLMs due to their limitations; for instance, PatentGPT is closed-source (Bai et al., 2024), and PatentGPT-J has a restricted context length (Lee, 2023). The small context length is a particular problem for patent texts, as it may fall short of typical patent claims. The average length of patent claims is more than 1,000 tokens (Suzgun et al., 2023), and Longformer can support up to 4,096 tokens of input. Thus, we use Longformer because it is open source, supports long input length (large enough for patent claims), and offers a controllable model size. Future work may investigate larger models with 7 or 8 billion parameters. The model encodes inputs and obtains the representation for a given input claim pair $( P , Q )$ per

$$
\mathbf { h } = { \mathcal { M } } ( [ P ; Q ] ) .\tag{1}
$$

It subsequently connects with a fully connected layer to get a quality score and a sigmoid function that maps the score to the range [0, 1] as

$$
s ( Q | P ) = \sigma ( \mathbf { w } ^ { \top } \mathbf { h } + b ) .\tag{2}
$$

## 4.2 Training Strategy

Our proposed training method draws inspiration from contrastive learning (Khosla et al., 2020) as the dataset presents relative relationships between samples. In the context of NLP, contrastive learning is used to align embeddings of related text pairs or to learn discriminative representations (Gao et al., 2021). Through contrastive loss functions, models can capture nuanced differences between text samples, making contrastive learning particularly suitable for tasks involving relative comparisons. Unlike traditional contrastive learning, which explicitly constructs positive and negative sample pairs, our method integrates label information directly to define optimization objectives tailored for different evaluation aspects.

The training data consists of quadruplets $( A , B , C , y )$ , where A is the reference claims, B and C are two generated claims, and $y \in$ 1, 0, 1 indicates their relative quality:

$$
y = \left\{ { \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } s ( B | A ) > s ( C | A ) } \\ { 0 , } & { { \mathrm { i f ~ } } s ( B | A ) = s ( C | A ) } \\ { - 1 , } & { { \mathrm { i f ~ } } s ( B | A ) < s ( C | A ) } \end{array} } \right.\tag{3}
$$

The model computes scores $s _ { B } = s ( B | A )$ and $s _ { C } = s ( C | A )$ . To optimize the model, we define the loss function as

$$
\mathcal { L } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \ell ( s _ { B _ { i } } , s _ { C _ { i } } , y _ { i } ) ,\tag{4}
$$

where $\ell ( s _ { B _ { i } } , s _ { C _ { i } } , y _ { i } )$ is defined as

$$
\ell = \left\{ \begin{array} { l l } { R e L U ( m - ( s _ { B _ { i } } - s _ { C _ { i } } ) ) , } & { \mathrm { i f ~ } y _ { i } = 1 , } \\ { R e L U ( | s _ { B _ { i } } - s _ { C _ { i } } | - n ) , } & { \mathrm { i f ~ } y _ { i } = 0 , } \\ { R e L U \left( m - \left( s _ { C _ { i } } - s _ { B _ { i } } \right) \right) , } & { \mathrm { i f ~ } y _ { i } = - 1 } \end{array} \right.\tag{5}
$$

where m is the margin hyper-parameter that enforces a minimum separation between scores for distinct quality levels, and n is a tolerance parameter that allows small differences between scores when the two claim sets are judged equally good.

By minimizing this loss, the model learns to align the predicted scores with the relative quality judgments. The margin m is a hyper-parameter that controls the separation between scores, ensuring that the model is confident in its predictions for cases where one claim set is clearly better than the other. This objective function allows the model to capture fine-grained distinctions in quality across diverse claim pairs. We introduce the training and evaluation details in Appendix C.

## 5 Experiments

## 5.1 Baselines

BLEU (Papineni et al., 2002) and ROUGE (Lin, 2004) are classic metrics widely used for evaluating text overlaps. BLEU measures n-gram precision by comparing candidate and reference texts, while ROUGE primarily evaluates recall-based overlap, commonly used in summarization. METEOR (Banerjee and Lavie, 2005) improves on BLEU by incorporating synonymy, stemming, and other linguistic features, thereby providing a more flexible approach to measuring textual overlap.

BERTScore (Zhang et al., 2019) computes similarity using contextualized embeddings from BERT (Devlin et al., 2019), enabling a more nuanced assessment of semantic similarity between reference and candidate sentences. BARTScore (Yuan et al., 2021), derived from the BART model (Lewis et al., 2020), uses a generative scoring approach to evaluate the likelihood of generating the candidate text from a reference. MoverScore (Zhao et al., 2019) measures the semantic similarity by calculating the minimum cost of transforming candidate embeddings to reference embeddings, effectively capturing semantic alignment. SimCSE (Gao et al., 2021) further enhances representation quality by using contrastive learning to generate sentence embeddings, which have been shown to perform well in semantic similarity tasks.

We also test recent multi-dimensional evaluation frameworks, including UniEval (Zhong et al., 2022) and AlignScore (Zha et al., 2023). UniEval provides a unified evaluation protocol for different aspects of natural language generation, such as coherence and fluency. Since our dataset does not include context information or source texts, we use UniEval to evaluate the relevance between generated and reference claims. Additionally, we use AlignScore as a representative to assess factual consistency between source and generated content.

The LLM-as-a-judge paradigm is becoming popular (Zheng et al., 2023), where LLMs are used as evaluators of generated content. This approach leverages the capabilities of pre-trained LLMs, such as GPT-4 (OpenAI, 2023), to serve as surrogate judges that can assess generated text for qualities like fluency, coherence, and factual consistency. In our experiments, we specifically focus on G-Eval-4 (Liu et al., 2023) because it has shown high agreement with human preference across multiple benchmarks (Zheng et al., 2023). Other LLMas-a-judge models are not tested because they use synthetic examples generated by GPT-4 for training, such as JudgeLM (Zhu et al., 2023) PandaLM (Wang et al., 2024). We ask GPT-4 to evaluate the given claims through chain-of-thought (CoT) prompting (Wei et al., 2022) by comparing them to the reference claims. The evaluation dimensions are the same as human expert metrics. We introduce detailed settings in Appendix D.

## 5.2 Evaluations

We used the Kendall-Tau correlation to assess the overall alignment with human judgment, following the approach of previous work by Zuo et al. (2024). This correlation metric evaluates the consistency of the global ranking while disregarding minor errors in individual predictions. We additionally report the Spearman correlation. Compared to Kendall-Tau, Spearman is more sensitive to large rank differences, providing a complementary perspective on the metric ability to predict relative claim quality.

Since the dataset originally presents a three-way classification problem, we also use accuracy and F1 scores to assess model performance. These metrics reflect the model’s ability to make precise decisions for individual input pairs, providing a more comprehensive view of its effectiveness. Classification labels can be obtained directly from G-Eval-4, while for other metrics, we assume quality scores to be equivalent if the score differences are less than 10−<sup>4</sup>.

## 6 Results

## 6.1 Correlations with Human Evaluations

Table 4 presents the Kendall-Tau and Spearman correlation between different automated metrics and human evaluation results across five criteria: feature completeness, conceptual clarity, terminology consistency, logical linkage, and overall quality.

Overall, PatClaimEval demonstrates the highest correlation with human evaluations across all criteria, suggesting its effectiveness in evaluating patent claim quality. For feature completeness, PatClaimEval achieves a correlation of $\tau = 0 . 4 0 0$ and $\rho = 0 . 5 0 4$ , which outperforms all other metrics. This finding holds consistently across other criteria, with correlations of $\tau = 0 . 4 6 1$ and $\rho =$ 0.518 for clarity, $\tau = 0 . 3 5 4$ and $\rho = 0 . 4 2 4$ for consistency, $\tau = 0 . 4 1 9$ and ρ = 0.518 for linkage, and $\tau = 0 . 4 7 7$ and $\rho = 0 . 6 0 2$ for overall quality. Notably, these values are not only the highest but also significantly surpass existing metrics in their alignment with human judgments. Particularly in overall quality, PatClaimEval outperforms the second-best result by approximately 41.5% and 58.0% for Kendall-Tau and Spearman correlation respectively.

In addition, N-gram-based metrics demonstrate relatively higher correlations than embedding-based methods in evaluating patent claims. While N-gram-based methods can sometimes achieve correlation scores exceeding 0.3 in different evaluation aspects, embedding-based metrics rarely surpass this threshold. For instance, ROUGE-L achieves the second-highest Spearman correlation in logical linkage with a score of 0.391. N-gram-based methods rely on surface-level overlap between generated and reference text, without capturing semantic information or contextual relevance. These methods typically underperform compared to embedding-based approaches, which calculate semantic similarities, in standard text evaluation tasks (Zhang et al., 2019; Zhao et al., 2019; Yuan et al., 2021). However, patent claim evaluation results diverge from these prior findings due to its unique focus on patent examination criteria. Both reference and candidate claims describe the same invention but often use different expressions. In this context, high semantic similarity does not necessarily indicate adherence to patent requirements, resulting in weak correlations with human judgments. In contrast, gold-standard patent claims use precise language and expressions designed to meet examination standards. Thus, more overlaps with these gold claims may better reflect higher quality, potentially explaining why simple overlapbased methods outperform embedding-based similarity approaches in this domain. These findings extend to metrics of UniEval and AlignScore. While AlignScore assesses factual consistency and shows less correlation, UniEval that measures relevance between candidate and reference claims performs relatively better.

<table><tr><td rowspan="2">Type</td><td rowspan="2">Metric</td><td colspan="2">Completeness</td><td colspan="2">Clarity</td><td colspan="2">Consistency</td><td colspan="2">Linkage</td><td colspan="2">Quality</td></tr><tr><td>T</td><td>ρ</td><td>T</td><td>ρ</td><td>T</td><td>ρ</td><td>T</td><td>ρ</td><td>T</td><td>ρ</td></tr><tr><td rowspan="5">N-gram</td><td>BLEU-1</td><td>0.305</td><td>0.345</td><td>0.359</td><td>0.401</td><td>0.284</td><td>0.329</td><td>0.335</td><td>0.376</td><td>0.326</td><td>0.369</td></tr><tr><td>BLEU-4</td><td>0.271</td><td>0.304</td><td>0.280</td><td>0.312</td><td>0.227</td><td>0.263</td><td>0.256</td><td>0.289</td><td>0.269</td><td>0.305</td></tr><tr><td>ROUGE-1</td><td>0.305</td><td>0.342</td><td>0.314</td><td>0.351</td><td>0.238</td><td>0.279</td><td>0.301</td><td>0.341</td><td>0.292</td><td>0.332</td></tr><tr><td>ROUGE-2</td><td>0.305</td><td>0.342</td><td>0.280</td><td>0.312</td><td>0.215</td><td>0.251</td><td>0.268</td><td>0.303</td><td>0.269</td><td>0.306</td></tr><tr><td>ROUGE-L METEOR</td><td>0.282 0.316</td><td>0.317 0.358</td><td>0.280 0.371</td><td>0.312 0.414</td><td>0.261 0.307</td><td>0.303 0.355</td><td>0.346</td><td>0.391</td><td>0.303 0.292</td><td>0.344</td></tr><tr><td rowspan="3">Embedding</td><td>BERTScore</td><td>0.241</td><td>0.279</td><td>0.251</td><td>0.281</td><td>0.242</td><td>0.283</td><td>0.324 0.272</td><td>0.364 0.303</td><td>0.239</td><td>0.331 0.268</td></tr><tr><td>BARTScore</td><td>0.165</td><td>0.188</td><td>0.130</td><td>0.146</td><td>0.211</td><td>0.242</td><td>0.196</td><td>0.219</td><td>0.164</td><td>0.185</td></tr><tr><td>MoverScore</td><td>0.199</td><td>0.227</td><td>0.199</td><td>0.217</td><td>0.223</td><td>0.264</td><td>0.231</td><td>0.265</td><td>0.210</td><td>0.243</td></tr><tr><td>Miscellaneous</td><td>SimCSE UniEval</td><td>0.177 0.339</td><td>0.196 0.383</td><td>0.165 0.337</td><td>0.173 0.375</td><td>0.143 0.261</td><td>0.165 0.302</td><td>0.220 0.301</td><td>0.246 0.338</td><td>0.165 0.337</td><td>0.186 0.381</td></tr><tr><td>LLM-as-a-judge</td><td>AlignScore G-Eval-4</td><td>0.146</td><td>0.162</td><td>0.145</td><td>0.160</td><td>0.261</td><td>0.305</td><td>0.200</td><td>0.226</td><td>0.224</td><td>0.255</td></tr><tr><td>Ours</td><td></td><td>0.377</td><td>0.410</td><td>0.412</td><td>0.481</td><td>0.276</td><td>0.353</td><td>0.350</td><td>0.385</td><td>0.277</td><td>0.310</td></tr><tr><td></td><td>PatClaimEval</td><td>0.400</td><td>0.504</td><td>0.461</td><td>0.518</td><td>0.354</td><td>0.424</td><td>0.419</td><td>0.518</td><td>0.477</td><td>0.602</td></tr></table>

Table 4: Kendall-Tau (τ) and Spearman (ρ) correlation of automated metrics with human evaluation results. The highest number in each criterion is in bold, and the second-best result is underlined. PatClaimEval shows the highest correlation with human assessments in all criteria.

G-Eval-4 shows strong performance in evaluating completeness, clarity, and linkage. G-Eval-4 achieves correlation scores of $\tau = 0 . 3 7 7$ and $\rho = 0 . 4 1 0$ for completeness, which surpasses all other metrics except for PatClaimEval and is consistent with findings from prior research (Jiang et al., 2025b). The high correlation in feature completeness can be attributed to GPT-4’s proven capabilities in information extraction (OpenAI, 2023; Li et al., 2023). In the context of claim evaluation, GPT-4 effectively extracts key features from both reference and candidate claims. In consequence, it enables accurate comparisons and reaches high scores in feature completeness assessments. However, its performance in terminology consistency and overall quality is less impressive. A plausible explanation is that GPT-4 is not extensively trained on patent-specific texts, which limits its ability to comprehend the unique linguistic and structural requirements of patent claims. Consequently, relying solely on prompting without further fine-tuning may be insufficient for accurately evaluating patent claims.

## 6.2 Classification Performance

Table 5 presents the accuracy and F1 scores of different metrics on each evaluation criterion as a classification problem, including feature completeness, conceptual clarity, terminology consistency, logical linkage, and overall quality.

PatClaimEval achieves the highest accuracy and F1 scores across nearly all evaluation criteria. Specifically, for conceptual clarity, Pat-ClaimEval achieves an accuracy of 60.3% and an F1 score of 59.5%, outperforming all other metrics. This superior performance extends to consistency and overall quality, where PatClaimEval consistently outperforms other methods. In feature completeness, G-Eval-4 demonstrates slightly better performance to PatClaimEval, with both accuracy and F1 scores of 54.8%. Despite these strengths, PatClaimEval’s absolute scores in some evaluation criteria, such as consistency, remain modest (50.0% accuracy and 49.3% F1 score). The moderate absolute scores indicate potential for improvement, such as expanding dataset sizes, larger models, or more sophisticated training strategies. Nonetheless, PatClaimEval represents a significant advancement as it achieves a 3.8% improvement in accuracy and a 10.5% increase in F1 score over the second-best method for overall quality evaluation. It currently stands as the most effective approach for patent claim evaluation.

<table><tr><td rowspan="2">Type</td><td rowspan="2">Metric</td><td colspan="2">Completeness</td><td colspan="2">Clarity</td><td colspan="2">Consistency</td><td colspan="2">Linkage</td><td colspan="2">Quality</td></tr><tr><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td></tr><tr><td rowspan="5">N-gram</td><td>BLEU-1</td><td>50.5</td><td>42.6</td><td>54.3</td><td>46.7</td><td>47.8</td><td>39.2</td><td>53.8</td><td>46.5</td><td>52.2</td><td>44.3</td></tr><tr><td>BLEU-4</td><td>48.9</td><td>41.4</td><td>50.5</td><td>43.5</td><td>45.1</td><td>37.0</td><td>50.0</td><td>43.2</td><td>49.5</td><td>42.0</td></tr><tr><td>ROUGE-1</td><td>50.5</td><td>42.8</td><td>52.2</td><td>44.8</td><td>45.7</td><td>37.5</td><td>52.2</td><td>45.0</td><td>50.5</td><td>42.9</td></tr><tr><td>ROUGE-2</td><td>50.5</td><td>42.8</td><td>50.5</td><td>43.4</td><td>44.6</td><td>36.6</td><td>50.5</td><td>43.6</td><td>49.5</td><td>42.0</td></tr><tr><td>ROUGE-L METEOR</td><td>49.5 51.1</td><td>41.8 43.2</td><td>50.5 54.9</td><td>43.5 47.2</td><td>46.7 48.9</td><td>38.3 40.1</td><td>54.3 53.3</td><td>46.9 46.0</td><td>51.1 50.5</td><td>43.4 42.8</td></tr><tr><td rowspan="4">Embedding</td><td>BERTScore</td><td>46.7</td><td>39.1</td><td>48.9</td><td>42.2</td><td>45.7</td><td>37.6</td><td>51.1</td><td>44.8</td><td>48.4</td><td>41.8</td></tr><tr><td>BARTScore</td><td>43.5</td><td>35.9</td><td>42.9</td><td>36.3</td><td>44.6</td><td>36.4</td><td>47.3</td><td>40.7</td><td>44.6</td><td>37.5</td></tr><tr><td>MoverScore</td><td>45.1</td><td>38.4</td><td>46.7</td><td>41.1</td><td>44.6</td><td>36.8</td><td>48.4</td><td>42.0</td><td>46.2</td><td>39.5</td></tr><tr><td>SimCSE</td><td>44.6</td><td>38.4</td><td>45.7</td><td>40.7</td><td>41.3</td><td>34.6</td><td>48.4</td><td>42.6</td><td>44.6</td><td>38.6</td></tr><tr><td rowspan="2">Miscellaneous</td><td>UniEval</td><td>52.2</td><td>44.1</td><td>53.3</td><td>45.8</td><td>46.7</td><td>38.3</td><td>52.2</td><td>45.0</td><td>52.7</td><td>44.8</td></tr><tr><td>AlignScore</td><td>42.9</td><td>36.4</td><td>44.0</td><td>38.0</td><td>46.7</td><td>38.3</td><td>47.3</td><td>40.8</td><td>47.3</td><td>40.1</td></tr><tr><td>LLM-as-a-judge</td><td>G-Eval-4</td><td>54.8</td><td>54.8</td><td>55.6</td><td>55.9</td><td>45.9</td><td>43.7</td><td>54.8</td><td>54.6</td><td>49.6</td><td>46.9</td></tr><tr><td>Ours</td><td>PatClaimEval</td><td>52.7</td><td>53.2</td><td>60.3</td><td>59.5</td><td>50.0</td><td>49.3</td><td>52.7</td><td>54.7</td><td>56.5</td><td>57.4</td></tr></table>

Table 5: Accuracy (Acc %) and F1 scores (F1 %) on each evaluation criterion. The highest number in each column is in bold, and the second-best result is underlined. PatClaimEval demonstrates relatively high and balanced accuracy and F1 scores across all evaluation criteria.

PatClaimEval and G-Eval-4 exhibit balanced performance between accuracy and F1 scores. Both models achieve similar accuracy and F1 scores across all five evaluation criteria, in contrast to other metrics, where F1 scores are normally notably lower than their accuracies. This balance reflects an effective trade-off between precision and recall. Although G-Eval-4 does not lead in accuracy across all aspects, its F1 scores are consistently higher than other metrics except for PatClaimEval. Based on a careful examination of the results, we attribute this strength to G-Eval-4’s ability to handle "equal cases", in which two candidate claims receive identical quality scores. Metrics such as Ngram-based and embedding-based methods struggle to evaluate such cases effectively, resulting in discrepancies between their accuracy and F1 scores. The balanced performance of PatClaimEval and G-Eval-4 highlights their robustness and reliability in patent claim evaluation.

## 6.3 Qualitative Analysis

We show an example of claim comparison in Table 7 to demonstrate the inherent challenges of this task, where A represents the gold claim, and B and C are candidate claims. We identify three types of differences between generated claims B and C. First, Claim C demonstrates higher clarity and language precision. It correctly uses an annular edge, whereas Claim B incorrectly uses a annular edge, a basic grammatical error. Furthermore, in Claim 3, C usesfurther comprising, which aligns with the gold claim and drafting conventions, while B uses the inappropriate comprises. Second, Claim C exhibits a stronger logical linkage between components. It introduces dependent clauses in Claim 3 properly using wherein that preserves structural relationships between features, whereas Claim B omits such linkages. Third, Claim C uses the phrase are configured to when describing some features. While this phrasing deviates from the gold claim, it does not degrade the quality. Overall, Claim C is better than Claim B. However, current metrics cannot capture such subtle and special differences, which could lead to unreliable performance in claim evaluation.

## 7 Conclusion

We introduce Patent-CE, the first comprehensive benchmark for evaluating patent claims. Patent-CE includes comparative evaluations annotated by patent experts, which focus on five key criteria that align with established patent examination standards: feature completeness, conceptual clarity, terminology consistency, logical linkage, and overall quality. Moreover, we propose Pat-ClaimEval, a novel multi-dimensional evaluation method specifically designed for patent claims. Extensive experiments demonstrate the effectiveness of PatClaimEval. It achieves the highest correlation with human expert evaluations across all assessment criteria when compared to existing metrics.

This research provides valuable resources for developing automated evaluation methods of patent claims and establishes a solid foundation for more reliable assessments of claim generation systems.

## Limitations

We acknowledge several limitations in this research. Firstly, the dataset used in this study includes only patents documented in English, which may affect the applicability to patents in other languages. In addition, the correlations of our method with human assessments are still somewhat low (Kendall-Tau < 0.5) and improvements are needed. Better semantic methods or different kinds of CoT prompting strategies may also be worth investigating. Furthermore, our evaluation approach relies on a gold standard. It provides a more reliable way to evaluate patent claims and is especially useful when developing related models for claim generation. However, real-world patent examinations by patent offices consider a range of criteria, including novelty, non-obviousness, and language requirements, without necessarily referencing a predefined gold standard. Patents are evaluated based on their intrinsic merit and their relation to prior art. Therefore, exploring reference-free evaluation approaches for patent claims is an important and worthwhile direction for future work.

## Ethics Statement

GPT-4 is under a commercial license provided by OpenAI, and we access it through its API. The use of existing artifacts, including models, evaluation metrics, and datasets, is consistent with their intended use. Our proposed dataset is used for patent claim generation evaluation and will be released under CC-BY-NC-4.0 license. This dataset does not include potential personal information or offensive content, and no ethics review board was involved.

## References

Zilong Bai, Ruiji Zhang, Linqing Chen, Qijun Cai, Yuan Zhong, Cong Wang, Yan Fang, Jie Fang, Jing Sun, Weikuan Wang, et al. 2024. Patentgpt: A large language model for intellectual property. arXiv preprint arXiv:2404.18255.

Satanjeev Banerjee and Alon Lavie. 2005. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In Proceedings ofthe ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, pages 65–72, Ann Arbor,

Michigan. Association for Computational Linguistics.

Iz Beltagy, Matthew E Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

European Patent Office. 2000. Epc - the european patent convention. https://www.epo.org/en/ legal/epc/2020/regulations.html. Accessed: 2023-06-12.

Alexander R. Fabbri, Wojciech Krysci ´ nski, Bryan Mc-´ Cann, Caiming Xiong, Richard Socher, and Dragomir Radev. 2021. SummEval: Re-evaluating summarization evaluation. Transactions ofthe Associationfor Computational Linguistics, 9:391–409.

Robert C Faber. 1990. Landis on mechanics of patent claim drafting. Practising Law Institute New York.

M. Frumkin. 1947. Early history of patents for innovation. Transactions of the Newcomen Society, 26(1):47–56.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple contrastive learning of sentence embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6894–6910, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Lekang Jiang and Stephan M Goetz. 2025. Natural language processing in the patent domain: a survey. Artificial Intelligence Review, 58(7):214.

Lekang Jiang, Chengzu Li, and Stephan Goetz. 2025a. Enriching patent claim generation with european patent dataset. arXiv preprint arXiv:2505.12568.

Lekang Jiang, Pascal A. Scherz, and Stefan Goetz. 2025b. Patent-CR: A dataset for patent claim revision. In Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2300– 2314, Albuquerque, New Mexico. Association for Computational Linguistics.

Lekang Jiang, Caiqi Zhang, Pascal A. Scherz, and Stefan Goetz. 2025c. Can large language models generate high-quality patent claims? In Findings of the Association for Computational Linguistics: NAACL 2025, pages 1272–1287, Albuquerque, New Mexico. Association for Computational Linguistics.

Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. 2020. Supervised contrastive learning. Advances in neural information processing systems, 33:18661–18673.

Jieh-Sheng Lee. 2020. Controlling patent text generation by structural metadata. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management, pages 3241–3244.

Jieh-Sheng Lee. 2023. Evaluating generative patent language models. World Patent Information, 72:102173.

Jieh-Sheng Lee and Jieh Hsiang. 2020. Patent claim generation by fine-tuning openai gpt-2. World Patent Information, 62:101983.

Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020. BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Proceedings of the 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 7871–7880, Online. Association for Computational Linguistics.

Bo Li, Gexiang Fang, Yang Yang, Quansen Wang, Wei Ye, Wen Zhao, and Shikun Zhang. 2023. Evaluating chatgpt’s information extraction capabilities: An assessment of performance, explainability, calibration, and faithfulness. arXiv preprint arXiv:2304.11633.

Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-eval: NLG evaluation using gpt-4 with better human alignment. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2511–2522, Singapore. Association for Computational Linguistics.

Cislo & Thomas LLP. 2023. Typical fees. Accessed: 2024-10-15.

Shikib Mehri and Maxine Eskenazi. 2020. USR: An unsupervised and reference free evaluation metric for dialog generation. In Proceedings of the 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 681–707, Online. Association for Computational Linguistics.

Adam Mossoff. 2000. Rethinking the development of patents: an intellectual history, 1550-1800. Hastings Lj, 52:1255.

OpenAI. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings ofthe

40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.

Ankur Parikh, Xuezhi Wang, Sebastian Gehrmann, Manaal Faruqui, Bhuwan Dhingra, Diyi Yang, and Dipanjan Das. 2020. ToTTo: A controlled table-to-text generation dataset. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1173–1186, Online. Association for Computational Linguistics.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9.

Mirac Suzgun, Luke Melas-Kyriazi, Suproteem Sarkar, Scott D Kominers, and Stuart Shieber. 2023. The harvard uspto patent dataset: A large-scale, wellstructured, and multi-purpose corpus of patent applications. Advances in neural information processing systems, 36:57908–57946.

Alex Wang, Kyunghyun Cho, and Mike Lewis. 2020. Asking and answering questions to evaluate the factual consistency of summaries. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 5008–5020, Online. Association for Computational Linguistics.

Yidong Wang, Zhuohao Yu, Zhengran Zeng, Linyi Yang, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, Xing Xie, Wei Ye, Shikun Zhang, and Yue Zhang. 2024. Pandalm: An automatic evaluation benchmark for llm instruction tuning optimization. International Conference on Learning Representations (ICLR).

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837.

Weizhe Yuan, Graham Neubig, and Pengfei Liu. 2021. Bartscore: Evaluating generated text as text generation. In Advances in Neural Information Processing Systems, volume 34, pages 27263–27277. Curran Associates, Inc.

Yuheng Zha, Yichi Yang, Ruichen Li, and Zhiting Hu. 2023. AlignScore: Evaluating factual consistency with a unified alignment function. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11328–11348, Toronto, Canada. Association for Computational Linguistics.

Saizheng Zhang, Emily Dinan, Jack Urbanek, Arthur Szlam, Douwe Kiela, and Jason Weston. 2018. Personalizing dialogue agents: I have a dog, do you have pets too? In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2204–2213,

Melbourne, Australia. Association for Computational Linguistics.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. 2019. Bertscore: Evaluating text generation with bert. In International Conference on Learning Representations.

Yu Zhang and Qiang Yang. 2021. A survey on multitask learning. IEEE transactions on knowledge and data engineering, 34(12):5586–5609.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223.

Wei Zhao, Maxime Peyrard, Fei Liu, Yang Gao, Christian M. Meyer, and Steffen Eger. 2019. MoverScore: Text generation evaluating with contextualized embeddings and earth mover distance. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 563–578, Hong Kong, China. Association for Computational Linguistics.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36:46595–46623.

Ming Zhong, Yang Liu, Da Yin, Yuning Mao, Yizhu Jiao, Pengfei Liu, Chenguang Zhu, Heng Ji, and Jiawei Han. 2022. Towards a unified multidimensional evaluator for text generation. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 2023– 2038, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Lianghui Zhu, Xinggang Wang, and Xinlong Wang. 2023. Judgelm: Fine-tuned large language models are scalable judges. arXiv preprint arXiv:2310.17631.

You Zuo, Kim Gerdes, Éric Clergerie, and Benoît Sagot. 2024. PatentEval: Understanding errors in patent generation. In Proceedings of the 2024 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 2687–2710, Mexico City, Mexico. Association for Computational Linguistics.

## A Human Annotations

We invite licensed patent attorneys for human evaluations. These professionals are provided with reference claims and candidate claims for assessment. They are informed about the intended use of the evaluation results. Table 8 outlines the detailed evaluation criteria, aligned with prior research (Jiang et al., 2025c). We compare the scores and construct the comparative evaluation dataset.

## B Dataset Statistics

We report the token length statistics of the Patent-CE dataset using the Longformer tokenizer in this section. The results are summarized as follows: the minimum length is 156 tokens, the maximum length is 1,461 tokens, the average length is 644 tokens, the median length is 631 tokens, and the standard deviation is 245 tokens. All claims fall within the token limit of Longformer (4096 tokens), and thus no truncation or segmentation strategies were used. This ensures that input length limitations do not affect the evaluation results. Since the dataset does not include very long claims, the proposed method may not generalize well to extremely long claims that exceed the model’s input capacity.

## C Experimental Details

All training and testing processes are conducted on NVIDIA A100 GPUs. The total running time is about 20 hours. We randomly select 10% samples from the training set as the validation set. During training, we use a batch size of 4, a learning rate of 5e-6, a weight decay of 0.01, and training epochs of 10. For BLEU, ROUGE, ME-TEOR, and BERTScore, we use the package from the HuggingFace evaluate library.<sup>4</sup> For Mover-Score<sup>5</sup>, BARTScore<sup>6</sup>, AlignScore<sup>7</sup>, SimCSE<sup>8</sup>, and UniEval<sup>9</sup>, we use their code from original repositories. We use the scipy Python library to calculate the correlation scores and scikit-learn for accuracy and F1 scores.

## D G-Eval-4

We use the following prompt for G-Eval consistent with previous research (Jiang et al., 2025b), as shown in Table 6. We use GPT-4 to evaluate feature completeness, conceptual clarity, terminology consistency, and logical linkage. The overall quality is calculated based on the same formula of human evaluation in Table 8.

![](images/617abb9e376591df684029e7c34142c7b89bc01ee04da33871a240785eeb106b.jpg)  
Table 6: G-Eval prompt used for claim evaluation originated from Jiang et al. (2025b)

## E Example Claim Comparison

We present an example of claim comparison in Table 7, where the differences between Claim B and C are marked in blue.

![](images/1309ef1883e63b3dfb0cbdb1140e0e7254758e6cff3246fca633da2c097e66ed.jpg)  
Table 7: An example of claim comparison. Differences between B and C are marked in blue, and C is better.

<table><tr><td>Criteria</td><td>Rating Description</td></tr><tr><td>Feature Completeness</td><td></td></tr><tr><td></td><td>• 0-2: Most essential features are missing or poorly described.</td></tr><tr><td></td><td>• 3-4: Some essential features are present but significant gaps remain.</td></tr><tr><td></td><td>• 5-6: Majority of essential features are covered but with minor omissions.</td></tr><tr><td></td><td>• 7-8: Almost all essential features are well described with very few gaps.</td></tr><tr><td></td><td>• 9-10: All essential features are thoroughly and comprehensively covered.</td></tr><tr><td>Conceptual Clarity</td><td></td></tr><tr><td></td><td>• 0-2: Claims are very unclear and ambiguous.</td></tr><tr><td></td><td>• 3-4: Claims have significant clarity issues, making them difficult to understand.</td></tr><tr><td></td><td>• 5-6: Claims are mostly clear but contain some ambiguous language.</td></tr><tr><td></td><td>• 7-8: Claims are clear with minimal ambiguity.</td></tr><tr><td></td><td>• 9-10: Claims are exceptionally clear and completely unambiguous.</td></tr><tr><td>Terminology</td><td></td></tr><tr><td>Consistency</td><td>• 0-2: Terminology is highly inconsistent.</td></tr><tr><td></td><td>• 3-4: Significant inconsistencies in terminology.</td></tr><tr><td></td><td>• 5-6: Some inconsistencies in terminology but mostly uniform.</td></tr><tr><td></td><td>• 7-8: Terminology is largely consistent with minor inconsistencies</td></tr><tr><td></td><td>• 9-10: Terminology is completely consistent throughout.</td></tr><tr><td>Logical Linkages</td><td></td></tr><tr><td></td><td>• 0-2: Features are poorly linked with many inaccuracies.</td></tr><tr><td></td><td>• 3-4: Significant issues with the linkages of features.</td></tr><tr><td></td><td>• 5-6: Mostly accurate linkages with some incorrect connections.</td></tr><tr><td></td><td>• 7-8: Accurate linkages with minor inaccuracies.</td></tr><tr><td></td><td>• 9-10: Features are accurately and correctly linked throughout.</td></tr><tr><td>Overall Quality</td><td></td></tr><tr><td></td><td>• Calculated by: (completeness * 4 + clarity * 2 + consistency * 2 + correctness * 3) ÷ 11</td></tr><tr><td></td><td>• 0-2: Very poor overall quality, fails to meet most criteria.</td></tr><tr><td></td><td>• 3-4: Low overall quality with significant issues across multiple criteria.</td></tr><tr><td></td><td>• 5-6: Average overall quality, meets criteria at a basic level.</td></tr><tr><td></td><td>• 7-8: High overall quality with minor issues.</td></tr><tr><td></td><td>• 9-10: Excellent overall quality, meets or exceeds all criteria.</td></tr></table>

Table 8: Rating criteria for human annotation deriving from Jiang et al. (2025c)