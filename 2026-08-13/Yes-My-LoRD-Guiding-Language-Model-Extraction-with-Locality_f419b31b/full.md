# “Yes, My LoRD.” Guiding Language Model Extraction with Locality Reinforced Distillation

Zi Liang† Qingqing Ye† Yanyun Wang<sup>[</sup> Sen Zhang† Yaxin Xiao† Ronghua Li† Jianliang Xu‡ Haibo Hu†\*

: The Hong Kong Polytechnic University, Hong Kong, China [: The Hong Kong University of Science and Technology (Guangzhou), China : Hong Kong Baptist University, Hong Kong, China {zi1415926.liang,20034165r,22041986r}@connect.polyu.hk {qqing.ye,senzhang,haibo.hu}@polyu.edu.hk ywang856@connect.hkust-gz.edu.cn, xujl@comp.hkbu.edu.hk

## Abstract

Model extraction attacks (MEAs) on large language models (LLMs) have received increasing attention in recent research. However, existing attack methods typically adapt the extraction strategies originally developed for deep neural networks (DNNs). They neglect the underlying inconsistency between the training tasks of MEA and LLM alignment, leading to suboptimal attack performance. To tackle this issue, we propose Locality Reinforced Distillation (LoRD), a novel model extraction algorithm specifically designed for LLMs. In particular, LoRD employs a newly defined policy-gradient-style training task that utilizes the responses of victim model as the signal to guide the crafting of preference for the local model. Theoretical analyses demonstrate that I) The convergence procedure of LoRD in model extraction is consistent with the alignment procedure of LLMs, and II) LoRD can reduce query complexity while mitigating watermark protection through our exploration-based stealing. Extensive experiments validate the superiority of our method in extracting various state-of-the-art commercial LLMs. Our code is available at: https://github.com/ liangzid/LoRD-MEA.

## 1 Introduction

In recent years, we have witnessed the remarkable success of large language models (LLMs) such as ChatGPT (Achiam et al., 2024), Gemini (Anil et al., 2024), and Claude (Anthropic, 2024), which are now widely employed in various consumer and industrial applications. Despite their success, these models may suffer from model extraction attacks (MEAs) (Krishna et al., 2020; Rafi et al., 2022; Li et al., 2023b), where their knowledge could be at risk of being stolen by an adversary through a local model that learns on the data collected from the victim model. Besides of some “open-source”

![](images/22b6227a89307de27e90ab81ce674bd176404343a312564301ce13a7dabd15cf.jpg)  
Figure 1: Comparison between vanilla MEAs on conventional DNNs (left) and MEAs on LLMs with alignments (right).

LLMs (e.g., Alpaca (Taori et al., 2023)), which are trained on the chat history of GPT-4, cases of commercial model theft among companies have also been reported recently (Heath, 2023).

Under such a real-world threat, instead of focusing on MEAs against conventional DNNs, which have been extensively studied theoretically (Saad and Solla, 1995; Tian, 2020; Zhou et al., 2021) and empirically (Jagielski et al., 2020; Tramèr et al., 2016; Papernot et al., 2017; Zheng et al., 2019; Xiao et al., 2022; Zhang et al., 2025), a few recent works turn to explore model extraction algorithms and theorems for LLMs. For example, Wallace et al. (2020) propose a monolingual-query-based imitation attack framework to steal machine translation knowledge from generative language models such as GPT-2. Li et al. (2023b) investigate threats of stealing the code-related knowledge from LLMs. However, these studies inherit those MEA algorithms from traditional fields, such as computer vision (Tramèr et al., 2016; Papernot et al., 2017), and train the local model via supervised learning like maximum likelihood estimation (MLE) (Bengio et al., 2000; Myung, 2003), while neglecting the inconsistency of training tasks between MEAs and the alignments (Ouyang et al., 2022; Glaese et al., 2022; Bai et al., 2022a,b; Perez et al., 2023) of modern LLMs. As shown in Figure 1, modern LLMs typically employ alignments using reinforced learning, which is missing in the local model training of conventional MEAs. As a result, these attacks usually suffer from poor performance.

In this paper, we challenge the effectiveness of MLE in stealing a reinforcement-learning-aligned LLM, by analyzing its following drawbacks: i) Low query efficiency. Current MEAs on LLMs suffer from unacceptably significant query times because they must collect enough generated responses, which entails exponential complexity in terms of generated tokens, resulting in low query efficiency. ii) Vulnerability against defenses. Directly learning from the responses of victim models can cause local models to inadvertently incorporate those watermarks (Cong et al., 2022; He et al., 2022; Zhao et al., 2022; He et al., 2021) embedded in the output of victim models. The residue of such watermarks makes the extraction less stealthy and even serves as provenance evidence of model theft.

Motivated by these limitations, we propose Locality Reinforced Distillation (LoRD), a queryefficient and watermark-resistant model extraction attack under a training paradigm similar to LLM’s alignments. Stealing LLMs via reinforcement learning (RL) paradigms is challenging. The main reason is that the alignment procedure of LLMs heavily relies on the feedback signal of human annotators (Bai et al., 2022a,b; Perez et al., 2023), which is difficult to reproduce directly in the context of MEAs. To tackle this challenge, we develop a policy-gradient-style extraction procedure. This approach regards the locality direction between the generations of local models and victim models as the implicit reward signal. It can thus achieve a human-feedback-free RL for our attack. From the theoretical perspective, we show why those existing MEAs using MLE and knowledge distillation (KD) are inconsistent with the optimization procedure in LLMs’ alignments. Along this way, we also demonstrate why LoRD can achieve stronger watermark resistance and higher query efficiency.

Extensive experiments on five downstream NLP tasks and two alignment tasks with 12 datasets demonstrate that it is feasible to steal a commercial LLM with 175 billion parameters by a pre-trained local model with only 8 billion parameters under a given domain. The resulting local model performs statistically similar to the victim model for tasks not requiring extra knowledge (e.g., data-to-text), and only $0 \sim 3$ percentage lower for tasks requiring it (e.g., translation and QAs). This result poses an immediate threat of task-specific and alignment extraction on commercial LLMs.

Our contribution are summarized as follows:

New Perspective of Large Language Model Extraction. We present LoRD, a novel model extraction attack algorithm for LLMs. To our best knowledge, it is the first effective and realistic extraction algorithm that takes LLM alignment into consideration for MEAs.

Theoretical Guarantee. We theoretically prove that the convergence procedure of LoRD in MEAs is consistent with the alignments of LLMs. Furthermore, we demonstrate that LoRD can reduce query complexity while mitigating watermark protection through exploration-based stealing.

Systematical Evaluation. Extensive experiments demonstrate that our method outperforms current extraction strategies across different downstream NLP tasks.

## 2 Background

## 2.1 Policy Gradient Models

Policy gradient models (PGM) are commonly used in reinforcement learning (RL) algorithms to optimize the agents based on the decided action of RL agents. Represented by TRPO (Schulman et al., 2015) and PPO (Schulman et al., 2017), policy gradient models minimize the the following objective function:

$$
\begin{array} { r } { \mathcal { L } _ { p g , j } = - \hat { \mathbb { E } } _ { j } [ p _ { j } ^ { r } ( \theta ) A _ { j } ] , } \end{array}\tag{1}
$$

where at each decision step $\begin{array} { r } { j , p _ { j } ^ { r } ( \theta ) = \frac { \pi _ { \theta } ( a _ { j } | s _ { j } ) } { \pi _ { \theta _ { o l d } } ( a _ { j } | s _ { j } ) } } \end{array}$ refers to the probability ratio defined by the optimized policy $\pi _ { \boldsymbol { \theta } } ( a _ { j } | \boldsymbol { s } _ { j } )$ and the initial policy $\pi _ { \theta _ { o l d } } ( a _ { j } | s _ { j } ) , s _ { j }$ denotes the state of the environment, $a _ { j }$ denotes the decided action of $\pi _ { \theta } .$ , and $A _ { j }$ is the de-biased reward of $a _ { j } . \ A _ { j }$ is estimated by the Q-value minus the V-value, i.e.,

$$
A _ { j } ( s _ { j } , a _ { j } ) = Q ( s _ { j } , a _ { j } ) - V ( s _ { j } ) .\tag{2}
$$

Intuitively, $Q \cdot$ -value refers to the reward if employing action $a _ { j }$ at the given environment state $s _ { j } .$ , which can be seen as the label of policy’s decision. V-value represents the estimation of the expected reward at $s _ { j } .$ . Consequently, $A _ { j }$ denotes the surprise when taking action $a _ { j }$

![](images/8f22e9d2cd659c3d360773aadde9c7a6f19e57cf60b7e51d0d07248ebdb79efc.jpg)  
Figure 2: The stealing procedure of LoRD.

## 2.2 Language Modeling

Supervised Training (SFT). Given a pre-trained model with parameters θ, supervised training is essentially the maximum likelihood estimation (MLE) task (Bengio et al., 2000; Myung, 2003), which fine-tunes θ on the labeled dataset $\mathcal { D } _ { t r } ^ { s }$ $\{ ( \mathbf { x } _ { i } , \mathbf { y } _ { i } ) | i = 1 , 2 , . . . , N _ { t r s } \}$ by minimizing the following objective function:

$$
\mathcal { L } _ { m l e } = - \prod _ { i } ^ { N _ { t r s } } P _ { \theta } ( \mathbf { y } _ { i } | \mathbf { x } _ { i } ) = - \prod _ { i } ^ { N _ { t r s } } \prod _ { j } ^ { N } P _ { \theta } ( y _ { i , j } | \mathbf { x } _ { i } , \mathbf { y } _ { i , < j } ) ,\tag{3}
$$

where N denotes the sequence length of ${ \bf y } _ { i } .$ $y _ { i , j }$ denotes the j-th token in ${ \bf y } _ { i } ,$ , and $\mathbf { y } _ { i , < j } =$ $\left\{ y _ { i , 0 } , . . . , y _ { i , j - 1 } \right\}$ . The logarithmic formula of Equation 3 can also be seen as a joint cross-entropy loss function:

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } } _ { c e } = - \sum _ { i } ^ { N _ { t r s } } \log P _ { \theta } ( \mathbf { y } _ { i } | \mathbf { x } _ { i } ) } \ ~ } \\ { { \displaystyle ~ = - \sum _ { i } ^ { N _ { t r s } } \sum _ { j } ^ { N } \log P _ { \theta } ( y _ { i , j } | \mathbf { x } _ { i } , \mathbf { y } _ { i , < j } ) } . } \end{array}\tag{4}
$$

Aligning from Preferences. Employing reinforcement learning in LLMs typically consists of three stages. First, the annotators construct a preference dataset ${ \mathcal D } ^ { p r e f } = \{ ( { \bf x } _ { i } , { \bf y } _ { i } ^ { + } , { \bf y } _ { i } ^ { - } ) \}$ by chatting with LLMs and rating their responses, where ${ \bf y } _ { i } ^ { + }$ and $\mathbf { y } _ { i } ^ { - }$ denote the rated positive and negative responses of the dialogue context $\mathbf { x } _ { i }$ , respectively. Then, a reward model $R _ { \theta _ { \phi } } ( \mathbf { x } , \mathbf { y } ) \to \mathbf { r }$ is trained based on $\mathcal { D } ^ { p r e f }$ to simulate the environment and predict the reward values of tokens in given texts. It is trained with a pair-wise loss,

$$
\mathcal { L } _ { r } = - \sum _ { ( \mathbf { x } , \mathbf { y } ^ { + } , \mathbf { y } ^ { - } ) \sim \mathcal { D } ^ { p r e f } } \sigma ( R _ { \theta _ { \phi } } ( \mathbf { x } , \mathbf { y } ^ { + } ) - R _ { \theta _ { \phi } } ( \mathbf { x } , \mathbf { y } ^ { - } ) ) ,\tag{5}
$$

where $\sigma ( \cdot )$ denotes the sigmoid function. Based on the reward model $R _ { \theta _ { \phi } } ( \mathbf { x } , \mathbf { y } )$ , we can finally train

the language models $P _ { \theta }$ by maximizing its reward:

$$
\operatorname* { m a x } _ { \theta } \sum _ { \mathbf { x } \sim \mathcal { D } _ { q } } R _ { \theta _ { \phi } } ( \mathbf { x } , \hat { \mathbf { y } } ) - \beta \mathbb { D } _ { K L } [ P _ { \theta } ( \hat { \mathbf { y } } | \mathbf { x } ) | | P _ { \theta _ { i n i t } } ( \hat { \mathbf { y } } | \mathbf { x } ) ] ,\tag{6}
$$

where $\mathcal { D } _ { q }$ denotes the dataset of text inputs, $\hat { \textbf { y } } \sim$ $P _ { \theta } ( \mathbf { y } | \mathbf { x } ) )$ denotes the sampled sequence of the training model, and $\theta _ { i n i t }$ is the initialized parameters of the model, e.g., the parameters after SFT. The Kullback-Leibler (KL) divergence term, $\beta \mathbb { D } _ { K L } [ P _ { \theta } ( \mathbf { y } | \mathbf { x } ) | | P _ { \theta _ { i n i t } } ( \mathbf { y } | \mathbf { x } ) ]$ , introduced by TRPO (Schulman et al., 2015), is incorporated to constrain the shift of distribution in generated texts yˆ, where $\beta$ is the hyperparameter.

Consequently, SFT shown in Equation 4 finetunes the pre-trained model with parameters $\theta _ { p r e }$ into an aligned model $\theta _ { s f t }$ through MLE, and RLHF outlined in Equation 6, further aligns $\theta _ { s f t }$ towards the target model $\theta _ { v i c } .$ . As this procedure is not consistent with the conventional training framework of DNNs, it remains unclear whether current MEAs (detailed in Appendix C.2) are effective and efficient in stealing a LLM. Specifically, we will first put forward a new stealing method in Section 3, and compare it with current MEAs in Section 4.

## 3 LoRD: Locality Reinforced Distillation

## 3.1 Overview

In this subsection, we delve into the details of our model extraction framework, LoRD (Locality Reinforced Distillation). As described in Algorithm 1, LoRD follows a reinforcement learning paradigm, that is, it consists of several periods, and in each period, the model will learn to explore new responses and attempt to enhance the model trained in the last period. However, different from LLMs alignments, the agent can neither obtain the reward from the reward model directly, nor label positive and negative responses manually. This motivates us to design a new RL method which can implicitly measure the reward for generated tokens under the guidance of victim model’s responses.

![](images/4667ea91c89c1ec564ad8a8b8587bbe81dc49d1cf6bb0f6fbbcb4e063fce7bc2.jpg)  
Figure 3: Determination of the positive and negative samples in LoRD. We sample $\mathbf { y } _ { t - 1 } ^ { + }$ and $\mathbf { y } _ { t - 1 } ^ { - }$ from $\bar { P _ { \theta _ { t - 1 } } } ( \cdot | \mathbf { x } )$ , and compute their conditional probabilities. The response with a higher probability increment on $\theta _ { t }$ is selected as the positive sample.

Illustrated by Figure 2, LoRD first requires the model to sample two sentences randomly at period $t - 1$ , which are denoted as $\mathbf { y } _ { t - 1 } ^ { + }$ and $\mathbf { y } _ { t - 1 } ^ { - } .$ respectively. In a new period t, it first computes the changes of likelihoods for these two sentences, among the old model $P _ { \theta _ { t - 1 } }$ and the current model $P _ { \theta _ { t } }$ . These changes of likelihoods, denoted as $\Delta _ { t } ^ { + }$ and $\Delta _ { t } ^ { - }$ , indicate whether a selected sentence is locally isotropic $( \Delta > 0 )$ to the optimization direction with victim model’s response ${ \bf y } _ { v i c }$ or not $( \Delta \le 0 )$ , which can be seen as the feedback signal for $P _ { \theta _ { t } }$ in the current optimization step. For convenience, we may swap $\mathbf { y } _ { t - 1 } ^ { + }$ with $\mathbf { y } _ { t - 1 } ^ { - }$ to make sure that $\Delta _ { t } ^ { + } > \Delta _ { t } ^ { - }$ always holds. In this way, for pairs $( { \bf x } , { \bf y } _ { v i c } )$ we can take $\mathbf { y } _ { t - } ^ { + }$ as a locality neighborhood of $\mathbf { y } _ { v i c }$ and $\mathbf { y } _ { t - 1 } ^ { - }$ as the negative sample, all of which can be utilized in the training of $P _ { \theta _ { t } }$ . Figure 3 illustrates this procedure. Additionally, LoRD takes $\mathbf { y } _ { t - 1 } ^ { + }$ as the positive label under the current scope only when $P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { + } | \mathbf { x } )$ or $\Delta ^ { + }$ exceed their respective fixed thresholds $\tau _ { 1 }$ and $\tau _ { 2 }$ . If these conditions are not met, it will use ${ \bf y } _ { v i c }$ as a substitute for $\mathbf { y } _ { t - 1 } ^ { + }$ to enable a cold start.

Based on $\mathbf { y } _ { v i c } , \mathbf { y } _ { t - 1 } ^ { + }$ , and $\mathbf { y } _ { t - 1 } ^ { - }$ , we now design LoRD’s loss function.

## 3.2 Design of Loss Functions

From Section 2.1, we know that the loss function of a policy gradient model can be expressed as an objective function to maximize the rewards of decisions (see Equation 1) and a regularization term to ensure the stability of training. Following this paradigm, the loss function of LoRD could be

$$
\mathcal { L } _ { \mathrm { L o R D } } = \mathcal { L } _ { o b j } + \mathcal { L } _ { r e g } .\tag{7}
$$

Objective function $\mathcal { L } _ { o b j }$ . Inspired by the reward model $R _ { \theta _ { \phi } }$ existed in Equation 6, which is trained to distinguish between positive and negative samples, we propose utilizing the logarithmic proportion of positive to negative samples as the means of achieving a de-biased reward, i.e.,

$$
\begin{array} { r l } & { \mathcal { L } _ { o b j } = - \displaystyle \sum _ { \mathbf { x } \in \mathcal { D } _ { q } } \log [ \frac { P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { + } | \mathbf { x } ) } { P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } | \mathbf { x } ) } ] } \\ & { \quad \quad \quad = - \displaystyle \sum _ { \mathbf { x } \in \mathcal { D } _ { q } } [ \log P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { + } | \mathbf { x } ) - \log P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } | \mathbf { x } ) ] . } \end{array}\tag{8}
$$

Equation 8 exhibits similarities to previous studies on RL-enhanced LLM (Peters and Schaal, 2007; Peng et al., 2019; Go et al., 2023; Korbak et al., 2022; Rafailov et al., 2023). We provide a theoretical explanation for its consistency with the learning procedure of RLHF and the deduction procedure, as detailed in Section 4 and Appendix B.1.

However, training the local model merely by $\mathcal { L } _ { o b j }$ is ineffective due to two reasons: $i )$ when $\mathcal { L } _ { \mathrm { \scriptsize ~ L o R D ~ } } : = \mathcal { L } _ { o b j }$ , no information from the victim model’s responses is incorporated into the selection of $\mathbf { y } _ { t - 1 } ^ { + }$ beyond the cold start phase, resulting in a meaningless self-reward-based learning loop for the stealing procedure; ii) the convergence of the local model’s training cannot be guaranteed.

To address these two issues simultaneously, we design the regularization term as follows.

Regularization loss $\mathcal { L } _ { r e g }$ . Different from LLM’s RLHF (Schulman et al., 2015; Rafailov et al., 2023; Bai et al., 2022a) that typically constrain $\theta _ { t }$ with initial model’s generating distribution $P _ { \theta _ { i n i t } } ( \cdot | \mathbf { x } )$ LoRD aims to directly constrain $\theta _ { t }$ with victim model’s distribution $P _ { \theta _ { v i c } } ( \cdot | \mathbf { x } )$

Unfortunately, $P _ { \theta _ { v i c } } ( \cdot | \mathbf { x } )$ is typically inaccessible within the APIs of commercial LLMs and is not feasible for our black-box scenarios. Consequently, we incorporate the regularization techniques employed in PPO and TRPO but tailor our regularization as a bounded contrastive term between the likelihood of $\theta _ { t }$ under the victim model’s response and the negative sample, i.e.,

$$
\begin{array} { r l } & { \mathcal { L } _ { r e g } = - \displaystyle \sum _ { \mathbf { x } \in \mathcal { D } _ { q } } c l i p ( \log [ \frac { P _ { \theta _ { t } } ( \mathbf { y } _ { v i c } | \mathbf { x } ) } { P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } | \mathbf { x } ) } ] ) } \\ & { \quad \quad \quad = - \displaystyle \sum _ { \mathbf { x } \in \mathcal { D } _ { q } } c l i p ( \log P _ { \theta _ { t } } ( \mathbf { y } _ { v i c } | \mathbf { x } ) - \log P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } | \mathbf { x } ) ) . } \end{array}\tag{9}
$$

In Equation 9, we utilize PPO’s $c l i p ( \cdot )$ function to limit the value of the regularization term, as we expect the regularization term could only be used to avoid the offthe cliff problem (Schulman et al.,

2017, 2015) in RL’s convergence. Besides, our contrastive term can be seen as a streamlined blackbox variant of the KL divergence in TRPO. This simplification offers two advantages: i) it alleviates the necessity of loading the initial model’s weights, leading to a substantial reduction in GPU memory usage; ii) it eliminates the need for $P _ { \theta _ { t } } ( \cdot | \mathbf { x } )$ , which would otherwise necessitate an additional exponential operation of log $P _ { \theta _ { t } } ( \cdot | \mathbf { x } )$ that slow down the forward process and increase extra consumption.<sup>1</sup>

Incorporating Equation 8 with Equation 9, we can reshape the loss function of LoRD as

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { L o R D } } = \mathcal { L } _ { o b j } + \mathcal { L } _ { r e g } } \\ & { \qquad = \displaystyle \sum _ { \mathbf { x } \in \mathcal { D } _ { q } } \log [ \frac { P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } | \mathbf { x } ) } { P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { + } | \mathbf { x } ) } ] + c l i p ( \log [ \frac { P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } | \mathbf { x } ) } { P _ { \theta _ { t } } ( \mathbf { y } _ { \mathrm { v i c } } | \mathbf { x } ) } ) ] . } \end{array}\tag{10}
$$

Finally, we wrap $\mathcal { L } _ { \mathrm { L o R D } }$ with a sigmoid function $\sigma ( \cdot )$ to normalize the loss to the interval (0, 1):

$$
\mathcal { L } = \sum _ { \mathbf { x } \sim \mathcal { D } _ { q } } \sigma ( \log [ \frac { P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } | \mathbf { x } ) } { P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { + } | \mathbf { x } ) } ] + c l i p ( \log [ \frac { P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } | \mathbf { x } ) } { P _ { \theta _ { t } } ( \mathbf { y } _ { v i c } | \mathbf { x } ) } ] ) ) .\tag{11}
$$

## 4 Theoretical Analysis

This section will compare LoRD with current model extraction methods from a theoretical perspective. We will first reveal the underlying inconsistency between the optimization of LLMs, which typically involves RL-based alignments, and the previous MEAs utilizing MLE and KD. Subsequently, we will demonstrate the reasons why LoRD can achieve stronger watermark resistance and higher query efficiency than existing methods.

## 4.1 Consistency Analysis on Learning Tasks

Based on the analysis of the four objective functions for MLE, KD, RLHF and LoRD, we reach the Proposition 1, and illustrate their convergence procedure exhibited in Figure 4. A detailed explanation for it can be found in Appendix B.1.

Proposition 1 (Consistency in Stealing Procedure). The learning procedure for LLMs’ alignments is consistent with the stealing procedure of LoRD, i.e., they both attempt to maximize the difference between the probabilities ofpositive and negative samples. Conversely, they are inconsistent with either MLE or KD. In MLE, the objective is maximizing the label probability, while KD aims to minimize the distance among all dimensions.

![](images/91f890c703434bfda438ba0fa0f80c7f2370944686400071a88dfef2bf2c7268.jpg)  
Figure 4: Illustrations for the converging procedure of probability distributions regarding four methods, namely MLE (a), KD (b), RLHF (c), and LoRD (d). Arrows indicate the expected optimization direction. We mark the distribution dimensions learned with labels in blue, and employ pink and yellow components to indicate the probabilities of positive and negative tokens, respectively.

Albeit the inconsistency in their training procedures, we put forward Proposition 2 to demonstrate that with enough samples, all these methods will reach the same distribution results.

Proposition 2 (Equivalence when Converged). Ideally, for any loss value of Equations 4, 5, 6, 10, or 11 converging to 0, we have $\mathbf { y } ^ { + } \equiv \mathbf { y } _ { v i c } .$ Meanwhile, the local model’s distribution $P _ { \theta } ( \cdot | \mathbf { x } )$ will approach that of the victim model $P _ { \theta _ { v i c } } ( \cdot | \mathbf { x } )$ on MEAs from all three discussed MEA methods, including LoRD, MLE, and KD.

Proposition 2 ensures that the local model will converge to the victim model regardless of the choice of MEA methods. So what is the benefit of LoRD? In Section 4.2, we will show that LoRD outperforms current MEAs with two aspects: the query time reduction, and the watermark resistance of the learned local model.

## 4.2 Comparative Analysis on Model Stealing

Query Efficiency. Let $N _ { Q }$ and $N _ { R }$ denote the sequence lengths of the query text and the response text, respectively. For MLE, the ideal query numbers to populate the entire text space are given by $\mathcal { O } ( V ^ { N _ { Q } } \cdot V ^ { N _ { R } } )$ , where V represents the size of the vocabulary. In contrast, LoRD possesses the capability to automatically explore the generation token space, thereby significantly reducing the query requirements about generation candidates to a constant level. Specifically, the complexity of LoRD’s query requirements is $\mathcal { O } ( V ^ { N _ { Q } } \cdot C )$ , where C is a constant that correlates with the capability

of local models.

Based on the above analysis, a straightforward concern with employing MLE in LLMs’ extraction is that, given the limited query times in real-world practices, it may suffer from incomplete learning, especially for text generation tasks. Consequently, the local model may tend to memorize some specific responses instead of achieving a broad understanding and generation. We call such a phenomenon preference overfitting $( P O )$ , which indicates that the local model is only effective on a limited set of explored samples, and yet does not generalize well to unseen scenarios. In such cases, the local model usually exhibits a more “rugged” decision surface, which appears to overfit the preference sentences in $\mathcal { D } _ { t r }$ , as shown in Figure 11 (b). Figure 10 provides a visualization of it.

Watermark Resistance. Another limitation of prevalent objective functions, such as MLE and KD, is their susceptibility to watermarks (Cong et al., 2022; He et al., 2022, 2021; Kirchenbauer et al., 2023) of output contents, i.e., while stealing knowledge from LLMs via responses ${ \bf y } _ { v i c } ,$ watermarks within them will also been passively inherited by the local model. Consequently, the generated sentences of the local model may possess some residual of watermarks, which might be detected as evidence of stealing.

Despite introducing current watermark removal techniques, we indicate that LoRD can mitigate the influences of watermarks naturally, as it does not learn the likelihood of victim models’ responses $\mathbf { y } _ { v i c } \sim \mathcal { D } _ { t r }$ directly, but relies on ${ \bf y } _ { v i c }$ to determine positive and negative labels from responses generated by the local model.

As depicted in Equation 8, LoRD guides the local model to learn the likelihood of $\mathbf { y } _ { t - 1 } ^ { + }$ instead of ${ \bf y } _ { v i c } ,$ which means that it will not been influenced by watermarks contained in $\mathbf { y } _ { v i c }$ explicitly. However, the regularization term $\mathcal { L } _ { r e g ; }$ , as well as the replacement $\mathbf { y } _ { t - 1 } ^ { + }  \mathbf { y } _ { v i c }$ for a cold start, will indeed introduce watermarks from ${ \bf y } _ { v i c }$ . To address this, we can reshape Equation 11 into a convex combination of the objective function and the regularization, i.e.,

$$
\begin{array} { r l } & { \mathcal { L } = \mathbb { E } [ ( 1 - \lambda _ { 1 } ) \cdot ( \log P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { + } | \mathbf { x } ) - \log P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } | \mathbf { x } ) ) } \\ & { \qquad + \left. \lambda _ { 1 } \cdot c l i p ( \log P _ { \theta _ { t } } ( \mathbf { y } _ { v i c } | \mathbf { x } ) - \log P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } | \mathbf { x } ) ) \right] , } \end{array}
$$

where $0 \leq \lambda _ { 1 } \leq 1$ is the hyperparameter.

When $\lambda _ { 1 }$ is small, the convergence of LoRD will substantially focus on maximizing $P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { + } | \mathbf { x } ) / P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } | \mathbf { x } )$ , with which the local model will exhibit a strong watermark resistance ability. When $\lambda _ { 1 }$ increases, LoRD will tend to rely more on the guidance of ${ \bf y } _ { v i c }$ , resulting in a higher risk of introducing watermarks. In the case of $\lambda _ { 1 } = 1$ , the local model will converge to the victim model without any exploration and watermark resistance, which might suffer from the same level of defense by watermarks.

From a global perspective, $\mathcal { L } _ { o b j }$ represents the exploration and the locality learning ability of LoRD, which can mitigate the influences of watermarks. On the other hand, $\mathcal { L } _ { { r e g } }$ ensures the stability of the training procedure. Therefore, characterizes a trade-off via $\lambda _ { 1 }$ between the stability and the diversity during stealing, and Equation 11 can be seen as a special case of $\mathcal { L }$ with $\lambda _ { 1 } = 0 . 5$

We provide an empirical comparison for query efficiency in Appendix A.2.1, and the comparison on watermark resistance in Appendix A.1.

## 5 Experiments

## 5.1 Settings

Datasets. We evaluate MEAs on six mainstream natural language generation (NLG) tasks, including safety alignment, machine translation, text summarization, question answering, structured text generation, and data-to-text. We select twelve representative datasets, including two for safety alignment and ten for domain-specific evaluation, as detailed in Table 9. We believe these datasets encompass the majority of downstream tasks and effectively capture the varying degrees of difficulty in model stealing across different task domains.

Baselines. As described in Section 2.2 and 4.1, we compare LoRD with two types of model extraction methods: maximum likelihood estimation (MLE) and knowledge distillation (KD). For MLE and LoRD, we conduct MEAs under pure black-box attack settings (see Appendix D for more details of the threat model). For KD, the predicted distributions are used specifically under grey-box settings. Metrics. For text generation tasks, we evaluate extracted models with a semantic-level and two lexical-level metrics, BERTScore (Zhang et al., 2020), BLEU (Papineni et al., 2002), and Rouge-L (Lin, 2004), all of which are commonly used in the NLG evaluation. Regarding reasoning tasks (e.g., QA), we use Precision, Recall, Accuracy, and F1 score as their evaluation metrics.

Implementation Details. We use Llama3-8B as the local model to learn the outputs generated by victim models. We set sequence length varying 128 to 4096 depending on the selected tasks, and learning rate $3 \times 1 0 ^ { - 5 }$ . Our experiments run on 2 80GB Nvidia Tesla A100. We execute each training five times and record the mean values and standard variances. For LoRD, we set $\tau _ { 1 }$ and $\tau _ { 2 }$ to 0.8 and -0.1, respectively. Besides, we set the period number $N _ { t }$ to 512, and use $\lambda _ { 1 } = 0 . 5$

<table><tr><td>Model/Metric</td><td>BLEU-1</td><td>BLEU-4</td><td>Rouge-L</td><td>BERTScore</td></tr><tr><td colspan="5">Czech to English with 16 query samples</td></tr><tr><td>Victim Model</td><td>0.611</td><td>0.313</td><td>0.604</td><td>0.957</td></tr><tr><td>Local Model</td><td>0.255</td><td>0.105</td><td>0.348</td><td>0.868</td></tr><tr><td>+MLE</td><td>0.535 ± 0.01</td><td>0.245 ± 0.01</td><td>0.526 ± 0.01</td><td> $\overline { { 0 . 8 9 9 \pm 0 . 0 0 } }$ </td></tr><tr><td>+LoRD</td><td>0.545 ± 0.01</td><td>0.249 ± 0.00</td><td> $0 . 5 3 8 \pm 0 . 0 1$ </td><td> $0 . 9 0 6 \pm 0 . 0 0$ </td></tr><tr><td colspan="5">German to English with 16 query sample</td></tr><tr><td>Victim Model</td><td>0.661</td><td>0.377</td><td>0.652</td><td>0.965</td></tr><tr><td>Local Model</td><td>0.276</td><td>0.130</td><td>0.359</td><td>0.877</td></tr><tr><td>+MLE</td><td>0.578 ± 0.02</td><td> $\begin{array} { c } { 0 . 3 0 2 \pm 0 . 0 1 } \\ { 0 . 3 0 8 \pm 0 . 0 0 } \end{array}$ </td><td> $0 . 5 7 3 \pm 0 . 0 2$ </td><td> $0 . 9 0 4 \pm 0 . 0 1$ </td></tr><tr><td>+LoRD</td><td>0.587 ± 0.00</td><td></td><td> $0 . 5 8 9 \pm 0 . 0 0$ </td><td> $0 . 9 1 7 \pm 0 . 0 0$ </td></tr><tr><td colspan="5">Finnish to English with 16 query samples</td></tr><tr><td>Victim Model</td><td>0.558</td><td>0.252</td><td>0.557</td><td>0.953</td></tr><tr><td>Local Model</td><td>0.242</td><td>0.085</td><td>0.320</td><td>0.866</td></tr><tr><td>+MLE</td><td>0.444 ± 0.03</td><td>0.173 ± 0.02</td><td> $0 . 4 4 9 \pm 0 . 0 3$ </td><td>0.905 ± 0.00</td></tr><tr><td>+LoRD</td><td>0.498 ± 0.01</td><td>0.196 ± 0.00</td><td> $0 . 4 8 5 \pm 0 . 0 1$ </td><td>0.905 ± 0.00</td></tr></table>

Table 1: MEA comparison on WMT16 (Bojar et al., 2016) among MLE and our LoRD methods, where we use GPT-3.5-turbo as the victim model, and Llama3- 8B (Grattafiori et al., 2024) as the local initial model.

## 5.2 Stealing Domain-Specific Knowledge

We first select GPT-3.5-turbo, a checkpoint of Chat-GPT, as the basic victim model. This is because its API provides probabilities of candidate words when generating responses. We employ Llama3- 8B (Grattafiori et al., 2024), a small LLM with only a 4.5% fraction of parameters than the victim model as our initial local model. Though this LaViSH (Large-Victim-Small-Heist) setting contradicts previous assumptions (Tramèr et al., 2016; Papernot et al., 2017; Jagielski et al., 2020) in MEA that the copy model should usually be “wider” or “larger” than the victim model to contain its knowledge, we believe this setting is more applicable in real world scenarios (Li et al., 2023b). Appendix D provides more detail for this setting. Besides, the number of query times selected in this section is less than 100, a significant degradation compared to previous studies (Li et al., 2023b). This is because, in our experiments, copy models can easily learn the knowledge with a few training samples and then exhibit only slight improvements afterward. More discussions on query times can be found in Appendix A.2.1.

Fidelity and limits on stealing. We first examine the fidelity and limits of a small LLM to steal commercial LLMs. As shown in Table 1, 2 and

<table><tr><td>Model/Metric</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1 Score</td></tr><tr><td colspan="5">PIQA (Bisk et al., 2020) with 64 query samples</td></tr><tr><td>Victim Model</td><td>0.828</td><td>0.828</td><td>0.827</td><td>0.827</td></tr><tr><td>Local Model</td><td>0.622</td><td>0.638</td><td>0.621</td><td>0.609</td></tr><tr><td>+MLE (baseline)</td><td> $\overline { { 0 . 7 6 0 \pm 0 . 0 2 } }$ </td><td> $\overline { { 0 . 7 7 1 \pm 0 . 0 1 } }$ </td><td>0.760 ± 0.02</td><td> $\overline { { 0 . 7 5 7 \pm 0 . 0 3 } }$ </td></tr><tr><td>+KD (gre-box)</td><td> $0 . 7 5 9 \pm 0 . 0 2$ </td><td> $0 . 7 6 0 \pm 0 . 0 2$ </td><td>0.759 ± 0.02</td><td> $0 . 7 5 9 \pm 0 . 0 2$ </td></tr><tr><td>+LoRD (ours)</td><td> $0 . 7 8 5 \pm 0 . 0 1$ </td><td>0.795 ± 0.01</td><td>0.785 ± 0.01</td><td> $0 . 7 8 3 \pm 0 . 0 2$ </td></tr><tr><td colspan="5">TruthfulQA (Lin et al., 2021) with 64 query samples</td></tr><tr><td>Victim Model</td><td>0.414</td><td>0.500</td><td>0.207</td><td>0.293</td></tr><tr><td>Local Model</td><td>0.391</td><td>0.500</td><td>0.195</td><td>0.281</td></tr><tr><td>+MLE (baseline)</td><td> $0 . 3 8 1 \pm 0 . 1 7$ </td><td> $0 . 5 0 0 \pm 0 . 0 0$ </td><td> $0 . 1 9 0 \pm 0 . 0 9$ </td><td> $0 . 2 6 6 \pm 0 . 0 9$ </td></tr><tr><td>+KD (gre-box)</td><td> $0 . 4 6 3 \pm 0 . 0 3$ </td><td> $0 . 5 0 0 \pm 0 . 0 0$ </td><td> $0 . 2 3 2 \pm 0 . 0 1$ </td><td> $0 . 3 1 6 \pm 0 . 0 1$ </td></tr><tr><td>+LoRD (ours)</td><td> $0 . 4 0 8 \pm 0 . 0 5$ </td><td></td><td>0.500 ± 0.00 0.204 ± 0.03</td><td> $0 . 2 8 9 \pm 0 . 0 3$ </td></tr></table>

Table 2: MEA comparison on QA tasks among MLE and our LoRD methods. More experiments are shown in Table 7.

7, we list the performance of the victim model and the local model on five tasks, and provide two MEA methods, local model fine-tuned with MLE (+MLE) and LoRD (+LoRD), respectively.

We can see that the original performance of the local model is significantly lower than the victim model, i.e., with a 50% decrease in BLEU-4 or 10 25 decrease in Rouge-L. Once we employ MEAs in the local model, its performance rapidly boosts to nearly the same as the victim model, with 0  40% points of gaps in BERTScore. These gaps are negligible $( \mathbf { e . g . \textit { \textbf { < } } 1 \% }$ in summarization) in some tasks, but remain eminent in other tasks such as reasoning, structured text generation, and machine translation. This phenomenon indicates that domain-specific model extractions can effectively learn domain-specific abilities from victim models but may perform poorly if downstream tasks require extra knowledge, such as machine translation and QA. We provide a stealing comparison among different local models in Table 9.

Comparison among stealing methods. Tables 1, 2, and 7 compare the stealing efficacy between MLE and our LoRD. The results consistently show that LoRD outperforms MLE under the same MEA settings. Besides, for challenging tasks such as reasoning and translation, LoRD exhibits much higher improvements, which demonstrates that it can address the preference overfitting problem discussed in Section 4.2 and do enable the local model to learn the task ability from victim models. However, we also observe that for some tasks (e.g., summarization), LoRD shows no statistical difference from MLE, probably because these tasks are relatively simple, where merely MLE has already achieved comparable results to victim models.

Tasks difficulties comparison. Based on previous analysis, we observe that the performance and limitations of MEA depend on the category of tasks.

<table><tr><td></td><td colspan="5">DiaSafety</td><td colspan="5">SafeRLHF</td></tr><tr><td>Model</td><td>Toxicity</td><td>Insult</td><td>Profanity</td><td>Severe Toxity</td><td>Threat</td><td>Toxicity</td><td>Insult</td><td>Profanity</td><td>Severe Toxity</td><td>Threat</td></tr><tr><td>Llama3-8B (initial)</td><td>14..20</td><td>7.94</td><td>8.35</td><td>1.58</td><td>2.29</td><td>7.92</td><td>2.71</td><td>2.80</td><td>0.30</td><td>1.49</td></tr><tr><td>+MLE</td><td>8.31</td><td>3.69</td><td>4.31</td><td>0.83</td><td>1.50</td><td>4.87</td><td>1.98</td><td>1.66</td><td>0.16</td><td>1.02</td></tr><tr><td>+LoRD</td><td>6.45</td><td>2.81</td><td>3.56</td><td>0.71</td><td>1.34</td><td>3.55</td><td>1.15</td><td>2.84</td><td>0.38</td><td>0.79</td></tr></table>

Table 3: Comparison on safety alignment extraction tasks.

Additionally, sometimes datasets in the same task exhibit significant differences in stealing. We put forward two metrics to measure task difficulties: thefidelity that measures extraction efficacy compared to victim models, and the performance-up, which assesses the performance gain before and after stealing for a given local model. Formally, given a test set $\mathcal { D } _ { t e } = \{ ( { \bf x } , { \bf y } ) \}$ and a corresponding metric $\mathcal { M } ( h y p o t h e s i s , r e f e r e n c e )$ , the fidelity (F) and performance-up (P) of the local model $\theta _ { N _ { t } }$ can be defined as:

$$
F = \frac { \displaystyle \sum _ { \mathbf { x } , \mathbf { y } \in \mathcal { D } _ { t e } } \mathcal { M } ( \mathbf { y } _ { N _ { t } } , \mathbf { y } ) } { \displaystyle \sum _ { \mathbf { x } , \mathbf { y } \in \mathcal { D } _ { t e } } \mathcal { M } ( \mathbf { y } _ { v i c } , \mathbf { y } ) } , P = \frac { \displaystyle \sum _ { \mathbf { x } , \mathbf { y } \in \mathcal { D } _ { t e } } \mathcal { M } ( \mathbf { y } _ { N _ { t } } , \mathbf { y } ) } { \displaystyle \sum _ { \mathbf { x } , \mathbf { y } \in \mathcal { D } _ { t e } } \mathcal { M } ( \mathbf { y } _ { 0 } , \mathbf { y } ) } ,\tag{12}
$$

where $\mathbf { y } _ { N _ { t } } \sim P _ { \theta _ { N _ { t } } } ( \cdot | \mathbf { x } ) , \mathbf { y } _ { 0 } \sim P _ { \theta _ { 0 } } ( \cdot | \mathbf { x } )$ , and $\mathbf { y } _ { v i c } \sim P _ { \theta _ { v i c } } ( \cdot | \mathbf { x } )$ denote the sampled responses from the trained local model $( \theta _ { N _ { t } } )$ , the initial local model $( \theta _ { 0 } )$ , and the victim model $( \theta _ { v i c } )$ , respectively. In Figure 5, we illustrate a “spectrum” of extracting various downstream tasks based on these two metrics defined in Equation 12. The figure can assist in recognizing and defending commercial LLM’s knowledge. From Figure 5, we observe five tasks forming the following three scenario groups and datasets coming from the same tasks are mostly in the same group:

High fidelity & performance-up (HFHP). These tasks are challenging for a pre-trained model but can be effectively learned with the guidance of victim models. This group includes two tasks: data-to-text and structured text generation.

High fidelity & low performance-up (HFLP). The initial local model already achieves a comparable performance to the victim model. QAs and summarization are in this group.

Low fidelity & high performance-up (LFHP). While MEAs significantly improve the local model’s performance, gaps between the local and victim models remain difficult to bridge with domain-specific extraction alone. Machine translation is a representative task whose reasons are explained in Section 5.2.

![](images/940fd57d418a69de0ba494a93d4141e801112b4dd0de1c4f2ba1995189426ef2.jpg)  
Figure 5: Spectrum of the fidelity and performance-up on extracting different downstream tasks.

## 5.3 Stealing Safety Alignments

Besides of the domain-specific model extraction, we also propose the safety alignment extraction. Specifically, we select two popular safety alignment datasets for the experiments, namely SafeRLHF (Ji et al., 2024) and DiaSafety (Sun et al., 2022), to assess the safety of the generated responses. We employed PerspectiveAPI <sup>2</sup> to automatically evaluate the safety of the responses. We select five key aspects of safety probabilities: Toxicity, Insult, Profanity, Severe Toxicity, and Threat. In these categories, a lower score indicates better safety performance. For the LoRD model, we have retained the same hyper-parameters as those used in our domain-specific experiments to ensure consistency. As shown in Table 3, we can see that both MLE and LoRD significantly reduce the harmful information after the stealing procedure. However, LoRD consistantly outperforms MLE on most of the indicators, suggesting that it can achieve better performance in the alignment task.

## 6 Conclusion

In this paper, we have focused on the extraction problem of commercial large language models. We proposed LoRD, a practical and realistic extraction algorithm which is consistent with the alignment procedure of large language models. Our analysis proved that LoRD can reduce the query time significantly and mitigate the certification of current watermarks naturally, surpassing existing MEA algorithms’ capabilities. Extensive experiments on domain-specific stealing and alignments demonstrated the superiority of our method.

## Acknowledgment

The authors would like to thank the reviewers for their detailed suggestions. This work was supported by the National Natural Science Foundation of China (Grant No: 92270123 and 62372122), the Research Grants Council, Hong Kong SAR, China (Grant No: 15203120, 15209922, 15210023, 15224124, and C2004-21GF), and the Innovation and Technology Fund (Grant No: ITS-140-23FP).

## Limitations and Future Works

MEAs on Multi-modal Models. While this paper delves into MEAs for large language models, it acknowledges the oversight of the multi-modal attribution of current commercial models (Anil et al., 2024; Achiam et al., 2024) that integrate various forms of data such as text, images, voice, and so on. The challenge of extending MEA algorithms to accommodate these models, which requires extra considerations on the unified representation of concepts, remains unexplored. Future work could focus on developing MEA methodologies sensitive to multi-modal data nuances.

Capacities beyond LaViSH Settings. We utilize the LaViSH setting to describe the model capacity of adversaries in our threat model (see Appendix D). However, sometimes, the adversary might possess comparable or superior training resources to the victims. Though this paper posits that our MEA algorithms and theoretical analysis are still compatible with such conditions, we concede that concrete experimental validation and results beyond LaViSH settings are not presented here.

Lower-level Extractions. This study evaluates MEAs at the performance level, i.e., it measures the extraction effectiveness simply through task performance metrics, or the similarity of learned distributions to the victim model. This setting is justified, as performance metrics are essential for evaluating task-related knowledge and the practical application of LLMs. However, it does not consider the lower-level similarities between the victim and local models. Can we achieve neuron-level alignments in LLM’s MEAs? How does a LaViSH setting hurt LLM’s MEAs? Is it compatible to extract a MoE (Mix-of-the-Expert) (Shazeer et al., 2017) victim model with a dense local model? These questions are not addressed in this research.

## Ethical Considerations

As discussed in Section 1, MEAs are becoming increasingly prevalent in industrial settings and have already been executed, yet there remains a critical gap in understanding which specific tasks are more susceptible and what capabilities are necessary for effective executions. This lack of knowledge exacerbates the challenges faced by LLM maintainers in safeguarding their systems. Our research can contribute to that. Besides, the theoretical problem we address (as shown in Section 4) offers a novel and insightful perspective on the nature of this threat. Based on these two points, we believe the benefits of our paper outweigh potential harms, which aligns with the principles of the Menlo Report (Bailey et al., 2012) on ethics.

Additionally, we have submitted an anonymous version of the paper to the maintainers of the victim models used in our study to assist in improving their model security.

It is important to acknowledge, however, that the algorithms we propose could inadvertently enhance the efficiency of illicit extraction efforts by adversaries. To mitigate this risk, we have introduced and analyzed two defensive strategies, assessing both their effectiveness and potential vulnerabilities under adaptive attack scenarios.

Potential defenses consist of:

Query Detection. One approach to effectively prevent the attack of LoRD is by detecting the distribution of query texts. This is because LoRD, similar to current MEA algorithms, makes no improvements to query samples, indicating that it can be detected by analyzing the statistical information of the adversary’s queries, such as the number of queries, distribution of query contents, and so on. However, this defense is usually resourceconsuming, as it requires the LLM provider to store all query texts of each user. Besides, the potential for false positives could adversely affect the user experience.

More Powerful Watermarks. While we highlight the watermark resistance of LoRD, watermarking remains one of the most effective solutions to mitigate MEAs. For example, some model-level watermarks, such as backdoor-based watermarking (Jia et al., 2021; Lv et al., 2024), can effectively certify the theft of DNNs. While model-level (e.g. backdoor-based) watermarks on pre-trained models raised increasing concerns recently (Peng et al., 2023; Gu et al., 2022; Li et al., 2023a), model-level watermarking on LLMs remains preliminary. Besides, this technique might not work when the adversary only steals a subset of knowledge in which no backdoor is embedded.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, and Anna-Luisa Brakman et al. 2024. Gpt-4 technical report. Preprint, arXiv:2303.08774.

Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, and Andrew M. Dai et al. 2024. Gemini: A family of highly capable multimodal models. Preprint, arXiv:2312.11805.

Anthropic. 2024. The claude 3 model family: Opus, sonnet, haiku.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, and Nelson Elhage et al. 2022a. Training a helpful and harmless assistant with reinforcement learning from human feedback. Preprint, arXiv:2204.05862.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, and Carol Chen et al. 2022b. Constitutional AI: harmlessness from AI feedback. CoRR, abs/2212.08073.

Michael Bailey, David Dittrich, Erin Kenneally, and Doug Maughan. 2012. The menlo report. IEEE Security and Privacy, 10(2):71–75.

Yoshua Bengio, Réjean Ducharme, and Pascal Vincent. 2000. A neural probabilistic language model. Advances in neural information processing systems, 13.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2020. Piqa: Reasoning about physical commonsense in natural language. In Thirty-Fourth AAAI Conference on Artificial Intelligence.

Ond rej Bojar, Rajen Chatterjee, Christian Federmann, Yvette Graham, Barry Haddow, Matthias Huck, Antonio Jimeno Yepes, Philipp Koehn, Varvara Logacheva, Christof Monz, Matteo Negri, Aurelie Neveol, Mariana Neves, Martin Popel, Matt Post, Raphael Rubino, Carolina Scarton, Lucia Specia, Marco Turchi, Karin Verspoor, and Marcos Zampieri. 2016. Findings of the 2016 conference on machine translation. In Proceedings of the First Conference on Machine Translation, pages 131–198, Berlin, Germany. Association for Computational Linguistics.

Tianshuo Cong, Xinlei He, and Yang Zhang. 2022. Sslguard: A watermarking scheme for selfsupervised learning pre-trained encoders. In Proceedings ofthe 2022 ACM SIGSAC Conference on Computer and Communications Security, CCS 2022, Los Angeles, CA, USA, November 7- 11, 2022, pages 579–593. ACM.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Ondˇrej Dušek, Jekaterina Novikova, and Verena Rieser. 2020. Evaluating the State-of-the-Art of End-to-End Natural Language Generation: The E2E NLG Challenge. Computer Speech & Language, 59:123–156.

Amelia Glaese, Nat McAleese, Maja Trebacz, John Aslanides, Vlad Firoiu, Timo Ewalds, Maribeth Rauh, Laura Weidinger, Martin Chadwick, Phoebe Thacker, Lucy Campbell-Gillingham, Jonathan Uesato, Po-Sen Huang, Ramona Comanescu, and Fan Yang et al. 2022. Improving alignment of dialogue agents via targeted human judgements. CoRR, abs/2209.14375.

Bogdan Gliwa, Iwona Mochol, Maciej Biesek, and Aleksander Wawer. 2019. SAMSum corpus: A human-annotated dialogue dataset for abstractive summarization. In Proceedings of the 2nd Workshop on New Frontiers in Summarization, pages 70–79, Hong Kong, China. Association for Computational Linguistics.

Dongyoung Go, Tomasz Korbak, Germán Kruszewski, Jos Rozen, Nahyeon Ryu, and Marc Dymetman. 2023. Aligning language models with preferences through f-divergence minimization. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 11546–11583. PMLR.

Aaron Grattafiori, Abhimanyu Dubey, and Abhinav Jauhri et al. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Chenxi Gu, Chengsong Huang, Xiaoqing Zheng, Kai-Wei Chang, and Cho-Jui Hsieh. 2022. Watermarking pre-trained language models with backdooring. Preprint, arXiv:2210.07543.

Xuanli He, Qiongkai Xu, Lingjuan Lyu, Fangzhao Wu, and Chenguang Wang. 2021. Protecting intellectual property of language generation apis with lexical watermark. Preprint, arXiv:2112.02701.

Xuanli He, Qiongkai Xu, Yi Zeng, Lingjuan Lyu, Fangzhao Wu, Jiwei Li, and Ruoxi Jia. 2022. CATER: intellectual property protection on text generation apis via conditional watermarks. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

Alex Heath. 2023. Bytedance is secretly using openai’s tech to build a competitor. [Online]. https://www. theverge.com/2023/12/15/24003151/ bytedance-china-openai-microsoft-compet

Karl Moritz Hermann, Tomás Kociský, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. 2015. Teaching machines to read and comprehend. In NIPS, pages 1693–1701.

Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean. 2015. Distilling the knowledge in a neural network. CoRR, abs/1503.02531.

Matthew Jagielski, Nicholas Carlini, David Berthelot, Alex Kurakin, and Nicolas Papernot. 2020.

High accuracy and high fidelity extraction of neural networks. In 29th USENIX Security Symposium (USENIX Security 20), pages 1345–1362.

Jiaming Ji, Donghai Hong, Borong Zhang, Boyuan Chen, Josef Dai, Boren Zheng, Tianyi Qiu, Boxun Li, and Yaodong Yang. 2024. Pkusaferlhf: Towards multi-level safety alignment for llms with human preference. arXiv preprint arXiv:2406.15513.

Hengrui Jia, Christopher A. Choquette-Choo, Varun Chandrasekaran, and Nicolas Papernot. 2021. Entangled watermarks as a defense against model extraction. In 30th USENIX Security Symposium, USENIX Security 2021, August 11-13, 2021, pages 1937–1954. USENIX Association.

John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. 2023. A watermark for large language models. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 17061–17084. PMLR.

Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, and Roberta Raileanu. 2023. Understanding the effects of rlhf on llm generalisation and diversity. arXiv preprint arXiv:2310.06452.

Tomasz Korbak, Hady Elsahar, Germán Kruszewski, and Marc Dymetman. 2022. On reinforcement learning and distribution matching for fine-tuning language models with no catastrophic forgetting. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information <sup>tor-llm.</sup>Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

Kalpesh Krishna, Gaurav Singh Tomar, Ankur P. Parikh, Nicolas Papernot, and Mohit Iyyer. 2020. Thieves on sesame street! model extraction of bert-based apis. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.

Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu,

Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, and Sushant Prakash. 2023. Rlaif: Scaling reinforcement learning from human feedback with ai feedback. Preprint, arXiv:2309.00267.

Peixuan Li, Pengzhou Cheng, Fangqi Li, Wei Du, Haodong Zhao, and Gongshen Liu. 2023a. Plmmark: A secure and robust black-box watermarking framework for pre-trained language models. Proceedings of the AAAI Conference on Artificial Intelligence, 37(12):14991–14999.

Zongjie Li, Chaozheng Wang, Pingchuan Ma, Chaowei Liu, Shuai Wang, Daoyuan Wu, Cuiyun Gao, and Yang Liu. 2023b. On extracting specialized code abilities from large language models: A feasibility study. Preprint, arXiv:2303.03012.

Zi Liang, Haibo Hu, Qingqing Ye, Yaxin Xiao, and Haoyang Li. 2025a. Why are my prompts leaked? unraveling prompt extraction threats in customized large language models. Preprint, arXiv:2408.02416.

Zi Liang, Pinghui Wang, Ruofei Zhang, Haibo Hu, Shuo Zhang, Qingqing Ye, Nuo Xu, Yaxin Xiao, Chen Zhang, and Lizhen Cui. 2025b. Exploring intrinsic alignments within text corpus. Proceedings of the AAAI Conference on Artificial Intelligence, 39(26):27455–27463.

Bill Yuchen Lin, Wangchunshu Zhou, Ming Shen, Pei Zhou, Chandra Bhagavatula, Yejin Choi, and Xiang Ren. 2020. CommonGen: A constrained text generation challenge for generative commonsense reasoning. In Findings of the Associationfor Computational Linguistics: EMNLP 2020, pages 1823–1840, Online. Association for Computational Linguistics.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2021. Truthfulqa: Measuring how models mimic human falsehoods. Preprint, arXiv:2109.07958.

P. Lv, H. Ma, K. Chen, J. Zhou, S. Zhang, R. Liang, S. Zhu, P. Li, and Y. Zhang. 2024. Mea-defender: A robust watermark against model extraction attack. In 2024 IEEE Symposium on Security

and Privacy (SP), pages 102–102, Los Alamitos, CA, USA. IEEE Computer Society.

Yu Meng, Mengzhou Xia, and Danqi Chen. 2024. Simpo: Simple preference optimization with a reference-free reward. arXiv preprint arXiv:2405.14734.

In Myung. 2003. Tutorial on maximum likelihood estimation. Journal ofMathematical Psychology, 47:90–100.

OpenAI. 2024. Openai api reference documentation: chat. [Online]. https://platform. openai.com/docs/api-reference/chat.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, and Alex Ray et al. 2022. Training language models to follow instructions with human feedback. ArXiv, abs/2203.02155.

Nicolas Papernot, Patrick McDaniel, Ian Goodfellow, Somesh Jha, Z Berkay Celik, and Ananthram Swami. 2017. Practical black-box attacks against machine learning. In Proceedings of the 2017 ACM on Asia conference on computer and communications security, pages 506–519.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Associationfor Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.

Wenjun Peng, Jingwei Yi, Fangzhao Wu, Shangxi Wu, Bin Bin Zhu, Lingjuan Lyu, Binxing Jiao, Tong Xu, Guangzhong Sun, and Xing Xie. 2023. Are you copying my model? protecting the copyright of large language models for EaaS via backdoor watermark. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7653–7668, Toronto, Canada. Association for Computational Linguistics.

Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. 2019. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. CoRR, abs/1910.00177.

Ethan Perez, Sam Ringer, Kamile Lukosiute, Karina Nguyen, Edwin Chen, Scott Heiner, Craig Pettit, Catherine Olsson, Sandipan Kundu, and et al. 2023. Discovering language model behaviors with model-written evaluations. In ACL 2023,, pages 13387–13434. Association for Computational Linguistics.

Jan Peters and Stefan Schaal. 2007. Reinforcement learning by reward-weighted regression for operational space control. In Machine Learning, Proceedings ofthe Twenty-Fourth International Conference (ICML 2007), Corvallis, Oregon, USA, June 20-24, 2007, volume 227 of ACM International Conference Proceeding Series, pages 745–750. ACM.

PyTorch. 2024. Softmax doesn’t work directly with nllloss, which expects the log to be computed between the softmax and itself. use log\_softmax instead (it’s faster and has better numerical properties). [Online]. https://pytorch.org/docs/stable/ generated/torch.nn.functional.softmax. html#torch.nn.functional.softmax.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Mujahid Al Rafi, Yuan Feng, and Hyeran Jeon. 2022. Revealing secrets from pre-trained models. CoRR, abs/2207.09539.

David Saad and Sara Solla. 1995. Dynamics of on-line gradient descent learning for multilayer neural networks. Advances in neural information processing systems, 8.

John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, and Pieter Abbeel. 2015. Trust region policy optimization. CoRR, abs/1502.05477.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. Preprint, arXiv:1707.06347.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. 2017. Outrageously large neural networks: The sparsely-gated mixture-ofexperts layer. arXiv preprint arXiv:1701.06538.

Hao Sun, Guangxuan Xu, Jiawen Deng, Jiale Cheng, Chujie Zheng, Hao Zhou, Nanyun Peng, Xiaoyan Zhu, and Minlie Huang. 2022. On the safety of conversational models: Taxonomy, dataset, and benchmark. In Findings of ACL 2022.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/ stanford\_alpaca.

Yuandong Tian. 2020. Student specialization in deep rectified networks with finite width and input dimension. In International Conference on Machine Learning, pages 9470–9480. PMLR.

Florian Tramèr, Fan Zhang, Ari Juels, Michael K Reiter, and Thomas Ristenpart. 2016. Stealing machine learning models via prediction apis. In 25th USENIX security symposium (USENIX Security 16), pages 601–618.

Eric Wallace, Mitchell Stern, and Dawn Song. 2020. Imitation attacks and defenses for blackbox machine translation systems. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 5531–5546. Association for Computational Linguistics.

Yaxin Xiao, Qingqing Ye, Haibo Hu, Huadi Zheng, Chengfang Fang, and Jie Shi. 2022. Mexmi: Pool-based active model extraction crossover membership inference. In Advances in Neural Information Processing Systems, volume 35, pages 10203–10216. Curran Associates, Inc.

Qiongkai Xu, Xuanli He, Lingjuan Lyu, Lizhen Qu, and Gholamreza Haffari. 2022. Student surpasses teacher: Imitation attack for black-box NLP apis. In Proceedings of the 29th International Conference on Computational Linguistics, COLING 2022, Gyeongju, Republic of Korea, October 12-17, 2022, pages 2849–2860. International Committee on Computational Linguistics.

Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, et al. 2018. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task. arXiv preprint arXiv:1809.08887.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. 2022. Opt: Open pre-trained transformer language models. Preprint, arXiv:2205.01068.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. Bertscore: Evaluating text generation with BERT. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.

Xinwei Zhang, Haibo Hu, Qingqing Ye, Li Bai, and Huadi Zheng. 2025. Mer-inspector: Assessing model extraction risks from an attack-agnostic perspective. In Proceedings of the ACM on Web Conference 2025, WWW ’25, page 4300–4315, New York, NY, USA. Association for Computing Machinery.

Xuandong Zhao, Lei Li, and Yu-Xiang Wang. 2022. Distillation-resistant watermarking for model protection in NLP. In Findings of the Association for Computational Linguistics: EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 5044–5055. Association for Computational Linguistics.

Xuandong Zhao, Yu-Xiang Wang, and Lei Li. 2023. Protecting language generation models via invisible watermarking. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org.

Huadi Zheng, Qingqing Ye, Haibo Hu, Chengfang Fang, and Jie Shi. 2019. Bdpl: A boundary differentially private layer against machine learning model extraction attacks. In Computer Security – ESORICS 2019, pages 66–83, Cham. Springer International Publishing.

Victor Zhong, Caiming Xiong, and Richard Socher. 2017. Seq2sql: Generating structured queries

from natural language using reinforcement learning. CoRR, abs/1709.00103.

Mo Zhou, Rong Ge, and Chi Jin. 2021. A local convergence theory for mildly over-parameterized two-layer neural network. In Conference on Learning Theory, pages 4577–4632. PMLR.

## A Supplemental Experiments

## A.1 Resistance to Watermarks

Current LLM watermarking methods have been shown (Kirchenbauer et al., 2023) to be robust against commonly used erasing strategies (e.g., rephrasing), making watermark removal a distinct challenge. In this section, we validate the inherent resistance of LoRD to watermarks, suggesting that LoRD is preliminarily resistant to text watermarking. As described in Section 4, we highlight that LoRD can extract the victim models’ knowledge with two terms: the straightforward likelihood learning term log $P _ { \theta _ { t } } ( \mathbf { y } _ { v i c } | \mathbf { x } ) - \log P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } | \mathbf { x } )$ and the exploration term log $P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { + } | \mathbf { x } ) - \log P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } | \mathbf { x } )$ where we can tune $\lambda _ { 1 }$ as shown in $\mathcal { L }$ to trade off the exploration and the convergence speed. Typically, a lower $\lambda _ { 1 }$ encourages the model for conducting a slower but more diverse and localized exploration from its own generated text ${ \bf y } _ { t - 1 } ^ { + } ,$ potentially enhancing watermark resistance. In this subsection, we evaluate this analysis empirically.

Watermarking Details. Unlike previous experimental settings in Section 5, here we cannot utilize commercial LLMs as victim models due to the inability to control token sampling inside LLMs. Instead, we employ Llama3-70B as the victim model and watermark its outputs based on “green” tokens selection. Following prior research (Kirchenbauer et al., 2023), we separate the predicted vocabulary into a green word set and a red word set , assigning them randomly with the seed derived from the hash values of generated tokens at the last generation step. Subsequently, we sample the next token exclusively from the green set, determined by a certain probability.

In this way, given the hypothesis $H _ { 0 }$ that texts are generated without the knowledge ofthe green word set, we can estimate the probability $H _ { 0 }$ occurs (P-value) and the Z-score of it for these texts. A high P-value, among with a low Z-score, indicates stronger watermark resistance for MEA algorithms. Result Analysis. As depicted in Figure 6, we evaluate the watermark resistance for both MLE and LoRD, and demonstrate how LoRD’s performance varies with different values of $\lambda _ { 1 }$ . The Z-score of LoRD witnesses a consistent increase as λ1 arises, indicating that the “confidence” in rejecting the hypothesis, i.e., the risk to be suspected, arises when $\lambda _ { 1 }$ increases. This finding coincides with the analysis in Section 4. Besides, we observe that the P-values of LoRD are generally higher than those of MLE when $\lambda _ { 1 }$ is below 0.8, indicating that LoRD typically exhibits stronger watermarking resistance than MLE in most situations. It is noteworthy that this enhanced resistance seems not a “tax” of MEAs efficacy, as the Rouge-L (F1) scores of LoRD consistently surpass those of MLE and do not exhibit a significant negative correlation with their P-values.

![](images/84a391c8f57b446dbef9cb6f60c8880bc0f468d440c54ce5e94e319aba6cb100.jpg)  
Figure 6: Comparison of watermarks resistance.

## A.2 Scaling the Stealing

In this subsection, we explore essential capacities to steal domain-specific knowledge from LLMs. We first analyze the influence of query times for the adversary, then compare the efficacy when utilizing different sizes of the local model, and finally compare the fidelity among different victim and local models.

## A.2.1 Query Times

We first investigate the influence of query numbers on MEAs. Specifically, we sample query examples randomly from the query dataset, starting from 4, and incrementally increase it until the performance of the learned model stabilizes. Figure 7 illustrates the stealing efficacy of LoRD and MLE on PiQA.

We observe that the scores of MLE and LoRD consistently increase as the query number rises, showing that a larger query number can improve stealing efficacy steadily until reaching their empirical upper bounds. Additionally, LoRD typically obtains a higher score than MLE with the same number of queries, and reaches bottlenecks earlier, which can reduce the required query numbers by 87% compared to MLE. Moreover, in Figure 7, the performance of LoRD exhibits a relatively lower standard variance than MLE, indicating a more stable training procedure.

![](images/5b89bd7be59cd894749e6b1b14b8d313f1a55e2eb5b7d510e94afd9b2b6b80a9.jpg)

![](images/6bb8b181e194c9d1252918e9fb869530225d8346cddf8f5c10e49a9391692af2.jpg)

![](images/a1c10e60fdbb5f3268835e70116457ea82a25f63e2db737a1a4e8c54afae455a.jpg)

![](images/3416c788c23d958bd552fac063fd74d6e5e0767e713f1ba4a732a359087b5712.jpg)  
Figure 7: Comparison of query efficiency between MLE and LoRD on PiQA, where the green horizontal line represents the performance of the initialized local model. We increase query times for each method until reaching their bottlenecks. It can be found that the model extracted by LoRD typically performs a higher accuracy than MLE under the same number of queries. At the same time, LoRD reaches bottlenecks significantly earlier, reducing about 87% query cost compared with MLE.

## A.2.2 Scales of Local Models

As shown in our threat model (see Appendix D), we assume the adversary is stealing existing commercial LLMs with a small local model. This raises the question of selecting an appropriate interval of the local model’s size. To address this concern, we illustrate the correlation between the local model’s size and extraction efficacy on two machine translation tasks, Russian-to-English (ru-en) and Germanto-English (de-en), as shown in Figure 8. Here, we employ seven OPT models (Zhang et al., 2022) as local models, with parameters ranging from 125 million to 30 billion, to minimize the interruptions of factors other than model size.

Figure 8 shows a sharp distinction between two machine translation tasks. In the de-en task, the performance of the local model increases steadily with model size, while this trend is not evident in the ru-en task with model size smaller than 30 billion. Nevertheless, the performance of a 30 billion parameter learned local model in ru-en cannot even be comparable to that of a 1.3 billion parameter local model in the de-en task. This phenomenon suggests that for tasks requiring commonsense knowledge, such as machine translation, the local model should at least possess foundational knowledge of the task (e.g., pre-trained on Russian texts) to learn from victim models effectively. Besides, experiments in BERTScore (F1) show that sometimes LoRD may underperform MLE when the local model has fewer than 1 billion parameters, demonstrating that it is challenging to bootstrap LoRD’s exploration with a very small local model. By summarizing the increase in LoRD’s curves, a model with 2.7 billion appears sufficient to steal domain-specific knowledge from commercial LLMs.

## A.2.3 Fidelity under Different Victim and Local Models

We evaluate the fidelity of extracting different victim models using various pre-trained local models. Specifically, we select GPT-3.5, GPT-4, and GPT-4o as victim models, and employ five state-of-theart open-source models, Phi-3 (3.8B), OPT (6.7B), Qwen-2 (7B), Mistral-V3 (7B), and Llama-3 (8B), as local models, as shown in Figure 9.

Horizontally, while GPT-4 exhibits a consistently lower extracted fidelity compared to the other two victim models, vulnerabilities of the three victim models are generally similar. Vertically, fidelity of different local models can be significantly impacted by their performance. For instance, OPT (6.7B) shows a noticeably lower score compared to the other four models, which indicates that the initial performance of the local model will affect the performance of MEAs. Besides, Phi-3 (3.8B) achieves a comparable fidelity to larger models like Llama-3 (8B), demonstrating that the size of a local model does not influence final fidelity in domain-specific stealing after 2.7 billion, which corroborates the observation in Appendix A.2.2.

## A.3 Visualization of Distributions

We also investigate the probability distributions in the generation procedure among different extraction methods. Specifically, we visualize these distributions for four models, the victim model (GPT-3.5-turbo), the initial local model (llama3-8B), and the learned local models with MLE and LoRD. As plotted in Figure 10, each row in the subfigures refers to the distribution when generating the i-th token, with each column element indicating the probability predicted for the corresponding token index. We limit the visualization to no more than five token probabilities as currently only GPT-3.5- turbo provides the token prediction probabilities during generation, with a maximum of 5 candidate tokens (OpenAI, 2024).

![](images/ee728292ce53e90d260f46b21c1a6242206bc903694830e548e5401d91849641.jpg)  
# Model Parameters (Billion) # Model Parameters (Billion) # Model Parameters (Billion) # Model Parameters (Billion)  
Figure 8: Experiments varying different model parameter scales.

![](images/9298c036a31a78ddb8e470277c7a55cd7248d8f74f233d5a618dbe54ee00f3ee.jpg)  
Figure 9: Fidelity of extracted models with different victim models (GPT-3.5-turbo, GPT-4, and GPT-4o) and different local models (Phi-3, OPT, Qwen2, MistralV3, and Llama3).

From Figure 10, we can see that both MLE and LoRD successfully redistribute the generation of the initial local model into a distribution similar to the victim model’s, where probabilities, especially Top-1 tokens, have been well inherited in the extraction. This phenomenon supports our analysis in Proposition 2. However, distributions of MLE extracted models are consistently sharper than LoRD’s, which aligns with our analysis in Section 4.2, where we claim that MLE leads local models to overfit to the preferred sentences (i.e., Top-1 tokens), namely PO, and thus to disrupt the original distributions, leveraging unusual low probabilities for other token indexes. The reason why LoRD can be resistant watermarks, i.e., tokens in Top-1, can also be derived from this discovery.

To compare MLE and LoRD accurately, we quantize the entropy of these distributions, and compute the KL divergence (DKL), and the Spearman Correlation (Spear. Corr.) with respect to the victim and initial local model. As shown in Table 5, while the MLE extracted model exhibits a lower KL divergence (i.e., high distribution similarity) with the victim model than LoRD’s on the training dataset, its KL divergence becomes comparable to LoRD’s on the test set. Meanwhile, its Spearman correlation significantly decreases from 0.78 to 0.27, which shows that MLE cannot effectively imitate victim model’s prediction behaviors when encountering data beyond the training dataset.

![](images/98e575736e45ef9fa8e3d3e9448ee1522abac203f3d42ccda9b29d4a7b3b7a4f.jpg)  
Figure 10: Token generation distributions of four models, namely the victim model, the (initial) local model, and the local model learned through LoRD and MLE, respectively. We visualize their logarithmic probability on examples sampled from the train set and test set, where a deeper color indicates a higher probability.

## A.4 Ablation Study

We conduct an ablation study to assess the impact of our proposed loss functions shown in Section 3. Specifically, we adopt the same experimental settings described in Section 5.1 and compare LoRD against the following variations on the WMT16 (de-en) dataset:

<table><tr><td rowspan="2">Method</td><td colspan="3">BLEU 2</td><td colspan="3">BERTScore</td><td rowspan="2">Rouge-L F1</td></tr><tr><td>1</td><td>3</td><td>4</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>LoRD (Q=16) (T=0.8)</td><td>54.40 42.18</td><td>33.56</td><td>27.06</td><td>89.89</td><td>94.06</td><td>91.44</td><td>56.09</td></tr><tr><td>SimPO (Q=16) (T=1.0)</td><td>44.80 34.80</td><td>27.94</td><td>22.83</td><td>89.79</td><td>93.50</td><td>91.57</td><td>48.39</td></tr><tr><td>SimPO (Q=16) (T=1.3)</td><td>44.19 33.45</td><td>26.31</td><td>21.18</td><td>88.49</td><td>92.65</td><td>90.47</td><td>47.09</td></tr><tr><td>SimPO (Q=16) (T=0.8)</td><td>42.99 31.81</td><td>24.85</td><td>19.82</td><td>90.37</td><td>88.32</td><td>92.64</td><td>44.04</td></tr><tr><td>SimPO (Q=256) (T=1.3)</td><td>3.09 0.13</td><td>0.00</td><td>0.00</td><td>68.04</td><td>81.54</td><td>74.17</td><td>11.22</td></tr><tr><td>SimPO (Q=256) (T=0.8)</td><td>20.99</td><td>10.75 7.01</td><td>5.04</td><td>85.56</td><td>87.52</td><td>86.50</td><td>21.08</td></tr></table>

Table 4: Comparison between LoRD and The Direct Prompting with SimPO. T denotes the temperature of local model’s sampling, and $Q$ denotes the query times.

<table><tr><td rowspan="2">Models\Metrics</td><td rowspan="2">Entropy</td><td colspan="2">To Victim Model</td><td colspan="2">To Initial Local Model</td></tr><tr><td> $\overline { { \mathbb { D } _ { \mathrm { K L } } \downarrow } }$ </td><td>Spear. Corr.↑</td><td> $\overline { { \mathbb { D } _ { \mathrm { K L } } } }$ </td><td>Spear. Corr.</td></tr><tr><td colspan="6">On training dataset</td></tr><tr><td>Initial Local Model</td><td>0.395</td><td>0.503</td><td>0.620</td><td>=</td><td></td></tr><tr><td>+ LoRD</td><td>0.209</td><td>0.051</td><td>0.880</td><td>0.169</td><td>0.680</td></tr><tr><td>+ MLE</td><td>0.271</td><td>0.029</td><td>0.780</td><td>0.051</td><td>0.540</td></tr><tr><td colspan="6">On the test dataset</td></tr><tr><td>Initial Local Model</td><td>0.269</td><td>0.471</td><td>0.680</td><td></td><td></td></tr><tr><td>+ LoRD</td><td>0.122</td><td>0.033</td><td>0.640</td><td>0.046</td><td>0.720</td></tr><tr><td>+MLE</td><td>0.275</td><td>0.032</td><td>0.274</td><td>0.001</td><td>0.740</td></tr></table>

Table 5: Quantization analysis on distributions. A low KL divergence or a high Spearman correlation indicates a high similarity.

w.o. $\sigma ( \cdot )$ : Removing the sigmoid function in Equation 11;

• <sup>Rep.</sup> $\mathbf { y } ^ { - } \ \mathbf { w } . \mathbf { y } ^ { + } ;$ Replacing $\mathbf { y } _ { t - 1 } ^ { - }$ with $\mathbf { y } _ { t - 1 } ^ { + }$ defined in Equation 9;

w.o. $\mathcal { L } _ { r e g } \mathrm { : }$ Eliminating the regularization term.

The ablation results are presented in Table 6. Our findings indicate that the sigmoid function used for normalization is not essential for the effectiveness of our extraction strategy. However, the regularization term proves to be crucial for ensuring the model’s convergence, which is consistent with our theoretical analysis.

## A.5 LoRD versus Direct Prompting

We notice that the victim model can serve as a feedback signal to explicitly determine $\mathbf { y } _ { t - 1 } ^ { + }$ and ${ \bf y } _ { t - 1 } ^ { - } ,$ thereby enabling a reinforcement learning (RL) approach based on direct prompting. This idea aligns with prior work on reinforcement learning with AI feedback (RLAIF), as discussed in Appendix C.1.

In this section, we present an empirical comparison between LoRD and direct prompting and argue that direct prompting is less suitable for MEAs than our LoRD.

Empirical Comparison. We design a prompt to obtain feedback from the victim model as “For a translation task involving the conversion of the given ‘Text‘ into English, the user will provide two translation versions labeled $^ \bullet \mathsf { A } ^ { \bullet }$ and $^ { \iota } \mathsf { B } ^ { \iota }$ Your task is to return the \*letter corresponding to the better translation\* without including any additional output.”. For direct prompting, we allow the victim model to determine the positive and negative responses generated by the local model. These responses are then used to fine-tune the local model using a DPO-inspired loss function. Specifically, we employ SimPO (Meng et al., 2024) as the loss function. To ensure a fair comparison, we maintain the same hyperparameter settings as in previous experiments.

<table><tr><td rowspan="2">Method</td><td rowspan="2">BLEU-4</td><td colspan="3">BERTScore</td><td rowspan="2">Rouge-L F1</td></tr><tr><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>LoRD</td><td>27.06</td><td>89.89</td><td>94.06</td><td>91.44</td><td>56.09</td></tr><tr><td>w.o. σ(·)</td><td>23.77</td><td>89.25</td><td>93.73</td><td>91.38</td><td>50.39</td></tr><tr><td>Rep.  $\mathbf { y } _ { t - 1 } ^ { - } \mathbf { w } . \mathbf { y } _ { t - } ^ { + }$  1</td><td>25.87</td><td>87.41</td><td>93.28</td><td>90.19</td><td>54.12</td></tr><tr><td>W.0.  $\underline { { \mathcal { L } _ { \boldsymbol { r } e g } } }$ </td><td>NC</td><td>NC</td><td>NC</td><td>NC</td><td>NC</td></tr></table>

Table 6: Ablation Study for LoRD. NC denotes that the model does not converged during training.

As shown in Table 4, we conducted experiments with various sampling temperatures for the direct prompting. However, the performance of the direct prompting still underperforms LoRD. This limitation may stem from the local model’s lack of guidance from correct answers. When the local model generates two suboptimal responses, a direct prompting-based method is compelled to select the "winner" of two inadequate response rather than an optimal response, which we believe is the crux of the issue.

RLHF tackles this challenge by incorporating a regularization term with the initial model, LoRD addresses it through our $\mathcal { L } _ { r e g } .$ , and DPO resolves it by employing the training corpus of the reward model. Unfortunately, a direct prompt-based method overlooks this point. To further investigate this problem, we increased the query number to 256, which resulted in the local model failing to converge and exhibiting poor performance.

Besides, we also observed a bias in the victim model’s selection between the first and second sentences. In a series of 256 queries, the model successfully provided an answer (either A or B) 255 times. However, it chose the first sentence only 84 times, which is a mere 32.94%, significantly deviating from the expected 50%. Given that the generated sentences are randomly sampled from the local model without any significant correlation to their order, we deduce that relying on the victim model to directly generate feedback might be, at least, an unreliable approach. It may necessitate additional considerations for the design of the prompt and the capabilities of the victim model to ensure the robustness of these algorithms.

<table><tr><td rowspan="3"></td><td colspan="4">BLEU</td><td colspan="4">BERTScore</td><td colspan="3">Rouge-L</td></tr><tr><td colspan="4">1 2</td><td colspan="4"> $\overline { { \mathrm { P r e . } } }$ </td><td colspan="3"> $\overline { { \mathrm { P r e . } } }$ </td></tr><tr><td></td><td></td><td>Text to SQL: WikiSQL (Zhong et al., 2017) with 64 query samples</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Rec.</td><td>F1.</td></tr><tr><td colspan="10"> $\overline { { { \bf \nabla } 9 3 . 5 } }$ </td><td>62.1</td></tr><tr><td>Victim Model Local Model</td><td>54.1  $2 0 . 2 \pm 0 . 2$ </td><td>41.4  $1 4 . 5 \pm 0 . 2$ </td><td>32.1  $1 0 . 9 \pm 0 . 1$ </td><td>24.4  $8 . 1 \pm 0 . 1$ </td><td>86.9  $8 2 . 5 \pm 0 . 0$ </td><td> $9 2 . 4 \pm 0 . 1$ </td><td> $8 7 . 1 \pm 0 . 0$ </td><td>90.1  $2 2 . 6 \pm 0 . 3$ </td><td>58.9</td><td>66.4 ± 0.4</td><td>59.7  $3 3 . 2 \pm 0 . 3$ </td></tr><tr><td>+MLE</td><td> $5 4 . 0 \pm 1 . 6$ </td><td> $3 7 . 5 \pm 2 . 1$ </td><td> $2 6 . 4 \pm 2 . 0$ </td><td> $1 8 . 8 \pm { 1 . 8 }$ </td><td>_  $8 3 . 1 \pm 0 . 2$ </td><td></td><td> $9 2 . 9 \pm 0 . 2$ </td><td> $8 7 . 7 \pm 0 . 2$ </td><td> $5 6 . 2 \pm 1 . 5$ </td><td> $5 6 . 1 \pm 0 . 9$ </td><td> $5 5 . 8 \pm 1 . 2$ </td></tr><tr><td>+LoRD</td><td> $5 5 . 1 \pm 2 . 3$ </td><td> $3 9 . 0 \pm 3 . 6$ </td><td> $2 8 . 0 \pm 4 . 0$ </td><td> $2 0 . 4 \pm 3 . 9$ </td><td> $8 3 . 4 \pm 0 . 4$ </td><td></td><td> $9 2 . 9 \pm 0 . 3$ </td><td> $8 7 . 9 \pm 0 . 4$ </td><td> $5 7 . 7 \pm 2 . 2$ </td><td> $5 6 . 3 \pm 2 . 0$ </td><td> $5 6 . 7 \pm 2 . 1$ </td></tr><tr><td></td><td></td><td></td><td></td><td>Text to SQL: Spider (Zhong et al., 2017) with 64 query samples</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">3.9</td><td></td><td></td><td></td></tr><tr><td>Victim Model</td><td>9.4</td><td></td><td>2.1</td><td>1.1</td><td>77.7</td><td></td><td>84.1</td><td>80.6</td><td>17.1</td><td>36.3</td><td>21.8</td></tr><tr><td>Local Model +MLE</td><td> $6 . 4 \pm 0 . 2$ </td><td> $2 . 1 \pm 0 . 1$ </td><td> $0 . 9 \pm 0 . 1$ </td><td> $0 . 5 \pm 0 . 0$ </td><td> $8 0 . 0 \pm 0 . 1$ </td><td> $8 2 . 6 \pm 0 . 1$ </td><td>1</td><td> $8 1 . 2 \pm 0 . 1$ </td><td> $1 0 . 0 \pm 0 . 3$ </td><td> $2 1 . 5 \pm 0 . 6$ </td><td> $1 2 . 7 \pm 0 . 4$ </td></tr><tr><td>+LoRD</td><td> $6 . 2 \pm 0 . 9$ </td><td> $1 . 3 \pm 0 . 5$ </td><td> $0 . 6 \pm 0 . 3$ </td><td> $0 . 2 \pm 0 . 2$ </td><td> $7 6 . 4 \pm 0 . 7$ </td><td></td><td> $8 1 . 8 \pm 0 . 4$ </td><td> $7 8 . 9 \pm 0 . 6$ </td><td> $1 2 . 7 \pm 1 . 6$ </td><td> $1 8 . 3 \pm 1 . 6$ </td><td> $1 4 . 3 \pm 1 . 6$ </td></tr><tr><td></td><td> $9 . 1 \pm 0 . 9$ </td><td> $2 . 8 \pm 0 . 5$ </td><td> $1 . 3 \pm 0 . 4$ </td><td> $0 . 6 \pm 0 . 2$ </td><td>77.7±0.4</td><td></td><td> $8 3 . 1 \pm 0 . 5$ </td><td> $8 0 . 2 \pm 0 . 3$ </td><td> $1 6 . 9 \pm 0 . 1$ </td><td> $2 4 . 1 \pm 0 . 2$ </td><td> $1 8 . 8 \pm 0 . 1$ </td></tr><tr><td colspan="10"> $\overline { { { D a t a ~ t o ~ T e x t } } } \cdot { \cal E } 2 { \cal E } N L G \left( D u \check { s } e k ~ e t ~ a l . ~ , ~ 2 0 2 0 \right) w i t h ~ 6 4 ~ q u e r y ~ s a m p l e s$ </td><td></td><td></td></tr><tr><td>Victim Model Local Model</td><td>51.8</td><td>27.0</td><td>26.8</td><td>19.1</td><td></td><td>93.9</td><td>94.6</td><td>94.2</td><td>49.6</td><td>54.6</td><td>51.4</td></tr><tr><td>+MLE</td><td> $3 1 . 1 \pm 0 . 1$ </td><td> $2 0 . 1 \pm 0 . 2$ </td><td> $1 3 . 5 \pm 0 . 2$   $2 7 . 5 \pm 0 . 5$ </td><td> $8 . 9 \pm 0 . 3$ </td><td> $8 6 . 1 \pm 0 . 1$ </td><td></td><td> $9 2 . 4 \pm 0 . 1$ </td><td> $8 9 . 1 \pm 0 . 1$ </td><td> $2 9 . 0 \pm 0 . 3$ </td><td> $4 9 . 4 \pm 0 . 4$ </td><td> $3 5 . 9 \pm 0 . 3$ </td></tr><tr><td>+LoRD</td><td> $5 3 . 0 \pm 0 . 9$ </td><td> $3 8 . 0 \pm 0 . 6$  38.2±0.9</td><td> $2 7 . 8 \pm 0 . 7$ </td><td>_  $1 9 . 9 \pm 0 . 4$ </td><td> $8 9 . 1 \pm 0 . 0$ </td><td></td><td> $9 4 . 5 \pm 0 . 0$ </td><td> $9 1 . 8 \pm 0 . 0$ </td><td> $4 8 . 3 \pm 0 . 5$ </td><td> $5 4 . 2 \pm 1 . 4$ </td><td> $5 0 . 4 \pm 0 . 9$ </td></tr><tr><td></td><td> $5 3 . 1 \pm 1 . 1$ </td><td></td><td></td><td> $2 0 . 2 \pm 0 . 5$ </td><td> $8 9 . 1 \pm 0 . 1$ </td><td></td><td> $9 4 . 5 \pm 0 . 1$ </td><td> $9 1 . 7 \pm 0 . 1$ </td><td> $4 8 . 3 \pm 0 . 7$ </td><td> $5 3 . 5 \pm 1 . 4$ </td><td> $5 0 . 2 \pm 0 . 9$ </td></tr><tr><td colspan="10"> $\overline { { { D a t a ~ t o ~ T e x t : ~ C o m m o n G e n ~ ( L i n ~ e t ~ a l . ~ 2 0 2 0 ) ~ w i t h ~ 6 4 ~ q u e r y ~ s a m p l e s } } }$  6.9</td><td></td><td>40.7</td></tr><tr><td>Victim Model Local Model</td><td>33.3</td><td>18.5</td><td>11.1</td><td></td><td>91.3</td><td></td><td>92.1</td><td>91.7</td><td>33.6</td><td></td><td>36.1</td></tr><tr><td>+MLE</td><td> $1 2 . 2 \pm 0 . 0$   $3 2 . 4 \pm 2 . 0$ </td><td> $6 . 5 \pm 0 . 1$ </td><td> $3 . 8 \pm 0 . 0$ </td><td> $2 . 3 \pm 0 . 0$ </td><td> $8 3 . 0 \pm 0 . 0$ </td><td> $8 9 . 7 \pm 0 . 0$ </td><td></td><td> $8 6 . 2 \pm 0 . 0$ </td><td> $1 4 . 6 \pm 0 . 1$ </td><td>46.2 ± 0.2</td><td> $2 1 . 6 \pm 0 . 0$ </td></tr><tr><td>+LoRD</td><td> $3 2 . 1 \pm 1 . 3$ </td><td> $1 8 . 3 \pm 1 . 3$ </td><td> $1 0 . 9 \pm 1 . 0$ </td><td> $6 . 6 \pm 0 . 7$ </td><td> $8 4 . 2 \pm 0 . 1$ </td><td></td><td> $9 1 . 7 \pm 0 . 0$ </td><td> $8 7 . 8 \pm 0 . 0$ </td><td> $3 1 . 7 \pm 2 . 4$ </td><td> $4 1 . 1 \pm 0 . 4$ </td><td> $3 5 . 1 \pm 1 . 6$ </td></tr><tr><td></td><td></td><td> $1 8 . 0 \pm 0 . 9$ </td><td> $1 0 . 7 \pm 0 . 5$ </td><td> $6 . 4 \pm 0 . 3$ </td><td> $8 4 . 1 \pm 0 . 0$ </td><td></td><td> $9 1 . 6 \pm 0 . 1$ </td><td> $8 7 . 7 \pm 0 . 0$ </td><td> $3 1 . 4 \pm 1 . 1$ </td><td> $4 0 . 3 \pm 0 . 9$ </td><td> $3 4 . 6 \pm 0 . 9$ </td></tr><tr><td colspan="10">Summarization: TLDR (Kirk et al., 2023) with 64 query samples 1.5</td><td></td></tr><tr><td>Victim Model Local Model</td><td>11.9</td><td>5.0</td><td>2.6</td><td></td><td>85.9</td><td></td><td>88.4  $\overline { { 8 7 . 1 } }$ </td><td></td><td>13.4</td><td>30.9</td><td>18.4</td></tr><tr><td>+MLE</td><td> $6 . 9 \pm 0 . 0$ </td><td> $3 . 2 \pm 0 . 1$ </td><td> $1 . 7 \pm 0 . 0$ </td><td> $1 . 0 \pm 0 . 0$ </td><td> $8 1 . 0 \pm 0 . 1$ </td><td></td><td> $8 7 . 6 \pm 0 . 0$ </td><td> $8 4 . 1 \pm 0 . 0$  </td><td> $1 0 . 5 \pm 0 . 1$ </td><td> $4 1 . 1 \pm 0 . 1$ </td><td> $1 6 . 4 \pm 0 . 1$ </td></tr><tr><td>+LoRD</td><td> $1 0 . 6 \pm 0 . 5$ </td><td> $4 . 8 \pm 0 . 2$ </td><td> $2 . 6 \pm 0 . 1$ </td><td> $1 . 6 \pm 1 . 1$ </td><td> $8 3 . 6 \pm 0 . 7$ </td><td></td><td> $8 8 . 4 \pm 0 . 2 $ </td><td> $8 5 . 9 \pm 0 . 5$ </td><td> $1 4 . 3 \pm 0 . 5$ </td><td> $3 2 . 7 \pm 1 . 1$ </td><td> $1 8 . 9 \pm 0 . 4$ </td></tr><tr><td></td><td> $1 0 . 2 \pm 0 . 3$ </td><td> $4 . 5 \pm 0 . 1$ </td><td> $2 . 4 \pm 0 . 1$ </td><td> $1 . 4 \pm 0 . 0$ </td><td> $8 4 . 1 \pm 0 . 1$ </td><td></td><td> $8 8 . 3 \pm 0 . 1$ </td><td> $8 6 . 2 \pm 0 . 1$ </td><td> $1 2 . 8 \pm 0 . 3$ </td><td> $3 3 . 2 \pm 0 . 9$ </td><td> $1 8 . 0 \pm 0 . 2$ </td></tr><tr><td colspan="10">Summarization: CNN Daily Mail (Hermann et al., 2015) with 64 query samples</td><td></td><td></td><td></td></tr><tr><td>Victim Model Local Model</td><td>20.4</td><td>10.8</td><td> $6 . 4$ </td><td></td><td></td><td>86.4</td><td>87.8</td><td>87.1</td><td>22.4</td><td>40.8</td><td>28.2</td></tr><tr><td>+MLE</td><td> $4 . 9 \pm 0 . 0$ </td><td> $3 . 6 \pm 0 . 0$ </td><td> $2 . 7 \pm 0 . 0$ </td><td> $2 . 1 \pm 0 . 0$ </td><td> $8 0 . 5 \pm 0 . 0$ </td><td></td><td> $8 8 . 3 \pm 0 . 0$ </td><td> $8 4 . 2 \pm 0 . 0$ </td><td> $1 0 . 9 \pm 0 . 0$ </td><td> $7 9 . 1 \pm 0 . 1$ </td><td> $1 8 . 8 \pm 0 . 0$ </td></tr><tr><td>+LoRD</td><td> $5 . 1 \pm 0 . 5$ </td><td> $3 . 7 \pm 0 . 0$ </td><td> $2 . 8 \pm 0 . 0$ </td><td> $2 . 2 \pm 0 . 0$ </td><td> $8 0 . 6 \pm 0 . 0$ </td><td></td><td> $8 8 . 3 \pm 0 . 0$ </td><td> $8 4 . 3 \pm 0 . 0$ </td><td> $1 1 . 3 \pm 0 . 1$ </td><td> $7 8 . 6 \pm 0 . 1$ </td><td> $1 9 . 3 \pm 0 . 1$ </td></tr><tr><td></td><td> $5 . 3 \pm 0 . 0$ </td><td> $3 . 9 \pm 0 . 0$ </td><td> $2 . 9 \pm 0 . 0$ </td><td> $2 . 3 \pm 0 . 0$ </td><td> $8 0 . 6 \pm 0 . 0$ </td><td></td><td> $8 8 . 4 \pm 0 . 0$ </td><td> $8 4 . 3 \pm 0 . 0$ </td><td> $1 1 . 3 \pm 0 . 1$ </td><td> $7 8 . 6 \pm 0 . 2$ </td><td> $1 9 . 1 \pm 0 . 1$ </td></tr><tr><td colspan="10">Summarization:  $S a m s u m \left( G l i w a e t a l . , 2 O I 9 \right) w i t h \ 6 4 \ q u e r y \ s a m p l e s$ </td></tr><tr><td>Victim Model Local Model</td><td>20.7</td><td>11.4</td><td>6.9</td><td> $4 . 4$ </td><td>88.1</td><td>91.7</td><td></td><td>89.8</td><td>24.2</td><td>50.5</td><td> $3 1 . 6$ </td></tr><tr><td>+MLE</td><td> $8 . 9 \pm 0 . 2$ </td><td> $5 . 2 \pm 0 . 1$ </td><td> $3 . 3 \pm 0 . 1$ </td><td> $2 . 1 \pm 0 . 1$ </td><td> $8 0 . 9 \pm 0 . 2$ </td><td> $9 0 . 1 \pm 0 . 1$ </td><td></td><td> $8 5 . 2 \pm 0 . 2$ </td><td> $1 7 . 0 \pm 0 . 3$ </td><td> $6 1 . 8 \pm 0 . 5$ </td><td> $2 5 . 5 \pm 0 . 4$ </td></tr><tr><td>+LoRD</td><td> $1 6 . 9 \pm 1 . 1$ </td><td> $9 . 4 \pm 0 . 7$ </td><td> $5 . 8 \pm 0 . 4$ </td><td> $3 . 7 \pm 0 . 3$ </td><td> $8 3 . 9 \pm 0 . 9$ </td><td> $9 0 . 9 \pm 0 . 6$ </td><td></td><td> $8 7 . 3 \pm 0 . 8$ </td><td> $2 5 . 2 \pm 0 . 8$ </td><td> $4 9 . 8 \pm 2 . 5$ </td><td> $3 1 . 0 \pm 1 . 7$ </td></tr><tr><td></td><td> $1 8 . 4 \pm 0 . 7$ </td><td> $1 0 . 1 \pm 0 . 3$ </td><td> $6 . 0 \pm 0 . 2$ </td><td> $3 . 7 \pm 0 . 1$ </td><td> $8 4 . 9 \pm 0 . 1$ </td><td></td><td> $9 1 . 5 \pm 0 . 1$ </td><td> $8 8 . 1 \pm 0 . 1$ </td><td> $2 3 . 2 \pm 0 . 8$ </td><td> $4 9 . 7 \pm 1 . 5$ </td><td> $3 0 . 2 \pm 0 . 6$ </td></tr></table>

Table 7: MEA comparison on three tasks, including structured text generation, data to text, and summarization. We use GPT-3.5-turbo as the victim model, and Llama3-8B (Grattafiori et al., 2024) as the local initial model. The intensity of the red or blue color corresponds to the degree of underperformance or outperformance relative to the victim model.

feedback is contingent upon the local model’s responses, which is query-inefficient. Specifically, for a given query sample, the algorithm would need to repeatedly query the victim model to distinguish between $\mathbf { y } _ { t - 1 } ^ { + }$ and $\mathbf { y } _ { t - 1 } ^ { - }$ across different learning periods. On the contrary, LoRD necessitates only a single query per sample to discriminate different $( \mathbf { y } _ { t - 1 } ^ { + } , \mathbf { y } _ { t - 1 } ^ { - } )$ pairs;

Discussion on the Feasibility. In addition to the empirical comparison, we provide a discussion supporting the proposition that direct prompting is unsuitable for model extraction attacks for the following reasons:

• The threat model will change if employing a direct prompting. As we know, both LoRD and MLE are currently trained under the same conditions, i.e. $( { \bf x } , { \bf y } _ { v i c } )$ paires. The fairness would be questioned when we compare methods under disparate query settings.

## B Theoretical Explanations and Proofs

• A direct feedback query will expose the intention of the adversary;

## B.1 Explanation of Proposition 1

• Unlike the current design of LoRD, direct

As we described in Section 2, both existing methods and LoRD are learned from the victim model’s response ${ \bf y } _ { v i c }$ and the corresponding probability distribution $P _ { \theta _ { v i c } } ( \cdot | \mathbf { x } ) \in \mathbb { R } ^ { V }$ , where V denotes the vocabulary size. Therefore, we first investigate how the local model is learned to emulate the distribution of the victim model, $P _ { \theta _ { v i c } } ( \cdot | \mathbf { x } )$ , under the following three stealing strategies.

![](images/85ea3df0d24993d5bdf970199a04494f526f0158a8fdef6f0a2c8430b9440e1c.jpg)  
Figure 11: Comparison of learned joint prediction distributions among the victim model (a), local models are learned with MLE (b) and LoRD (c). Simply obtaining the tokens from the victim model (solid black squares), MLE may only memorize specific responses and build a complicated decision surface, resulting in preference overfitting. In contrast, LoRD further explores the candidate generation paths (dashed arrows and squares) under the guidance of the victim’s generation, which is expected to better approximate the victim model in terms of generalization ability, especially under a limited query budget.

Expected Distribution of MLE. We can first reshape the MLE loss into a special formation of Kullback-Leibler divergence with labels of one-hot distributions, that is,

$$
\begin{array} { l } { \mathcal { L } _ { c e } = - \displaystyle \sum _ { \mathbf { x } , \mathbf { y } \sim \mathcal { D } _ { t r } } \log P _ { \theta } ( \mathbf { y } _ { v i c } \vert \mathbf { x } ) } \\ { = \displaystyle \sum _ { \mathbf { x } , \mathbf { y } \sim \mathcal { D } _ { t r } } \sum _ { j } ^ { N } \mathbb { D } _ { K L } [ \mathbf { 1 } _ { y _ { v i c } , j } \vert \vert P _ { \theta } ( \cdot \vert \mathbf { x } , \mathbf { y } _ { v i c } , \vert _ { { j } } ) ] , } \end{array}\tag{13}
$$

where $\mathbf { 1 } _ { y _ { v i c , j } }$ is a one-hot vector in which only $\mathbf { 1 } _ { y _ { v i c , j } } [ y _ { v i c , j } ] = 1$ and all the other elements are 0. Equation 13 demonstrates that MLE learns to maximize the probability of ${ \bf y } _ { v i c , j }$ , without explicit constraints on probabilities across other dimensions.

Expected Distribution of KD. Following a previous work (Hinton et al., 2015), the objective function of KD is

$$
\begin{array} { r l } & { \mathcal { L } _ { k d } = \mathbb { D } _ { K L } \big [ P _ { \theta _ { v i c } } ( \cdot | \mathbf { x } ) | | P _ { \theta } ( \cdot | \mathbf { x } ) \big ] } \\ & { \qquad + \boldsymbol { T } ^ { 2 } \cdot \mathbb { D } _ { K L } \big [ \mathbf { S } \mathbf { M } ( P _ { \theta _ { v i c } } ( \cdot | \mathbf { x } ) / T ) | | \mathbf { S } \mathbf { M } ( P _ { \theta } ( \cdot | \mathbf { x } ) / T ) \big ] , } \end{array}\tag{14}
$$

where $\mathbf { S M } ( \cdot )$ represents the softmaxfunction, and $T \ > \ 1$ denotes the temperature to smooth the targeted distribution $P _ { \theta _ { v i c } } ( \cdot | \mathbf { x } )$ . As described in Equation 14, knowledge distillation aims to align $P _ { \theta } ( \cdot | \mathbf { x } )$ with $P _ { \theta _ { v i c } } ( \cdot | \mathbf { x } )$ in both the original and the smoothed probability across all dimensions, which is exceptionally comprehensive among these MEA loss functions.

Expected Distribution of Alignments. Replacing Equation 6 with Equation 5, we can merge the optimization target of LLMs’ alignments as

$$
\begin{array} { r l } & { \underset { \theta \ast } { \operatorname* { m i n } } - \underset { ( \mathbf { x } , \mathbf { y } ^ { + } , \mathbf { y } ^ { - } ) \sim \mathcal { D } ^ { p r e f } } { \sum } \sigma \left( \log \frac { P _ { \theta \ast } ( \mathbf { y } ^ { + } \vert \mathbf { x } ) / P _ { \theta \ast } ( \mathbf { y } ^ { - } \vert \mathbf { x } ) } { P _ { \theta _ { i n i t } } ( \mathbf { y } ^ { + } \vert \mathbf { x } ) / P _ { \theta _ { i n i t } } ( \mathbf { y } ^ { - } \vert \mathbf { x } ) } \right) } \\ & { \Rightarrow \underset { \theta \ast } { \operatorname* { m a x } } \underset { ( \mathbf { x } , \mathbf { y } ^ { + } , \mathbf { y } ^ { - } ) \sim \mathcal { D } ^ { p r e f } } { \sum } \log P _ { \theta \ast } ( \mathbf { y } ^ { + } \vert \mathbf { x } ) - \log P _ { \theta \ast } ( \mathbf { y } ^ { - } \vert \mathbf { x } ) , } \end{array}\tag{15}
$$

where $\theta *$ denotes the expected parameters of the models as

$$
P _ { \theta * } ( \mathbf { y } | \mathbf { x } ) = \frac { 1 } { Z ( x ) } P _ { \theta _ { i n i t } } ( \mathbf { y } | \mathbf { x } ) \cdot e ^ { \frac { 1 } { \beta } R _ { \phi } ( \mathbf { x } , \mathbf { y } ) } .\tag{16}
$$

We provide a detailed derivation for Equation 16 in Appendix B.2. By replacing Equation 15 with Equation 16, the expected distribution can be represented as $\mathbf { r } _ { i , j } { \cdot } P _ { \theta _ { i n i t } } ( { \cdot } | \mathbf { x } )$ , in which $\mathbf { r } _ { i , j }$ indicates the wrapped distribution gain. This distortion aims to maximize the ratio $P _ { \theta } ( y _ { j } ^ { + } | \mathbf { x } , \mathbf { y } _ { < j } ^ { + } ) / P _ { \theta } ( y _ { j } ^ { - } | \mathbf { x } , \mathbf { y } _ { < j } ^ { - } )$ and leave the probabilities in other dimensions unconstrained directly.

Expected Distribution of LoRD. Similar to alignments, the expected converging procedure by the objective function $\mathcal { L } _ { o b j }$ is also intended to maximize the ratio between positive samples and negative samples, i.e., $P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { + } | \mathbf { x } ) / P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } | \mathbf { x } )$ Meanwhile, the regularization term $P _ { \theta _ { t } } ( \mathbf { y } _ { v i c } \vert \mathbf { x } ) / P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } \vert \mathbf { x } )$ will guide the models to maximize the ratio between ${ \bf y } _ { v i c }$ and $\mathbf { y } _ { t - 1 } ^ { - }$ . As the “standard response” to be learned, ${ \bf y } _ { v i c }$ can be viewed sufficiently as a positive example. Therefore, we can derive that the optimization target of LoRD is consistent with RLHF’s optimization, i.e., both encourage local models to maximize the probability proportion between positive and negative samples.

Similar to Equation 16 in which the optimized model can be seen as the distortion of the original model $P _ { \theta _ { i n i t } }$ , in LoRD the optimized model can be regarded as the distortion of the local model $P _ { \theta _ { 0 } } ,$ , with $P _ { \theta _ { t } } ( \cdot | \mathbf { x } ) = \mathbf { r } _ { i , j } ^ { t } P _ { \theta _ { t - 1 } } ( \cdot | \mathbf { x } )$ at each step $t ,$ where the distortion term $\mathbf { r } _ { i , j } ^ { t }$ is intended to jointly maximize $P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { + } | \mathbf { x } ) / P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } | \mathbf { x } )$ and $P _ { \theta _ { t } } ( \mathbf { y } _ { v i c } \vert \mathbf { x } ) / P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } \vert \mathbf { x } )$ , while leaving the probabilities in other dimensions unconstrained directly.

## B.2 The Deduction of Equation 16 in Proposition 1

From Equation 6, we can get that

$$
\begin{array} { r l } & { \underset { \theta \leq \pi } { \operatorname* { m a x } } \sum _ { \theta _ { \theta } \leq \pi , \theta \leq \pi } \int _ { \theta \leq \pi } \langle \boldsymbol { \mathsf { S } } \boldsymbol { \mathsf { I } } _ { \mathcal { K } } \boldsymbol { \mathsf { Z } } \rangle - \delta \log \big [ p _ { \theta } \langle \boldsymbol { \mathsf { S } } \boldsymbol { \mathsf { I } } \rangle \big ] / \hat { \mathcal { I } } _ { \hat { \mathcal { K } } _ { \epsilon \omega } } \langle \boldsymbol { \mathsf { I } } \boldsymbol { \mathsf { S } } \boldsymbol { \mathsf { I } } \rangle \big ] } \\ & { = \underset { \theta \leq \pi } { \operatorname* { m a x } } \underset { \theta \leq \pi } { \sum } \underset { \theta \leq \pi } { \sum } \underset { \theta \leq \pi } { \sum } R _ { \theta \leq \pi } { \operatorname* { m a x } } \int _ { \theta \leq \pi } \boldsymbol { \mathsf { S } } \boldsymbol { \mathsf { I } } } \\ & { \quad - \beta \log \big [ P _ { \theta } P _ { \theta } ( \boldsymbol { \mathsf { I } } ) \big ] \times 1 } \\ & { = \underset { \theta \leq \pi } { \operatorname* { m i n } } \underset { \theta \leq \pi } { \sum } \underset { \theta \leq \pi } { \sum } - \frac { 1 } { \alpha } R _ { \theta \leq \pi } \mathcal { S } _ { \theta \omega \times \pi } \boldsymbol { \mathsf { S } } \big \} + \log \frac { \beta \mathsf { I } _ { \theta } ( \boldsymbol { \mathsf { I } } ) \times 1 } { P _ { \theta \leq \pi } ( \boldsymbol { \mathsf { I } } ) \big \vert \sqrt { \boldsymbol { S } } } } \\ & { \leq \operatorname* { m i n } \underset { \theta \leq \pi } { \operatorname* { m a x } } \underset { \theta \leq \pi } { \sum } \underset { \theta \leq \pi } { \sum } - \frac { 1 } { \alpha } R _ { \theta \leq \pi } ( \exp \big \vert \frac { 1 } { \beta } R _ { \theta \leq \pi } ( \boldsymbol { \mathsf { I } } ) \big \vert ) } \\ &  \quad \times \operatorname* { m i n } \underset { \theta \leq \pi } { \operatorname* { m a x } } \underset  \theta  \end{array}
$$

If we define a partition function $Z ( \mathbf { x } )$ with the formation of

$$
Z ( \mathbf { x } ) = \sum _ { \mathbf { y } } P _ { i n i t } ( \mathbf { y } | \mathbf { x } ) \mathrm { e x p } ( \frac { 1 } { \beta } R _ { \theta _ { \phi } } ( \mathbf { x } , \mathbf { y } ) ) ,\tag{17}
$$

we can reformat the optimization target as

$$
\begin{array} { l } { { \displaystyle \operatorname* { m i n } _ { \theta } \sum _ { \mathbf { x } \sim \mathcal { D } _ { q } } \sum _ { \mathbf { y } \sim P _ { \theta } ( \cdot | \mathbf { x } ) } \log \frac { P _ { \theta } ( \mathbf { y } | \mathbf { x } ) } { \exp \left( \frac { 1 } { \beta } R _ { \theta _ { \phi } } ( \mathbf { x } , \mathbf { y } ) \right) \cdot P _ { \theta _ { i n i t } } ( \mathbf { y } | \mathbf { x } ) } } \ ~ } \\ { \displaystyle \Rightarrow \operatorname* { m i n } _ { \theta } \sum _ { \mathbf { x } \sim \mathcal { D } _ { q } } \sum _ { \mathbf { y } \sim P _ { \theta } ( \cdot | \mathbf { x } ) } \log \frac { Z ( \mathbf { x } ) \cdot P _ { \theta } ( \mathbf { y } | \mathbf { x } ) } { \exp \left( \frac { 1 } { \beta } R _ { \theta _ { \phi } } ( \mathbf { x } , \mathbf { y } ) \right) \cdot P _ { \theta _ { i n i t } } ( \mathbf { y } | \mathbf { x } ) } \ ~ } \\ { \displaystyle ~ \qquad \quad - \log Z ( \mathbf { x } ) . } \end{array}
$$

If we mark $\begin{array} { r } { \frac { 1 } { Z ( \mathbf { x } ) } \mathrm { e x p } ( \frac { 1 } { \beta } R _ { \theta _ { \phi } } ( \mathbf { x } , \mathbf { y } ) ) \cdot P _ { \theta _ { i n i t } } ( \mathbf { y } | \mathbf { x } ) } \end{array}$ as $P _ { \theta * } ( \mathbf { y } | \mathbf { x } )$ , then we have

$$
\begin{array} { r l } & { \displaystyle \underset { \boldsymbol { \theta } } { \mathrm { m i n } } \sum _ { \mathbf { x } \sim \mathcal { D } _ { q } } \sum _ { \mathbf { y } \sim P _ { \boldsymbol { \theta } } ( \cdot | \mathbf { x } ) } \log \frac { Z ( \mathbf { x } ) \cdot P _ { \boldsymbol { \theta } } ( \mathbf { y } | \mathbf { x } ) } { \exp ( \frac { 1 } { \beta } R _ { \boldsymbol { \theta } _ { \phi } } ( \mathbf { x } , \mathbf { y } ) ) \cdot P _ { \boldsymbol { \theta } _ { i n i t } } ( \mathbf { y } | \mathbf { x } ) } } \\ & { \qquad - \log Z ( \mathbf { x } ) } \\ & { \displaystyle \Rightarrow \operatorname* { m i n } _ { \boldsymbol { \theta } \sim \mathcal { D } _ { q } \mathbf { y } \sim P _ { \boldsymbol { \theta } } ( \cdot | \mathbf { x } ) } \log \frac { P _ { \boldsymbol { \theta } } ( \mathbf { y } | \mathbf { x } ) } { P _ { \boldsymbol { \theta } \ast } ( \mathbf { y } | \mathbf { x } ) } - \log Z ( \mathbf { x } ) . } \end{array}
$$

Because $Z ( \mathbf { x } )$ is independent to y, we can deduct that

$$
\begin{array} { r l } & { \displaystyle \underset { \theta } { \mathop { \operatorname* { m i n } } } \displaystyle \sum _ { \mathbf { x } \sim \mathcal { D } _ { q } } \sum _ { \mathbf { y } \sim P _ { \theta } ( \cdot \vert \mathbf { x } ) } \log \frac { P _ { \theta } ( \mathbf { y } \vert \mathbf { x } ) } { P _ { \theta \ast } ( \mathbf { y } \vert \mathbf { x } ) } - \log Z ( \mathbf { x } ) } \\ & { \displaystyle \Rightarrow \underset { \theta } { \mathop { \operatorname* { m i n } } } \displaystyle \sum _ { \mathbf { x } \sim \mathcal { D } _ { q } } \left[ \sum _ { \mathbf { y } \sim P _ { \theta } ( \cdot \vert \mathbf { x } ) } \log \frac { P _ { \theta } ( \mathbf { y } \vert \mathbf { x } ) } { P _ { \theta \ast } ( \mathbf { y } \vert \mathbf { x } ) } \right] - \log Z ( \mathbf { x } ) } \\ & { \displaystyle \Rightarrow \underset { \theta \sim \mathcal { D } _ { q } } { \mathop { \operatorname* { m i n } } } \frac { \mathrm { B } } { \mathbf { x } \sim \mathcal { D } _ { q } } \mathbb { D } _ { K L } [ P _ { \theta } ( \mathbf { y } \vert \mathbf { x } ) \vert | P _ { \theta \ast } ( \mathbf { y } \vert \mathbf { x } ) ] - \log Z ( \mathbf { x } ) . } \end{array}\tag{18}
$$

As we know that $Z ( \mathbf { x } )$ does not contain $\theta ,$ the above optimization target actually minimizes the KL-divergence between the distribution of $P _ { \theta }$ and $P _ { \theta * }$ , demonstrating that θ is the optimal value of θ that satisfies

$$
P _ { \theta * } ( \mathbf { y } \vert \mathbf { x } ) = \frac { 1 } { Z ( \mathbf { x } ) } \mathrm { e x p } ( \frac { 1 } { \beta } R _ { \theta _ { \phi } } ( \mathbf { x } , \mathbf { y } ) ) \cdot P _ { \theta _ { i n i t } } ( \mathbf { y } \vert \mathbf { x } ) .\tag{19}
$$

Based on equation 19, we can see that the optimal distribution of θ is built upon $P _ { \theta _ { i n i t } }$ with a distortion, as we discussed in Section 4.1.

## B.3 The Proof of Proposition 2

Guarantee of MLE. From Equation 13 we can obtain that when $\mathcal { L } _ { c e }$ decreases to 0, the KL divergence between $P _ { \theta } ( \cdot | \mathbf { x } )$ and $P _ { \theta _ { v i c } } ( \cdot | \mathbf { x } )$ decreases to 0, indicating that $P _ { \theta } ( \cdot | \mathbf { x } )$ equals to $P _ { \theta _ { v i c } } ( \cdot | \mathbf { x } )$

Guarantee of KD. As we know, D ${ } ^ { \prime } { \cal K } { \cal L } ( p , q ) \geq$ $0 \forall p$ and $q .$ Therefore, if $\mathcal { L } _ { k d }$ shown in Equation 14 equals to 0, then both $\mathbb { D } _ { K L } [ P _ { \theta } ( \cdot | \mathbf { x } ) | | P _ { \theta _ { v i c } } ( \cdot | \mathbf { x } ) ]$ and $\mathbb { D } _ { K L } [ \mathbf { S M } ( P _ { \theta } ( \cdot | \mathbf { x } ) / T ) | | \mathbf { S M } ( P _ { \theta _ { v i c } } ( \cdot | \mathbf { \bar { x } } ) / T ) ]$ equal to 0. For the latter one, we can derive that only when $P _ { \theta } ( \cdot | \mathbf { x } )$ equals to $P _ { \theta _ { v i c } } ( \cdot | \mathbf { x } )$ can this term reduce to 0 based on the property of KL divergence. Integrating the analysis of these two terms, we can obtain that $\mathcal { L } _ { k d } = 0$ represents the local model’s distribution converge to that of the victim model.

Guarantee of LoRD. When $\mathcal { L }$ shown in Equation 11 equals to 0, the proportion of $P _ { \theta _ { t } } ( \mathbf { y } _ { v i c } \vert \mathbf { x } ) / P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } \vert \mathbf { x } )$ and $P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { + } | \mathbf { x } ) / P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } | \mathbf { x } )$ should limit to $- \infty .$ . As we know that i) in a distribution $\begin{array} { r l r } { \sum P _ { \theta _ { t } } ( \cdot | \mathbf { x } ) } & { { } = } & { 1 } \end{array}$ and ii) $\mathbf { y } _ { t - 1 } ^ { + }$ is a dynamic positive response generated at each period, we can deduct that when $ { \mathcal { L } } \quad = \quad 0$ there must be $\mathbf { y } _ { v i c } = \mathbf { y } _ { t - 1 } ^ { + } , \mathrm { i . e . , } P _ { \theta _ { t } } ( \mathbf { y } _ { v i c } | \mathbf { x } ) = P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { + } | \mathbf { x } ) = 1$ and $P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } | \mathbf { x } ) = 0$

Note that this is merely a theoretical limit for LoRD that cannot be reached, because $\mathbf { y } _ { t - 1 } ^ { - }$ will not be sampled if its probability is 0, and $\mathbf { y } _ { t - 1 } ^ { + }$ usually doesn’t exhibit a significant distinction to $\mathbf { y } _ { t - 1 } ^ { - }$ when sampling.

<table><tr><td>Task</td><td>Instruction</td></tr><tr><td>WMT16 PiQA &amp; TruthfulQA</td><td>Please translate the sentence from [source language] to English. Please select the correct answer for the “Question&quot; of Users. Question:</td></tr><tr><td></td><td>[question] Selection 1: [Selection1] Selection 2:[Selection2].</td></tr><tr><td>E2E NLG CommonGen</td><td>Please translate the information to a sentence in natural language. Please generate a sentence based on the words provided by Users.</td></tr><tr><td>WikiSQL&amp; Spider</td><td>Please return to me the SQL sentence based on the text (i.e., Question)</td></tr><tr><td></td><td>and the table information (i.e., Table) provided by the User.</td></tr><tr><td>TLDR&amp; SamSUM</td><td>Please **summarize** the content given by the user.</td></tr><tr><td>CNN Daily Mail</td><td>Please **summarize** the content given by the user.</td></tr></table>

Table 8: Instructions used in the different downstream datasets.

## C Supplemental Related Works

## C.1 Human-Feedback-Free Alignments

There are several alternatives to the standard RLHF approach (Lee et al., 2023; Liang et al., 2025b). Lee et al. (2023) propose reinforcement learning with AI feedback (RLAIF) as a means to diminish the annotation burden associated with the preference assessments. Besides, there are some approaches, such as direct preference optimization (DPO) (Rafailov et al., 2023), that conceptualize the language model itself as the reward model and thus consolidate Equation 5 and Equation 6 into a unified supervised and preference-based training task. Since they do not change the primary targets (i.e., maximizing rewards) and optimization strategies of LLM’s alignments, we only consider the standard formation of alignments for simplicity in our theoretical analysis.

## C.2 Language Models Extraction

Studies to steal language models originated from the natural language understanding (NLU) models, such as BERT(Devlin et al., 2019), and then evolved to generative language models, especially large language models recently.

Krishna et al. (2020) highlights early recognition of model extraction threats in language models. By constructing text inputs with randomly vocabulary sampling, they successfully extract the weights from BERT-based APIs. Besides, Rafi et al. (2022) investigate the feasibility of side-channel model extraction attacks, revealing that by analyzing extra signals from GPU kernels, one could accurately steal the model architecture and its parameters. Subsequent research (Xu et al., 2022) has thoroughly investigated the strategy of ensembling victim models to train a competitor model that surpasses its teachers.

The exploration of generative language model extraction is still in its infant stage, with only a handful of studies thus far. Wallace et al. (2020) investigate imitation attacks on natural language models. By designing monolingual query texts and collecting responses, they successfully extract the knowledge from a simulated machine translation model under the black-box settings. This research exhibits that slight architectural differences will not influence the extraction between language models. Li et al. (2023b) also explores the potential risks of stealing the code-generation abilities of LLMs into smaller downstream models. Unlike previous research (Wallace et al., 2020), this is the first study that selects LLMs as targets. By collecting large-scale domain-specific samples, they fine-tune a 7-billion local pre-trained model with them and show the similarity between the victim and local models in both performances and adversarial samples. However, these two studies employ the MLE loss (Equation 3) as the MEA method, neither considering whether MLE is compatible with LLMs’s training, especially the alignment procedure shown in Section 2.2, nor addressing optimizations related to query efficiency and the watermark resistance. Besides, the scope of these studies is limited to stealing specific knowledge in a few downstream domains. At the same time, most of the critical aspects of LLMs and the required extraction capabilities, such as query numbers and local model scales, remain unresolved. Besides, while various other extraction attacks target LLMs (e.g., prompt extraction (Liang et al., 2025a)), these lie beyond the scope of our current discussion.

<table><tr><td>Datasets \ Models</td><td>Links</td></tr><tr><td>SafeRLHF</td><td>https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF</td></tr><tr><td>DiaSafety</td><td>https://huggingface.co/datasets/thu-coai/diasafety</td></tr><tr><td>PIQA</td><td>https://huggingface.co/datasets/piqa</td></tr><tr><td>TruthfulQA</td><td>https://huggingface.co/datasets/truthful_qa</td></tr><tr><td>WMT16</td><td>https://huggingface.co/datasets/wmt16</td></tr><tr><td>E2E NLG</td><td>https://huggingface.co/datasets/e2e_nlg</td></tr><tr><td>CommonGen</td><td>https://huggingface.co/datasets/allenai/common_gen</td></tr><tr><td>WikiSQL</td><td>https://huggingface.co/datasets/wikisql</td></tr><tr><td>Spider</td><td>https://huggingface.co/datasets/spider</td></tr><tr><td>TLDR</td><td>https://huggingface.co/datasets/UCL-DARK/openai-tldr-filtered</td></tr><tr><td>SamSUM</td><td>https://huggingface.co/datasets/samsum</td></tr><tr><td>CNN Daily Mail</td><td>https://huggingface.co/datasets/cnn_dailymail</td></tr><tr><td>Llama3-8B</td><td>https://huggingface.co/meta-1lama/Meta-Llama-3-8B-Instruct</td></tr><tr><td>Llama3-70B</td><td>https://huggingface.co/meta-1lama/Meta-Llama-3-70B-Instruct</td></tr><tr><td>Phi3-3.8B</td><td>https://huggingface.co/microsoft/Phi-3-mini-4k-instruct</td></tr><tr><td>OPT-6.7B</td><td>https://huggingface.co/facebook/opt-6.7b</td></tr><tr><td>Qwen2-7B</td><td>https://huggingface.co/Qwen/Qwen2-7B-Instruct</td></tr><tr><td>MistralV3-7B</td><td>https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3</td></tr></table>

Table 9: Datasets and pre-trained model checkpoints used in the paper. Specifically, we select twelve representative datasets: SafeRLHF (Ji et al., 2024), DiaSafety (Sun et al., 2022), WMT16 (Bojar et al., 2016), TLDR (Kirk et al., 2023), CNN Daily Mail (Hermann et al., 2015), Samsum (Gliwa et al., 2019), WikiSQL (Zhong et al., 2017), Spider (Yu et al., 2018), E2E-NLG (Dušek et al., 2020), CommonGen (Lin et al., 2020), PIQA (Bisk et al., 2020), and TruthfulQA (Lin et al., 2021) as benchmarks for our evaluation. These datasets cover most of the downstream tasks in natural language generation.

## C.3 Text Watermarks

In contrast to stealing LLMs, IP protection methods have received considerable attention recently. By sampling a stealthy but representative “greed word set” on the vocabulary distribution, these methods (Cong et al., 2022; He et al., 2022, 2021; Kirchenbauer et al., 2023) can remap the generated words into their synonyms or add the “watermarked” token automatically, and thus effectively certify the output. Besides, strategies such as integrating embeddings into the representation as the backdoor (Peng et al., 2023) or manipulating the probabilities with crafted sinusoidal noises (Zhao et al., 2022, 2023) are also proposed. However, these approaches often presume more stringent conditions regarding the victim and the suspected models. This paper will further assess the effectiveness of LoRD and current MEAs in evading these blackbox watermarking strategies.

## D A Detailed Threat Model

Adversary’s Objective. The adversary’s objective is to steal the targeted knowledge from LLMs. Specifically, we select machine translation, reasoning, data-to-text, structured text generation, and summarization as the downstream domain-specific tasks. The adversary aims to develop a queryefficient MEA algorithm, since the amount of input and generated tokens will be counted as the costs. Additionally, the MEA methods are expected to be watermark-resistant, i.e., they are highly desired to reduce the risks of exposure to unauthorized stealing.

Targeted Models. We select Llama3-70B, GPT-3.5-turbo, and GPT-4o as the victim models in this paper. Unlike previous works that only deployed simulated local victim models (e.g., OPT (Zhang et al., 2022)), our selections aim to expose the stealing threat on realistic AI services. Besides, our target models are specifically constrained to LLMs fine-tuned with alignment methods (e.g., RLHF) since they are not only state-of-the-art solutions now but also more valuable due to their humanbased alignments.

Adversary’s Capabilities. In accordance with the LLM-based AI service APIs, we identify two attack scenarios: black-box and grey-box attacks. In the black-box scenario, only textual responses the adversary is allowed to obtain. At the same time, all other information, such as the temperature, sampling strategies, and the hidden states of LLMs, are unseen and inaccessible. On the contrary, a grey-box attack allows the adversary to access the generation probabilities distribution of tokens. Notice that both MLE and our LoRD method are under black-box settings, and we only adopt grey-box settings on some particular stealing methods, such as knowledge distillation.

Besides, this paper posits that the adversary usually has worse training conditions than the victims. Specifically, query times and the scale of the local model available to the adversary are much smaller than the victims’ training datasets and model parameters. This setting has been adopted in previous LLMs’ extraction (Li et al., 2023b). We call it a LaViSH (Large-Victim-Small-Heist) framework, which allows us to estimate the upper bound of MEA risks empirically. For adversaries with more substantial resources, they can train more powerful MEA-based LLMs by leveraging MEA algorithms under our LaViSH settings.

Algorithm 1 LoRD Algorithm   
1: Input:Query dataset $\mathcal { D } _ { q } ,$ local language model $\theta _ { i n i t } ,$ , in  
terface of the victim model $P _ { \theta _ { v i c } } ( \cdot | \cdot )$ , train period number   
$N _ { t } ,$ threshold values τ and $\tau _ { 2 } .$   
2: $/ /$ Initialization.   
3: $\theta _ { 0 }  \theta _ { i n i t } , \mathcal { D } _ { t r }  \emptyset , \mathcal { D } _ { 0 } ^ { + }  \emptyset , \mathcal { D } _ { 0 } ^ { - }  \emptyset , t  0 ;$   
4: // Query the victim model.   
5: for x $\sim \mathcal { D } _ { q }$ do   
6: $\mathbf { y } _ { v i c }  \mathbf { \dot { P } } _ { \theta _ { v i c } } ( \cdot | \mathbf { x } ) ;$   
7: $\mathcal { D } _ { t r }  \mathcal { D } _ { t r } \cup \{ ( \mathbf { x } , \mathbf { y } _ { v i c } , P _ { \theta _ { v i c } } ( \mathbf { y } _ { v i c } | \mathbf { x } ) ) \}$   
8: end for   
9: // Train local model.   
10: // Initialize the positive and negative   
datasets.   
11: $\mathcal { D } _ { 0 } ^ { + }  \mathcal { D } _ { t r } ;$   
12: for $( \mathbf { x } , \mathbf { y } _ { v i c } , P _ { \theta _ { v i c } } ( \mathbf { y } _ { v i c } | \mathbf { x } ) ) \sim \mathcal { D } _ { t r }$ do   
13: $\mathbf { y } _ { 0 } ^ { - } \sim P _ { \theta _ { t } } ( \cdot | \mathbf { x } ) ;$   
14: $\mathcal { D } _ { 0 } ^ { - }  \mathcal { D } _ { 0 } ^ { - } \cup \{ ( \mathbf { x } , \mathbf { y } _ { 0 } ^ { - } ) \}$   
15: end for   
16: while $t < N _ { t }$ do   
17: $t \gets t + 1 ;$   
18: ${ \theta } _ { t } \gets { \theta } _ { t - 1 } ;$   
19: // Forward.   
20: for $\mathbf x , \mathbf y _ { v i c } , \mathbf y _ { t - 1 } ^ { + } , \mathbf y _ { t - 1 } ^ { - } \sim ( \mathscr D _ { t r } , \mathscr D _ { t - 1 } ^ { + } , \mathscr D _ { t - 1 } ^ { - } )$ do   
21: // Compute loss with Equation 10 or 11.   
22: $\begin{array} { r } { \mathcal { L } \gets \log [ \frac { P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } | \mathbf { x } ) } { P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { + } | \mathbf { x } ) } ] + c l i p ( \log [ \frac { P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } | \mathbf { x } ) } { P _ { \theta _ { t } } ( \mathbf { y } _ { \mathrm { v i c } } | \mathbf { x } ) } ] ] ; } \end{array}$   
23: $/ /$ Backward.   
24: $\theta _ { t } \gets \mathrm { s t e p U p d a t e } ( \theta _ { t } , \mathcal { L } ) ;$   
25: end for   
$\mathcal { D } _ { t } ^ { + }  \emptyset ; \mathcal { D } _ { t } ^ { - }  \emptyset ;$   
26: for $( \mathbf { x } , \mathbf { y } _ { v i c } , P _ { \theta _ { v i c } } ( \mathbf { y } _ { v i c } | \mathbf { x } ) ) \sim \mathcal { D } _ { t }$ <sub>r</sub> do   
27: $\mathbf { y } _ { t - 1 } ^ { + } , \mathbf { y } _ { t - 1 } ^ { - } \sim P _ { \theta _ { t - 1 } } ( \cdot | \mathbf { x } ) ;$   
28: $\begin{array} { r } { \Delta ^ { + }  \log P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { + } | x ) - \log P _ { \theta _ { t - 1 } } ( \mathbf { y } _ { t - 1 } ^ { + } | x ) ; } \end{array}$   
29: $\begin{array} { r } { \Delta ^ { - } \gets \log P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { - } | x ) - \log P _ { \theta _ { t - 1 } } ( \mathbf { y } _ { t - 1 } ^ { - } | x ) ; } \end{array}$   
30: if $\Delta ^ { + } < \Delta ^ { - }$ then   
31: $\operatorname { s w a p } ( \mathbf { y } _ { t - 1 } ^ { + } , \mathbf { y } _ { t - 1 } ^ { - } ) ;$   
32: $\operatorname { s w a p } ( \Delta ^ { + } , \Delta ^ { - } ) ;$   
33: end if   
34: if $P _ { \theta _ { t } } ( \mathbf { y } _ { t - 1 } ^ { + } | x ) < \tau _ { 1 }$ && $\Delta ^ { + } < \tau _ { 2 }$ then   
35: $\mathbf { y } _ { t - 1 } ^ { + }  \mathbf { y } _ { v i c } ;$   
36: end if   
37: $\mathcal { D } _ { t } ^ { + }  \mathcal { D } _ { t } ^ { + } \cup \{ ( \mathbf { x } , \mathbf { y } _ { t - 1 } ^ { + } ) \} ;$   
38: $\mathcal { D } _ { t } ^ { - }  \mathcal { D } _ { t } ^ { - } \cup \{ ( \mathbf { x } , \mathbf { y } _ { t - 1 } ^ { - } ) \}$   
39: end for   
40: end while   
41: return $\theta _ { t }$