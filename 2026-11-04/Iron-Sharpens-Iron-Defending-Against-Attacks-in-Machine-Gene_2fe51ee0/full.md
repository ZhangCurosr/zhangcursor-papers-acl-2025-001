# Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training

Yuanfan Li<sup>1,</sup> †, Zhaohan Zhang<sup>2,</sup> †, Chengzhengxu Li<sup>1,</sup> †, Chao Shen<sup>1</sup>, Xiaoming Liu<sup>1,</sup> ∗

<sup>1</sup>Faculty of Electronic and Information Engineering, Xi’an Jiaotong University

<sup>2</sup>Queen Mary University of London

† Equal contribution, ∗ Corresponding author

liyuan7716@gmail.com, czx.li@stu.xjtu.edu.cn zhaohan.zhang@qmul.ac.uk, {chaoshen, xm.liu}@xjtu.edu.cn

## Abstract

Machine-generated Text (MGT) detection is crucial for regulating and attributing online texts. While the existing MGT detectors achieve strong performance, they remain vulnerable to simple perturbations and adversarial attacks. To build an effective defense against malicious perturbations, we view MGT detection from a threat modeling perspective, that is, analyzing the model’s vulnerability from an adversary’s point of view and exploring effective mitigations. To this end, we introduce an adversarial framework for training a robust MGT detector, named GREedy Adversary PromoTed DefendER (GREATER). The GREATER consists of two key components: an adversary GREATER-A and a detector GREATER-D. The GREATER-D learns to defend against the adversarial attack from GREATER-A and generalizes the defense to other attacks. GREATER-A identifies and perturbs the critical tokens in embedding space, along with greedy search and pruning to generate stealthy and disruptive adversarial examples. Besides, we update the GREATER-A and GREATER-D synchronously, encouraging the GREATER-D to generalize its defense to different attacks and varying attack intensities. Our experimental results across 10 text perturbation strategies and 6 adversarial attacks show that our GREATER-D reduces the Attack Success Rate (ASR) by 0.67% compared with SOTA defense methods while our GREATER-A is demonstrated to be more effective and efficient than SOTA attack approaches. Codes and dataset are available in https:// github.com/Liyuuuu111/GREATER.

## 1 Introduction

The rapid development of large language models (LLM) (Achiam et al., 2023; Dubey et al., 2024; Anthropic, 2024; Guo et al., 2025) enables the model to generate highly human-like texts, which has raised broad concerns about the unrestricted dissemination of non-attributed textual contents including misinformation, fabricate news, and phishing emails. These negative impacts of MGTs lead to extensive works on MGT detection (Mitchell et al., 2023; Verma et al., 2024; Liu et al., 2024b; Bao et al.; Liu et al., 2024b) to accurately attribute the authorship of textual content and inform the readers.

![](images/952076a9c110ceb1b4bd55e5aa101d3a4dafee1784d650a41096661d9680d902.jpg)  
Figure 1: Performance drop of different MGT detectors and defense methods under text perturbation<sup>1</sup>.

Despite the superb performance of current MGT detectors, a recent study (Wang et al., 2024a) finds an astonishing fact that all detectors exhibit different loopholes in robustness, that is, existing detectors suffer great performance drop when facing different text perturbation strategies including editing (Kukich, 1992; Gabrilovich and Gontmakher, 2002), paraphrasing (Shi et al., 2024), prompting (Zamfirescu-Pereira et al., 2023), and co-generating (Kushnareva et al., 2024), etc. As illustrated in Figure 1, the detection accuracy of current detectors drops by around 30%-50% when confronted with simple perturbations, and the defense methods for general text classification cannot be simply adapted to the MGT detection scenario. More seriously, the vulnerability of MGT detectors is also unveiled by adversarial attacks that exploit the internal state (Yoo and Qi, 2021) or outputs (Liu et al., 2024a; Yu et al., 2024; Hu et al., 2024) of the detectors through multiple queries. Alas, there are few works on improving the robustness against adversarial attacks for MGT detectors.

Motivation. We rely on threat modeling to advance the robustness of the MGT detectors against perturbation and adversarial attacks. As the proverb says ’Iron sharpens iron’, we focus on constructing powerful adversarial examples which mislead the prediction of the detector to facilitate the post-training of MGT detectors and defend against different attacks. Existing text perturbation strategies (Wang et al., 2024a) adjust token distribution without accessing information from the target MGT detector, resulting in low-quality and non-targeted adversarial examples. The adversarial attacks are only effective in white-box setting (Yoo and Qi, 2021) or require excessive queries to target detectors (Hu et al., 2024; Yu et al., 2024; Liu et al., 2024a). Moreover, Wang et al. (2024c) find that the defense built by adversarial training cannot generalize well to the attacks on which it was not originally trained. To overcome these limitations, we propose an efficient adversarial training framework that works in a black-box setting and builds generalizable defense against a wide variety of perturbations and attacks for MGT detectors.

Our Work. In this paper, we propose an adversarial framework for training robust MGT detector, namely GREedy Adversary PromoTed DefendeR. (GREATER). GREATER consists of an adversary (GREATER-A) and a detector (GREATER-D). The GREATER-D learns to discern MGTs from the human-written texts (HWTs), while the GREATER-A, which queries the output of the detector, aims to imply minimum perturbation on MGTs to deceive the detector. Restricted by the scenario where only outputs from the target detector are available, we use an open-sourced surrogate model to retrieve gradient information to identify important tokens in the prediction. Afterwards, we introduce a gradient ascent perturbation on the embedding of MGTs from the surrogate model to enhance both the quality and stealthiness of generated adversarial text. To reduce the number of queries needed for building effective adversarial examples, we design a greedy search and pruning strategy. In the training stage, we update the GREATER-A and GREATER-D in the same training step so that GREATER-D learns from a curriculum of adversarial examples to generalize its defense. The experiment results demonstrate that our method achieves an average ASR of 5.53% against various attacks, which is 0.67% lower compared to the SOTA defense methods. We also find that GREATER-A achieves the most effective attack, achieving an ASR of 96.58%, which surpasses SOTA attack methods by 8.45% while requiring 4 times fewer queries.

Our contributions are as follows:

• Adversarial Training Framework. We propose an adversarial training framework GREATER to improve the robustness of MGT detectors, in which the adversary maliciously perturbs the MGTs to construct hard adversarial examples, while the detector is trained to maintain correct prediction towards the adversarial examples. We update the detector and the adversary generator in the same training step for better generalization on defense.

• Adversarial Examples Generation. We propose a strong and efficient adversarial examples construction method in the black-box setting. We retrieve gradient information from a surrogate model to rank the important tokens in MGT detection and design a greedy search and pruning strategy to reduce the query times needed for adversarial attacks.

• Outstanding Performance. Testing results across 16 attack methods demonstrate that our detector outperforms 10 existing SOTA defense methods in robustness, while our adversary achieves significant improvements in both attack efficiency and effectiveness compared to 13 SOTA attack approaches.

## 2 Related Work

Machine-Generated Text (MGT) Detection. There have been attempts to detect and attribute the MGTs in the pre-LLM era (Zhong et al., 2020; Uchendu et al., 2020). Nowadays, many works (Mitchell et al., 2023; Wang et al., 2023; Liu et al., 2022; Kushnareva et al., 2024; Guo et al.) aim to accurately annotate online texts as LLMs’ astonishing ability to generate fluent, logical, and human-like content, which helps the proliferation of unchecked information. Despite the achievements made in MGT detection, some works indicate the MGT detectors are vulnerable to simple perturbation or adversarial attacks. For example,

Wang et al. (2024a) test the robustness of eight MGT detectors with twelve perturbation strategies and they surprisingly find that none of the existing detectors remain robust under all the attacks. Moreover, MGT detectors’ defense against adversarial attacks is also questioned (Fishchuk, 2023). Other studies also reveal the fact that MGT detectors suffer from authorship obfuscation (Macko et al., 2024) and biased decision (Liang et al., 2023). To mitigate the vulnerability of MGT detectors, our work focuses on improving detector robustness against text perturbations and adversarial attacks.

Adversarial Training. Adversarial training aims to optimize the model toward maintaining correct predictions against adversarial examples that are misleading data constructed for malicious purposes. Earlier works first augment the training set with adversarial examples for defense against specific attacks (Huang et al., 2024; Zeng et al., 2023). These methods are shown to be hard to generalize to unseen attacks (Wang et al., 2024c). Yoo and Qi (2021) update the adversary and the target model in the same step to generalize the defense to unseen attacks. However, they rely on the availability of explicit first-order gradients, which is not applicable in the real-world case. OUTFOX (Koike et al., 2024) identifies adversarial attack with in-context learning, which requires the demonstrations of adversarial samples as prompts. RADAR (Hu et al., 2023a) uses a paraphraser as the adversary. Both OUTFOX and RADAR are designed to defend against known attacks. Different from previous works, we propose an effective adversarial training framework for MGT detectors that builds a generalizable defense against a variety of attacks in the black-box setting.

## 3 Threat Model

We follow the standard threat modeling framework outlined in prior work (Biggio and Roli, 2018) and describe our assumptions about the adversary’s goal and adversary’s capability.

Adversary’s Goal. Given a piece of MGT, the goal of the adversary is to make trivial changes to the original MGT so as to mislead the prediction of the detector. We refer the changed texts as adversarial examples. Ideally, the adversarial examples should satisfy three requirements: i) Low Perturbation Rate. Only trivial changes should be applied on the adversarial examples and the semantics of original texts should be retained. ii) High Readability. Adversarial examples should exhibit high readability so that the attack is most invisible to humans. iii) Less Query Requirements. The adversary should be query-efficient to reduce the time and budget needed to construct each adversarial example.

Adversary’s Capability. We assume the adversary’s capability in a real-world setting. First, an adversary only maliciously edits the MGTs but would not make any changes to the HWTs. This is because HWTs are trustworthy and there is no need for the adversary to change the prediction on HWTs. Second, since most commercial MGT detectors $( e . g . , \mathrm { G P T } \ Z \mathrm { e r o } ^ { 2 } )$ are close-sourced, the adversary should not have access to model weights and internal states of the target model. The only information the adversary is permitted to query is the output of the detector. Third, the adversary is allowed to access any open-sourced models.

## 4 Methodology

We introduce the framework of GREATER in this section. The architecture of GREATER is shown in Figure 2. In the following subsections, we first describe the workflow of the adversary and detector, respectively. Then we systematically outline the adversarial training process.

## 4.1 GREATER-A for Generating Adversarial Examples

To achieve the Adversary’s Goal outlined in §3, we developed an effective and efficient adversary. Specifically, the adversary achieves these requirements through two stages: Identify & Perturb and Replace & Refine.

## 4.1.1 Identify & Perturb

In this module, we design a token importance estimation module and apply a targeted perturbation on the embeddings of important tokens.

Important Token Identification. We consider a black-box setting where the internal state of the target detector $\mathcal { M } _ { t a r } ( . )$ is inaccessible. Given an original MGT $X = [ x _ { 1 } , x _ { 2 } , \dots , x _ { T } ]$ consisting of $T$ tokens, we utilize a surrogate model $\mathcal { M } _ { s u r } ( . )$ instead to obtain the last layer hidden state of each token in the text:

$$
H = \mathcal { M } _ { s u r } ( X ) = [ \mathbf { h } _ { 1 } , \mathbf { h } _ { 2 } , \ldots , \mathbf { h } _ { T } ] ,\tag{1}
$$

where $\mathbf { h } _ { t }$ represents the last layer hidden state of the t-th token $x _ { t }$ generated by $\mathcal { M } _ { s u r } ( . )$ . To obtain the importance score of each token $s _ { t } ,$ we train a simple scoring network $\mathcal { F } _ { \theta } ( . )$ which takes the feature embeddings as input and outputs the prediction of importance scores for each token:

![](images/04bfa242d49c44c6448380b817300a415559ceb7bb1c9bcc3fd6d8b92e47b410.jpg)  
Figure 2: Pipeline of GREATER. The adversary identifies important tokens in the original MGT and generates candidates for important tokens (§4.1.1). The adversary conducts and refines the attack by greedy search and pruning (§4.1.2). The final adversarial examples are feeded to the target detector (§4.2) and participate in the adversarial training process (§4.3).

$$
s _ { t } = \mathcal { F } _ { \theta } ( \mathbf { h } _ { t } ) ,\tag{2}
$$

where $\theta$ are learnable parameters. Then, we select the top-k tokens with the highest importance scores in text X and construct the important-token set I:

$$
\mathbf { I } = \mathrm { t o p } { - k \left( \left[ \left( x _ { t } , s _ { t } \right) \mid t = 1 , 2 , \ldots , T \right] \right) . }\tag{3}
$$

To mitigate the impact of discrepancies between the $\mathcal { M } _ { s u r } ( . )$ and $\mathcal { M } _ { t a r } ( . )$ , we leverage the predictions of the target detector $\mathcal { M } _ { t a r } ( . )$ to guide $\mathcal { F } _ { \theta } ( . )$ in more accurately identifying important tokens during adversarial training process. We detail the adversarial training process in §4.3.

Embedding-level Perturbation. We apply a targeted perturbation on the embedding of the tokens in I to improve the attack effectiveness while preserving semantic integrity. Formally, we introduce perturbations to the tokens in set I within the embeddings $E = [ \mathbf { e } _ { 1 } , \mathbf { e } _ { 2 } , \ldots , \mathbf { e } _ { T } ]$ of the surrogate model to obtain the perturbed embedding $\tilde { E } = [ \tilde { \mathbf { e } } _ { 1 } , \tilde { \mathbf { e } } _ { 2 } , \dots , \tilde { \mathbf { e } } _ { T } ]$

$$
\tilde { \mathbf { e } } _ { t } = \mathbf { e } _ { t } + \mathbf { 1 } _ { [ t \in \mathbf { I } ] } \delta _ { t } ,\tag{4}
$$

where $\delta _ { t }$ represents the perturbation of the t-th token, and $\mathbf { 1 } _ { [ t \in \mathbf { I } ] }$ represents an indicator function with a value of 1 if and only if the condition $t \in \mathbf { I }$ is satisfied, otherwise, it is 0. For the calculation of $\delta _ { t } .$ we first initialize the perturbation from a normalized uniform distribution, and then design a single-step gradient ascent strategy to optimize the perturbation. Specifically, we update the perturbation towards the direction where the KL divergence between the output distributions with respect to the original embedding E and initial perturbed embedding $\tilde { E } ^ { 0 }$ increases most steeply. This process is formulated as:

$$
\begin{array} { r } { \delta _ { t } ^ { 0 } \sim \mathcal { U } ( a , b ) , \quad \hat { \delta } _ { t } ^ { 0 } = \xi \frac { \delta _ { t } ^ { 0 } } { \lVert \delta _ { t } ^ { 0 } \rVert _ { 2 } } , \qquad } \\ { \delta _ { t } = \epsilon \frac { \nabla _ { \hat { \delta } _ { t } ^ { 0 } } \mathrm { K L } \Big ( P _ { \mathrm { s u r } } ( \mathbf { y } \mid E ) \lVert P _ { \mathrm { s u r } } ( \mathbf { y } \mid \tilde { E } ^ { 0 } ) \Big ) } { \left. \nabla _ { \hat { \delta } _ { t } ^ { 0 } } \mathrm { K L } \Big ( P _ { \mathrm { s u r } } ( \mathbf { y } \mid E ) \lVert P _ { \mathrm { s u r } } ( \mathbf { y } \mid \tilde { E } ^ { 0 } ) \Big ) \right. _ { 2 } } , \qquad } \end{array}\tag{5}
$$

where $\hat { \delta } _ { t } ^ { 0 }$ represents the normalized value of $\delta _ { t } ^ { 0 }$ and $\xi$ is a scaling factor. The parameters a and b define the lower and upper bounds of the uniform distribution $\textstyle { \mathcal { U } } ( a , b )$ , ϵ is a scaling factor, $P _ { \mathrm { s u r } } ( \mathbf { y } | E )$ and $P _ { \mathrm { s u r } } ( \mathbf { y } | \tilde { E } ^ { 0 } )$ are label distribution before and after perturbation, respectively.

Afterwards, we project the $\tilde { E }$ back to the vocabulary with the language modeling head and select the top-m tokens with the highest probabilities as candidates for replacing the important tokens:

Algorithm 1 Greedy Search Procedure   
1: Input: Target Detector $\mathcal { M } _ { t a r }$ , Original MGT X, Label $^ { c , }$   
Important-token Set I.   
2: round $ 0$ and ${ \tilde { X } }  X .$   
3: while round $<$ round $l _ { m a x }$ do   
4: Get the token $x _ { t } = \mathbf { I } [ $ [round] to be perturbed.   
5: Compute corresponding perturbation $\delta _ { t }$ using Eq $. ( 5 )$   
6: Get candidates $\scriptstyle { \dot { \mathbf { C } } } _ { t }$ for replacing token using Eq.(6).   
7: Replace x<sub>t</sub> with the token in $\mathbf { C } _ { t }$ to update $\tilde { X }$   
8: Classify $\tilde { X }$ via $\mathcal { M } _ { t a r }$ and obtain the output c˜.   
9: if $\tilde { c } \neq \dot { c }$ then   
10: break // Attack success   
11: else   
12: round round + 1 // Attack failed   
13: end if   
14: end while   
15: Output: Adversarial Example $\tilde { X } .$   
Algorithm 2 Greedy Pruning Procedure   
1: Input: Adversarial Example X<sup>˜</sup> from Algorithm 1, Label   
c, Important-token Set I and Target Detector $\mathcal { M } _ { t a r }$   
2: for each perturbed token $\tilde { x } _ { t }$ in $\tilde { X }$ do   
3: Revert $\tilde { x } _ { t }$ to corresponding token $x _ { t }$ in $X .$   
4: Classify $\tilde { X }$ via $\mathcal { M } _ { t a r }$ and obtain the output ${ \tilde { c } } .$   
5: if c˜ $\neq { \dot { c } } .$ then   
6: continue $/ /$ revert successful   
7: else   
8: Replace $x _ { t }$ with $\tilde { x } _ { t }$ again. // revert failed   
9: end if   
10: end for   
11: Output: Final Adversarial Example ${ \tilde { X } } .$

$$
\mathbf { C } _ { t } = \mathrm { t o p } { - m } ( \mathrm { S o f t m a x } ( \mathrm { L M H e a d } ( \mathcal { M } _ { s u r } ( \tilde { E } ) ) _ { t } ) ) ,\tag{6}
$$

where LMHead $( \mathcal { M } _ { s u r } ( \tilde { E } ) ) _ { t }$ represents the output of the LMHead(.) at position t. Following POS Constraints (Zhou et al., 2024), which require that the candidate words must match the part of speech of the words they replace, we filter the candidate tokens so that the candidate set contains the tokens with the same POS as the original one.

## 4.1.2 Replace & Refine

Based on the token importance calculated in §4.1.1, we introduce a greedy search and a greedy pruning strategy to efficiently construct powerful adversarial examples facilitating adversarial training.

Greedy Search. We present the process of greedy search in Algorithm 1. For a piece of original MGT X, we substitute the token $x _ { t }$ with the most possible candidate token in $\mathbf { C } _ { t }$ sequentially according to the descending order of importance scores. After each replacement, we query the target model $\mathcal { M } _ { t a r }$ if the adversarial example in the current step is machine-generated. We iterate this token-replacing procedure until the adversarial example deceives the $\mathcal { M } _ { t a r }$ or all the tokens in I are replaced. We then use the successful adversarial example (if the attack succeeds) or the text that is perturbed the most as training data for $\mathcal { M } _ { t a r }$ . Compared to existing methods (Liu et al., 2024a; Yu et al., 2024; Hu et al., 2024), our method perturbs only the tokens in set I incrementally with the guidance of importance score, enabling the generation of adversarial examples with fewer queries and lower perturbation rates.

Greedy Pruning. Due to the local optimality characteristic of greedy search (Yu et al., 2024; Prim, 1957), the adversarial examples constructed by greedy search contain redundant perturbations. We apply the greedy pruning algorithm, as shown in Algorithm 2, to further reduce the perturbation rate and make the attack stealthy without sacrificing its effectiveness. Given an adversarial example $\tilde { X }$ generated with greedy search, we sample the perturbed token by order of importance scores and replace the selected token with its corresponding original token one-by-one. After each restoration, we query the target model $\mathcal { M } _ { t a r }$ if $\tilde { X }$ is still a successful adversarial example. If the attack remains successful after restoration, the token is converted to the original one; otherwise, we preserve the perturbed token. We loop over all perturbed tokens and produce the final adversarial example $\tilde { X }$

To further validate the efficiency of our method, we provide a detailed theoretical analysis of query complexity and perturbation rate in GREATER-A in Appendix E.

## 4.2 GREATER-D for Defending Attacks

Our GREATER-D contains a target detector $\mathcal { M } _ { t a r } ( . )$ . For the target detector $\mathcal { M } _ { t a r } ( . )$ , we expect it to learn to defend against adversarial attacks from the adversary and generalize the defense ability to other attacks. Specifically, given the target detector $\mathcal { M } _ { t a r } ( . )$ , for each original MGT $X _ { i } \in \mathbf { X }$ and the corresponding adversarial example ${ \tilde { X _ { i } } }$ generated by the adversary, which share the same label $c _ { i } ,$ the optimization objective of the target detector is to make correct predictions for both the original MGT and the adversarial example:

$$
\operatorname* { m i n } _ { \theta } \big ( \sum _ { X _ { i } \in { \bf X } } \left( \mathcal { L } ( \mathcal { M } _ { t a r } ^ { \theta } ( X _ { i } ) , c _ { i } ) + \mathcal { L } ( \mathcal { M } _ { t a r } ^ { \theta } ( \tilde { X } _ { i } ) , c _ { i } ) ) \right) ,\tag{7}
$$

where $\theta$ represents the learnable parameters of the target detector $\mathcal { M } _ { t a r } ( . )$ and $\mathcal { L }$ is the loss function. This optimization objective aims to guide the $\mathcal { M } _ { t a r } ( . )$ to simultaneously enhance its performance on both original MGT and adversarial examples, thereby compelling it to learn robust features of samples before and after attacks. Regardless of whether the samples are subjected to adversarial interference, these features ensure that the detector can accurately classify the samples. As a result, the detector is better equipped to handle various types of attacks.

## 4.3 Adversarial Training

We propose to train the adversary and detector in a co-training manner, that is, the two main components in GREATER are updated in the same training step. Unlike the previous methods (Zhang et al., 2024b; Huang et al., 2024; Zeng et al., 2023) that rely on static training set augmented with adversarial examples, synchronously updating the adversary and the detector allows the detector to learn from easy adversarial examples to hard ones, facilitating the defense to generalize to different attacks.

Adversary Loss. The goal of the adversary is to precisely estimate the importance of each token in the detection to guide it to undertake a successful attack. Thus, we use the gradient with respect to the input in calculating the cross-entropy loss on training data as the golden label for the token importance score. However, since we are constrained in a black-box setting, we utilize the $\mathcal { M } _ { s u r } ( . )$ to obtain the gradient. This process is formulated as:

$$
\begin{array} { c } { \displaystyle { s _ { x } ^ { * } = \Big \| \nabla _ { x } \mathcal { L } _ { \mathrm { s u r } } \Big \| _ { 2 } , } } \\ { \displaystyle { \mathcal { L } _ { \mathrm { s u r } } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } [ - \log P _ { \mathrm { s u r } } ( c _ { i } \mid X _ { i } ) ] , } } \end{array}\tag{8}
$$

where ${ \mathcal { L } } _ { \mathrm { s u r } }$ is the cross-entropy classification loss of the surrogate model to the original MGTs. Subsequently, we update the scoring network $\mathcal { F } _ { \theta } ( . )$ with the mean squared error loss:

$$
\mathcal { L } _ { \mathrm { i m p } } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \frac { 1 } { | X _ { i } | } \sum _ { x \in X _ { i } } ( s _ { x } - s _ { x } ^ { * } ) ^ { 2 } ,\tag{9}
$$

where $\left| X _ { i } \right|$ is the number of tokens in sample $X _ { i }$ In addition, we leverage the output information from the target detector to guide the training of the $\mathcal { F } _ { \theta } ( . )$ , thereby mitigating the impact of discrepancies between the surrogate and target detector. Specifically, we replace the true labels in the crossentropy loss with misleading labels to direct the $\mathcal { F } _ { \theta } ( . )$ to produce samples that are capable to deceive the detector:

$$
\mathcal { L } _ { \mathrm { a d v } } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \left[ - \mathrm { l o g } P _ { \mathrm { t a r } } \big ( 1 - c _ { i } | \tilde { X } _ { i } \big ) \right] .\tag{10}
$$

Finally, we balance the influence of various losses on the adversary by adjusting the weight parameter λ. The total loss for the adversary is given as follows:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { A } } = \lambda \mathcal { L } _ { \mathrm { a d v } } + ( 1 - \lambda ) \mathcal { L } _ { \mathrm { i m p } } . } \end{array}\tag{11}
$$

Detector Loss. The detector’s goal is to maintain correct predictions in all circumstances. Based on this, we optimize the detector towards minimizing a cross-entropy loss:

$$
\mathcal { L } _ { \mathrm { D } } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \left[ - \mathrm { l o g } P _ { \mathrm { t a r } } ( c _ { i } | X _ { i } ) - \mathrm { l o g } P _ { \mathrm { t a r } } ( c _ { i } | \tilde { X } _ { i } ) \right] ,\tag{12}
$$

where $P _ { \mathrm { t a r } } ( c _ { i } | X _ { i } )$ is the probability of the target detector output at the correct label $c _ { i }$

Training Process. We update the adversary and the detector alternatively in the same training step. Specifically, at each training step t, we first update the adversary with loss function (10) while keeping the detector frozen. The updated adversary generates adversarial example ${ \tilde { X } } _ { i }$ in the current step. Then we update the detector with loss function (12). We detail the adversarial training process in form of pseudocode in Appendix F.

## 5 Experiment Results

We conduct extensive experiments to comprehensively evaluate the defense performance of our GREATER-D and also reveal the vulnerability of the current defense strategy with the adversarial examples generated by GREATER-A.

## 5.1 Defense Performance for GREATER-D

Experiment Setting. We evaluate our defense model GREATER-D against 16 text perturbation and adversarial attack methods, whose detailed introductions are outlined in the Appendix A.5. The competitors include i) data augmentation methods: Editing Pretrained (EP) (Wang et al., 2024c), Paraphrasing Pretrained (PP) (Wang et al., 2024c), CERT-ED (Huang et al., 2024), RanMask (Zeng et al., 2023), Text-RS (Zhang et al., 2024b), Text-CRS (Zhang et al., 2024a). ii) adversarial training methods: Virtual Adversarial Training (VAT) (Miyato et al., 2016), Token Aware Virtual Adversarial Training (TAVAT) (Li et al., 2021), RADAR (Hu et al., 2023a), OUTFOX (Koike et al., 2024). Detailed introduction and implementation are presented in Appendix A.4.1 and A.2. The dataset we use is presented in Appendix A.1.

<table><tr><td rowspan="2">Category</td><td rowspan="2">Method</td><td rowspan="2">Metric</td><td rowspan="2">Baseline</td><td colspan="6">Adversarial Data Augmentation</td><td colspan="5">Adversarial Training</td></tr><tr><td>F.t.XLM-RoBERTa-Base EP</td><td>PP</td><td>CERT-ED</td><td>RanMask</td><td>Text-RS</td><td>Text-CRS</td><td>VAT</td><td>TAVAT</td><td>RADAR</td><td>OUTFOX</td><td>GREATER-D</td></tr><tr><td rowspan="10">Text Perturbation</td><td>Mixed Edit</td><td>ASR(%)↓</td><td>34.65</td><td>4.58</td><td>32.33</td><td>16.74</td><td>17.00</td><td>22.81</td><td>15.34</td><td>33.19</td><td>37.50</td><td>18.76</td><td>11.33</td><td>8.85</td></tr><tr><td>HMGC</td><td>ASR(%)↓</td><td>30.00</td><td>6.53</td><td>27.34</td><td>12.52</td><td>11.74</td><td>16.58</td><td>13.37</td><td>25.19</td><td>31.50</td><td>8.53</td><td>3.74</td><td>2.28</td></tr><tr><td>Paraphrasing</td><td>ASR(%)↓</td><td>70.58</td><td>54.65</td><td>6.27</td><td>32.10</td><td>40.20</td><td>59.58</td><td>26.67</td><td>67.98</td><td>65.90</td><td>2.08</td><td>14.96</td><td>3.45</td></tr><tr><td>Code-switching MF*</td><td>ASR(%)↓</td><td>50.58</td><td>38.37</td><td>34.08</td><td>29.19</td><td>27.78</td><td>48.80</td><td>27.15</td><td>36.71</td><td>47.52</td><td>32.57</td><td>5.46</td><td>1.13</td></tr><tr><td>Code-switching MR*</td><td>ASR(%)↓</td><td>47.91</td><td>31.23</td><td>30.21</td><td>5.30</td><td>10.80</td><td>16.98</td><td>14.36</td><td>38.03</td><td>46.46</td><td>14.14</td><td>5.37</td><td>1.02</td></tr><tr><td>Human Obfuscation</td><td>ASR(%)↓</td><td>18.42</td><td>22.19</td><td>27.00</td><td>13.64</td><td>18.86</td><td>18.05</td><td>15.24</td><td>25.35</td><td>24.17</td><td>16.26</td><td>5.74</td><td>0.86</td></tr><tr><td>Emoji-cogen</td><td>ASR(%)↓</td><td>32.19</td><td>46.55</td><td>44.17</td><td>11.41</td><td>22.23</td><td>17.72</td><td>27.10</td><td>52.35</td><td>40.33</td><td>33.13</td><td>7.72</td><td>0.47</td></tr><tr><td>Typo-cogen</td><td>ASR(%)↓</td><td>60.10</td><td>61.57</td><td>59.72</td><td>27.29</td><td>38.82</td><td>37.09</td><td>44.79</td><td>70.26</td><td>63.04</td><td>41.52</td><td>9.29</td><td>1.08</td></tr><tr><td>ICL</td><td>ASR(%)↓</td><td>1.40</td><td>1.41</td><td>1.30</td><td>1.72</td><td>0.83</td><td>1.89</td><td>0.67</td><td>1.13</td><td>1.88</td><td>2.77</td><td>0.59</td><td>0.20</td></tr><tr><td>Prompt Paraphrasing</td><td>ASR(%)↓</td><td>0.00</td><td>0.69</td><td>0.00</td><td>0.70</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.94</td><td>0.65</td><td>0.23</td></tr><tr><td></td><td>CSGen</td><td>ASR(%)↓</td><td>25.44</td><td>22.81</td><td>6.14</td><td>10.65</td><td>1.70</td><td>2.41</td><td>23.95</td><td>26.88</td><td>27.52</td><td>17.07</td><td>3.40</td><td>0.00</td></tr><tr><td rowspan="10">TextFooler</td><td>Avg.</td><td>ASR(%)↓</td><td>33.75</td><td>26.42</td><td>24.41</td><td>14.66</td><td>17.27</td><td>21.99</td><td>18.97</td><td>34.28</td><td>35.07</td><td>17.07</td><td>6.20</td><td>1.78</td></tr><tr><td rowspan="3">PWWS</td><td>ASR(%)↓</td><td>61.45</td><td>48.47</td><td>49.06</td><td>11.07</td><td>14.83</td><td>16.13</td><td>19.15</td><td>53.56</td><td>62.68</td><td>12.58</td><td></td><td>5.91</td></tr><tr><td>Queries†</td><td>1197.95</td><td>1237.62</td><td>1235.01</td><td>1371.28</td><td>1361.15</td><td>1356.67</td><td>1342.15</td><td>1226.53</td><td>1180.87</td><td>1366.85</td><td></td><td>1390.69</td></tr><tr><td>ASR(%)↓</td><td>72.29</td><td>70.76</td><td>70.02</td><td>11.07</td><td>17.43</td><td>19.56</td><td>22.98</td><td>69.25</td><td>94.02</td><td>14.46</td><td></td><td>6.11</td></tr><tr><td></td><td>Queries†</td><td>690.92</td><td>724.23</td><td>713.77</td><td>1362.17</td><td>1291.35</td><td>1271.34</td><td>1237.18</td><td>780.51</td><td>440.28</td><td>1301.84</td><td></td><td>1396.52</td></tr><tr><td>BERTAttack</td><td>ASR(%)↓</td><td>71.49</td><td>69.34</td><td>63.52</td><td>6.04</td><td>12.42</td><td>12.30</td><td>16.94</td><td>69.45</td><td>93.81</td><td>13.08</td><td></td><td>5.70</td></tr><tr><td rowspan="3">A2T</td><td>Queries↑</td><td>411.54</td><td>446.64</td><td>425.68</td><td>710.56</td><td>682.29</td><td>680.14</td><td>667.82</td><td>450.57</td><td>279.02</td><td>677.24</td><td></td><td>718.84</td></tr><tr><td>ASR(%)↓</td><td>45.47</td><td>37.45</td><td>50.10</td><td>6.24</td><td>11.22</td><td>14.52</td><td>14.31</td><td>29.12</td><td>54.64</td><td>8.58</td><td></td><td>5.09</td></tr><tr><td>Queries↑</td><td>293.26</td><td>320.88</td><td>302.18</td><td>516.07</td><td>499.89</td><td>493.25</td><td>488.91</td><td>361.88</td><td>207.15</td><td>505.54</td><td></td><td>519.77</td></tr><tr><td></td><td>T-PGD</td><td>ASR(%)↓</td><td>49.90</td><td>34.15</td><td>25.28</td><td>8.52</td><td>12.07 13.15</td><td>14.92</td><td></td><td>42.14</td><td>45.58 7.79</td><td></td><td></td><td>5.61</td></tr><tr><td rowspan="2"></td><td></td><td>Queries↑</td><td>354.75</td><td>515.62</td><td>712.08</td><td>799.62</td><td>766.53</td><td>752.04</td><td>744.17</td><td>404.85</td><td>377.69</td><td>813.76</td><td></td><td>824.58</td></tr><tr><td>GREATER-A(ours)</td><td>ASR(%)↓</td><td>96.58</td><td>87.08</td><td>84.34</td><td>62.17</td><td>63.58</td><td>75.25</td><td>82.29</td><td>89.26</td><td>85.02</td><td>71.04</td><td></td><td>46.08</td></tr><tr><td rowspan="2"></td><td></td><td>Queries↑</td><td>62.63</td><td>66.71</td><td>68.53</td><td>99.56</td><td>98.92</td><td>106.08</td><td>75.98</td><td>63.57</td><td>67.17</td><td>100.74</td><td></td><td>190.64</td></tr><tr><td>Avg.</td><td>ASR(%)↓</td><td>66.20</td><td>57.87</td><td>57.05</td><td>17.52</td><td>21.92</td><td>25.15</td><td>28.43</td><td>58.80</td><td>72.62</td><td>21.26</td><td></td><td>12.42</td></tr><tr><td>Total Avg.</td><td></td><td>Queries†</td><td>501.84</td><td>551.95</td><td>576.21</td><td>809.88</td><td>783.36</td><td>776.59</td><td>759.37</td><td>547.98</td><td>425.36</td><td>794.33</td><td></td><td>840.17</td></tr></table>

Table 1: Performance of defense methods under different attacks. The best results are highlighted in green background.  means that Code-switching MF and Code-switching MR are two variations of Code-switching method. Since the detector in OUTFOX is a closed-source LLM, it cannot be attacked by adversarial attack. Therefore, we only evaluate OUTFOX under text perturbation attacks.

Experiment results. We present the defense performance of GREATER-D in Table 1 and unveil the following three key insights: 1) Best defense performance. Our method exhibits the best defense performance against different attacks among all competitors. The average Attack Success Rate (ASR) drops to 1.78% for text perturbation attacks and 12.42% for adversarial attacks, which are lower than the second-best methods, RADAR (6.20%) and CERT-ED (17.52%), by 4.42% and 5.10%, respectively. 2) Most effort needed for adversarial attack. We observe that it takes adversarial attack more resources to conduct a successful attack to GREATER-D. 840.17 queries are required on average, which is 30.29 more than other defense methods. Moreover, the average ASR against GREATER-D is only 12.42% for adversarial attacks. It illustrates that GREATER-D makes adversarial attacks both inefficient and ineffective. 3) Generalized Defense to Different Attacks. We notice that GREATER-D significantly reduces the ASR of different kinds of attacks even though it is trained with GREATER-A. As an exemplary method, EP performs better than GREATER-D when defending Mixed Edit Attack but cannot withstand other attacks. The defense against a wide variety of attacks demonstrates the generalized defense effect of GREATER-D.

## 5.2 Attack Performance for GREATER-A

In this section, we evaluate the effectiveness of GREATER-A in the black-box setting using the metrics detailed in Appendix A.3. We categorize the comparison methods into two classes: i) Query-based methods, which query the target model for output to adjust attack strategy, including PWWS (Ren et al., 2019), TextFooler (Jin et al., 2020), BERTAttack (Li et al., 2020), HQA (Liu et al., 2024a), ABP (Yu et al., 2024), T-PGD (Yuan et al., 2023), and FastTextDodger (Hu et al., 2024). ii) Zero-query methods, which conducts attack without any information from the target model, including WordNet (Zhou et al., 2024), Back Translation (Zhou et al., 2024), Rewrite (Zhou et al., 2024), T-PGD (Yuan et al., 2023), and HMGC (Zhou et al., 2024). To further demonstrate the effectiveness of GREATER-A, we also incorporate A2T (Yoo and Qi, 2021), a SOTA white-box method in querybased methods. Detailed introduction of these attack methods are listed in Appendix A.4.2. Note that GREATER-A is a query-based method. However, for a fair comparison, we also implement our method in a zero-query setting where we query the surrogate model for feedback. Among all the attacks, we employ a fine-tuned XLM-RoBERTa-Base model (Conneau et al., 2019) as the target detector.

Experiment Results. We show the experiment results in Table 2. We find GREATER-A performs the best in three dimensions: effectiveness, efficiency, and stealthy. GREATER-A achieves 96.58% and 69.11% in terms of ASR in query and zeroquery settings, respectively, which significantly outperforms all other methods. Moreover, in the query-based setting, it only takes 62.63 queries for GREATER-A to conduct a successful attack, which is four times fewer than its competitors. As for the stealthy, the texts edited by GREATER-A achieves the lowest perplexity and has the best readability implied by the highest USE and lowest readability change in query-based scenario. In the zeroquery setting, GREATER-A performs second-best in terms of readability after Back Translation which can rarely conduct a successful attack.

<table><tr><td>Attack Type</td><td>Method</td><td>Avg Queries ↓</td><td>ASR (%) ↑</td><td>Pert. (%) ↓</td><td>∆PPL↓</td><td>USE↑</td><td>∆r↓</td></tr><tr><td rowspan="9">Query-based</td><td>PWWS</td><td>1197.95</td><td>61.45</td><td>4.71</td><td>37.85</td><td>0.9488</td><td>12.76</td></tr><tr><td>TextFooler</td><td>690.92</td><td>72.29</td><td>6.26</td><td>46.89</td><td>0.9302</td><td>21.07</td></tr><tr><td>BERTAttack</td><td>411.54</td><td>71.49</td><td>5.79</td><td>36.15</td><td>0.9402</td><td>15.78</td></tr><tr><td>HQA</td><td>283.89</td><td>88.13</td><td>23.57</td><td>102.87</td><td>0.8854</td><td>72.16</td></tr><tr><td>FastTextDodger</td><td>745.75</td><td>63.78</td><td>13.29</td><td>76.15</td><td>0.9188</td><td>55.14</td></tr><tr><td>ABP</td><td>785.18</td><td>75.65</td><td>14.63</td><td>39.61</td><td>0.8709</td><td>26.40</td></tr><tr><td>T-PGD</td><td>354.75</td><td>49.90</td><td>38.01</td><td>181.31</td><td>0.8197</td><td>35.09</td></tr><tr><td>GREATER-A</td><td>62.63</td><td>96.58</td><td>7.26</td><td>35.22</td><td>0.9506</td><td>9.21</td></tr><tr><td>A2T (White Box)</td><td>293.26</td><td>45.47</td><td>7.01</td><td>62.84</td><td>0.9215</td><td>32.07</td></tr><tr><td rowspan="6">Zero-query</td><td>WordNet</td><td></td><td>42.60</td><td></td><td>26.27</td><td>0.90</td><td>4.26</td></tr><tr><td>Back Translation</td><td></td><td>2.40</td><td></td><td>6.40</td><td>0.91</td><td>3.06</td></tr><tr><td>Rewrite</td><td></td><td>36.47</td><td></td><td>92.08</td><td>0.79</td><td>15.62</td></tr><tr><td>T-PGD</td><td></td><td>62.40</td><td></td><td>140.13</td><td>0.82</td><td>44.99</td></tr><tr><td>HMGC</td><td></td><td>30.00</td><td></td><td>12.94</td><td>0.84</td><td>36.90</td></tr><tr><td>GREATER-A</td><td></td><td>69.11</td><td></td><td>43.11</td><td>0.92</td><td>4.55</td></tr></table>

Table 2: Attack results of the query and zero-query attack methods on the target model. Note that perturbation rates are not reported for zero-query methods because the zero-query methods rewrite the whole text. The best result in each group is highlighted with a green background.

![](images/c3002fd70915af8858db9973c0d77209a6aa69ab3272273a356cff93d62d0b1e.jpg)  
Figure 3: Impact of attack strength in GREATER. We normalize the attack strength for better visualization of the results.

## 6 Discussion

## 6.1 Impact of Attack Strength of GREATER-A on GREATER-D

In this section, we investigate how the strength of adversarial training affects the performance of both our GREATER-D and GREATER-A. We define the attack strength as the max number of query the adversary is allowed to make in the training process. Generally, if the adversary queries the target model more frequently, its final output tends to be more effective. We increase attack strength in GREATER and evaluate the performance of GREATER-D under 7 attacks and GREATER-A on the fine-tuned XLM-RoBERTa-Base detector and present the results in Figure 3. The ASR significantly increases as the attack strength grows, which proves the rationale of the attack strength measure. We observe that the GREATER-D becomes more robust with the increasing of attack strength, indicated by the decreasing ASR under all attacks. Notably, the ASR under Paraphrasing Attack decreases from 10.80 to 3.45 as the attack strength increases from 0.0 to 1.0, which is 3.13 times lower.

The experimental results indicate that attack strength is a key factor influencing the robustness of the MGT detector. However, an increased number of queries comes with more cost on time and budget. Thus, there exists a trade-off between the effectiveness and efficiency in adversarial training.

## 6.2 Defense Adaptation to Different Backbones

To demonstrate the generalizability and effectiveness of our GREATER across different model architectures, we replace the backbone model with seven state-of-the-art transformer-based models, including both base and large variants of AL-BERT (Lan et al., 2019), DeBERTa (He et al., 2020), RoBERTa (Liu et al., 2019), and XLM-RoBERTa (Conneau et al., 2019). We report the ASR of each model under the Paraphrasing Attack with a budget of 0.74, both before defense (Baseline) and after applying our GREATER-D. We compare it with the best-performing defense method CERT-ED and show the result in Table 3.

<table><tr><td>Model</td><td>Metric</td><td>Baseline</td><td>CERT-ED</td><td>GREATER-D</td></tr><tr><td>ALBERT Base (12M)</td><td>ASR(%)↓</td><td>63.58</td><td>46.71</td><td>35.62</td></tr><tr><td>ALBERT Large (18M)</td><td>ASR(%)↓</td><td>97.14</td><td>45.45</td><td>43.40</td></tr><tr><td>DeBERTa Base (86M)</td><td>ASR(%)↓</td><td>22.95</td><td>29.37</td><td>4.10</td></tr><tr><td>DeBERTa Large (304M)</td><td>ASR(%)↓</td><td>51.72</td><td>25.07</td><td>10.25</td></tr><tr><td>RoBERTa Base (125M)</td><td>ASR(%)↓</td><td>81.88</td><td>26.78</td><td>15.32</td></tr><tr><td>RoBERTa Large (355M)</td><td>ASR(%)↓</td><td>30.21</td><td>34.66</td><td>5.38</td></tr><tr><td>XLM-RoBERTa Large (561M)</td><td>ASR(%)↓</td><td>81.99</td><td>40.04</td><td>1.73</td></tr></table>

Table 3: Experiment results of various models before and after applying GREATER-D under Paraphrasing attack. The best result in each group is highlighted with a green background.

The results presented in Table 3 demonstrate the efficacy of our adversarial training method across a diverse set of transformer-based models. Notably, XLM-RoBERTa-Large exhibits the most substantial improvement, with ASR decreasing by 80.26%, 38.31% compared with baseline and CERT-ED, highlighting the significant impact of adversarial training on models with initially lower performance metrics. Similarly, both RoBERTa Base and DeBERTa Large demonstrate substantial improvements. Specifically, RoBERTa Base achieves a 66.56% reduction in ASR compared to the baseline and outperforms the CERT-ED with an additional 11.46% decrease. Likewise, DeBERTa Large exhibits a 41.47% drop in ASR relative to the baseline and surpasses CERT-ED by reducing ASR by an additional 14.82%. These results underscore the robustness and versatility of our adversarial training approach across different model sizes and architectures. While smaller models like ALBERT Base and ALBERT Large exhibit more modest gains, the consistent upward trends across all evaluated models affirm that our adversarial training method effectively enhances model resilience and performance against Paraphrasing Attack. This versatility makes our approach a valuable tool for improving a wide range of transformer-based models in adversarial settings.

## 6.3 Impact of Surrogate Model of GREATER-A

We use different surrogate model and dataset for training GREATER-A to demonstrate that GREATER-A’s performance is independent of surrogate model selection and training data. Specifically, we use RoBERTa-Large and GPT2 as surrogate models, and use SemEval (same dataset for training GREATER-D, abbreviated as IND) and M4 (Wang et al., 2024b) (different dataset from training GREATER-D, abbreviated as IND) to validate our claim. The results are shown in Table 4.

<table><tr><td>Surrogate Model</td><td>Dataset (Type)</td><td>Avg Queries ↓</td><td>ASR (%) ↑</td><td>Pert. (%) ↓</td><td>∆PPL↓</td><td>USE↑</td><td>∆r ↓</td></tr><tr><td>RoBERTa-Large</td><td>SemEval (IND)</td><td>62.63</td><td>96.58</td><td>7.26</td><td>35.22</td><td>0.9506</td><td>9.21</td></tr><tr><td>RoBERTa-Large</td><td>M4 (OOD)</td><td>65.77</td><td>95.72</td><td>7.33</td><td>38.84</td><td>0.9503</td><td>9.65</td></tr><tr><td>GPT2</td><td>SemEval (IND)</td><td>65.60</td><td>96.17</td><td>7.29</td><td>36.87</td><td>0.9550</td><td>12.73</td></tr><tr><td>GPT2</td><td>M4 (OOD)</td><td>63.80</td><td>96.36</td><td>7.27</td><td>37.14</td><td>0.9536</td><td>10.64</td></tr></table>

Table 4: The attack results of different surrogate models and datasets that the surrogate models are trained with. The best result in each group is highlighted with a green background.

As shown in Table 4, the attack effectiveness remains stable regardless of the surrogate model or training dataset. This consistency in attack success rates indicates that GREATER-A’s performance is not sensitive to changes in the surrogate model or training data, supporting the claim that the use of a surrogate model from the same family as the target model does not leak crucial information.

## 7 Conclusion

In this paper, we proposed an adversarial training framework GREedy Adversary PromoTed DefendER (GREATER) to enhance the robustness of MGT detector under different text perturbation and adversarial attacks. We design a novel attack strategy for the adversary including Identify & Perturb and Replace & Refine to construct effective adversarial examples efficiently. In GREATER, we update the adversary and the detector alternatively in the same training step for better defense generalization. Our experiment results demonstrate the efficacy of our detector GREATER-D under 16 attacks along with the leading performance of adversary GREATER-A compared with 13 methods. The discussion on the relationship between attack strength and defense performance reveals the importance of a powerful adversary in adversarial training for robust MGT detectors.

## Limitations

Despite the promising results achieved by GREATER, there are three primary limitations to this study. First, our defense method can generalize to different attacks but its application on texts in different languages remains a challenge. Second, the computational cost of training the adversarial framework, particularly for the adversary and detector, is substantial, requiring significant hardware resources that could limit its deployment in resource-constrained settings. Third, the detector presented in our work is only able to attribute the origin of the text on the document-level. However, more works are needed for fine-grained (e.g., sentence-level, token-level) detection in Human-AI co-authored texts.

## Ethics Statement

This study seeks to improve the robustness and security of machine-generated text (MGT) detection, with a focus on defending against adversarial threats. While the proposed GREATER framework enhances detection capabilities, it also introduces potential risks of misuse, such as creating adversarial examples to evade detection systems. To mitigate such risks, our experiments were conducted in controlled environments, and details that could enable misuse were abstracted. We emphasize that this work is intended solely for advancing detection technologies and defending against malicious applications. Ethical use of these findings is imperative, and any misuse for harmful purposes is strongly discouraged. The artifacts used in our work are all under the restriction of the license.

## Acknowledgment

We thank all the reviewers and the area chair for their helpful feedback, which aided us in greatly improving the paper. This work is supported by National Key R&D Program (2023YFE0209800), National Natural Science Foundation of China (62272371, 62103323, U24B20185, T2442014, 62161160337, 62132011, U21B2018), Initiative Postdocs Supporting Program (BX20190275, BX20200270), China Postdoctoral Science Foundation (2019M663723, 2021M692565), Fundamental Research Funds for the Central Universities under grant (xzy012024144, xzy012025043), and Shaanxi Province Key Industry Innovation Program (2023-ZDLGY-38, 2021ZDLGY01-02).

Thanks to the New Cornerstone Science Foundation and the Xplorer Prize.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Anthropic. 2024. Claude 3: A conversational ai model.

Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi Yang, and Yue Zhang. Fast-detectgpt: Efficient zeroshot detection of machine-generated text via conditional probability curvature. In The Twelfth International Conference on Learning Representations.

Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi Yang, and Yue Zhang. 2023. Fast-detectgpt: Efficient zero-shot detection of machine-generated text via conditional probability curvature. arXiv preprint arXiv:2310.05130.

Battista Biggio and Fabio Roli. 2018. Wild patterns: Ten years after the rise of adversarial machine learning. In Proceedings ofthe 2018 ACM SIGSAC Conference on Computer and Communications Security, pages 2154–2156.

Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St John, Noah Constant, Mario Guajardo-Cespedes, Steve Yuan, Chris Tar, et al. 2018. Universal sentence encoder. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 169–174.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Unsupervised cross-lingual representation learning at scale. arXiv preprint arXiv:1911.02116.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Steffen Eger, Gözde Gül ¸Sahin, Andreas Rücklé, Ji-Ung Lee, Claudia Schulz, Mohsen Mesgar, Krishnkant Swarnkar, Edwin Simpson, and Iryna Gurevych. 2019. Text processing like humans do: Visually attacking and shielding nlp systems. In Proceedings of the 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 1634–1647.

Vitalii Fishchuk. 2023. Adversarial attacks on neural text detectors. B.S. thesis, University of Twente.

Rudolf Flesch. 1948. A new readability yardstick. Journal of Applied Psychology, 32(3):221–233.

Evgeniy Gabrilovich and Alex Gontmakher. 2002. The homograph attack. Communications of the ACM, 45(2):128.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Xun Guo, Yongxin He, Shan Zhang, Ting Zhang, Wanquan Feng, Haibin Huang, and Chongyang Ma. Detective: Detecting ai-generated text via multi-level contrastive learning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Abhimanyu Hans, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi, Aniruddha Saha, Micah Goldblum, Jonas Geiping, and Tom Goldstein. 2024. Spotting llms with binoculars: Zero-shot detection of machine-generated text. arXiv preprint arXiv:2401.12070.

Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. 2020. Deberta: Decoding-enhanced bert with disentangled attention. arXiv preprint arXiv:2006.03654.

Helsinki-NLP. 2020. Helsinki-nlp machine translation models. https://huggingface.co/ Helsinki-NLP.

Xiaomeng Hu, Pin-Yu Chen, and Tsung-Yi Ho. 2023a. Radar: Robust ai-text detection via adversarial learning. Advances in neural information processing systems, 36:15077–15095.

Xiaoxue Hu, Geling Liu, Baolin Zheng, Lingchen Zhao, Qian Wang, Yufei Zhang, and Minxin Du. 2024. Fasttextdodger: Decision-based adversarial attack against black-box nlp models with extremely high efficiency. IEEE Transactions on Information Forensics and Security.

Zhengmian Hu, Lichang Chen, Xidong Wu, Yihan Wu, Hongyang Zhang, and Heng Huang. 2023b. Unbiased watermark for large language models. arXiv preprint arXiv:2310.10669.

Zhuoqun Huang, Neil G Marchant, Olga Ohrimenko, and Benjamin IP Rubinstein. 2024. Cert-ed: Certifiably robust text classification for edit distance. arXiv preprint arXiv:2408.00728.

Di Jin, Zhijing Jin, Joey Tianyi Zhou, and Peter Szolovits. 2020. Is bert really robust? a strong baseline for natural language attack on text classification and entailment. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 8018–8025.

Ryuto Koike, Masahiro Kaneko, and Naoaki Okazaki. 2024. Outfox: Llm-generated essay detection through in-context learning with adversarially generated examples. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 21258–21266.

Kalpesh Krishna, Yixiao Song, Marzena Karpinska, John Wieting, and Mohit Iyyer. 2024. Paraphrasing evades detectors of ai-generated text, but retrieval is an effective defense. Advances in Neural Information Processing Systems, 36.

Karen Kukich. 1992. Techniques for automatically correcting words in text. ACM computing surveys (CSUR), 24(4):377–439.

Laida Kushnareva, Tatiana Gaintseva, German Magai, Serguei Barannikov, Dmitry Abulkhanov, Kristian Kuznetsov, Eduard Tulchinskii, Irina Piontkovskaya, and Sergey Nikolenko. 2024. Ai-generated text boundary detection with roft. In 1st Conference on Language Modeling (COLM), volume 2024.

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. 2019. Albert: A lite bert for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942.

Vladimir I. Levenshtein. 1966. Binary codes capable of correcting deletions, insertions, and reversals. Soviet Physics Doklady, 10(8):707–710.

Linyang Li, Ruotian Ma, Xiaonan Guo, Qing Wang, Xipeng Qiu, and Xuanjing Tang. 2020. Bert-attack: Adversarial attack against bert using bert. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6193–6202.

Yuan Li, Shuhuai Ren, and Zhouxing Shi. 2021. Tavat: Token-aware virtual adversarial training for language understanding. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 2924–2936.

Weixin Liang, Mert Yuksekgonul, Yining Mao, Eric Wu, and James Zou. 2023. Gpt detectors are biased against non-native english writers. Patterns, 4(7).

Han Liu, Zhi Xu, Xiaotong Zhang, Feng Zhang, Fenglong Ma, Hongyang Chen, Hong Yu, and Xianchao Zhang. 2024a. Hqa-attack: toward high quality blackbox hard-label adversarial attack on text. Advances in Neural Information Processing Systems, 36.

Shengchao Liu, Xiaoming Liu, Yichen Wang, Zehua Cheng, Chengzhengxu Li, Zhaohan Zhang, Yu Lan, and Chao Shen. 2024b. Does detectgpt fully utilize perturbation? bridging selective perturbation to finetuned contrastive learning detector would be better. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1874–1889.

Xiaoming Liu, Zhaohan Zhang, Yichen Wang, Hang Pu, Yu Lan, and Chao Shen. 2022. Coco: Coherenceenhanced machine-generated text detection under data limitation with contrastive learning. arXiv preprint arXiv:2212.10341.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized BERT pretraining approach. arXiv preprint arXiv:1907.11692.

Dominik Macko, Robert Moro, Adaku Uchendu, Ivan Srba, Jason Samuel Lucas, Michiharu Yamashita, Nafis Irtiza Tripto, Dongwon Lee, Jakub Simko, and Maria Bielikova. 2024. Authorship obfuscation in multilingual machine-generated text detection. arXiv preprint arXiv:2401.07867.

Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D Manning, and Chelsea Finn. 2023. Detectgpt: Zero-shot machine-generated text detection using probability curvature. In International Conference on Machine Learning, pages 24950–24962. PMLR.

Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, and Shin Ishii. 2016. Virtual adversarial training: A regularization method for supervised and semisupervised learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(8):1979– 1993.

Robert Clay Prim. 1957. Shortest connection networks and some generalizations. The Bell System Technical Journal, 36(6):1389–1401.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners. OpenAI Technical Report, 1(8):1–24.

Shuhuai Ren, Yihe Zheng, Yizhan Chen, Bin Yu, Zhiyuan Liu, and Maosong Sun. 2019. Generating natural language adversarial examples through probability weighted word saliency. In Proceedings ofthe 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 1085–1097.

Herbert Robbins and Sutton Monro. 1951. A stochastic approximation method. The annals ofmathematical statistics, pages 400–407.

Zhouxing Shi, Yihan Wang, Fan Yin, Xiangning Chen, Kai-Wei Chang, and Cho-Jui Hsieh. 2024. Red teaming language model detectors with language models. Transactions of the Association for Computational Linguistics, 12:174–189.

M. Siino. 2024. Badrock at semeval-2024 task 8: Distilbert to detect multigenerator, multidomain and multilingual black-box machine-generated text. In Proceedings ofthe 18th International Workshop on Semantic Evaluation.

Jinyan Su, Terry Zhuo, Di Wang, and Preslav Nakov. 2023. Detectllm: Leveraging log rank information for zero-shot detection of machine-generated text. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 12395–12412.

Jörg Tiedemann. 2012. Parallel data, tools and interfaces in OPUS. In Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC’12), pages 2214–2218, Istanbul, Turkey. European Language Resources Association (ELRA).

Adaku Uchendu, Thai Le, Kai Shu, and Dongwon Lee. 2020. Authorship attribution for neural text generation. In 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, pages 8384–8395. Association for Computational Linguistics (ACL).

Vivek Verma, Eve Fleisig, Nicholas Tomlin, and Dan Klein. 2024. Ghostbuster: Detecting text ghostwritten by large language models. In Proceedings of the 2024 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 1702–1717.

Pengyu Wang, Linyang Li, Ke Ren, Botian Jiang, Dong Zhang, and Xipeng Qiu. 2023. Seqxgpt: Sentencelevel ai-generated text detection. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 1144–1156.

Yichen Wang, Shangbin Feng, Abe Bohan Hou, Xiao Pu, Chao Shen, Xiaoming Liu, Yulia Tsvetkov, and Tianxing He. 2024a. Stumbling blocks: Stress testing the robustness of machine-generated text detectors under attacks. arXiv preprint arXiv:2402.11638.

Yuxia Wang, Jonibek Mansurov, Petar Ivanov, Jinyan Su, Artem Shelmanov, Akim Tsvigun, Chenxi Whitehouse, Osama Mohammed Afzal, Tarek Mahmoud, Toru Sasaki, et al. 2024b. M4: Multi-generator, multi-domain, and multi-lingual black-box machinegenerated text detection. In Proceedings ofthe 18th Conference of the European Chapter of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1369–1407.

Zaitian Wang, Pengfei Wang, Kunpeng Liu, Pengyang Wang, Yanjie Fu, Chang-Tien Lu, Charu C. Aggarwal, Jian Pei, and Yuanchun Zhou. 2024c. A comprehensive survey on data augmentation. arXiv preprint arXiv:2405.09591.

Genta Winata, Alham Fikri Aji, Zheng Xin Yong, and Thamar Solorio. 2023. The decades progress on codeswitching research in nlp: A systematic survey on trends and challenges. In Findings of the Association for Computational Linguistics: ACL 2023, pages 2936–2978.

Jin Yong Yoo and Yanjun Qi. 2021. Towards improving adversarial training of nlp models. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 1001–1013.

Zhen Yu, Zhenhua Chen, and Kun He. 2024. Queryefficient textual adversarial example generation for black-box attacks. In Proceedings ofthe 2024 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 556–569.

Lifan Yuan, Yichi Zhang, Yangyi Chen, and Wei Wei. 2023. Bridge the gap between cv and nlp! a gradientbased textual adversarial attack framework. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, pages 7132–7146.

JD Zamfirescu-Pereira, Richmond Y Wong, Bjoern Hartmann, and Qian Yang. 2023. Why johnny can’t prompt: how non-ai experts try (and fail) to design llm prompts. In Proceedings ofthe 2023 CHI Conference on Human Factors in Computing Systems, pages 1–21.

Jiehang Zeng, Xiaoqing Zheng, Jianhan Xu, Linyang Li, Liping Yuan, and Xuanjing Huang. 2023. Certified robustness to text adversarial attacks by randomized [mask]. Computational Linguistics, 49(2):345–373.

Biao Zhang, Philip Williams, Ivan Titov, and Rico Sennrich. 2020a. Improving massively multilingual neural machine translation and zero-shot translation. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics, pages 1628– 1639, Online. Association for Computational Linguistics.

Jingqing Zhang, Yao Zhao, Mohammad Saleh, and Peter Liu. 2020b. Pegasus: Pre-training with extracted gap-sentences for abstractive summarization. In International conference on machine learning, pages 11328–11339. PMLR.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2019. Bertscore: Evaluating text generation with bert. arXiv preprint arXiv:1904.09675.

Xinyu Zhang, Hanbin Hong, Yuan Hong, Peng Huang, Binghui Wang, Zhongjie Ba, and Kui Ren. 2024a. Text-crs: A generalized certified robustness framework against textual adversarial attacks. In Proceedings of the 2024 IEEE Symposium on Security and Privacy, pages 53–53.

Zeliang Zhang, Wei Yao, Susan Liang, and Chenliang Xu. 2024b. Random smooth-based certified defense against text adversarial attack. In Findings of the Association for Computational Linguistics: EACL 2024, pages 1251–1265.

Wanjun Zhong, Duyu Tang, Zenan Xu, Ruize Wang, Nan Duan, Ming Zhou, Jiahai Wang, and Jian Yin. 2020. Neural deepfake detection with factual structure of text. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2461–2470.

Ying Zhou, Ben He, and Le Sun. 2024. Humanizing machine-generated content: Evading ai-text detection through adversarial attack. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 8427–8437.

## A Experimental Setting

## A.1 Dataset

We used the Semeval Task8 dataset (Siino, 2024) as the primary dataset for training the detector. This large-scale dataset includes MGTs from ChatGPT 4, Davinci, Bloomz, Dolly, and Cohere, as well as HWTs from WikiHow, Reddit, arXiv, Wikipedia, and PeerRead. Moreover, the average token length of the dataset is 623.2521. For each scenario, we employed distinct datasets for testing, as detailed in Tabel 5.

<table><tr><td>Scenario</td><td>Dataset</td><td>Number of MGTs</td></tr><tr><td>Mixed Edit</td><td>Semeval Task8 (Siino, 2024)</td><td>1000</td></tr><tr><td>HMGC</td><td>Semeval Task8 (Siino, 2024)</td><td>1000</td></tr><tr><td>Paraphrasing</td><td>Semeval Task8 (Siino, 2024)</td><td>1000</td></tr><tr><td>Code-switching</td><td>Semeval Task8 (Siino, 2024)</td><td>1000</td></tr><tr><td>Human Obfuscation</td><td>Semeval Task8 (Siino, 2024)</td><td>1000</td></tr><tr><td>Emoji-cogen</td><td>Wang et al. (Wang et al., 2024a)</td><td>500</td></tr><tr><td>Typo-cogen</td><td>Wang et al. (Wang et al., 2024a)</td><td>500</td></tr><tr><td>ICL</td><td>Wang et al. (Wang et al., 2024a)</td><td>500</td></tr><tr><td>Prompt Paraphrasing</td><td>Wang et al. (Wang et al., 2024a)</td><td>500</td></tr><tr><td>CSGen</td><td>Wang et al. (Wang et al., 2024a)</td><td>500</td></tr><tr><td>Adversarial Attack</td><td>Semeval Task8 (Siino, 2024)</td><td>500</td></tr></table>

Table 5: Experimental scenarios and corresponding datasets.

## A.2 Implementation

GREATER is deployed on a server equipped with 4 NVIDIA A100 GPUs, running on Ubuntu 22.04. The adversarial framework uses the xlm-robertabase model (279M) as its base detector. For the evaluation of Mixed Edit Attack, Paraphrasing Attack, Code-switching Attack, and Human Obfuscation, we adopt the concept of "budget" to control the intensity of the attacks for a more fine-grained investigation of model robustness, following the methodology in Wang et al. (2024a). Specifically, Mixed Edit Attack uses character edit distance (Levenshtein, 1966) as the budget, Paraphrasing Attack and Code-switching Attack utilize BERTScore (Zhang et al., 2019) as the budget, and Human Obfuscation Attack employs the confusion ratio as the budget.

For all defense methods, we use xlm-roberta-base as the base detector. The sizes of the training set, validation set, and test set are 8000, 1000, and 1000, respectively. The learning rate is set to 1e-5, and the number of epochs is

fixed at 6.

For all data augmentation-based methods, we use 20% of MGT for data augmentation. If the method has hyperparameters, they are set according to the reference values in the original paper. For all adversarial training-based methods, we use 20% of MGT for adversarial training. If the method has hyperparameters, they are set according to the reference values in the original paper. For our method, our detector is trained using a label smoothing loss function with a smoothing factor of α. We use a trained RoBERTa Large as the Surrogate Model $\mathcal { M } _ { \mathrm { s u r } } ( . )$ due to its strong generalization ability and precise understanding of English text. Our method’s selected hyperparameters are shown in Table 6.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Weight Parameter λ</td><td>0.05</td></tr><tr><td>Scaling Factor €</td><td>0.3</td></tr><tr><td>Scaling Factor ξ</td><td>0.01</td></tr><tr><td>Lower Bound of the Uniform Distribution a Upper Bound of the Uniform Distribution b</td><td>0.5</td></tr><tr><td></td><td>-0.5</td></tr><tr><td>Batch Size M</td><td>50</td></tr><tr><td>Epoch N</td><td>6</td></tr><tr><td>Label Smoothing Factor α</td><td>0.1</td></tr></table>

Table 6: Hyperparameters for our GREATER.

## A.3 Evaluation Metrics

We use attack effectiveness metrics and text quality metrics to comprehensively evaluate the performance of defense and attack methods.

1) Attack Effectiveness Metrics

Attack Success Rates (ASR). The Attack Success Rate (ASR) measures the proportion of successful attacks relative to the total number of attempted attacks. Note that we only attack text that was detected as machine-written before the attack. ASR is calculated as follows:

$$
\mathrm { A S R } = { \frac { \mathrm { T e x t ~ d e t e c t e d ~ a s ~ H W T ~ a f t e r ~ a t t a c k } } { \mathrm { T e x t ~ d e t e c t e d ~ a s ~ M G T ~ b e f o r e ~ a t t a c k } } } .
$$

For detector, a lower ASR indicates better defense performance, while for adversary, a higher ASR signifies a stronger attack effectiveness.

2) Text Quality Metrics

Perturbation Rate (Pert.). Pert. measures the lexical difference between the adversarial text and the original text. It is defined as the ratio of the number of perturbed tokens to the total number of tokens in the text. A lower perturbation rate indicates that the adversarial text remains more similar to the original text.

Perplexity Variation (∆PPL). ∆PPL measures the change of perplexity, which represents the consistency and fluency of the adversarial examples. The PPL is calculated as:

$$
\mathrm { P P L } = e x p ( - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log P ( w _ { i } | w _ { 1 } , w _ { 2 } , \dots , w _ { i - 1 } ) ) ,
$$

where N is the total number of tokens, $w _ { i }$ represents the i-th token, and $P ( w _ { i } | w _ { 1 } , w _ { 2 } , \dots , w _ { i - 1 } )$ is the probability assigned by the language model. Generally, a lower PPL variation indicates that the quality of the adversarial text is closer to that of the original text. We use GPT-2 (Radford et al., 2019) to compute PPL.

Universal Sentence Encoder (USE) score(Cer et al., 2018). USE evaluates the semantic similarity between the adversarial example and the original text. The USE score is computed as:

$$
\operatorname { U S E } \operatorname { S c o r e } = { \frac { \mathbf { E } _ { \mathrm { o r i g } } \cdot \mathbf { E } _ { \mathrm { a d v } } } { \| \mathbf { E } _ { \mathrm { o r i g } } \| \| \mathbf { E } _ { \mathrm { a d v } } \| } } ,
$$

where $\mathbf { E } _ { \mathrm { o r i g } }$ and $\mathbf { E } _ { \mathrm { a d v } }$ are the sentence embeddings of the original and adversarial texts, respectively, generated by the USE model, and E denotes the Euclidean norm. A higher USE score indicates greater semantic similarity. We use USE (Universal Sentence Encoder) to calculate USE score.

Flesch Reading Ease score (∆r) (Flesch, 1948). $\Delta r$ measures the variation in text readability. The Flesch Reading Ease score is calculated as:

$$
r = 2 0 6 . 8 3 5 \mathrm { - } 1 . 0 1 5 \times \frac { N _ { \mathrm { w o r d s } } } { N _ { \mathrm { s e n t e n c e s } } } \mathrm { - } 8 4 . 6 \times \frac { N _ { \mathrm { s y l l a b l e s } } } { N _ { \mathrm { w o r d s } } } ,
$$

where $N _ { \mathrm { w o r d s } }$ is the total number of words in the text, $N _ { \mathrm { s e n t e n c e s } }$ is the total number of sentences, and $N _ { \mathrm { s y l l a b l e s } }$ is the total number of syllables. A smaller $\Delta r$ indicates less change in readability.

## A.4 Experimental Comparison Methods

This section provides a detailed introduction to the SOTA methods included in our comparisons.

## A.4.1 Defense Methods

Edit Pretraining (EP) (Wang et al., 2024c): An adversarial data augmentation method that blends Mixed Edit Attack into the training set.

Paraphrasing Pretraining (PP) (Wang et al., 2024c): An adversarial data augmentation method that blends Paraphrasing Attack into the training set.

CERT-ED (Huang et al., 2024): An adversarial data augmentation method that adapts randomized deletion to effectively safeguard natural language classification models from diverse edit-based adversarial operations, including synonym substitution, insertion, and deletion.

RanMask (Zeng et al., 2023): An adversarial data augmentation method that randomly masks a proportion of words in the input text, thereby mitigating both word- and character-level adversarial perturbations without assuming prior knowledge of the adversaries’ synonym generation.

Text-RS (Zhang et al., 2024b): An adversarial data augmentation method that treats discrete word substitutions as continuous perturbations in the embedding space to reduce the complexity of searching through large discrete vocabularies and bolster the model’s robustness.

Text-CRS (Zhang et al., 2024a): An adversarial data augmentation method that is built on randomized smoothing, encompassing various word-level adversarial manipulations—such as synonym substitution, insertion, deletion, and reordering—by modeling them in both embedding and permutation spaces.

Virtual Adversarial Training (VAT) (Miyato et al., 2016): An Adversarial Training method that defines the adversarial direction without label information and employs the robustness of the conditional label distribution around each data point against local perturbation as the adversarial loss.

Token Aware Virtual Adversarial Training (TAVAT) (Li et al., 2021): An Adversarial Training method that uses token-level accumulated perturbation to better initialize the noise and applies token-level normalization.

RADAR (Hu et al., 2023a): An Adversarial Training method that uses a paraphraser (such as DIP-PER) as adversary and a fine-tuned model as detector. The adversary and detector learn from each other and improve the robustness of the detector when facing paraphrasing attack.

OUTFOX (Koike et al., 2024): An adversarial training method that iteratively improves a detector by generating in-context adversarial examples via a co-evolving attacker. The attacker crafts examples to evade the detector, which are then used to strengthen the detector through in-context learning.

## A.4.2 Attack Methods

TextFooler (Jin et al., 2020): A black-box attack method targeting text classification and natural language inference tasks. It ranks words by their importance to the model’s prediction and replaces them with semantically similar and grammatically correct synonyms to generate adversarial examples while preserving sentence meaning and structure.

BERTAttack (Li et al., 2020): A black-box attack method that utilizes a pre-trained BERT model to generate adversarial examples. It replaces certain words in the target sentence with high-probability candidates predicted by BERT, ensuring the adversarial examples remain semantically and grammatically similar to the original while misleading the model.

PWWS (Ren et al., 2019): A black-box attack method designed for text classification models. It computes the saliency of each word in the sentence, ranks them accordingly, and replaces high-saliency words with synonyms from WordNet, generating adversarial examples that mislead the model while preserving meaning.

A2T (Yoo and Qi, 2021): A white-box attack method aimed at improving adversarial training. It leverages gradient-based word importance estimation, performing a greedy search from least to most important words, and uses word embedding models or masked language models to generate candidate replacements, ensuring adversarial examples remain semantically similar while misleading the model.

HQA (Liu et al., 2024a): A black-box adversarial attack method for text classification task. It initializes adversarial examples by minimizing perturbations and iteratively substitutes words using synonym sets to optimize both semantic similarity and adversarial effectiveness while reducing query consumption.

ABP (Yu et al., 2024): A black-box adversarial attack method leveraging prior knowledge to guide word substitutions efficiently. It introduces Adversarial Boosting Preference (ABP) to rank word importance and proposes two query-efficient strategies: a query-free attack (ABPfree) and a guided search attack (ABPguide), significantly reducing query numbers while maintaining high attack success rates.

T-PGD (Yuan et al., 2023): A black-box zero-query adversarial attack method extending optimization-based attack techniques from computer vision to NLP. It applies perturbations to the embedding layer and amplifies them through forward propagation, then uses a masked language model to decode adversarial examples.

FastTextDodger (Hu et al., 2024): A black-box adversarial attack designed for query-efficient adversarial text generation. It generates grammatically correct adversarial texts while maintaining strong attack effectiveness with minimal query consumption.

## A.5 Detailed Information of Text Perturbation Method

In this section, we introduce 10 Text Perturbation Methods mentioned in Table 1. Detailed information of Adversarial Attack Methods can be found in Appendix A.4.2.

Mixed Edit (Wang et al., 2024a): A text modification strategy combining homograph substitution, formatting edits, and case conversion to evade detection. It manipulates character representation, encoding, and capitalization to introduce imperceptible variations while preserving readability. Homograph Substitution exploits visually similar graphemes, characters, or glyphs with different meanings for imperceptible text modifications. We use VIPER (Eger et al., 2019) and Easy Character Embedding Space (ECES) to obtain optimal homoglyph alternatives. Formatting Edits introduces human-invisible disruptions using special escape characters and format-control Unicode symbols to evade detection. We employ newline (\n), carriage return (\r), vertical tab (\v), zero-width space (\u200B), and line tabulation (\u000B) to fragment text at the encoding level while preserving visual coherence. Case Conversion alters letter capitalization within a word by converting uppercase letters to lowercase and vice versa. For example, transforming PaSsWoRd into pAsSwOrD disrupts case-sensitive detection while preserving readability.

HMGC (Zhou et al., 2024): A black-box zeroquery attack framework. It uses a surrogate model to approximate the detector, ranks words by gradient sensitivity and PPL, and replaces highimportance words via an encoder-based masked language model. Constraints ensure fluency and semantic consistency, while dynamic adversarial learning refines the attack strategy.

Paraphrasing (Wang et al., 2024a): A paragraphlevel attack that reorganizes sentence composition to hinder detection. It utilizes Dipper (Krishna et al., 2024) to reorder, merge, and split multiple sentences, increasing textual variance while preserving meaning.

Code-Switching (Winata et al., 2023): A linguistic modification strategy that substitutes words with their synonyms in different languages. It includes a model-free (MF) approach using a static dictionary (Zhang et al., 2020a; Tiedemann, 2012) for replacements in German, Arabic, or Russian, and a model-required (MR) approach employing the Helsinki-NLP (Helsinki-NLP, 2020) model to translate selected words.

Human Obfuscation: A semantic alteration technique inspired by Semeval Task8 (Siino, 2024), where the initial segment of MGT is replaced with an equally long HWT. The confusion ratio measures the extent of content substitution to increase ambiguity.

Emoji-Cogen (Wang et al., 2024a): A cogeneration attack method that inserts emojis into text generation to perturb the output. Emojis are introduced immediately after a token is sampled, before generating the next token, and are removed post-generation, ensuring natural readability while confusing automated detectors.

Typo-Cogen (Wang et al., 2024a): A cogeneration attack method that introduces typos during text generation to manipulate lexical structure. Artificial typos are injected into the generated text and subsequently corrected post-generation, preserving overall coherence while disrupting detection models.

In-Context Learning (ICL) (Wang et al., 2024a): A prompt attack method designed to produce human-like outputs that evade detection. It provides the generator with a related HWT as a positive example and a vanilla MGT as a negative example, guiding the model to generate more natural and deceptive text.

Prompt Paraphrasing (Wang et al., 2024a): A prompt attack method rewriting technique that enhances textual variation while maintaining semantic integrity. It utilizes the Pegasus paraphraser (Zhang et al., 2020b) to restructure input prompts.

Character-Substituted Generation (CS-Gen) (Wang et al., 2024a): A prompt attack method that incorporates character substitution strategies within the prompt. The prompt explicitly specifies replacement rules, such as substituting all occurrences of ‘e’ with ‘x’ during generation. For example, given the prompt: “Continue 20 words with all ‘e’s substituted with ‘x’s and all ‘x’s substituted with ‘e’s: The evening breeze carried a gentle melody. . . ”, the model generates text following these constraints. A post-processing step then restores the original characters, ensuring a natural final output.

## B Experiment on Defense

## B.1 Defense Performance under Different Attack Strengths

We evaluate the resistance of defense methods to increasing attack strengths. For text perturbation strategies, we employ four text perturbation strategies: Mixed Edit, Paraphrasing, Code-Switching, and Human Obfuscation in the experiment. Following Wang et al. (2024a), we use Character Edit Distance, BERT Score, BERT Score, and Confusion Ratio as the measure of attack strength for the methods mentioned above, respectively. For adversarial attacks, we choose PWWS TextFooler, BERTAttack, and A2T to attack MGT detectors, and we utilize max query count to quantify the attack strength. In the implementation, we change the limit on the attack strength measures to vary the attack intensity. We show the experimental results in Figure 4.

Our experimental results demonstrate that as the attack strength increases, the ASR on our GREATER-D remains consistently close to zero, whereas other defense methods exhibit significant vulnerabilities under more intensive adversarial attacks. However, it is worth noticing that under Mixed Edit perturbation, GREATER-D performs second-best after EP. This is because EP is originally trained on Mix Edit perturbation and obtains stronger defense against it. The consistent effective defense against varying attack strengths proves the steadiness of our defense method.

## B.2 Impact of Synchronous Update of GREATER-D

The synchronous update mechanism plays a crucial role in enhancing the robustness of the target detector. To explicitly demonstrate its contribution, we ablate the generator-side updates and instead use static texts to update the GREATER-D detector. The results are presented in Table 7.

As shown in Table 7, the robustness of the GREATER-D detector significantly degrades when it is trained with static adversarial examples. The ASR converges to approximately 34% after around

![](images/b3592c34da33440d36c98c7eacbf1a40e837bd84d90c40df3af6571281847f84.jpg)  
Figure 4: Defense performance under attack with different strengths. A lower ASR(%) indicates better defensive performance. A larger character edit distance indicates greater attack intensity in the Mixed Edit Attack. A lower BERT score corresponds to stronger attacks in the Paraphrasing Attack and Code-Switching Attack. A higher obfuscation ratio reflects greater intensity in the Human Obfuscation Attack. Similarly, for PWWS, BERTAttack, TextFooler, and A2T, a larger maximum query count signifies a stronger attack.

<table><tr><td>Epoch</td><td>GREATER-D (w/o synchronous update)</td><td>GREATER-D (w/ synchronous update)</td></tr><tr><td>1</td><td>59.04</td><td>57.64</td></tr><tr><td>2</td><td>52.17</td><td>46.08</td></tr><tr><td>3</td><td>45.52</td><td>21.85</td></tr><tr><td>4</td><td>40.17</td><td>10.28</td></tr><tr><td>5</td><td>34.52</td><td>4.77</td></tr><tr><td>6</td><td>34.08</td><td>3.45</td></tr></table>

Table 7: Defense performance in ASR (%) of GREATER-D against paraphrasing attacks across different training epochs.

5 epochs, which is substantially worse than the result achieved with dynamic updates. This demonstrates the critical role of the dynamic update mechanism in enhancing model robustness.

## B.3 Defense Performance of Different Epochs

To demonstrate the effectiveness of the adversarial training procedure, we analyze the GREATER-A and GREATER-D performance after each training round. The detailed analysis of training dynamics is in the Table 8. We present the ASR of GREATER-A and defense performance of GREATER-D against Paraphrasing Attack at each epoch from 1 to 6. We find that as training progresses, the GREATER-A becomes stronger at attacking XLM-Roberta detector. Moreover, GREATER-D exhibits more robust defense per-

formance on the attack it is not trained with.
<table><tr><td>Epoch</td><td>GREATER-A ASR ↑</td><td>GREATER-D ASR ↓</td></tr><tr><td>1</td><td>77.04</td><td>57.64</td></tr><tr><td>2</td><td>79.03</td><td>46.08</td></tr><tr><td>3</td><td>84.61</td><td>21.85</td></tr><tr><td>4</td><td>88.94</td><td>10.28</td></tr><tr><td>5</td><td>94.06</td><td>4.77</td></tr><tr><td>6</td><td>96.58</td><td>3.45</td></tr></table>

Table 8: ASR (%) of the Adversary against F.t.XLM-RoBERTa-Base detector and the Detector against Paraphrasing Attack across different epochs.

## B.4 Experiments on Multilingual data

We evaluate the multilingual performance of different defense methods using German and Urdu datasets (Siino, 2024), and the results are shown in Table 9. As observed from the table, our model achieves strong performance on both languages and yields the best overall results, indicating its effectiveness in multilingual detection scenarios.

## C Evaluations on GREATER-A Performance

## C.1 Attack on the real-world detector

Speaking of the fact that the adversary and target model share the same training dataset (but different sets), it is necessary to evaluate if the attack conducted by GREATER-A would generalize to other detectors which trained on different datasets. To validate this, we attack a close-sourced commercial detector $\mathbf { G P T Z e r o } ^ { 3 }$ with GREATER-A, which is a strict black-box setting because no other information except for the input and output of the target model is available. The results are in the Table 10. It demonstrates that our GREATER-A remains high ASR compared with other SOTA methods, though we are not aware of the architecture, parameters, or training data at all.

<table><tr><td>Source</td><td>Metric</td><td>F.t.XLM-RoBERTa</td><td>EP</td><td>PP</td><td>VAT</td><td>TAVAT</td><td>CERT-ED</td><td>RanMask</td><td>Text-RS</td><td>Text-CRS</td><td>GREATER-D</td></tr><tr><td rowspan="2">German</td><td>Acc</td><td>68.30</td><td>59.84</td><td>73.06</td><td>67.44</td><td>59.37</td><td>55.60</td><td>61.13</td><td>65.04</td><td>56.80</td><td>72.86</td></tr><tr><td>Fl</td><td>75.36</td><td>71.13</td><td>78.40</td><td>75.30</td><td>71.08</td><td>68.90</td><td>71.40</td><td>73.51</td><td>69.55</td><td>77.76</td></tr><tr><td rowspan="2">Urdu</td><td>Acc</td><td>56.01</td><td>55.09</td><td>56.88</td><td>63.33</td><td>59.04</td><td>66.60</td><td>64.03</td><td>70.62</td><td>70.21</td><td>66.24</td></tr><tr><td>Fl</td><td>65.27</td><td>69.14</td><td>69.85</td><td>69.07</td><td>68.56</td><td>74.97</td><td>69.81</td><td>76.10</td><td>76.21</td><td>74.89</td></tr><tr><td rowspan="2">Overall</td><td>Acc</td><td>62.16</td><td>57.47</td><td>64.97</td><td>65.39</td><td>59.21</td><td>61.10</td><td>62.58</td><td>67.83</td><td>63.51</td><td>69.55</td></tr><tr><td>Fl</td><td>70.32</td><td>70.14</td><td>74.13</td><td>72.19</td><td>69.82</td><td>71.94</td><td>70.61</td><td>74.81</td><td>72.88</td><td>76.33</td></tr></table>

Table 9: Performance of different defense methods in a multilingual setting. The best result in each group is highlighted with a green background.

<table><tr><td>Method</td><td>Avg Queries ↓</td><td>ASR (%) ↑</td><td>Pert. (%) ↓</td><td>∆PPL ↓</td><td>USE ↑</td><td>∆r ↓</td></tr><tr><td>HQA</td><td>172.38</td><td>63</td><td>11.23</td><td>133.58</td><td>0.8312</td><td>66.24</td></tr><tr><td>FastTextDodger</td><td>162.58</td><td>70</td><td>10.58</td><td>96.25</td><td>0.9096</td><td>42.15</td></tr><tr><td>ABP</td><td>133.46</td><td>82</td><td>9.13</td><td>72.78</td><td>0.8613</td><td>36.18</td></tr><tr><td>T-PGD</td><td>189.36</td><td>42</td><td>13.12</td><td>168.54</td><td>0.8077</td><td>60.51</td></tr><tr><td>GREATER-A (ours)</td><td>42.57</td><td>100</td><td>7.82</td><td>67.13</td><td>0.9393</td><td>13.39</td></tr></table>

Table 10: The attack results of different attack methods on the GPTZero detector. For each sample, the maximum number of queries is limited to 200. The best result in each group is highlighted with a green background.

## C.2 Generalization to Other Detector

To demonstrate the generalization of our GREATER-A, we conduct additional experiments with two zero-shot detectors: Fast-DetectGPT (Bao et al., 2023) and Binoculars (Hans et al., 2024). Moreover, to further demonstrate the generalization ability of GREATER-A, we also test it on a decoderonly detector F.t.GPT2 (Radford et al., 2019) which shares a different architecture with XLM-RoBERTa. As shown in the Table 11, GREATER-A outperforms all comparison methods when attacking zero-shot and decoder-only detectors, which demonstrates the universal effectiveness of GREATER-A.

<table><tr><td>Method</td><td>Fast-DetectGPT</td><td>Binoculars</td><td>F.t.GPT2</td></tr><tr><td>WordNet</td><td>51.36</td><td>37.35</td><td>42.66</td></tr><tr><td>Back Translation</td><td>37.50</td><td>19.13</td><td>1.26</td></tr><tr><td>Rewrite</td><td>10.97</td><td>8.88</td><td>30.02</td></tr><tr><td>T-PGD</td><td>56.83</td><td>41.91</td><td>70.46</td></tr><tr><td>HMGC</td><td>17.39</td><td>31.43</td><td>35.08</td></tr><tr><td>GREATER-A (ours)</td><td>61.14</td><td>65.83</td><td>74.25</td></tr></table>

Table 11: The ASR (%) across different detectors under different attack methods. The best result in each group is highlighted with a green background.

<table><tr><td>Method</td><td>Avg Queries ↓</td><td>ASR (%) ↑</td><td>Pert. (%) ↓</td><td>∆PPL ↓</td><td>USE ↑</td><td>∆r↓</td></tr><tr><td>R+NP</td><td>26.61</td><td>75.77</td><td>11.56</td><td>106.29</td><td>0.9324</td><td>14.41</td></tr><tr><td>R+P</td><td>53.22</td><td>75.77</td><td>9.51</td><td>58.72</td><td>0.9497</td><td>12.13</td></tr><tr><td>S+NP</td><td>31.32</td><td>96.58</td><td>11.28</td><td>85.18</td><td>0.9136</td><td>21.80</td></tr><tr><td>Mask-T</td><td>303.82</td><td>96.38</td><td>8.33</td><td>27.12</td><td>0.9696</td><td>4.73</td></tr><tr><td>GREATER-W</td><td>33.21</td><td>96.38</td><td>11.49</td><td>65.16</td><td>0.9482</td><td>10.04</td></tr><tr><td>GREATER-WordNet</td><td>28.41</td><td>80.89</td><td>16.68</td><td>53.82</td><td>0.9199</td><td>13.21</td></tr><tr><td>GREATER-A</td><td>62.63</td><td>96.58</td><td>7.26</td><td>35.22</td><td>0.9506</td><td>9.21</td></tr></table>

Table 12: Ablation study on the GREATER-A. The best result in each group is highlighted with a green background.

## D Ablation Study

To demonstrate the effectiveness of each component in the design of GREATER-D and GREATER-A, we conduct ablation experiments. The ablation models are as follows:

R+NP: Randomly select tokens to perturb and not apply pruning.

R+P: Randomly select tokens to perturb and apply pruning.

S+NP: Select tokens to perturb with the important token identification module but not apply pruning. Mask-T: Select tokens to perturb with the important token identification but mask them instead of adding perturbation.

GREATER-W: Select words to perturb with the important token identification module instead of tokens.

GREATER-WordNet: Select words to perturb with the important token identification module and substitute them with synonyms.

As shown in Table 12, the original GREATER-A exhibits the most balanced and effective performance in generating adversarial examples. Comparing R+P to GREATER-A, we find the ASR drops by 20.81, indicates that identifying important tokens is of great significance for the attack to be successful. Moreover, greedy pruning is important for maintaining the text quality of adversarial examples. S+NP achieves the same ASR with GREATER-A but is left far behind in terms of PPL, USE, and $\Delta \mathbf { r } .$ The perturbation strategy also plays a crucial role in the adversarial attack. Masking the important token instead of perturbing the embedding greatly increases the number of queries. Applying perturbation on the word level or substituting important words with synonyms also degrades the performance of the adversary.

## E Mathematical Analysis of Perturbation Rate and Query Complexity

In this section, we provide the theoretical analysis of our proposed adversarial example generation framework, focusing on two crucial metrics: (i) the perturbation rate, which characterizes the fraction of modified tokens in an adversarial example; and (ii) the query complexity, which measures the number of queries made to the target detector in a black-box setting. We establish strict upper and lower bounds on both metrics, illustrating the efficiency and effectiveness of our method.

## E.1 Perturbation Rate Analysis

Definition 1. Let Z denote the maximum number oftokens allowed to be modified. Let P be the total number of tokens (out of T) perturbed by greedy search, taking integer values in the interval [1, Z], where $1 \leq P \leq Z $ . Let Y be thefraction ofthose P tokens retained (not reverted) by greedy pruning, taking real values in [0, 1]. Formally,

$$
P \in \{ 1 , 2 , \ldots , Z \} , \qquad Y \in [ 0 , 1 ] .
$$

We assume P and Y exhibit weak dependence, meaning they have a small but nonzero covariance $\operatorname { C o v } ( P , Y )$

Definition 2. Let P  Y be the expected count of tokens that remain modified. The perturbation rate $\rho$ is defined as

$$
\rho = \frac { P \cdot Y } { T } ,
$$

where $T$ is the length of the original text.

Theorem 1. Let P and Y modeled as truncated normal random variables on [1, Z] and [0, 1], respectively. Then under mild assumptions on the truncation intervals, we have

$$
\mathbb { E } [ P ] \approx { \frac { Z + 1 } { 2 } } a n d \mathbb { E } [ Y ] \approx { \frac { 1 } { 2 } } .
$$

Proof. (1) Truncated Normal for $P$ and $Y .$ . In adversarial text attacks, $P$ often emerges from an aggregation of (approximately) Bernoulli decisions: each of the $T$ tokens has a non-negligible probability of being deemed “important,” subject to a global cap of $Z .$ By the Central Limit Theorem (CLT), this sum is close to normally distributed with mean $\mu _ { P }$ and variance $\sigma _ { P } ^ { 2 }$ . Because $P$ cannot exceed Z (and must be at least 1 to induce misclassification), we say

$$
P \sim \mathcal { N } _ { t } ( \mu _ { P } , \sigma _ { P } ^ { 2 } ; 1 , Z ) ,
$$

where $\mathcal { N } _ { t } ( \cdot )$ denotes a normal distribution truncated to the integer range [1, Z]. Analogously, once $P$ tokens are selected, greedy pruning decides to keep or revert each token, again creating a sum of i.i.d. Bernoulli-like indicators. Dividing by $P$ yields

$$
\begin{array} { l } { { Y ~ = ~ { \frac { \mathrm { n u m b e r ~ o f ~ r e t a i n e d ~ t o k e n s } } { P } } } } \\ { { ~ \sim ~ \mathcal { N } _ { t } ( \mu _ { Y } , ~ \sigma _ { Y } ^ { 2 } ; ~ 0 , ~ 1 ) , } } \end{array}
$$

a (truncated) normal over [0, 1].

(2) Standard Formulas for Truncated Normal Means. From truncated normal theory, if X ${ \mathcal { N } } ( \mu , \sigma ^ { 2 } )$ is restricted to $[ a , b ]$ , then its mean is

$$
\operatorname { \mathbb { E } } [ X ] = \mu + \sigma { \frac { \phi \left( { \frac { a - \mu } { \sigma } } \right) - \phi \left( { \frac { b - \mu } { \sigma } } \right) } { \Phi \left( { \frac { b - \mu } { \sigma } } \right) - \Phi \left( { \frac { a - \mu } { \sigma } } \right) } } ,
$$

where $\phi$ and $\Phi$ are the standard normal PDF and CDF, respectively. Thus, for $P \sim$ $\mathcal { N } _ { t } ( \mu _ { P } , \sigma _ { P } ^ { 2 } ; 1 , \bar { Z } ) , \mathbb { E } [ P ]$ depends on how far $\mu _ { P }$ is from the boundaries $\{ 1 , Z \}$ . Similarly, E[Y ] depends on truncation at [0, 1].

(3) Approximate Symmetry in Practice. For symmetric truncated normal distributions $P$ and $Y .$ , it is evident that the following expectations hold:

$$
\mu _ { P } = \frac { Z + 1 } { 2 } , \mu _ { Y } = \frac { 1 } { 2 } .
$$

So long as $\mu { } _ { P }$ and $\sigma _ { P }$ ensure that $\frac { 1 - \mu _ { P } } { \sigma _ { P } }$ and $\displaystyle \frac { Z - \mu _ { P } } { \sigma _ { P } }$ are not extreme, the truncation does not drastically

shift the mean from $\mu _ { P } .$ . Concretely, if $Z$ is sufficiently large and $\begin{array} { r } { \mu _ { P } \approx \frac { Z + 1 } { 2 } } \end{array}$ , then

$$
\mathbb { E } [ P ] \approx \mu _ { P } \approx \frac { Z + 1 } { 2 } ,
$$

Likewise, if $\textstyle \mu _ { Y } \approx { \frac { 1 } { 2 } }$ and $\sigma _ { Y }$ is moderate, $\mathbb { E } [ Y ]$ ≈ $\frac { 1 } { 2 }$ despite the boundaries [0, 1].

Combining (1), (2), and (3), Theorem 1 is proved. □

Theorem 2. Let T be the length of the original text and M be the maximum number oftokens that can be perturbed. Then, the perturbation rate $\rho$ satisfies $\begin{array} { r } { \frac { 1 } { T } \leq \rho \leq \frac { Z } { T } . } \end{array}$ , and its expected value is approximately $\begin{array} { r } { \mathbb E [ \rho ] \approx \frac { Z } { 4 T } } \end{array}$

Proof. (1) Basic Bounds. Since $P$ is at least 1 and at most M, and $Y$ is at least 0 and at most 1, the product P  Y satisfies

$$
\begin{array} { r l r } { 1 \leq P \leq Z , } & { 0 \leq Y \leq 1 } \\ { } & { } & { } \\ { \implies } & { 1 \leq P \cdot Y \leq Z . } \end{array}
$$

Dividing by T yields the strict bounds

$$
{ \frac { 1 } { T } } \leq { \frac { P \cdot Y } { T } } \leq { \frac { Z } { T } } .
$$

The lower limit $1 / T$ reflects that at least one token must change to induce a misclassification; the upper limit $\textstyle { \frac { Z } { T } }$ follows from the maximal $Z$ token modifications.

(2) Expected Product with Weak Dependence. By definition, the covariance between P and $Y$ is

$$
\operatorname { \mathbb { E } } [ P Y ] = \operatorname { \mathbb { E } } [ P ] \operatorname { \mathbb { E } } [ Y ] + \operatorname { C o v } ( P , Y ) .
$$

When the detector is relatively robust, greedy search skews $P$ toward larger values (close to $M )$ and pruning skews $Y$ toward retention (close to 1). In that case, E[ $P ]$ is large, $\mathbb { E } [ Y ]$ is near 1, and a small positive covariance implies

$$
\mathbb { E } [ P \cdot Y ] > \mathbb { E } [ P ] \mathbb { E } [ Y ] .
$$

However, the final mean number of changed tokens cannot exceed $M$ . Conversely, if the detector is weak, $\mathbb { E } [ P ]$ may approach 1, and $\mathbb { E } [ Y ]$ might be relatively modest, implying

$$
\mathbb { E } [ P Y ] < \mathbb { E } [ P ] \mathbb { E } [ Y ] + | { \mathrm { C o v } } ( P , Y ) | .
$$

In all cases, the weak dependence ensures $\operatorname { C o v } ( P , Y )$ is bounded such that

$$
1 \leq P \cdot Y \leq Z ,
$$

preventing the expected perturbation rate from falling below $\textstyle { \frac { 1 } { T } }$ or above $\frac { \dot { Z } } { T } .$

(3) Characteristic Mean $\textcircled { 4 T }$ . By Theorem 1, we have E[ $\textstyle P ] \approx { \frac { Z + 1 } { 2 } }$ and $\textstyle \mathbb { E } [ \dot { Y } ] \approx { \frac { 1 } { 2 } }$ . If $\operatorname { C o v } ( P , Y )$ is small or near zero, then

$$
{ \begin{array} { r l } & { \mathbb { E } [ P \cdot Y ] = \mathbb { E } [ P ] \mathbb { E } [ Y ] + \operatorname { C o v } ( P , Y ) } \\ & { } \\ & { \approx { \frac { Z + 1 } { 2 } } \times { \frac { 1 } { 2 } } = { \frac { Z + 1 } { 4 } } . } \end{array} }
$$

For large $\begin{array} { r } { Z , \frac { Z + 1 } { 4 T } \approx \frac { Z } { 4 T } } \end{array}$ . If $\operatorname { C o v } ( P , Y )$ modestly raises or lowers this sum, the final mean $\mathbb { E } [ P Y ]$ still cannot breach the fundamental $[ 1 , Z ]$ interval. This shows that $\frac { Z } { 4 T }$ is a natural approximate pivot for the average perturbation rate under moderate parameters.

In Section 5.2, we set $Z = 0 . 3 T$ . If the number of iterations exceeds this upper bound, the attack is considered a failure. Therefore, the expected perturbation rate is

$$
\frac { 0 . 3 T } { 4 T } \approx 7 . 5 \% ,
$$

which is very close to the result obtained in the Section 5.2 (7.26%).

Combining (1), (2), and (3), Theorem 2 is proved. □

Note that if the detector forces greedy search to repeatedly fail early (producing a right-skewed $P -$ distribution concentrated near $Z )$ and greedy prun-$i n g$ is left-skewed (so that $\mathbb { E } [ Y ]$ is close to 1), then the mean $\mathbb { E } [ P \cdot Y ]$ exceeds $\frac { Z \dot { + } 1 } { 4 }$ , but still cannot exceed $Z .$ Conversely, if greedy search finds success quickly, giving a left-skewed P-distribution near 1, then $\mathbb { E } [ P ]$ might drop below $\textstyle { \frac { Z + 1 } { 2 } }$ but never below 1; similarly, if pruning is so aggressive that $\mathbb { E } [ Y ]$ is significantly below ${ \frac { 1 } { 2 } } ;$ , the mean also decreases. Consequently, the expected perturbation rate remains strictly within the $\textstyle \left[ { \frac { 1 } { T } } , \sum _ { T } \right]$ interval, but its exact value depends on the interplay of these skewed distributions. For moderate skewness on both $P$ and $Y , { \frac { Z } { 4 T } }$ is a characteristic outcome.

## E.2 Query Complexity Analysis

Definition 3. Let $Q _ { \mathrm { G } }$ be the total number of queries made by greedy search and $Q _ { \mathrm { P } }$ be the total number made by greedy pruning. Define the overall query complexity:

$$
Q = Q _ { \mathrm { G } } + Q _ { \mathrm { P } } .
$$

Theorem 3. Let $T$ be the length of the text, and let $Z = 0 . 3 T$ be the maximum iteration countfor both greedy search and greedy pruning. Then the total number ofqueries $Q$ used to construct one adversarial example lies within $1 \leq Q \leq 2 Z ,$ , which implies $Q = O ( T )$ and $Q = \Omega ( 1 )$ . Moreover, if the average perturbation rate is relatively low, then the number of required iterations typically shrinks, resulting in a correspondingly smaller $Q .$

Proof. (1) Basic Range of Queries. The minimal number of queries is 1, occurring if greedy search succeeds in its very first attempt, requiring no greedy pruning. Conversely, if both greedy search and greedy pruning use their maximum of $Z$ single-query iterations, we have

$$
Q = Q _ { \mathrm { G } } + Q _ { \mathrm { P } } \ \le \ Z + Z = \ 2 Z .
$$

Since $Z ~ = ~ 0 . 3 T , Q ~ \leq ~ 2 Z$ translates to $Q \ =$ $O ( T )$ . The trivial lower bound of 1 implies $Q =$ Ω(1).

(2) Perturbation Rate and Query Trade-Off. Let $\rho$ denote the fraction of tokens altered in a final adversarial example, as described in the perturbation rate analysis. A smaller $\rho$ typically indicates that the detector is fooled by changing fewer tokens, implying fewer iterative steps in greedy search. Thus, a lower $\rho$ correlates with fewer queries: once the necessary (small) set of tokens is found, misclassification is often achieved without exhausting all Z iterations. On the other hand, a higher $\rho$ suggests more alterations, potentially requiring more query rounds to finalize a successful attack.

(3) Expected Complexity under Moderate Perturbation. From the previous analysis, the average perturbation rate can hover around $\frac { Z } { 4 T }$ under typical conditions. This moderate $\rho$ implies that greedy search rarely needs the full $Z$ steps to identify the required modifications, and greedy pruning likewise terminates without iterating over all $Z$ . Consequently, the expected $Q$ is substantially below the worst-case $2 Z$ . Formally, we have

$$
\mathbb { E } [ Q _ { \mathrm { G } } ] < Z , \quad \mathbb { E } [ Q _ { \mathrm { P } } ] < Z ,
$$

$$
\implies \mathbb { E } [ Q ] = \mathbb { E } [ Q _ { \mathrm { G } } ] + \mathbb { E } [ Q _ { \mathrm { P } } ] < 2 Z .
$$

thus still preserving $O ( T )$ upper complexity while being strictly smaller on average.

Combining (1), (2), and (3), Theorem 3 is proved.

## F Adversarial Training Process

Algorithm 3 shows the detailed optimization process of GREATER-A and GREATER-D. The two components are updated alternatively in the same training step.

Algorithm 3 Adversarial Training Procedure   
1: Input: Training set $D _ { t r a i n } .$ , surrogate detector $\mathcal { M } _ { s u r } .$   
\*\*\*\* training phase begins \*\*\*\*   
2: Initialize: target detector $\mathcal { M } _ { t a r } .$ , importance scoring net  
work $\mathcal { F } _ { \theta }$ and epoch $ 0 .$   
3: while epoch $<$ epoc $_ { \ i _ { m a x } }$ do   
4: for each batch samples $\{ X _ { i } , c _ { i } \} _ { i = 0 } ^ { N }$ in $D _ { t r a i n }$ do   
5: $D _ { a d v } \gets \{ \}$   
6: for each MGT in $\{ X _ { i } , c _ { i } \} _ { i = 0 } ^ { N }$ do   
7: Obtain the last layer hidden state by Eq.(1).   
8: Obtain the importance score by Eq.(2).   
9: Construct the Important-token Set I by Eq.(3).   
10: Execute the Greedy Search Algorithm1.   
11: Execute the Greedy Pruning Algorithm2.   
12: Get the adversarial example:   
$D _ { a d v } \gets D _ { a d v } \cup \{ X _ { i } , c _ { i } \} _ { i = 0 } ^ { N } .$   
13: end for   
14: $D _ { a d v }  D _ { a d v } \cup \{ \tilde { X } _ { i } , c _ { i } \} .$   
15: Classify $D _ { a d v }$ via $\mathcal { M } _ { t a r }$ and obtain the output.   
16: Calculate loss $\mathcal { L } _ { \mathrm { A } }$ by Eq.(11).   
17: Calculate loss $\mathcal { L } _ { \mathrm { D } }$ by Eq.(12).   
18: Update $\mathcal { M } _ { s a r }$ and $\dot { \mathcal { F } } _ { \theta }$ via SGD (1951).   
19: end for   
20: end while   
21: Output: Trained detector $\mathcal { M } _ { t a r }$ and ${ \mathcal { F } } _ { \theta } .$

## G Case Study

Table 13 presents a case study of our adversary GREATER-A, in which our approach GREATER-A outperforms other SOTA methods regarding semantic preservation and reduction in perturbation rate.

<table><tr><td>Method</td><td>Text</td><td>Result</td><td>Pert. (%)</td><td>Queries</td></tr><tr><td>Original MGT</td><td>Suggested by the Scottish Parliamentary Constituencies Com- mission in 2003, and adopted at Holyrood on 1 May 2004. Area of Scotland&#x27;s 32nd largest council area - covering parts of East Renfrewshire Council Area</td><td>Machine-written </td><td></td><td></td></tr><tr><td>PWWS</td><td>Suggested by the Scottish Parliamentary Constituencies Com- mission in 2003, and adoptions at Holyrood on 1 Probability 2004. Area of Scotland&#x27;s 32nd highest council area - covering item of East Renfrewshire Council Area</td><td>Human-written 3 (Succeeded)</td><td>11.76</td><td>107</td></tr><tr><td>TextFooler</td><td>Suggested by the Scottish Parliamentary Constituencies Com- mission in 2003, and adopted at Holyrood on 1 May 2004. Area of Scotland&#x27;s 32nd largest council area - covering parts of East Renfrewshire Council Area</td><td>Machine-written (Failed) 88</td><td></td><td></td></tr><tr><td>BERTAttack</td><td>Suggested by the Scottish rural Constituencies Commission in 2003, and abolished at Holyrood on 1 May 2004. Area of ward 32nd most council constituency - almost all of East Renfrewshire Council Area</td><td>Human-written 3 (Succeeded)</td><td>20.59</td><td>60</td></tr><tr><td>A2T</td><td>Suggested by the Scottish Parliamentary Constituencies Com- mission in 2003, and adopted at Holyrood on 1 May 2004. Area of Scotland&#x27;s 32nd largest council area - covering parts of East Renfrewshire Council Area</td><td>Machine-written 88 (Failed)</td><td></td><td></td></tr><tr><td>FastTextDodger</td><td>Suggested by the Scottish Parliamentary Constituencies Com- mission in 2003: and adopted at Holyrood on 1 May 2004. Area of’&#x27;s 32nd largest council area - covering parts of East council Council</td><td>Human-written 3 (Succeeded)</td><td>14.71</td><td>73</td></tr><tr><td>ABP</td><td>Suggested aside the Scottish Parliamentary Constituencies Com- mission indium 2003, and adopted atomic number 85 Holyrood on ace Crataegus oxycantha 2004. Area of Scotland&#x27;s 32nd largest council area - covering parts of East Renfrewshire Coun- cil Area</td><td>Human-written 3 (Succeeded)</td><td>23.53</td><td>71</td></tr><tr><td>HQA</td><td>Suggested by the Scottish Parliamentary Constituencies Com- mission in 2003, and adopted at Holyrood on 1 May 2004. Area of Scotland&#x27;s 2004. sphere council area thirty-second prominent council of East Renfrewshire Council Area</td><td>Human-written 3 (Succeeded)</td><td>20.59</td><td>44</td></tr><tr><td>T-PGD</td><td>Introduced. by. Scottish Parliamentary Resituiances Commis- sion in 2002, and adopted at Holyrood on 1 May 2004. All of Scotland&#x27;s 32 The largest councils area - covering parts of East Renfrewshire Council Area</td><td>Human-written 3 (Succeeded)</td><td>26.47</td><td>56</td></tr><tr><td>GREATER-A</td><td>Suggested by the Grave Parliamentary Constituencies Commis- sion in 2003, and adopted at Holyrood on 1 May 2004. Area Scotland 32nd biggest council area - covering parts of East Renfrewshire Council Area</td><td>Human-written (Succeeded)</td><td>5.88</td><td>12</td></tr></table>

Table 13: Case study of semantic preservation of the adversarial texts generated by various attack methods. Words modified during the attacks are highlighted in red.