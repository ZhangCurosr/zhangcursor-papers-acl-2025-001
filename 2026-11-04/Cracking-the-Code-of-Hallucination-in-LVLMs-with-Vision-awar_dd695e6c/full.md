# Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence

Jinghan He<sup>1,2</sup>, Kuan Zhu<sup>1,2</sup>, Haiyun Guo<sup>1,2</sup>∗, Junfeng Fang<sup>3</sup>, Zhenglin Hua<sup>4</sup>, Yuheng Jia<sup>4</sup>, Ming Tang<sup>1</sup>, Tat-Seng Chua<sup>3</sup>, Jinqiao Wang<sup>1,2,5</sup>∗

<sup>1</sup>Foundation Model Research Center, Institute of Automation, Chinese Academy of Sciences

<sup>2</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences

<sup>3</sup>National University of Singapore <sup>4</sup>Southeast University <sup>5</sup>Wuhan AI Research hejinghan2022@ia.ac.cn, {kuan.zhu, haiyun.guo, jqwang}@nlpr.ia.ac.cn

## Abstract

Large vision-language models (LVLMs) have made substantial progress in integrating large language models (LLMs) with visual inputs, enabling advanced multimodal reasoning. Despite their success, a persistent challenge is hallucination—where generated text fails to accurately reflect visual content—undermining both accuracy and reliability. Existing meth ods focus on alignment training or decoding refinements but primarily address symptoms at the generation stage without probing the underlying causes. In this work, we investigate the internal mechanisms driving hallucination in LVLMs, with an emphasis on the multi-head attention module. Specifically, we introduce Vision-aware Head Divergence (VHD), a metric that quantifies the sensitivity of attention head outputs to visual context. Based on this, our findings reveal the presence of vision-aware attention heads that are more attuned to visual information; however, the model’s overreliance on its prior language patterns is closely related to hallucinations. Building on these insights, we propose Vision-aware Head Reinforcement (VHR), a training-free approach to mitigate hallucination by enhancing the role of visionaware attention heads. Extensive experiments demonstrate that our method achieves superior performance compared to state-of-the-art approaches in mitigating hallucinations, while maintaining high efficiency with negligible additional time overhead. The code is available at https://github.com/jinghan1he/VHR.

## 1 Introduction

Large vision-language models (LVLMs) (Dai et al., 2023; Liu et al., 2024b) represent a notable advancement in artificial intelligence by enabling large language models (LLMs) to understand visual inputs. However, LVLMs still face the challenge of hallucination (Rohrbach et al., 2018), where generated text does not accurately correspond to visual content. This misalignment can compromise the accuracy and reliability of LVLMs across a wide range of vision and language tasks, limiting their practical applications (You et al., 2024).

![](images/a0fbc7c6034e15d48086880e57c3f1a3bab82a53d88721557a00d0c1b86812e2.jpg)  
Figure 1: (a) An example indicating the connection between hallucination in LVLMs and language bias. When hallucination occurs (chat 1), we remove the image input and prompt the model to complete the description (chat 2). The output closely resembles the hallucinated content<sup>1</sup>. (b) The proposed VHD metric measures the sensitivity of the attention head outputs to image inputs, reflecting the degree of language bias. Hallucinated words generally correspond to lower T-VHD scores.

To mitigate this issue, several approaches incorporate additional information or models for alignment training (Zhao et al., 2023; Yu et al., 2024) or post-processing (Zhou et al., 2023; Yin et al., 2023), which incur higher training or inference costs. Recently, another line of research focuses on refining decoding strategies, employing methods like contrastive decoding (Leng et al., 2024; Zhu et al., 2024; Kim et al., 2024; Gong et al., 2024) or beam search (Huang et al., 2024) to adjust the logits distribution during inference. However, these approaches merely intervene at the output level to rectify hallucinations after they occur, without directly targeting and adjusting the internal mechanisms that drive hallucinations. This work aims to fill this research gap.

One of the factors contributing to hallucination in LVLMs is their tendency to prioritize language patterns (Ghosh et al., 2024; Parcalabescu and Frank, 2024), which can lead to the generation of fluent but inaccurate content. We further investigate this phenomenon and present an example in Figure 1. Specifically, when prompted to continue generating an image description, the model generates highly consistent outputs, irrespective of whether an image is provided. This problem may arise from biased language patterns in the training data, which are incorporated into the model’s parameters (Liu et al., 2024a), causing output to rely more on internal knowledge than image context. Yu et al., 2023 analyzed similar biases in language models and revealed that the multi-head attention module contains both in-context and memory heads. Manipulating these heads can influence whether the output is driven by contextual information or internal knowledge.

Building on previous findings, we are inspired to investigate the relationship between hallucination in LVLMs and the multi-head attention mechanism. To this end, we introduce a novel metric, Vision-aware Head Divergence (VHD), to quantify how the output of each attention head changes when the image context is removed in a generation step of LVLMs. Our analysis reveals that only a few heads show significant sensitivity to the image context, while the majority exhibit minimal variation. Based on this, we aggregate the VHD values from the most prominent attention heads in a generation step, resulting in the Token-VHD (T-VHD) metric. This metric allows us to evaluate the model’s reliance on visual content versus language priors when predicting each token, as illustrated in the bottom-right part of Figure 1. By examining the T-VHD scores, we observe that words and sentences associated with hallucinations generally correspond to lower values, further supporting the role of language bias in hallucination in LVLMs.

Leveraging the insights above, we propose Vision-aware Head Reinforcement (VHR), a training-free approach aimed at enhancing the model’s reliance on visual context rather than language priors. This method proactively mitigates hallucination in LVLMs by first identifying key attention heads based on their VHD scores and then amplifying their contributions during generation. Theoretical analysis demonstrates that this scalingup operation effectively re-orients the output of the multi-head attention module towards the reinforced head component, improving the alignment of the model’s output with visual context. Experiments on established LVLM hallucination benchmarks show that VHR outperforms existing decoding strategies, validating its effectiveness and efficiency in alleviating hallucinations.

Our main contributions can be summarized as follows:

• We propose the VHD metric to probe the attention heads in LVLMs for the language-bias tendency, and the T-VHD metric to analyze the relationship between language-biased generation and hallucination in LVLMs.

• We propose VHR, a training-free method that proactively mitigates hallucinations by adaptively identifying and reinforcing key attention heads during generation.

• Extensive experiments demonstrate that VHR outperforms existing decoding methods on widely-adopted hallucination benchmarks with negligible additional time cost.

## 2 Preliminary

LVLM generation. The LVLMs take both image and text as input. The image is encoded into vision tokens using an image encoder and projected to the text embedding space through a connector. These vision tokens $x _ { V }$ are then combined with tokenized text input $x _ { T }$ and passed into the LLM component for autoregressive generation:

$$
y _ { t } = \arg \operatorname* { m a x } p _ { \theta } ( y _ { t } | y _ { < t } , x _ { V } , x _ { T } ) ,\tag{1}
$$

where $y _ { < t }$ and $y _ { t }$ denote the earlier and the currently generated text tokens, respectively.

Multi-head attention. The multi-head attention mechanism is a core component of transformer models with each attention head performing the self-attention operation among tokens:

$$
\boldsymbol { A _ { l , i } } ( \boldsymbol { X _ { l , i } } ) = \mathrm { A t t e n t i o n } ( \boldsymbol { X _ { l , i } } \boldsymbol { W } _ { l , i } ^ { Q } , \boldsymbol { X _ { l , i } } \boldsymbol { W } _ { l , i } ^ { K } , \boldsymbol { X _ { l , i } } \boldsymbol { W } _ { l , i } ^ { V } ) ,
$$

$$
{ \mathrm { w h e r e ~ A t t e n t i o n } } ( Q , K , V ) = { \mathrm { s o f t m a x } } ( { \frac { Q K ^ { T } } { \sqrt { d _ { k } } } } ) V .\tag{2}
$$

![](images/c91375ad982302f298d31277c0af22c1a7218ca9c93c950704fb2b15013ae42d.jpg)  
Figure 2: VHD scores of all the attention heads among all layers at one generation step.

$X _ { l , i }$ and $A _ { l , i }$ represent the input and output of the i-th attention head in the l-th layer, respectively. $W ^ { Q } , W ^ { K }$ , and $W ^ { V }$ denote the learned weight matrices for the query, key, and value transformations, respectively. $d _ { k }$ is the dimension of the query (Q) and key (K) vectors. The outputs of all the attention heads in the l-th layer are then concatenated and linearly transformed into the output space of this module:

$$
\mathbf { M H A } _ { l } ( X _ { l } ) = [ A _ { l , 1 } ( X _ { l , 1 } ) , \cdots , A _ { l , n _ { h } } ( X _ { l , n _ { h } } ) ] W _ { l } ^ { O } ,\tag{3}
$$

where $n _ { h }$ denotes the number of attention heads in each layer, $X _ { l }$ is the input to the MHA module in the l-th layer, and $W ^ { \hat { O } }$ is the learned weight matrices for the output linear transformation.

Attention head output during generation. To more clearly delineate the correspondence between the model’s intermediate outputs and its inputs, we combine Equation 1 and 2 to introduce the notation $A _ { l , i } ( y _ { t } | y _ { < t } , x _ { V } , x _ { T } )$ . This notation represents the output of the i-th head in the l-th layer for generation step t, given the inputs $x _ { V }$ and $x _ { T }$ , and the generation history $y _ { < t }$

## 3 Method

## 3.1 Vision-aware Head Identification

Vision-aware head divergence (VHD). Inspired by the presence of in-context and memory heads in the model (Yu et al., 2023), we investigate whether different attention heads exhibit significantly different degrees of sensitivity to visual content. Specifically, we propose the vision-aware head divergence metric, which measures the change in the output of attention head for generation step t when the image

![](images/b82b2c9bf6bf0d450c37e6e7d6061d609429782e6b76667279e7c4ba65a9f251.jpg)  
(a) Histogram of sentence-level T-VHD scores

![](images/b1bf4ea28f60c6d28a4b0310e3c57519d41c5b7a9e693a708b265658da60961f.jpg)  
(b) Histogram of word-level T-VHD scores  
Figure 3: Relationship between T-VHD scores and hallucinations in LVLMs. Sentences and words associated with hallucinations generally correspond to lower T-VHD scores. Best viewed in color.

context is removed:

$$
\begin{array} { r } {  { \boldsymbol \nabla }  { \mathrm { H } }  { \mathbf Ḋ \ Ḍ } _ { l , i } =  { \mathrm { d } } \left( A _ { l , i } ( y _ { t } | y _ { < t } , x _ { V } , x _ { T } ) , A _ { l , i } ( y _ { t } | y _ { < t } , x _ { T } ) \right) , } \end{array}\tag{4}
$$

where d represents the Euclidean distance measure (Tabak, 2014).

Figure 2 visualizes the VHD scores for each attention head in the model. Specifically, we prompt LLaVA-1.5 with an image and the instruction "Please describe the image in detail" to generate descriptions, calculating the VHD scores when predicting the first token. The results show that a few attention heads exhibit notably higher VHD scores, while the others show minimal sensitivity. This suggests the presence of vision-aware attention heads that are more attuned to visual information. More examples of VHD scores during the generation process are presented in Appendix E.

Token-VHD. Beyond the varying degrees of vision awareness within the model, we further explore whether the VHD scores vary across different token generation steps. To this end, the VHD scores of the most prominent attention heads in each layer of the model are aggregated into the Token-VHD metric:

$$
\mathrm { T } \mathrm { \cdot } \mathrm { V H D } = \sum _ { l } \sum _ { i } \mathrm { t o p k } _ { i } ( \mathrm { V H D } _ { l , i } , k ) .\tag{5}
$$

Note that we only consider the top k VHD scores at each layer to ensure that the large number of insensitive attention heads does not dilute the aggregation metric. Eventually, T-VHD metric serves as an indicator of the model’s reliance on visual information over language priors when predicting a specific token.

Leveraging this metric, we can quantitatively analyze the relationship between hallucination in

![](images/0d399390211cf7f0678c5413c4f9f165003b1f506b5744380b41b5d54cad0590.jpg)  
Figure 4: The illustration of the proposed VHD metric and the VHR approach to mitigate hallucinations in LVLM. We select the attention heads that are sensitive to visual information for a given layer based on the VHD metric, i.e. $H _ { l } .$ , and then amplify their outputs to reinforce their contributions.

LVLMs and language bias at different levels of granularity, specifically at the sentence and word levels. To this end, we conduct an experiment on a random sample of 500 images from the CHAIR benchmark (Rohrbach et al., 2018), tracking the T-VHD scores at each generation step. Objectrelated words in the generated descriptions are classified as either hallucinated or correct, depending on whether they appear in the annotated object set for the given image. Sentences are then labeled according to whether they contain any hallucinated words. Figure 3 presents the experimental results, highlighting the distributional differences in T-VHD scores between hallucinated and correct instances. These findings provide statistical evidence that language bias is closely related to hallucinations in LVLMs.

## 3.2 Vision-aware Head Reinforcement

Since only a small subset of attention heads within the model are sensitive to visual information, we can amplify their contributions during generation to strengthen the model’s reliance on visual cues and counteract language bias. As discussed in Section 3.1, the VHD metric effectively captures the sensitivity of attention heads to visual information, making it a suitable indicator for selecting key attention heads for reinforcement. However, we observed that some high VHD values stem from a surge in the activation of attention heads upon the removal of visual context, indicating negative vision sensitivity. Amplifying the contributions of such heads would diverge from our objective. Therefore, we propose to zero out these undesired outliers, i.e.

$\mathrm { V H D } _ { l , i } = 0 .$ , if the following condition satisfies:

$$
\left\{ \begin{array} { l l } { \mathrm { V H D } _ { l , i } > \mu ( \mathrm { V H D } _ { l , * } ) + \sigma ( \mathrm { V H D } _ { l , * } ) , } \\ { \mathrm { T A } _ { l , i } > \mu ( \mathrm { T A } _ { l , * } ) + \sigma ( \mathrm { T A } _ { l , * } ) , } \\ { \mathrm { w h e r e ~ T A } _ { l , i } = \Vert A _ { l , i } ( y _ { t } | y _ { < t } , x _ { T } ) \Vert ^ { 2 } . } \end{array} \right.\tag{6}
$$

$\mu$ and σ represent the mean and standard deviation. Next, for the multi-head attention module in a given layer of the model, we select the first half of the attention heads based on their VHD scores and directly scale up their outputs by a factor of α:

$$
\begin{array} { r l } & { \widetilde { A } _ { l , i } = \left\{ \begin{array} { l l } { \alpha \cdot A _ { l , i } , } & { \mathrm { ~ i f ~ } i \in H _ { l } , } \\ { A _ { l , i } , } & { \mathrm { ~ o t h e r w i s e } , } \end{array} \right. } \\ & { \mathrm { ~ w h e r e ~ } H _ { l } = \{ i | \mathrm { ~ V H D } _ { l , i } > \mathrm { m e d i a n } ( \mathrm { V H D } _ { l , * } ) \} . } \end{array}\tag{7}
$$

Apply VHR layer by layer. This specific implementation allows for the selection and reinforcement of attention heads within a single forward pass, as opposed to first selecting the heads in all layers and then reinforcing them in two separate passes. Additionally, when VHR is applied across multiple layers simultaneously, the reinforcement in earlier layers can influence the VHD scores of subsequent layers. The layer-by-layer VHR strategy helps to avoid such inconsistencies, as the previous layers are already reinforced when calculating the VHD scores for a given layer.

Determine the heads at the first generation step. Although we can compute the VHD scores and select the key heads at each generation step, reinforcing different heads at different steps may bring negative effects. Specifically, LVLMs rely on KV caching to speed up inference, which means that the keys and values of the previous tokens will not be recalculated in subsequent generation steps. Therefore, the important heads should be determined at the beginning of the generation process to ensure consistency in the Q, K, and $V$ of all tokens in the attention module. Our experimental results show that this approach is sufficient to mitigate hallucinations.

Comparison with other head identification methods. Different from existing attention head identification methods in the field of model interpretability (Yu et al., 2023; Li et al., 2024; Zhou et al., 2024; Fang et al., 2024a; He et al., 2024), VHR does not require any annotation and can adaptively detect the key heads for each sample. Furthermore, rather than identifying and reinforcing all key attention heads in the model in two forward passes, VHR ensures computational efficiency and metric consistency by iteratively applying the select-thenreinforce approach across the layers. The complete procedure of VHR is provided in Algorithm 1.

## 3.3 Attention Output Reorientation

Scaling up the outputs of certain attention heads within a layer to reinforce its contribution is a straightforward and intuitive operation, and we present a theoretical analysis to substantiate its rationale. Consider the input to the FFN module following the MHA module in layer l, which can be expressed as follows:

$$
\begin{array} { r l } & { Z _ { l } = \mathrm { R M S N o r m } ( \hat { X } _ { l } + \mathbf { M H A } _ { l } ( X _ { l } ) ) } \\ & { \quad \quad = \hat { g } _ { l } \cdot \frac { \hat { X } _ { l } + \mathbf { M H A } _ { l } ( X _ { l } ) } { \| \hat { X } _ { l } + \mathbf { M H A } _ { l } ( X _ { l } ) \| } , } \end{array}\tag{8}
$$

where $\hat { g } _ { l }$ is a fixed constant after training, and $\hat { X _ { l } }$ is the input to the l-th layer before RMSNorm. Due to the normalization operation, only the direction of the overall output from earlier modules is crucial.

Proposition 1 Consider a layer l within an LVLM, and let h be the index ofthe attention head to be reinforced. Let $\widetilde { Z } _ { l }$ be the input to the FFN module obtained with $\hat { A } _ { l , h } = \alpha \cdot A _ { l , h } ( \alpha > 1 ) ,$ $Z _ { l }$ be the origeinal input obtained with $A _ { l , h } , Z _ { l , h }$ be the pseudoinput obtained with only the $A _ { l , h }$ component. Then it holds that cos $( \widetilde { Z } _ { l } , Z _ { l , h } ) > c o s ( Z _ { l } , Z _ { l , h } )$

The proof is detailed in Appendix B. Proposition 1 implies that amplifying the output of a specific head in the MHA module effectively reorients the direction of $Z _ { l }$ towards the output direction of the reinforced head component. This provides theoretical support for the mechanism underlying the reinforcement of the key attention head. An overview of the proposed VHD metric and the VHR method is presented in Figure 4.

Algorithm 1 VHR   
Input image $x _ { V }$ , instruction $x _ { T }$ , generation step t,   
scale factor α, layers to reinforce $L _ { r }$   
1: for layer $l \in L _ { r }$ do   
2: if $t = 0$ then   
3: Compute $\mathrm { V H D } _ { l , * }$ ▷ Equation 4   
4: Zero out $\mathrm { V H D } _ { l , i }$ if Equation 6 holds   
5: Select the heads as $H _ { l }$ ▷ Equation 7   
6: end if   
7: Reinforce the heads in $H _ { l }$ ▷ Equation 7   
8: end for

## 4 Experiments

## 4.1 LVLMs

We conduct experiments on three of the most representative LVLMs, i.e. InstructBLIP-7b (Dai et al., 2023), LLaVA-1.5-7b (Liu et al., 2024b), and LLaVA-NeXT-7b (Liu et al., 2024c). LVLMs are typically composed of an image encoder, a connector, and an LLM. Specifically, LLaVA-1.5-7b and LLaVA-NeXT-7b leverage MLP to align the visual and textual embedding space and feed all the image tokens from the image encoder to the LLM. In contrast, InstructBLIP uses Q-Former to reduce the number of image tokens before passing them to the LLM. LLaVA-NeXT differs from LLaVA-1.5 by offering a higher image resolution, allowing it to capture more visual details.

## 4.2 Benchmarks

CHAIR. The Caption Hallucination Assessment with Image Relevance (CHAIR) metric (Rohrbach et al., 2018) evaluates object hallucination in image captioning by comparing generated captions with ground truth data. It identifies objects mentioned in captions but absent in images and calculates their proportion to quantify hallucination. Specifically, CHAIR includes two metrics at both caption level (CHAIR<sub>S</sub>) and object level (CHAIR<sub>I</sub>):

$$
\begin{array} { r l } & { \mathrm { C H A I R } _ { S } = \frac { \left| \left\{ \mathrm { c a p t i o n ~ w / \ h a l l u c i n a t e d ~ o b j e c t s } \right\} \right| } { \left| \left\{ \mathrm { a l l ~ c a p t i o n s } \right\} \right| } , } \\ & { \mathrm { C H A I R } _ { I } = \frac { \left| \left\{ \mathrm { h a l l u c i n a t e d ~ o b j e c t s } \right\} \right| } { \left| \left\{ \mathrm { a l l ~ m e n t i o n e d ~ o b j e c t s } \right\} \right| } . } \end{array}\tag{9}
$$

<table><tr><td rowspan="2"></td><td colspan="3">InstructBLIP</td><td colspan="3">LLaVA-1.5</td><td colspan="3">LLaVA-NeXT</td></tr><tr><td> $\operatorname { C H A I R } _ { S } \downarrow$ </td><td> $\mathrm { C H A I R } _ { I } \downarrow$ </td><td>Len</td><td> $\operatorname { C H A I R } _ { S } \downarrow$ </td><td> $\mathrm { C H A I R } _ { I } \downarrow$ </td><td>Len</td><td> $\operatorname { C H A I R } _ { S } .$  →</td><td> $\mathrm { C H A I R } _ { I } \downarrow$ </td><td>Len</td></tr><tr><td>Greedy</td><td> $4 5 . 3 2 \substack { \pm 2 . 2 4 }$ </td><td> $\underline { { 1 2 . 9 8 } } \mathrm { \pm 0 . 7 6 }$ </td><td>91.06</td><td> $4 9 . 6 8 _ { \pm 1 . 4 7 }$ </td><td> $1 4 . 3 2 { \scriptstyle \pm 0 . 7 8 }$ </td><td>83.06</td><td> $2 9 . 0 8 { \scriptstyle \pm 2 . 0 9 }$ </td><td> $8 . 0 8 { \scriptstyle \pm 0 . 7 4 }$ </td><td>157.06</td></tr><tr><td>Beam</td><td> $4 8 . 5 6 { \scriptstyle \pm 1 . 6 6 }$ </td><td> $1 3 . 5 0 { \scriptstyle \pm 0 . 4 4 }$ </td><td>94.87</td><td> $5 3 . 8 4 _ { \pm 2 . 4 1 }$ </td><td> $1 5 . 6 0 { \scriptstyle \pm 0 . 4 6 }$ </td><td>87.47</td><td> $2 5 . 7 2 \substack { \pm 2 . 1 7 }$ </td><td> $6 . 9 2 _ { \pm 0 . 8 8 }$ </td><td>160.64</td></tr><tr><td>DoLa</td><td> $4 6 . 0 0 { \scriptstyle \pm 1 . 8 7 }$ </td><td> $1 3 . 0 0 { \scriptstyle \pm 0 . 9 1 }$ </td><td>90.75</td><td> $5 0 . 8 8 _ { \pm 2 . 3 4 }$ </td><td> $1 4 . 6 4 _ { \pm 0 . 9 0 }$ </td><td>82.41</td><td> $2 8 . 7 6 { \scriptstyle \pm 2 . 5 8 }$ </td><td> $8 . 1 2 _ { \pm 0 . 7 8 }$ </td><td>155.75</td></tr><tr><td>VCD</td><td> $5 0 . 7 2 _ { \pm 2 . 4 4 }$ </td><td> $1 4 . 4 2 _ { \pm 0 . 9 9 }$ </td><td>90.39</td><td> $5 1 . 9 2 _ { \pm 1 . 8 7 }$ </td><td> $1 5 . 4 2 _ { \pm 0 . 8 4 }$ </td><td>83.12</td><td> $3 0 . 8 0 { \scriptstyle \pm 2 . 4 8 }$ </td><td> $8 . 7 2 _ { \pm 0 . 9 4 }$ </td><td>157.72</td></tr><tr><td>OPERA</td><td> $4 5 . 7 6 { \scriptstyle \pm 2 . 3 2 }$ </td><td> $1 3 . 0 6 _ { \pm 0 . 8 8 }$ </td><td>92.46</td><td> $4 4 . 2 8 _ { \pm 0 . 9 5 }$ </td><td> $1 3 . 3 6 _ { \pm 0 . 4 7 }$ </td><td>75.88</td><td></td><td></td><td></td></tr><tr><td>CODE</td><td> $5 0 . 7 6 { \scriptstyle \pm 2 . 0 6 }$ </td><td> $1 4 . 1 2 _ { \pm 0 . 9 3 }$ </td><td>88.57</td><td> $4 7 . 9 6 _ { \pm 0 . 8 0 }$ </td><td> $1 4 . 2 6 _ { \pm 0 . 5 7 }$ </td><td>78.52</td><td> $2 7 . 8 4 _ { \pm 2 . 7 3 }$ </td><td> $7 . 9 8 _ { \pm 0 . 9 2 }$ </td><td>151.51</td></tr><tr><td>EAH</td><td> $4 6 . 4 0 _ { \pm 1 . 1 5 }$ </td><td> $1 3 . 1 3 _ { \pm 0 . 6 0 }$ </td><td>92.33</td><td> $3 8 . 7 6 { \scriptstyle \pm 2 . 4 7 }$ </td><td> $1 1 . 0 5 _ { \pm 0 . 8 1 }$ </td><td>86.28</td><td> $2 8 . 1 3 _ { \pm 1 . 1 3 }$ </td><td> ${ \bf 6 . 6 2 _ { \pm 0 . 4 9 } }$ </td><td>142.75</td></tr><tr><td>VHR</td><td> $\mathbf { 3 7 . 7 6 _ { \pm 2 . 7 6 } }$ </td><td> $\mathbf { 9 . 7 5 _ { \pm 0 . 9 8 } }$ </td><td>106.49</td><td> $3 3 . 3 2 _ { \pm 1 . 3 1 }$ </td><td> ${ \bf 9 . 7 1 _ { \pm 0 . 4 5 } }$ </td><td>81.33</td><td> $\pm \mathbf { 4 . 9 6 _ { \pm 2 . 0 9 } }$ </td><td> $\underline { { 6 . 8 0 } } \mathrm { \underline { { { \pm 0 . 5 9 } } } }$ </td><td>156.92</td></tr></table>

Table 1: CHAIR evaluation results on MSCOCO dataset averaged over 5 random splits, with best in bold and second-best underlined. Len represents the average number of words in the generated descriptions.

We randomly sample 500 images from the COCO 2014 validation set and repeat the experiments for five times with different random seeds. The LVLMs are prompted with "Please describe this image in detail." to get the descriptions. We report the average results for each metric along with the standard deviation.

POPE. POPE (Li et al., 2023) is a dataset for evaluating object hallucinations by having models answer true or false questions about the presence of objects in images. The dataset includes 500 images from MSCOCO (Lin et al., 2014), with each image paired with questions like "Is there a <object> in the image?". The evaluation consists of three splits—random, popular, and adversarial—where objects are sampled in different ways. The evaluation metrics include Accuracy, Precision, Recall, and F1 scores, with the results averaged across all three splits.

LLaVA-Bench. LLaVA-Bench (In-the-Wild) (Liu et al., 2024b) is a comprehensive benchmark designed to evaluate the performance of visionlanguage models on a wide range of challenging tasks. It includes 24 images across diverse domains, such as indoor and outdoor scenes, memes, accompanied by 60 carefully crafted questions covering simple question answering, detailed descriptions, and complex reasoning. Due to the open-ended nature and complexity of the responses, we prompt the GPT-4V model to evaluate the LVLMs’ outputs in terms of accuracy, detailedness, and naturalness.

## 4.3 Baselines

We compare VHR with the popular training-free methods that do not introduce external information or models: DoLa (Chuang et al., 2023) derives the next-token distribution by contrasting the logits from later and earlier layers; VCD (Leng et al.,

2024) contrasts the output distribution generated from the original and distorted image; OPERA (Huang et al., 2024) mitigates over-trust in previous summary tokens in beam-search decoding; CODE (Kim et al., 2024) uses self-generated descriptions as contrast references to improve alignment with the actual visual content; EAH (Zhang et al., 2024) enhances the attention sinks on image tokens in shallow layers. In addition, we also compare the performance of base LVLMs using greedy and beam search decoding.

## 4.4 Implementation Details

We set α to 2 to strike a balance between effectively correcting hallucinations and minimizing the invasiveness of hidden states manipulation. VHR is applied to the second and last 14 layers for LLaVA series and the last 18 layers for InstructBLIP. We faithfully reproduced all baseline methods based on their open-source repositories and set the hyperparameters according to the values reported in the papers. The results of all methods are reported under consistent conditions of base models, prompts, and generation parameters to ensure a fair comparison. Specifically, the max\_new\_token is set to 512, and the number of beams is set to 5 for all methods involving beam search.

## 4.5 Results

CHAIR. Table 1 presents the performance of VHR in comparison to all baseline approaches on the CHAIR benchmark. The results for OPERA on LLaVA-NeXT are absent due to its excessive memory requirements. VHR demonstrates robust performance across all three LVLMs, achieving reductions of up to 16.36 in CHAIR<sub>S</sub> and 4.61 in CHAIR<sub>I</sub> on LLaVA-1.5. Notably, with increased image resolution and enhanced model capabilities, LLaVA-NeXT already exhibits a significant reduction in hallucinations compared to other base LVLMs, but VHR continues to exhibit notable effectiveness in mitigating its hallucinations. Moreover, VHR consistently outperforms baseline methods with greater stability, requiring only minor trade-offs in the length or richness of the generated description.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>InstructBLIP</td><td rowspan=1 colspan=1>LLaVA-1.5</td><td rowspan=1 colspan=1>LLaVA-NeXT</td></tr><tr><td rowspan=1 colspan=1>Greedy</td><td rowspan=1 colspan=1>85.36</td><td rowspan=1 colspan=1>84.98</td><td rowspan=2 colspan=1>88.5187.97</td></tr><tr><td rowspan=1 colspan=1>Beam</td><td rowspan=1 colspan=1>84.40</td><td rowspan=1 colspan=1>85.30</td></tr><tr><td rowspan=1 colspan=1>DoLa</td><td rowspan=1 colspan=1>85.21</td><td rowspan=1 colspan=1>85.07</td><td rowspan=1 colspan=1>88.46</td></tr><tr><td rowspan=1 colspan=1>VCD</td><td rowspan=1 colspan=1>84.67</td><td rowspan=1 colspan=1>84.41</td><td rowspan=2 colspan=1>88.11</td></tr><tr><td rowspan=1 colspan=1>OPERA</td><td rowspan=1 colspan=1>84.41</td><td rowspan=1 colspan=1>85.45</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>CODE</td><td rowspan=1 colspan=1>84.80</td><td rowspan=1 colspan=1>84.63</td><td rowspan=1 colspan=1>88.44</td></tr><tr><td rowspan=1 colspan=1>EAH</td><td rowspan=1 colspan=1>85.18</td><td rowspan=1 colspan=1>85.03</td><td rowspan=1 colspan=1>84.28</td></tr><tr><td rowspan=1 colspan=1>VHR</td><td rowspan=1 colspan=1>85.52</td><td rowspan=1 colspan=1>85.47</td><td rowspan=1 colspan=1>88.87</td></tr></table>

Table 2: F1 scores on POPE averaged over popular, adversarial, and random splits, with best in bold and second-best underlined.

<table><tr><td></td><td>Accuracy</td><td>Detailedness</td><td>Naturalness</td></tr><tr><td>InstructBLIP</td><td>4.917</td><td>5.017</td><td>6.717</td></tr><tr><td>w/VHR</td><td>5.250</td><td>5.117</td><td>6.733</td></tr><tr><td>LLaVA-1.5</td><td>6.017</td><td>6.100</td><td>7.400</td></tr><tr><td>w/ VHR</td><td>6.333</td><td>6.217</td><td>7.333</td></tr><tr><td>LLaVA-NeXT</td><td>5.383</td><td>6.750</td><td>7.900</td></tr><tr><td>w/ VHR</td><td>5.783</td><td>6.700</td><td>7.833</td></tr></table>

Table 3: LLaVA-Bench (In-the-Wild) evaluation results, scored by GPT-4o via pairwise response comparison.

POPE. As shown in Table 2, VHR outperforms all other decoding methods across all LVLMs. While the binary (yes/no) benchmark limits the full demonstration of VHR’s strength in handling language bias, our method still consistently improves performance across models of varying capabilities. This indicates that VHR is a robust and effective training-free strategy for enhancing models at different performance levels.

LLaVA-Bench. The GPT-4V evaluation results on LLaVA-Bench (In-the-Wild) are presented in Table 3. These results demonstrate that VHR improves model accuracy on highly diverse and challenging tasks while preserving a consistent level of detailedness and naturalness. Additionally, the Len metric in Table 1 and the qualitative results in Figure 7 also indicate that VHR has only a minor impact on language generation quality, despite its enhancement of vision-aware attention heads.

<table><tr><td>Model</td><td>Method</td><td>CHAIRS↓</td><td>CHAIRI↓</td></tr><tr><td rowspan="3">InstructBLIP</td><td>VHR</td><td>37.76</td><td>9.75</td></tr><tr><td>fixed VHR</td><td>45.40</td><td>13.57</td></tr><tr><td>outlier VHR</td><td>37.76</td><td>10.18</td></tr><tr><td rowspan="3">LLaVA-1.5</td><td>VHR</td><td>33.32</td><td>9.71</td></tr><tr><td>fixed VHR</td><td>44.72</td><td>13.81</td></tr><tr><td>outlier VHR</td><td>36.88</td><td>10.36</td></tr><tr><td rowspan="3">LLaVA-NeXT</td><td>VHR</td><td>24.96</td><td>6.80</td></tr><tr><td>fixed VHR</td><td>36.96</td><td>9.80</td></tr><tr><td>outlier VHR</td><td>24.64</td><td>6.37</td></tr></table>

Table 4: Ablation study on adaptively determining key heads per sample and removing outlier VHD scores.

![](images/4b94b8c568533b68e777ca70fe51d1e3a30db095d85caac41bec6c5240862d52.jpg)  
Figure 5: Results of VHR applied across different numbers of the last few layers in LLaVA-1.5.

## 4.6 Ablation Study

Impact of Adaptive Attention Head Selection. Since VHR reinforces different attention heads for each sample, we conduct an ablation study to validate the necessity of this strategy. As shown in Table 4, fixing a set of attention heads identified by one sample for reinforcement across all samples leads to a significant performance drop.

Impact of Outlier VHD Score Removal. VHR relies on VHD scores to identify critical attention heads, but these scores measure absolute influence, conflating both positive and negative vision sensitivity. To selectively enhance heads that improve visual context awareness, we propose excluding outlier VHD scores during selection (Equation 6). As shown in Table 4, this refinement consistently matches or outperforms the baseline, demonstrating that pruning unhelpful sensitivity is essential for optimal head selection.

Impact of Reinforced Layers. Figure 5 shows the ablation study results on the number of the last few layers for reinforcement. Increasing the number of reinforced layers continuously alleviates hallucinations, with optimal performance reached at the last 14 layers. However, further reinforcement degrades the model’s generation quality and fails to effectively mitigate hallucinations. More discussion on the choice of reinforced layers and scale factor can be found in Appendix C and D.

![](images/be21bc311fb9367bb73180ca29bb2f4d595ef71a84232086c96b6bd052ae167b.jpg)  
Figure 6: Comparison of inference time for different methods.

## 4.7 Further Analysis

Additional Time Analysis. For the first generation step, VHR requires an additional forward pass, removing the image context to calculate the VHD scores. In subsequent generation steps, only the scaling operation is needed. As a result, the extra computation introduced by VHR is negligible throughout the entire generation process. A detailed inference time comparison between VHR and baseline methods is presented in Figure 6.

Qualitative Results. To clearly demonstrate the effect of VHR in reducing hallucinations, we provide a concrete example in Figure 7. Without VHR, the LVLM generates content that is absent from the image, such as mentioning people watching the game in the background. This could stem from inherent language bias in the training data. When VHR is applied, the outputs are more accurately aligned with the actual content of the image.

## 5 Related Work

## 5.1 Hallucination in LVLMs

Hallucination in LVLMs refers to discrepancies between the generated text and the actual content present in the corresponding image. This issue stems from multiple factors, including data bias, insufficient image grounding by vision encoders, and misalignment between modalities (Liu et al., 2024a). Current approaches to mitigating hallucinations in LVLMs can be broadly classified into three categories: training alignment (Zhao et al.,

![](images/37652dc799200d859fafd76a5c3bbd9cd4fdcf253fd4949af822ac06960e8a67.jpg)  
Figure 7: An example of VHR eliminating the hallucinated content.

2023; Yu et al., 2024), post-processing (Zhou et al., 2023; Yin et al., 2023; Fang et al., 2024b), and decoding strategies (Leng et al., 2024; Zhu et al., 2024; Huang et al., 2024; Kim et al., 2024; Gong et al., 2024). The first two categories often rely on external resources or models to improve performance, whereas our proposed VHR method operates solely based on the model’s internal states, without requiring additional external information. In contrast to existing decoding strategies, VHR intervenes proactively within the model to address hallucinations before they occur, rather than modifying the logits distribution at the output stage. A recent approach, EAH (Zhang et al., 2024), also targets the attention mechanism to reduce hallucinations, but it focuses specifically on the attention sink phenomenon. In comparison, VHR reduces language bias in LVLMs by exploiting cross-modal contrast and re-adjusting the contributions of attention heads across multiple layers. Consequently, the underlying motivations and methodologies of the two works differ significantly.

## 5.2 Language Bias in LVLMs

Language bias refers to the tendency of models to prioritize language patterns or prior knowledge over the actual visual context presented in the input. This issue has a long-standing research history predating the emergence of LVLMs and was studied through methods like balanced multimodal training (Goyal et al., 2017) and causal inference (Niu et al., 2021). With the advent of LVLMs, language bias has become an even more pressing concern. These models are typically pre-trained on massive corpora of text data, further exacerbating the potential for language overfitting. Studies have identified that LVLMs, despite their capabilities in visual recognition, struggle to fully integrate visual context (Parcalabescu and Frank, 2024), leading to hallucinations, particularly in reasoning tasks (Ghosh et al., 2024). To address this growing challenge, several recent methods have proposed solutions such as contrastive decoding (Leng et al., 2024; Zhu et al., 2024) and visual description grounding decoding (Ghosh et al., 2024). However, these approaches directly manipulate the output logits, which introduces instability during generation and lacks a thorough analysis of the model’s internal mechanisms. In contrast, our approach intervenes directly within the model, providing a more interpretable and effective way to address the internal factors driving language bias in LVLMs while complementing existing methods.

## 6 Conclusion

This work investigates the connection between hallucination in LVLMs and the multi-head attention mechanism. We introduce the VHD metric, which quantifies the sensitivity of attention head outputs to visual context, revealing that language bias can contribute to hallucinations in LVLMs. Building on these findings, we propose VHR, a trainingfree approach that strengthens the role of visionaware attention heads to mitigate hallucinations. Extensive experiments demonstrate that VHR outperforms existing methods, significantly improving the alignment of LVLMs with visual information.

## Limitations

Our analysis and mitigation strategy primarily focus on the multi-head attention mechanism of LVLMs. While this is a critical component influencing hallucinations, there may be other architectural factors—such as those in the vision encoder and the FFN module in the LLMs—that contribute to hallucinations but were not directly addressed in this study. Future work could focus on more comprehensive interventions that span the entire model, going beyond attention head manipulation.

## Acknowledgements

This work was supported by National Key R&D Program of China under Grant No.2024YFC3015501, also partly supported by National Natural Science Foundation of China under Grant No.62276260, Beijing Municipal Science and Technology Project under Grant Z231100007423004, Beijing Natural Science Foundation under Grant 4244099, Aeronautical Science Foundation of China under Grant 2024M0710M0002.

## References

Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. 2025. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In European Conference on Computer Vision, pages 19–35. Springer.

Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, and Pengcheng He. 2023. Dola: Decoding by contrasting layers improves factuality in large language models. arXiv preprint arXiv:2309.03883.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. 2023. Instructblip: Towards general-purpose visionlanguage models with instruction tuning. Preprint, arXiv:2305.06500.

Junfeng Fang, Zac Bi, Ruipeng Wang, Houcheng Jiang, Yuan Gao, Kun Wang, An Zhang, Jie Shi, Xiang Wang, and Tat-Seng Chua. 2024a. Towards neuron attributions in multi-modal large language models. Advances in Neural Information Processing Systems, 37:122867–122890.

Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Shi Jie, Xiang Wang, Xiangnan He, and Tat-Seng Chua. 2024b. Alphaedit: Null-space constrained knowledge editing for language models. arXiv preprint arXiv:2410.02355.

Sreyan Ghosh, Chandra Kiran Reddy Evuru, Sonal Kumar, Utkarsh Tyagi, Oriol Nieto, Zeyu Jin, and Dinesh Manocha. 2024. Visual description grounding reduces hallucinations and boosts reasoning in lvlms. arXiv preprint arXiv:2405.15683.

Xuan Gong, Tianshi Ming, Xinpeng Wang, and Zhihua Wei. 2024. Damro: Dive into the attention mechanism of lvlm to reduce object hallucination. arXiv preprint arXiv:2410.04514.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2017. Making the v in vqa matter: Elevating the role of image understanding

in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913.

Jinghan He, Haiyun Guo, Kuan Zhu, Zihan Zhao, Ming Tang, and Jinqiao Wang. 2024. Seekr: Selective attention-guided knowledge retention for continual learning of large language models. arXiv preprint arXiv:2411.06171.

Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. 2024. Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13418– 13427.

Junho Kim, Hyunjun Kim, Yeonju Kim, and Yong Man Ro. 2024. Code: Contrasting self-generated description to combat hallucination in large multi-modal models. arXiv preprint arXiv:2406.01920.

Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. 2024. Mitigating object hallucinations in large visionlanguage models through visual contrastive decoding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13872–13882.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2024. Inferencetime intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems, 36.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. 2023. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In Computer Vision– ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer.

Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and Wei Peng. 2024a. A survey on hallucination in large vision-language models. arXiv preprint arXiv:2402.00253.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024b. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. 2024c. Llavanext: Improved reasoning, ocr, and world knowledge.

Yulei Niu, Kaihua Tang, Hanwang Zhang, Zhiwu Lu, Xian-Sheng Hua, and Ji-Rong Wen. 2021. Counterfactual vqa: A cause-effect look at language bias. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12700– 12710.

Letitia Parcalabescu and Anette Frank. 2024. Do vision & language decoders use images and text equally? how self-consistent are their explanations? arXiv preprint arXiv:2404.18624.

Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. 2018. Object hallucination in image captioning. arXiv preprint arXiv:1809.02156.

John Tabak. 2014. Geometry: the language of space andform. Infobase Publishing.

Shukang Yin, Chaoyou Fu, Sirui Zhao, Tong Xu, Hao Wang, Dianbo Sui, Yunhang Shen, Ke Li, Xing Sun, and Enhong Chen. 2023. Woodpecker: Hallucination correction for multimodal large language models. arXiv preprint arXiv:2310.16045.

Junwei You, Haotian Shi, Zhuoyu Jiang, Zilin Huang, Rui Gan, Keshu Wu, Xi Cheng, Xiaopeng Li, and Bin Ran. 2024. V2x-vlm: End-to-end v2x cooperative autonomous driving through large vision-language models. arXiv preprint arXiv:2408.09251.

Qinan Yu, Jack Merullo, and Ellie Pavlick. 2023. Characterizing mechanisms for factual recall in language models. arXiv preprint arXiv:2310.15910.

Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu, Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun, et al. 2024. Rlhf-v: Towards trustworthy mllms via behavior alignment from finegrained correctional human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13807–13816.

Xiaofeng Zhang, Yihao Quan, Chaochen Gu, Chen Shen, Xiaosong Yuan, Shaotian Yan, Hao Cheng, Kaijie Wu, and Jieping Ye. 2024. Seeing clearly by layer two: Enhancing attention heads to alleviate hallucination in lvlms. arXiv preprint arXiv:2411.09968.

Zhiyuan Zhao, Bin Wang, Linke Ouyang, Xiaoyi Dong, Jiaqi Wang, and Conghui He. 2023. Beyond hallucinations: Enhancing lvlms through hallucinationaware direct preference optimization. arXiv preprint arXiv:2311.16839.

Yiyang Zhou, Chenhang Cui, Jaehong Yoon, Linjun Zhang, Zhun Deng, Chelsea Finn, Mohit Bansal, and Huaxiu Yao. 2023. Analyzing and mitigating object hallucination in large vision-language models. arXiv preprint arXiv:2310.00754.

Zhenhong Zhou, Haiyang Yu, Xinghua Zhang, Rongwu Xu, Fei Huang, Kun Wang, Yang Liu, Junfeng Fang, and Yongbin Li. 2024. On the role of attention

heads in large language model safety. arXiv preprint arXiv:2410.13708.

Lanyun Zhu, Deyi Ji, Tianrun Chen, Peng Xu, Jieping Ye, and Jun Liu. 2024. Ibd: Alleviating hallucinations in large vision-language models via imagebiased decoding. arXiv preprint arXiv:2402.18476.

## A A Case Study of Language Bias

Figure 8 illustrates an example that reflects language bias in LVLMs. We first prompted the model to describe the image, resulting in a complete description. We observed that the latter part of this description contained hallucinated content. To investigate the role of language bias in this case, we removed the image input and re-prompted the model using only the original text prompt and the nonhallucinated part of the initial response. This allowed us to observe how the model would continue the text based solely on its internal knowledge. We found that the continuation closely resembled the previously hallucinated content, suggesting it was generated based on internal language priors rather than visual evidence. This example clearly demonstrates the connection between language bias and hallucination in LVLMs.

In the lower part of Figure 8, we visualize the T-VHD scores for each word in the generated description. The word color intensity reflects the T-VHD scores, with darker shades indicating higher sensitivity to visual input. The results show that words within fixed phrases (e.g., surrounded by, on the right/left side) generally have lower T-VHD scores, suggesting stronger reliance on language priors. In contrast, object terms (e.g. wooden dining table, vase, red roses) typically exhibit higher T-VHD scores upon first mention, indicating greater dependence on visual context. Notably, among all object terms, hallucinated items (e.g., cup, bowl, chair) tend to have lower T-VHD scores, highlighting reduced visual grounding. This example demonstrates how the proposed T-VHD metric effectively captures the model’s reliance on visual information versus language priors at the token level.

## B Attention Output Reorientation

Proposition 1 Consider a layer l within an LVLM, and let h be the index ofthe attention head to be reinforced. Let $\widetilde { Z } _ { l }$ be the input to the FFN module obtained with $\widetilde { A } _ { l , h } = \alpha \cdot A _ { l , h } ( \alpha > 1 ) ,$ , Z<sub>l</sub> be the origeinal input obtained with $A _ { l , h } , Z _ { l , h }$ be the pseudoinput obtained with only the $A _ { l , h }$ component. Then it holds that cos $( \widetilde { Z } _ { l } , Z _ { l , h } ) > c o s ( Z _ { l } , Z _ { l , h } )$

Please describe this image in detail.

The image features a wooden dining table with a vase filled with red roses placed in the center. The vase is white, and the roses are arranged in a bouquet. The table is surrounded by various items, including a bottle on the left side, Generating..

<table><tr><td rowspan=1 colspan=8>T-VHD Scores</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>wooden</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>vase</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1><img src="images/1b6acadaa8b543beb2c972d2792ebfdfa8177deebf30887ad93b80f4b56e0707.jpg"/></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Additional</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>positioned</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr></table>

Figure 8: An example indicating the connection between hallucination in LVLMs and language bias. The proposed T-VHD metric reflects the model’s reliance on visual content versus language priors at the token level.

Proof. By partitioning the projection matrix $W _ { l } ^ { O }$ we can further express the output of MHA as the sum of the contributions from each attention head:

$$
\begin{array} { r l } & { \mathbf { M H A } _ { l } ( X _ { l } ) } \\ & { \quad = [ A _ { l , 1 } ( X _ { l , 1 } ) , \cdots , A _ { l , n _ { h } } ( X _ { l , n _ { h } } ) ] W _ { l } ^ { O } } \\ & { \quad = A _ { l , 1 } ( X _ { l , 1 } ) W _ { l , 1 } ^ { O } + \cdots + A _ { l , n _ { h } } ( X _ { l , n _ { h } } ) W _ { l , h } ^ { O } . } \end{array}\tag{10}
$$

To simplify the notation , we define x and y as follows:

$$
\begin{array} { r l } & { \mathbf { x } = \hat { X } _ { l } + \mathbf { M H A } _ { l } ( X _ { l } ) , } \\ & { \mathbf { y } = A _ { l , h } ( X _ { l , h } ) W _ { l , h } ^ { O } . } \end{array}\tag{11}
$$

We then substitute x and y for the variables in Equation 8 to derive $Z _ { l } , \widetilde { Z } _ { l } , Z _ { l , h } \colon$

$$
\begin{array} { l } { { \displaystyle Z _ { l } = \hat { g } _ { t } \frac { { \bf x } } { \| { \bf x } \| } } , } \\ { ~ } \\ { { \displaystyle \widetilde { Z } _ { l } = \hat { g } _ { t } \frac { { \bf x } + ( \alpha - 1 ) { \bf y } } { \| { \bf x } + ( \alpha - 1 ) { \bf y } \| } } , } \\ { { \displaystyle Z _ { l , h } = \hat { g } _ { t } \frac { { \bf y } } { \| { \bf y } \| } } . } \end{array}\tag{12}
$$

Lastly, we prove that cos $( \widetilde { Z } _ { l } , Z _ { l , h } )$ is greater than cos $( Z _ { l } , Z _ { l , h } )$

$$
\begin{array} { r l } & { \cos ( \tilde { \mathcal { T } } _ { i } , \mathcal { T } _ { i , k } ) - \cos ( \mathcal { T } _ { j } , \mathcal { T } _ { j , k } ) } \\ & { = \frac { \langle \mathbf { x } + ( \boldsymbol { \alpha } - 1 ) \mathbf { y } , \mathbf { y } \rangle } { \| \mathbf { x } + \boldsymbol { \alpha } - 1 \rangle \mathbf { y } \| } \cdot \frac { \langle \mathbf { x } , \mathbf { y } \rangle } { \| \mathbf { x } \| \| \mathbf { y } \| } } \\ & { = \frac { \langle \mathbf { x } + ( \boldsymbol { \alpha } - 1 ) \mathbf { y } , \boldsymbol { ( \alpha - 1 ) } \mathbf { y } \rangle } { \| \mathbf { x } + \boldsymbol { ( \alpha - 1 ) } \mathbf { y } \| } - \frac { \langle \mathbf { x } , ( \boldsymbol { \alpha } - 1 ) \mathbf { y } \rangle } { \| \mathbf { x } \| \| \boldsymbol { ( \alpha - 1 ) } \mathbf { y } \| } } \\ & { = \frac { \langle \mathbf { x } , \mathbf { y } \rangle + \| \mathbf { y } \| ^ { 2 } } { \| \mathbf { x } + \mathbf { y } \| \| \mathbf { y } \| } - \frac { \langle \mathbf { x } , \mathbf { y } \rangle } { \| \mathbf { x } \| \| \mathbf { y } \| } } \\ & { > \frac { \langle \mathbf { x } , \mathbf { y } \rangle + \| \mathbf { y } \| } { \| \mathbf { x } + \mathbf { y } \| } - \frac { \langle \mathbf { x } , \mathbf { y } \rangle } { \| \mathbf { x } \| \| \mathbf { y } \| } } \\ & { > \frac { - \| \mathbf { x } \| + \| \mathbf { y } \| } { \| \mathbf { x } + \mathbf { y } \| } + 1 } \\ & { = \frac { \| \mathbf { x } + \mathbf { y } \| + \| - \mathbf { y } \| - \| \mathbf { x } \| } { \| \mathbf { x } + \mathbf { y } \| } } \\ & { > 0 , } \end{array}\tag{13}
$$

which concludes the proof.

## C Choice of Reinforced Layers

Since Chen et al., 2025 and Zhang et al., 2024 have highlighted the unique role of the second layer in integrating visual information through attention map analysis, we include this layer for VHR and further validated its significance in mitigating hallucinations. Table 5 presents the results of ablation experiments conducted on this layer and the last few layers on LLaVA-1.5. The results show that enhancing layer1 and the deeper layers both significantly alleviate hallucinations, with the combination of both yielding even better results. This suggests that VHR in the shallow and deep layers alleviates hallucinations through distinct mechanisms. However, we note that applying VHR to the second layer does not universally improve performance; in some challenging benchmarks, its benefits diminish or even introduce instability. Further analysis of the layer-specific mechanisms in LVLMs remains an important direction for future work.

## D Choice of Scale Factor

Table 6 shows the results of ablation experiments on the scale factor α in VHR. When α is set to 2 or 3, hallucinations are effectively alleviated. However, as α increases to 4, excessive intervention disrupts the model’s behavior, causing anomalies in the hallucination metric. Conversely, when $\alpha < 1$ which weakens the contribution of attention heads sensitive to visual information, hallucinations become significantly more pronounced. This further confirms the crucial role of the attention heads identified based on the VHD scores in mitigating hallucinations in LVLMs.

<table><tr><td></td><td>CHAIRS↓</td><td>CHAIRI↓</td></tr><tr><td>LLaVA-1.5</td><td>49.68</td><td>14.32</td></tr><tr><td>w/ VHR on layer1</td><td>40.80</td><td>12.00</td></tr><tr><td>w/ VHR on last 14 layers</td><td>41.96</td><td>12.56</td></tr><tr><td>w/ VHR on both</td><td>33.32</td><td>9.71</td></tr></table>

Table 5: Ablation study on the reinforced layers.
<table><tr><td></td><td>CHAIRS↓</td><td>CHAIRI↓</td><td>Len</td></tr><tr><td>LLaVA-1.5</td><td>49.68</td><td>14.32</td><td>83.06</td></tr><tr><td>w/ VHR α = 0.2</td><td>63.28</td><td>21.04</td><td>86.42</td></tr><tr><td>w/ VHR α = 0.5</td><td>55.80</td><td>17.32</td><td>84.47</td></tr><tr><td>w/ VHR α = 2</td><td>33.32</td><td>9.71</td><td>81.33</td></tr><tr><td>w/ VHR α = 3</td><td>27.04</td><td>8.68</td><td>88.31</td></tr><tr><td>w/ VHR α = 4</td><td>3.64</td><td>2.01</td><td>144.54</td></tr></table>

Table 6: Ablation study on the scale factor.

## E Examples of VHD Scores

As shown in Figure 10, the first row presents the VHD scores during the first generation step across different samples, while the second row shows the VHD scores for different object terms generated within the same sample. It can be observed that VHD scores vary across different samples and generation steps; however, significant differences between the VHD scores of the attention heads within the model are consistently present.

## F More Qualitative Results

As shown in Figure 9, we present more examples that illustrate the effect of VHR in eliminating hallucinated objects. After incorporating VHR, the descriptions generated by the LVLMs faithfully align with the content of the images, while preserving the richness of the descriptions.

## G Details on the GPT-4V Evaluation

To evaluate the performance of LVLMs on LLaVA-Bench (In-the-Wild), we use GPT-4o as the evaluator. The prompt template adapted from (Gong et al., 2024) is shown in Table 7, with an additional metric, Naturalness, introduced to assess the fluency and coherence of the generated language. For each sample, GPT-4o is provided with the original image, the baseline LVLM output, and the output from the VHR-enhanced model. The evaluation focuses on three key aspects: accuracy, detailedness, and naturalness, with particular emphasis on the reduction of hallucinations in the VHR-enhanced responses compared to the baseline.

![](images/791fc5b4cf901c86b984beb4cf08874f7d9dbde402019abcda155650093b97fb.jpg)

Please describe this image in detail.

![](images/73353ed539e3ff33948fd868fd604369f8eed1edd2a0127c3c886ec45789bb3b.jpg)

![](images/15e03d395658eee75b1ca7e47983968554d8906dcf367c9cade909843d84ade2.jpg)  
Please describe this image in detail.

![](images/fa1862609d22dbe9ef72aa35c35e31f82bdb47315626441ab0c403827ecc8885.jpg)

![](images/113eff59068b0507fa663c2902fcb6b23ba3ebf98480633b3d9c7b7e77e0b234.jpg)

The image captures a beautiful beach scene with two people walking along the shoreline. Both individuals are carrying surfboards, likely preparing to enjoy a day of surfing. The surfboards are visible in their hands, and the people are positioned close to the water. In the background, there are several cars parked near the beach, indicating that this is a popular spot for visitors. The cars are scattered around the area, with some closer to the water and others further back.

![](images/4ffc936b52309d3cb372f49349ce07c0b6d17613a4d3ac526917195a8241df53.jpg)

![](images/f9b7cbb96947e2343979ae699018bbe88ff506475965c752084ffc2531786b38.jpg)

The image captures a beautiful sunset over a beach, with two people walking along the water's edge. Both individuals are holding surfboards, likely preparing for a surfing session. The surfboards are positioned horizontally, with one person's surfboard on the left side and the other person's surfboard on the right side. In the background, there are several buildings, including a tall one on the left side and another one on the right side. The combination of the sunset, the beach, and the surfboards creates a serene and picturesque scene.

The image features a red and yellow train traveling down the tracks. The train is quite large, occupying a significant portion of the scene. The train is passing under a bridge, which can be seen in the background. There are several people visible in the image, likely passengers or bystanders, scattered around the train and the surrounding area. Some of them are closer to the train, while others are further away, possibly waiting for the train to come to a stop.

![](images/190fe46a8871e50f94655a2ca74e543051bb40a274fbc4d8a996d60da4cccab6.jpg)

![](images/74eb751bcfd00bb5bb8216ab2c3ca098b9cf6181ae7bc29b51f85c7891fb128d.jpg)

The image features a red and gray train traveling down the tracks, with a large red train car positioned in the middle of the scene. The train is moving through a city, with buildings visible in the background. The train appears to be a DB Schenker train, which is a German rail company. The train is traveling on a steel track, providing a sense of speed and movement.

Figure 9: More examples of VHR eliminating the hallucinated content.  
![](images/2dae43651be937a64d75aca694369917bbb6ac86f14b9cd81da6b0b08ea5d5e2.jpg)

![](images/7b1caa29a72228a8808b8c2f28b99d5cde8ac5792c7ae125be0d57dbf256bd02.jpg)

![](images/4349e6d76afefe4412501e8deb096f92138102a66309951ef4442d134c76d91c.jpg)

![](images/bfc3a19db94add90cc0cc1039f3b6c9477e988d250992d70457b72313ce3df68.jpg)

![](images/6c3c0ecc76063373e884880ca7bfea50766e9a1a40111f7917bcc90c1dcca2a7.jpg)

![](images/e9f2d06243644f84d928307a3ce212dc6987b1560550af13327c7b210d46d9d0.jpg)  
Figure 10: More examples of the VHD scores during different generation steps of different samples.

![](images/a221d8cdd0488795545949988b64a9ea75fcb1483546e19725210f8f99663935.jpg)  
Table 7: The prompt used for GPT-4V evaluation.