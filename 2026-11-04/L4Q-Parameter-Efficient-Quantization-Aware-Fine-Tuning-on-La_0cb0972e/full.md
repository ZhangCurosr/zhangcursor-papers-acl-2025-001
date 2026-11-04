# L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models

Hyesung Jeon<sup>1</sup> Yulhwa Kim<sup>2</sup>\* Jae-Joon Kim<sup>1</sup>∗ <sup>1</sup>Seoul National University <sup>2</sup>Sungkyunkwan University {hjeon2k, kimjaejoon}@snu.ac.kr yulhwakim@skku.edu

## Abstract

Due to the high memory and computational costs associated with large language models (LLMs), model compression techniques such as quantization, which reduces inference costs, and parameter-efficient fine-tuning (PEFT) methods like Low-Rank Adaptation (LoRA), which reduce training costs, have gained significant popularity. This trend has spurred active research into quantization-aware PEFT techniques, aimed at maintaining model accuracy while minimizing memory overhead during both inference and training. Previous quantization-aware PEFT methods typically apply post-training quantization (PTQ) to pre-trained LLMs, followed by PEFT to recover accuracy loss. Meanwhile, this approach has limitations in recovering the accuracy loss. In this paper, we propose L4Q, a method that integrates Quantization-Aware Training (QAT) with LoRA. By employing a memory-optimized layer design, L4Q signifi cantly reduces QAT’s memory overhead, making its training cost comparable to LoRA, while preserving the advantage of QAT in producing fully quantized LLMs with high accuracy. Our experiments demonstrate that this combined approach to quantization and fine-tuning achieves superior accuracy compared to decoupled finetuning schemes, particularly in 4-bit and 3-bit quantization, positioning L4Q as an efficient QAT solution. Using the LLaMA and Mistral models with instructional datasets, we showcase L4Q’s capabilities in language tasks and few-shot learning.

## 1 Introduction

Given their impressive scalability, Large Language Models (LLMs) such as GPT, OPT, PaLM, and LLaMA (Brown et al., 2020; Zhang et al., 2022; Chowdhery et al., 2024; Touvron et al., 2023a,b) have become popular in natural language processing. However, their substantial memory and computational demands pose challenges for practical deployment, making model compression (Han et al., 2015) crucial for LLM deployment. Quantization is a prominent method that reduces model size by lowering the bit precision of model parameters (Hubara et al., 2017), so LLM quantization has been actively studied (Liu et al., 2024; Xiao et al., 2023; Frantar et al., 2023; Dettmers and Zettlemoyer, 2023). These quantization methods are generally divided into two categories: quantization-aware training (QAT) and post-training quantization (PTQ). QAT effectively reduces the quantization error by integrating quantization into the training process, where both the model weights and the quantization parameters are trained together (Esser et al., 2020; Bhalgat et al., 2020). However, applying QAT to LLM is challenging due to its significant memory overhead. As a result, PTQ, which applies quantization without retraining the entire pre-trained model weights and with minimal calibration of the quantization parameters, is widely adopted for LLM quantization (Xiao et al., 2023; Lin et al., 2024; Heo et al., 2024).

![](images/4b2cfa3d5c45180a44f069256fd49caf62328d3930a52f28da00a9a56260d996.jpg)  
Figure 1: A diagram of accuracy and inference speedup of the quantized or fine-tuned LLaMA-1 7B models. L4Q produces fast and accurate quantized model.

Concurrently, to enhance the problem-solving abilities of LLMs for specific applications, finetuning pre-trained LLMs on downstream tasks is crucial as it improves accuracy on target tasks and related tasks (Wei et al., 2022; Scialom et al., 2022). However, fine-tuning is a resource-intensive process due to the large number of model weights involved. Parameter-efficient fine-tuning (PEFT) addresses this issue (Hu et al., 2022; Li and Liang, 2021; Liu et al., 2022a; Wang et al., 2023) by training a small subset of parameters while freezing the majority of pre-trained weights. One of the most prominent techniques within PEFT is Low-Rank Adaptation (LoRA) (Hu et al., 2022), which inserts trainable rank decomposition matrices into each layer to represent updates to the frozen weights.

The integration of quantization and PEFT holds significant potential for developing efficient and accurate LLMs for downstream tasks. Recent research has introduced quantization-aware PEFT approaches to achieve high-quality quantized models (Dettmers et al., 2024b; Kim et al., 2024; Xu et al., 2023; Li et al., 2024). Previous works involve a two-stage optimization strategy: first, a PTQ technique, such as GPTQ (Frantar et al., 2023), is applied to pre-trained LLMs for compression. Then, these quantized LLMs undergo PEFT, such as LoRA, where quantized weights are kept fixed and only the LoRA parameters are fine-tuned. While fine-tuning can mitigate the effects of quantization errors, separating quantization and finetuning into distinct stages hinders the models from achieving the best accuracy. Furthermore, as highprecision LoRA parameters are adopted alongside the quantized weight matrix, these methods eventually produce mixed-precision models, which limits the efficiency of full quantization during inference. Recently, QA-LoRA (Xu et al., 2023) addresses this issue by strictly constraining the LoRA parameter structure to integrate with quantization parameters, but this constraint can limit the fine-tuning capability.

In this paper, we propose a novel quantizationaware fine-tuning technique, named L4Q (Lowrank adaptive Learning quantization for LLMs). L4Q addresses the limitations of PTQ-based PEFT methods by introducing QAT alongside LoRA. While QAT have advantages in reducing quantization error and LoRA enables memory-efficient training, their straightforward integration diminishes the benefits of each approach. Therefore, L4Q carefully integrates these two approaches to properly leverage their advantages. First, L4Q applies the quantization process after fully combining the model weights and LoRA parameters in the linear layer. This approach produces a fully-quantized model that enables memory-efficient and fast inference without limiting the training capabilities of either QAT or LoRA. Moreover, to preserve the memory-efficient nature of LoRA during training, we design the backpropagation path of L4Q to eliminate the need to store weight gradients required for QAT. Finally, the full integration of QAT and LoRA in the proposed L4Q allows for the joint optimization of both the quantization and LoRA parameters, thereby improving the quality of the quantized LLMs. As a result, L4Q significantly improves the accuracy of quantized models while maintaining low memory costs during both inference and training, and achieves inference speed comparable to state-of-the-art approaches, making it a more effective solution compared to previous works, as illustrated in Figure 1.

## 2 Backgrounds

## 2.1 PEFT with LoRA

LoRA inserts the rank-decomposition matrices composed of a pair of parameter matrices $A \ \in$ $R ^ { r \times i }$ and $B \in R ^ { o \times r }$ . Here, i and o represent the size of input and output dimensions of the original weight matrix, respectively. $r \ll i ,$ o is the rank of the LoRA matrices, and α is a constant that adjusts the influence of the LoRA matrices. During the fine-tuning process, the pre-trained weight matrix $W _ { 0 } \in R ^ { o \times i }$ is frozen, preserving the pre-trained features. For a given input activation $X \in R ^ { i \times s \times b }$ (s: sequence length, $b \colon$ batch size), the output $Y \in \bar { R } ^ { o \times s \times b }$ of a layer utilizing LoRA is computed as follows:

$$
Y = W _ { 0 } X + \alpha B A X\tag{1}
$$

The fine-tuning of the LoRA parameters is guided by the gradient of a loss function $L ,$ which is calculated with respect to each parameter matrix. The gradients are derived as follows:

$$
\frac { \partial L } { \partial A } = \alpha \frac { \partial L } { \partial \tilde { X } } X ^ { \top } , \quad \frac { \partial L } { \partial B } = \alpha \frac { \partial L } { \partial Y } \tilde { X } ^ { \top }\tag{2}
$$

Here, $\tilde { X } : = A X$ represents the intermediate input activation of B, which is the transformation of X by A. These gradients guide the adjustment of the LoRA parameters to minimize the loss and more accurately approximate the necessary updates to the original model weights.

## 2.2 Quantization

Uniform quantization is a widely used quantization scheme due to its simplicity and broad compatibility with various computing kernels and hardware units (Liu et al., 2022b). Therefore, we refer to the term ‘quantization’ specifically as uniform quantization throughout this paper. A common practice is to organize a quantization group consisting of a certain number of consecutive weight elements that share the same quantization scale s and zero-point (bias) b.

The weights W within the quantization group are quantized according to the following equation:

$$
\tilde { w } = \mathrm { r o u n d } ( \mathrm { c l a m p } ( \frac { W - b } { s } , Q _ { N } , Q _ { P } ) )\tag{3}
$$

Here, w˜ denotes the quantized integer value. Clamping is applied within the range $Q _ { N } = - 2 ^ { n - 1 }$ to $Q _ { P } = 2 ^ { n - 1 } - 1$ , where n is the bit-width, followed by the rounding function. We also note that $W _ { q } : = \tilde { w } \times s + b$ represents the dequantized version of the quantized weight, which is adjusted using s and b from w˜ to approximate the original weight.

During QAT, the straight-through estimator (STE) approximates the derivative of the rounding function with an identity function (Bengio et al., 2013; Choi et al., 2018; Esser et al., 2020), enabling gradients to propagate through non-differentiable rounding operations and allowing effective weight parameter training. LSQ (Esser et al., 2020) and LSQ+ (Bhalgat et al., 2020) extend this process by training quantization parameters s and b, alongside the model weights. This tuning scheme provides finer control over quantization, improving model accuracy. The quantization parameters tuning during backpropagation proceeds by using the chain rule via $W _ { q }$ . This is denoted as follows:

$$
\frac { \partial L } { \partial s } = \frac { \partial L } { \partial W _ { q } } \frac { \partial W _ { q } } { \partial s } , \quad \frac { \partial L } { \partial b } = \frac { \partial L } { \partial W _ { q } } \frac { \partial W _ { q } } { \partial b }\tag{4}
$$

As a consequence, the backpropagation process requires the weight gradient $\frac { \bar { \partial } L } { \partial W _ { q } }$ and the computation of the terms $\frac { \partial W _ { q } } { \partial s } , \frac { \partial W _ { q } } { \partial b }$ regarding the non-linear STE function.

$$
\frac { \partial W _ { q } } { \partial s } = - w + \tilde { w } , \quad \mathrm { i f } Q _ { N } \leq w \leq Q _ { P }\tag{5}
$$

$$
\frac { \partial W _ { q } } { \partial b } = 1 , \mathrm { i f } w < Q _ { N } o r w > Q _ { P }\tag{6}
$$

More details on QAT with LSQ and LSQ+ are provided in Appendix A.1.

## 2.3 LLM Quantization

Quantization compress LLMs by lowering the bit precision of model parameters (Hubara et al., 2017). A key challenge is the introduction of quantization errors that reduce model accuracy, leading to extensive research aimed at mitigating these losses through calibration or training. A notable examples of PTQ for LLM compression are GPTQ (Frantar et al., 2023) and OmniQuant (Shao et al., 2023). In contrast, QAT integrates quantization into the training process, adaptively training model parameters to account for quantization errors during training, ensuring that the quantized model retains much of its accuracy and functionality through training. Despite its advantages, QAT faces challenges, primarily due to its high training overhead, which limits its use in resource-intensive models like LLMs. The memory overhead of QAT stems from storing weight gradients and their optimizer states, each requiring multiple times the memory of the weights. Hence, even applying QAT to a 7B model requires approximately 80GB of memory.

## 2.4 Quantization-Aware PEFT

To improve the accuracy of quantized LLMs, recent research has introduced quantization-aware PEFT approaches (Dettmers et al., 2024b; Kim et al., 2024; Xu et al., 2023; Li et al., 2024). Among these, QLoRA (Dettmers et al., 2024b), QA-LoRA (Xu et al., 2023), and LoftQ (Li et al., 2024) stand out as notable methods. As illustrated in Figure 2, QLoRA begins by applying PTQ to a pre-trained model. After this initial quantization, LoRA finetuning is performed, with the quantized weight parameters kept frozen. This allows the method to correct quantization errors during the fine-tuning. However, QLoRA introduces computational inefficiencies during inference due to the additional forward path on LoRA parameters. This inefficiency arises because the high-precision LoRA parameters and low-precision quantized weights cannot be merged into low-precision values. Advanced methods (Li et al., 2024; Qin et al., 2024) that build upon QLoRA and share its layer structure also suffer from this issue. We further examine the impact of this unmerged LoRA path on inference efficiency by comparing the speed of fully-quantized models with mixed-precision models in Section 4.

QA-LoRA (Xu et al., 2023) addresses the issue of high-precision LoRA parameters by modifying the structure of the LoRA matrix, allowing these parameters to be integrated with the quantized weights after training (Figure 2). The input dimension of the LoRA matrix A is set to the number of weight quantization groups. This adjustment ensures that each element of BA corresponds directly with individual quantization groups, enabling the LoRA parameters to be seamlessly integrated into the quantization bias as b′ = b αBA at the end of training. Hence, QA-LoRA shares the same objective as our work. However, this solution presents a new challenge: the constrained LoRA structure in this setup limits the model’s ability to achieve optimal accuracy during the PEFT stage.

![](images/26c00d6734efc452f8e87509ec7d308f112703991cf05181472e45d96c0ceff3.jpg)  
Figure 2: A categorization of training scheme and inference strategy of QA-LoRA, QLoRA, QAT-LoRA, and L4Q. Compared to QA-LoRA, L4Q utilizes higher optimization ability with non-constrained LoRA parameters and quantization parameters. Additionally, compared to QLoRA and QAT-LoRA, L4Q exploit fully-quantized weights rather than the mixed-precision weights during inference and perform a solid co-optimization of parameters.

A broader issue with existing quantization-aware PEFT methods is that fine-tuning begins from a prequantized model with inherent quantization errors, which is suboptimal compared to starting from a pre-trained model. LoftQ attempts to mitigate these errors by approximating them with LoRA using iterative singular-value decomposition (SVD). However, this approach still cannot achieve a single forward path due to the high-precision LoRA parameters, limiting subsequent adaptation. These challenges underscore the need for further research to improve quantization-aware PEFT techniques, focusing on enhancing both quantization and PEFT processes for better accuracy and inference efficiency in LLMs.

## 3 Methods

## 3.1 Straight Integration of QAT and LoRA

QAT-LoRA One of the key principles in our proposed L4Q scheme is the integration of the QAT and LoRA to facilitate the simultaneous calibration for quantization and fine-tuning on downstream tasks. To achieve this, we begin with a straightforward integration of QAT and LoRA, referred to as QAT-LoRA, which serves as our baseline approach for combining QAT and PEFT.

In QAT-LoRA, pre-trained weights are frozen, while LoRA parameters are added to the linear layers (Figure 2). Additionally, quantization scales and bias parameters are introduced, similar to advanced QAT techniques like LSQ, which are crucial for calibrating the quantization function. Freezing the weights reduces the need for optimizer states, while a small number of LoRA and quantization parameters are introduced to approximate updates to the weight matrix and to update the quantization function, respectively. This results in more efficient memory usage compared to standard QAT. Detailed analysis results of the memory efficiency of QAT-LoRA is further discussed in Section 4.

Limitations of QAT-LoRA While the previous section introduced a straightforward integration of QAT and LoRA—where quantized weights and

LoRA parameters are maintained separately—this approach presents several limitations. First, although freezing the pre-trained weights eliminates the need for optimizer states, weight gradients $\frac { \partial L } { \partial W _ { q } }$ must still be stored to update the quantization parameters, as shown in Equation 5 and Equation 6. As a result, QAT-LoRA still incurs memory overhead from weight gradients, undermining the memory efficiency benefits of LoRA finetuning. Secondly, QAT-LoRA produces a mixedprecision model with both quantized weights and high-precision LoRA parameters. This mixedprecision approach negates the advantages of LLM quantization, similar to previous methods such as QLoRA and LoftQ discussed in Section 2.4. Lastly, the gradient updates for quantization and LoRA parameters are decoupled, limiting the potential for comprehensive optimization across the model. As outlined in Equations 5 and 6, updates to the quantization parameters rely on the quantized weight matrix $W _ { q } ,$ , while updates to the LoRA parameters depend on weights A and B. This limits the effectiveness of model training, as it prevents holistic adjustments where changes in LoRA parameters could directly influence quantization adjustments and vice versa.

To address these challenges, we introduce L4Q, an enhanced integration of QAT and LoRA. L4Q features an advanced layer design that seamlessly integrates QAT and LoRA. By applying quantization after merging the weights and LoRA parameters, along with a custom backpropagation path that reduces the memory overhead from the complex quantization and LoRA processes, L4Q effectively overcomes the primary challenges encountered with QAT-LoRA.

## 3.2 L4Q: Low-rank Adaptive Quantization-Aware Fine-tuning

Fully-Quantized Linear Layer As highprecision LoRA weights introduces inference overhead, it is crucial to design a fully-quantized linear layer. In this context, L4Q first combines the original weights $W _ { 0 }$ and the LoRA parameters BA into a unified parameter matrix:

$$
W _ { c o m b } = W _ { 0 } + \alpha B A\tag{7}
$$

Then, quantization is applied to the fully combined weight $W _ { c o m b }$ and produces w˜ and $W _ { q } { \mathrm { : } }$

$$
\tilde { w } = \mathrm { r o u n d } ( \mathrm { c l a m p } ( \frac { W _ { c o m b } - b } { s } , Q _ { N } , Q _ { P } ) )\tag{8}
$$

$$
W _ { q } = \tilde { w } \times s + b\tag{9}
$$

In this way, during inference, L4Q only uses quantized weights $W _ { q } ,$ simplifying the forward path of the linear layer from Equation 1 to as follows:

$$
Y = W _ { q } X\tag{10}
$$

While QA-LoRA also achieves fully-quantized linear layers by introducing constraints on the LoRA parameter structure, the proposed L4Q imposes no such restrictions. This flexibility allows L4Q to fully leverage the benefits of LoRA-based finetuning, all with fully-quantized linear layers.

Memory Efficient QAT As discussed in the previous section, QAT requires weight gradients to train quantization parameters s and b. Since weight gradients are a major source of memory overhead during training, we compute these gradients locally in the backpropagation path as follows:

$$
\frac { \partial L } { \partial W _ { q } } = \frac { \partial L } { \partial Y } X ^ { \top }\tag{11}
$$

We then use these weight gradients to calculate gradients of s and b with Equation 5 and 6. Once the gradient computation for the linear layer is complete, the weight gradients are immediately flushed to conserve memory.

Efficient LoRA Training Unlike the conventional LoRA backward path which does not involve the weight $W _ { 0 }$ as described in Equation 2, computing the gradients of the LoRA parameters in the L4Q linear layer follows a more complicated backpropagation path, tracing back from Equation 10 to Equation 7. Because a non-linear quantization function is applied after LoRA during quantization in L4Q, the gradients of the LoRA parameters depend on both the weights (in their quantized form w) and the weight gradients. This process can be described as follows:

$$
\frac { \partial L } { \partial A } = \frac { \partial L } { \partial W _ { q } } \frac { \partial W _ { q } } { \partial A } , \quad \frac { \partial L } { \partial B } = \frac { \partial L } { \partial W _ { q } } \frac { \partial W _ { q } } { \partial B }\tag{12}
$$

We reuse the weight gradient $\frac { \partial L } { \partial W _ { q } }$ that have been computed previously for quantization parameter update. Therefore, we only compute $\frac { \partial W _ { q } } { \partial A }$ and $\frac { \partial W _ { q } } { \partial B }$ to obtain the gradients of LoRA parameters. Both terms are derived by applying the chain rule from Equation 8 to Equation 7. Since Equation 8 contains a rounding function, we apply STE and clamping with conditional gradient propagation to $\frac { \partial { W } _ { q } } { \partial A }$ and $\frac { \partial W _ { q } } { \partial B }$ . This leads to the following expressions for $\frac { \partial \bar { W _ { q } } } { \partial A }$ and $\frac { \partial W _ { q } } { \partial B }$

$$
\frac { \partial W _ { q } } { \partial A } = \left\{ \begin{array} { l l } { \alpha B ^ { \top } , } & { \mathrm { i f } Q _ { N } \leq w \leq Q _ { P } } \\ { 0 , } & { o t h e r w i s e } \end{array} \right.\tag{13}
$$

$$
\frac { \partial W _ { q } } { \partial B } = \left\{ \begin{array} { l l } { \alpha A ^ { \top } , } & { \quad \mathrm { i f } Q _ { N } \leq w \leq Q _ { P } } \\ { 0 , } & { \quad o t h e r w i s e } \end{array} \right.\tag{14}
$$

Therefore, the proposed L4Q efficiently processes LoRA training by simply reusing the weight gradients computed for QAT parameter training. For more detailed explanations of the gradient calculation in L4Q, please refer to Appendix A.2, and the memory efficiency of L4Q will be further examined in Section 4.

Joint Quantization and Low-rank Adaptation Since $\frac { \partial L } { \partial W _ { q } }$ is involved in the gradient calculation for the LoRA parameters (Equation 12), the proposed L4Q ensures that the impact of quantization is directly reflected in the updates to the LoRA parameters. This enables the joint optimization of LoRA parameters and the quantization process, enhancing the accuracy of the fully-quantized LLMs.

In summary, the proposed L4Q produces a fullyquantized model for memory-efficient and fast LLM inference by fully integrating the model weights and LoRA parameters prior to the quantization process. Additionally, the training process of L4Q is memory-efficient due to careful handling of gradient computation for quantization. Finally, L4Q can improve the accuracy of quantized LLMs through the joint optimization of the quantization and LoRA parameters.

## 3.3 Quantization Parameter Initialization

LSQ+, a previous QAT approach, sets the quantization range based on the weight standard deviation. This is effective for CNNs, but it does not work well on LLMs because activation outliers and their corresponding salient weights are crucial for model performance (Xiao et al., 2023; Dettmers et al., 2024a; Lin et al., 2024). Hence, when initializing quantization parameters, it is important to address the outlier sensitivity of LLMs. To address this, we propose $\mathrm { L 4 Q } _ { \mathrm { i n i t } }$ , a symmetric quantization scheme that minimizes clipping errors by using a conservative scale to capture both minimum and maximum outliers. The quantization scale is defined by the following equation:

$$
s = M a x ( | \frac { M i n ( W ) } { Q n } | , | \frac { M a x ( W ) } { Q p } | )\tag{15}
$$

![](images/bda288e3d451153b4f3a6367fabef403fe501a2e8c6279ea96b10a210a3b6f55.jpg)  
Figure 3: MMLU 5-shot results and clipping errors for LLaMA-2 7B models after 100 training steps. The results include clipping errors at both initialization and post-training for the LSQ+, symmetric, asymmetric, and L4Q initialization methods.

We evaluate models trained with L4Q using various initialization methods, including conventional min/max-based symmetric and asymmetric schemes, and ${ \mathrm { L S Q } } + _ { \mathrm { i n i t } }$ schemes. We compare the accuracy and the sum of clipping errors caused by overflowed outliers at both initialization and after training. As shown in Figure 3, $\mathrm { L S Q + _ { \mathrm { i n i t } } }$ and symmetric initialization result in higher clipping errors. While asymmetric initialization avoids clipping initially, its tight range, defined by minimum-tomaximum values, becomes insufficient as weights evolve, leading to increased clipping errors during fine-tuning. In contrast, $\mathrm { L 4 Q } _ { \mathrm { i n i t } }$ accounts for the broader weight distribution in LLMs, effectively reducing clipping errors. As a result, $\mathrm { L 4 Q } _ { \mathrm { i n i t } }$ achieves the highest accuracy, whereas $\mathrm { L S Q + _ { \mathrm { i n i t } } }$ struggles to recover from quantization errors. A detailed explanation of $\mathrm { L S Q + i n i t } ,$ symmetric, asymmetric, and L4Q initialization methods is provided in Appendix B.

## 4 Experiments

## 4.1 Experimental Settings

Target Foundation Models OpenLLaMA<sup>1</sup> 3B model, LLaMA family models (Touvron et al., 2023a,b) 7B to 33B, and Mistral-v0.1 7B (Jiang et al., 2023) model are used for the evaluation.

Baselines We compare the proposed L4Q with previous quantization methods and quantizationaware PEFT methods. The baseline quantization methods considered are LSQ for QAT, and GPTQ and OmniQuant for PTQ. For quantization-aware PEFT baselines, we include QLoRA, QA-LoRA, and LoftQ. We apply uniform quantization in our experiments to ensure consistency across methods. The methods that were originally deployed with non-uniform quantization are denoted with an asterisk (\*). For methods with LoRA adapters, including L4Q, the adapters are initialized following the original scheme: A is initialized using Kaiminguniform, while B is initialized to zeros. We also note that the term fully-quantized refers to a condition in which all linear layers are quantized, while non-linear components and the head layer remain in higher precision.

Evaluation Setups We establish the L4Q framework based on the open-source frameworks: Lit-$\mathrm { G P T } ^ { 2 }$ and huggingface transformers<sup>3</sup>. The models are symmetrically quantized with quantization group size of 128 for the LLaMA and Mistral models, and 64 for the OpenLLaMA models due to its small channel size. Also, the models are finetuned with a LoRA rank r of 4 by default, and 8 for Mistral. We double the rank r in QA-LoRA for a fair comparison in terms of the number of LoRA parameters. Further details on the effect of rank size and quantization group size can be found in Appendix C. We use the Stanford-Alpaca (Taori et al., 2023), a dataset that consists of 50k training samples and 2k validation samples generated from the GPT 3.5 (Brown et al., 2020). We use bfloat16 precision for numerically stable fine-tuning. All experiments are conducted on an NVIDIA A100 80GB GPU. Detailed hyperparameter settings for fine-tuning are provided in Appendix D.

Evaluation Metrics We evaluate the accuracy of LLMs on Commonsense QA (CSQA) (Gao et al., 2021) and MMLU (Hendrycks et al., 2021) benchmarks. The CSQA benchmark comprises seven multiple-choice tasks designed to evaluate language models (Zellers et al., 2019; Bisk et al., 2020; Clark et al., 2018; Sakaguchi et al., 2020; Clark et al., 2019; Talmor et al., 2019). The MMLU benchmark spans four subject categories made up of 57 subcategories of language tasks.

## 4.2 Evaluation Results

Memory Cost for Fine-Tuning We measure the peak memory usage during fine-tuning of 4-bit LLMs, including L4Q and QAT-based baselines, as shown in Table 1. While QAT and QAT-LoRA incur significantly higher memory costs compared to LoRA, L4Q’s memory usage remains comparable to that of LoRA. This analysis shows that L4Q effectively balances the advantages of QAT and the memory efficiency of LoRA. Further analysis of the training efficiency of L4Q and the baseline methods can be found in Appendix E.

<table><tr><td colspan="2">OpenLLaMA</td><td rowspan="2">7B</td><td rowspan="2">LLaMA 13B</td><td rowspan="2">33B</td></tr><tr><td>Methods</td><td>3B</td></tr><tr><td>LoRA</td><td>15.1</td><td>25.1</td><td>43.8</td><td>71.9</td></tr><tr><td>QAT</td><td>44.2</td><td>79.5</td><td>00M</td><td>00M</td></tr><tr><td>QAT-LoRA</td><td>22.6</td><td>41.9</td><td>70.6</td><td>00M</td></tr><tr><td>L4Q</td><td>15.3</td><td>25.4</td><td>44.3</td><td>73.2</td></tr></table>

Table 1: Memory cost (GB) for fine-tuning LLMs on NVIDIA A100 GPU. (OOM: Out of Memory)

![](images/128a0b72686878aaba9888d554a089131af0c7b89662d14397643fde8778e195.jpg)  
Figure 4: The average inference speedup of quantized models compared to pre-trained models.

Inference Speedup We measure the inference speed of 16-bit pre-trained models and quantized models using LLaMA-1 models. The average speedup of quantized models compared to fullprecision 16-bit models is reported in Figure 4. The quantized models include fully-quantized 4- bit models (L4Q and QA-LoRA), which contain only quantized parameters, and mixed-precision 4&16-bit models ${ { ( \mathrm { L o f t } 0 } ^ { * } }$ , QLoRA<sup>\*</sup>, and QAT-LoRA), which use additional 16-bit LoRA parameters. The inference speed was measured with input batch sizes ranging from 1 to 64. The 4-bit models achieve a speedup of 1.8 to 2.3 over the pre-trained models. Moreover, these 4-bit models achieve a 1.4 to 1.6 speedup compared to mixed-precision models, which are also quantized versions of LLMs. This demonstrates that the full integration of QAT and LoRA in L4Q plays a crucial role in inference speedup. Further analysis and details on speedup can be found in Appendix F.

Accuracy Results We compare the CSQA and MMLU benchmark accuracy of baselines and L4Q. Table 2 and Table 3 present a comprehensive comparison between baselines and the proposed L4Q. Since previous quantization-aware PEFT methods involve a fine-tuning stage after PTQ, they generally achieve higher accuracy compared to PTQ methods. L4Q further enhances accuracy by incorporating the QAT strategy, achieving highest accuracy compared to the baselines, and attaining 4-bit model accuracy comparable to that of 16-bit models. Moreover, L4Q consistently outperforms QAT-LoRA that keeps quantization and LoRA parameters decoupled. This highlights the advantage of L4Q in accuracy through the joint optimization of quantization and LoRA parameters. This impact is more pronounced in 3-bit quantization, as some PTQ-based PEFT approaches experience significant accuracy degradation. The detailed results are presented in Appendix G.

Table 2: Accuracy (%) evaluation results with 4-bit quantization. We present the bit precision under each methods. The notation ‘4&16’ refers to the use of 16-bit LoRA parameters with the quantized weights for inference.
<table><tr><td>Model</td><td>Benchmark</td><td>Pre-trained 16</td><td>LoRA 16</td><td>GPTQ 4</td><td>OmniQ 4</td><td>LoftQ* 4&amp;16</td><td>QLoRA 4&amp;16</td><td>QA-LoRA 4</td><td>QAT-LoRA 4&amp;16</td><td>L4Q 4</td></tr><tr><td>OpenLLaMA 3B</td><td>CSQA</td><td>54.8</td><td>55.9</td><td>50.7</td><td>54.1</td><td>54.2</td><td>54.4</td><td>54.5</td><td>54.6</td><td>55.0</td></tr><tr><td>LLaMA-3 8B</td><td>CSQA</td><td>65.6</td><td>67.2</td><td>57.9</td><td>64.9</td><td>63.3</td><td>58.6</td><td>58.0</td><td>65.7</td><td>66.8</td></tr><tr><td>LLaMA-1 7B</td><td>CSQA</td><td>61.7</td><td>63.4</td><td>59.4</td><td>58.1</td><td>62.6</td><td>61.3</td><td>61.3</td><td>62.4</td><td>62.7</td></tr><tr><td></td><td> $\mathbf { M M L U } _ { \mathrm { 0 - s h o t } }$ </td><td>32.5</td><td>36.3</td><td>28.3</td><td>30.9</td><td>33.0</td><td>32.8</td><td>34.5</td><td>33.8</td><td>34.9</td></tr><tr><td></td><td> $\mathbf { M M L U } _ { 5 \mathrm { - s h o t } }$ </td><td>35.1</td><td>36.7</td><td>32.7</td><td>33.3</td><td>35.1</td><td>33.6</td><td>35.6</td><td>34.8</td><td>35.7</td></tr><tr><td>LLaMA-1 13B</td><td>CSQA</td><td>63.8</td><td>65.2</td><td>63.5</td><td>60.4</td><td>64.2</td><td>63.8</td><td>62.5</td><td>64.4</td><td>64.5</td></tr><tr><td></td><td> $\mathbf { M M L U } _ { \mathrm { 0 - s h o t } }$ </td><td>43.6</td><td>44.3</td><td>40.1</td><td>42.6</td><td>42.4</td><td>42.1</td><td>42.4</td><td>42.0</td><td>43.2</td></tr><tr><td></td><td> $\mathbf { M M L U } _ { 5 \mathrm { - s h o t } }$ </td><td>46.3</td><td>47.0</td><td>45.7</td><td>45.7</td><td>45.4</td><td>45.9</td><td>45.8</td><td>45.5</td><td>46.0</td></tr><tr><td>LLaMA-1 33B</td><td>CSQA</td><td>67.4</td><td>68.3</td><td>65.7</td><td>62.9</td><td>67.4</td><td>66.2</td><td>65.3</td><td>67.3</td><td>67.5</td></tr><tr><td></td><td> $\mathbf { M M L U } _ { \mathrm { 0 - s h o t } }$ </td><td>53.0</td><td>54.4</td><td>51.4</td><td>52.0</td><td>51.8</td><td>51.0</td><td>48.9</td><td>52.3</td><td>53.3</td></tr><tr><td></td><td> $\mathbf { M M L U } _ { 5 \mathrm { - s h o t } }$ </td><td>56.4</td><td>57.6</td><td>55.7</td><td>55.8</td><td>56.4</td><td>55.6</td><td>55.0</td><td>56.7</td><td>56.7</td></tr><tr><td>LLaMA-2 7B</td><td>CSQA</td><td>61.9</td><td>63.3</td><td>60.7</td><td>59.5</td><td>61.7</td><td>61.3</td><td>61.0</td><td>61.9</td><td>63.6</td></tr><tr><td></td><td>MMLU0-shot</td><td>41.6</td><td>43.9</td><td>37.1</td><td>41.0</td><td>38.5</td><td>38.6</td><td>38.9</td><td>37.9</td><td>40.9</td></tr><tr><td></td><td>MMLU5-shot</td><td>45.4</td><td>46.0</td><td>42.9</td><td>45.4</td><td>43.7</td><td>44.6</td><td>44.4</td><td>43.8</td><td>45.5</td></tr><tr><td>LLaMA-2 13B</td><td>CSQA</td><td>65.0</td><td>66.5</td><td>64.4</td><td>59.9</td><td>64.9</td><td>64.0</td><td>64.5</td><td>64.7</td><td>65.8</td></tr><tr><td></td><td> $\mathbf { M M L U } _ { \mathrm { 0 - s h o t } }$ </td><td>52.1</td><td>52.5</td><td>50.0</td><td>51.8</td><td>51.7</td><td>50.7</td><td>50.4</td><td>50.7</td><td>51.9</td></tr><tr><td></td><td> $\mathbf { M M L U } _ { 5 \mathrm { - s h o t } }$ </td><td>54.8</td><td>55.7</td><td>54.7</td><td>54.7</td><td>54.5</td><td>54.2</td><td>54.1</td><td>53.8</td><td>55.2</td></tr><tr><td>Mistral-v0.1 7B</td><td>CSQA</td><td>66.2</td><td>66.4</td><td>65.3</td><td>64.7</td><td>60.7</td><td>65.8</td><td>65.4</td><td>64.5</td><td>66.1</td></tr><tr><td></td><td> $\mathbf { M M L U } _ { \mathrm { 0 - s h o t } }$ </td><td>60.2</td><td>60.6</td><td>57.6</td><td>58.4</td><td>45.2</td><td>58.7</td><td>56.5</td><td>58.8</td><td>59.0</td></tr><tr><td></td><td> $\mathbf { M M L U } _ { 5 \mathrm { - s h o t } }$ </td><td>62.6</td><td>62.9</td><td>61.0</td><td>61.0</td><td>45.7</td><td>61.1</td><td>61.2</td><td>60.2</td><td>61.4</td></tr></table>

Table 3: Accuracy (%) evaluation results with 3-bit quantization. We present the bit precision under each methods. The notation ‘3&16’ refers to the use of 16-bit LoRA parameters with the quantized weights for inference.
<table><tr><td>Model</td><td>Benchmark</td><td>Pre-trained 16</td><td>LoRA 16</td><td>GPTQ 3</td><td>OmniQ 3</td><td>LoftQ 3&amp;16</td><td>QLoRA* 3&amp;16</td><td>QA-LoRA 3</td><td>QAT-LoRA 3&amp;16</td><td>L4Q 3</td></tr><tr><td>OpenLLaMA 3B</td><td>CSQA</td><td>54.8</td><td>55.9</td><td>52.2</td><td>50.0</td><td>38.1</td><td>51.0</td><td>51.5</td><td>53.2</td><td>54.0</td></tr><tr><td>LLaMA-3 8B</td><td>CSQA</td><td>65.6</td><td>67.2</td><td>53.5</td><td>58.7</td><td>48.6</td><td>55.7</td><td>56.6</td><td>63.1</td><td>63.5</td></tr><tr><td>LLaMA-1 7B</td><td>CSQA</td><td>61.7</td><td>63.4</td><td>53.4</td><td>56.5</td><td>49.8</td><td>59.1</td><td>58.7</td><td>60.7</td><td>61.2</td></tr><tr><td></td><td>MMLU0-shot</td><td>32.5</td><td>36.3</td><td>23.7</td><td>29.0</td><td>23.4</td><td>27.7</td><td>28.0</td><td>30.6</td><td>30.6</td></tr><tr><td></td><td> $\mathbf { M M L U } _ { 5 \mathrm { - s h o t } }$ </td><td>35.1</td><td>36.7</td><td>27.3</td><td>31.6</td><td>23.1</td><td>31.5</td><td>29.1</td><td>31.5</td><td>31.8</td></tr><tr><td>LLaMA-1 13B</td><td>CSQA</td><td>63.8</td><td>65.2</td><td>61.0</td><td>58.9</td><td>54.0</td><td>61.3</td><td>61.1</td><td>63.2</td><td>63.4</td></tr><tr><td></td><td> $\mathbf { M M L U } _ { \mathrm { 0 - s h o t } }$ </td><td>43.6</td><td>44.3</td><td>33.1</td><td>34.8</td><td>25.0</td><td>36.1</td><td>37.5</td><td>38.8</td><td>40.7</td></tr><tr><td></td><td> $\mathbf { M M L U } _ { 5 \mathrm { - s h o t } }$ </td><td>46.3</td><td>47.0</td><td>38.2</td><td>41.6</td><td>25.3</td><td>40.4</td><td>38.2</td><td>40.9</td><td>41.8</td></tr><tr><td>LLaMA-1 33B</td><td>CSQA</td><td>67.4</td><td>68.3</td><td>65.1</td><td>62.3</td><td>54.8</td><td>64.3</td><td>64.6</td><td>67.4</td><td>67.4</td></tr><tr><td></td><td> $\mathbf { M M L U } _ { \mathrm { 0 - s h o t } }$ </td><td>53.0</td><td>54.4</td><td>50.0</td><td>50.2</td><td>24.6</td><td>45.6</td><td>46.1</td><td>50.1</td><td>50.5</td></tr><tr><td></td><td> $\mathbf { M M L U } _ { 5 \mathrm { - s h o t } }$ </td><td>56.4</td><td>57.6</td><td>51.9</td><td>52.4</td><td>24.0</td><td>50.1</td><td>48.7</td><td>50.6</td><td>53.1</td></tr><tr><td>LLaMA-2 7B</td><td>CSQA</td><td>61.9</td><td>63.3</td><td>57.6</td><td>57.9</td><td>34.7</td><td>57.6</td><td>56.3</td><td>57.4</td><td>61.3</td></tr><tr><td></td><td> $\mathbf { M M L U } _ { \mathrm { 0 - s h o t } }$ </td><td>41.6</td><td>43.9</td><td>31.3</td><td>34.3</td><td>22.9</td><td>32.5</td><td>31.0</td><td>31.5</td><td>34.9</td></tr><tr><td></td><td> $\mathbf { M M L U } _ { 5 \mathrm { - s h o t } }$ </td><td>45.4</td><td>46.0</td><td>37.5</td><td>37.7</td><td>24.2</td><td>37.6</td><td>37.5</td><td>36.8</td><td>38.0</td></tr><tr><td>LLaMA-2 13B</td><td>CSQA</td><td>65.0</td><td>66.5</td><td>61.7</td><td>59.9</td><td>39.3</td><td>62.5</td><td>61.7</td><td>64.3</td><td>65.1</td></tr><tr><td></td><td> $\mathbf { M M L U } _ { \mathrm { 0 - s h o t } }$ </td><td>52.1</td><td>52.5</td><td>46.3</td><td>46.3</td><td>23.5</td><td>46.8</td><td>46.4</td><td>45.9</td><td>47.1</td></tr><tr><td></td><td> $\mathbf { M M L U } _ { 5 \mathrm { - s h o t } }$ </td><td>54.8</td><td>55.7</td><td>50.4</td><td>50.2</td><td>26.0</td><td>50.6</td><td>49.9</td><td>48.9</td><td>50.0</td></tr><tr><td>Mistral-v0.1 7B</td><td>CSQA</td><td>66.2</td><td>66.4</td><td>61.8</td><td>61.4</td><td>58.5</td><td>63.0</td><td>62.3</td><td>61.6</td><td>63.1</td></tr><tr><td></td><td> $\mathbf { M M L U } _ { \mathrm { 0 - s h o t } }$ </td><td>60.2</td><td>60.6</td><td>50.5</td><td>54.3</td><td>35.8</td><td>52.2</td><td>50.5</td><td>52.4</td><td>54.5</td></tr><tr><td></td><td> $\mathbf { M M L U } _ { 5 \mathrm { - s h o t } }$ </td><td>62.6</td><td>62.9</td><td>49.6</td><td>55.9</td><td>37.1</td><td>53.6</td><td>51.7</td><td>54.0</td><td>56.2</td></tr></table>

## 5 Conclusion

In this work, we introduce L4Q, a parameterefficient quantization-aware fine-tuning method for large language models. L4Q enables element-wise adaptation of model weights for downstream tasks while simultaneously optimizing quantization parameters. This concurrent optimization ensures that the adaptation parameters effectively account for quantization errors. We demonstrate the efficiency of L4Q, which significantly reduces training resource requirements compared to traditional QAT. Moreover, since the L4Q layer is designed to produce fully quantized low-bit model weights, it maintains inference efficiency, unlike QLoRA, LoftQ, or QAT-LoRA, which results in mixed precision models. The effectiveness of L4Q as a QAT framework is further supported by experimental results in various task evaluations. L4Q consistently achieves superior quality maintenance in language tasks, demonstrating its enhanced adaptability compared to the QAT-LoRA and PTQ-based PEFT methods.

## Limitations

Our work focuses on efficient weight quantization methods for large language models (LLMs), but there are vertical approaches that could further enhance inference efficiency and effectiveness, integrated with our work. Activation quantization offers a chance to further reduce computation costs when combined with weight quantization. Similarly, KV cache compression could minimize memory overhead and latency, especially for longcontext applications. Finally, refinement of LoRA initialization schemes for quantized models may improve accuracy of the fine-tuned models. We believe that integrating these approaches with L4Q could further improve LLM inference efficiency and effectiveness, which we leave to future work.

## Ethical Considerations

While our research contributes to the development and application of machine learning, particularly in language models, we recognize the potential societal implications associated with this work. Based on the claims of the referenced sources — including datasets, code, and models — we believe that the artifacts do not violate personal identification rights or contain offensive content. All datasets, code, and models cited in this paper are publicly accessible and processed solely for research purposes. Our use of these artifacts is consistent with their intended purpose.

## Acknowledgments

This work was supported in part by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.RS-2025-02273157: Development of Low Power Training/Inference Technologies based on AI Semiconductors, RS-2024-00395134, DPU-Centric Datacenter Architecture for Next-Generation AI Devices, No. 2021- 0-01343: Artificial Intelligence Graduate School

Program (Seoul National University)), Samsung Research Funding Center under Project SRFC-TC1603-53, and BK21 FOUR program at Seoul National University. (Corresponding Author: Jae-Joon Kim).

## References

Yoshua Bengio, Nicholas Léonard, and Aaron Courville. 2013. Estimating or propagating gradients through stochastic neurons for conditional computation. Preprint, arXiv:1308.3432.

Yash Bhalgat, Jinwon Lee, Markus Nagel, Tijmen Blankevoort, and Nojun Kwak. 2020. Lsq+: Improving low-bit quantization through learnable offsets and better initialization. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW).

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2020. Piqa: Reasoning about physical commonsense in natural language. In The Associationfor the Advancement ofArtificial Intelligence (AAAI).

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Proceedings ofthe 34th International Conference on Neural Information Processing Systems (NeurIPS).

Jungwook Choi, Zhuo Wang, Swagath Venkataramani, Pierce I-Jen Chuang, Vijayalakshmi Srinivasan, and Kailash Gopalakrishnan. 2018. Pact: Parameterized clipping activation for quantized neural networks. In International Conference on Learning Representations (ICLR).

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sashank Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira,

Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. 2024. Palm: Scaling language modeling with pathways. Journal ofMachine Learning Research (JMLR), 24(1):240:1–240:113.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. Boolq: Exploring the surprising difficulty of natural yes/no questions. In Annual Conference of the North American Chapter of the Associationfor Computational Linguistics (NAACL).

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. Preprint, arXiv:1803.05457.

Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. 2024a. Llm.int8(): 8-bit matrix multiplication for transformers at scale. In Proceedings of the 36th International Conference on Neural Information Processing Systems, NIPS ’22.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2024b. Qlora: Efficient finetuning of quantized llms. In Proceedings of the 37th International Conference on Neural Information Processing Systems (NeurIPS).

Tim Dettmers and Luke Zettlemoyer. 2023. The case for 4-bit precision: k-bit inference scaling laws. In Proceedings ofthe 40th International Conference on Machine Learning (ICML).

Steven K. Esser, Jeffrey L. McKinstry, Deepika Bablani, Rathinakumar Appuswamy, and Dharmendra S. Modha. 2020. Learned step size quantization. In International Conference on Learning Representations (ICLR).

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2023. Gptq: Accurate post-training quantization for generative pre-trained transformers. International Conference on Learning Representations (ICLR), abs/2210.17323.

Leo Gao, Jonathan Tow, Stella Biderman, Shawn Black, Anthony DiPofi, Charlie Foster, Leo Golding, Jasmine Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Lucy Reynolds, Eric Tang, Alex Thite, Ben Wang, Kevin Wang, and Amanda Zou. 2021. A framework for few-shot language model evaluation. Zenodo Repository.

Song Han, Huizi Mao, and William J. Dally. 2015. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. arXiv: Computer Vision and Pattern Recognition.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt.

2021. Measuring massive multitask language understanding. In International Conference on Learning Representations (ICLR).

Jung Hwan Heo, Jeonghoon Kim, Beomseok Kwon, Byeongwook Kim, Se Jung Kwon, and Dongsoo Lee. 2024. Rethinking channel dimensions to isolate outliers for low-bit weight quantization of large language models. In International Conference on Learning Representations (ICLR).

J. Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, and Weizhu Chen. 2022. Lora: Low-rank adaptation of large language models. International Conference on Learning Representations (ICLR).

Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, and Yoshua Bengio. 2017. Quantized neural networks: Training neural networks with low precision weights and activations. Journal ofMachine Learning Research (JMLR), 18(1):6869–6898.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. Preprint, arXiv:2310.06825.

Jeonghoon Kim, Jung Hyun Lee, Sungdong Kim, Joonsuk Park, Kang Min Yoo, Se Jung Kwon, and Dongsoo Lee. 2024. Memory-efficient fine-tuning of compressed large language models via sub-4-bit integer quantization. In Proceedings of the 37th International Conference on Neural Information Processing Systems (NeurIPS).

Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Associationfor Computational Linguistics (ACL) and the 11th International Joint Conference on Natural Language Processing (IJCNLP), pages 4582–4597. Association for Computational Linguistics.

Yixiao Li, Yifan Yu, Chen Liang, Pengcheng He, Nikos Karampatziakis, Weizhu Chen, and Tuo Zhao. 2024. Loftq: Lora-fine-tuning-aware quantization for large language models. In International Conference on Learning Representations (ICLR).

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. 2024. Awq: Activation-aware weight quantization for llm compression and acceleration. In Annual Conference on Machine Learning and Systems (MLSys).

Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin A. Raffel. 2022a. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, pages 1950–1965.

Z. Liu, K. Cheng, D. Huang, E. Xing, and Z. Shen. 2022b. Nonuniform-to-uniform quantization: Towards accurate quantization via generalized straightthrough estimation. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4932–4942. IEEE Computer Society.

Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, and Vikas Chandra. 2024. LLM-QAT: Data-free quantization aware training for large language models. In Findings ofthe Associationfor Computational Linguistics (ACL 2024), pages 467–484. Association for Computational Linguistics.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In International Confer ence on Learning Representations (ICLR).

Haotong Qin, Xudong Ma, Xingyu Zheng, Xiaoyang Li, Yang Zhang, Shouda Liu, Jie Luo, Xianglong Liu, and Michele Magno. 2024. Accurate lorafinetuning quantization of llms via information retention. In International Conference on Machine Learning (ICML).

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2020. Winogrande: An adversarial winograd schema challenge at scale. In The Associationfor the Advancement ofArtificial Intelligence (AAAI).

Thomas Scialom, Tuhin Chakrabarty, and Smaranda Muresan. 2022. Fine-tuned language models are continual learners. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6107–6122. Association for Computational Linguistics.

Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, and Ping Luo. 2023. Omniquant: Omnidirectionally calibrated quantization for large language models. arXiv preprint arXiv:2308.13137.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. Commonsenseqa: A question answering challenge targeting commonsense knowledge. In North American Chapter ofthe Association for Computational Linguistics (NAACL).

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xinyun Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford alpaca: An instruction-following llama model. GitHub Repository.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a. Llama: Open and efficient foundation language models. Preprint, arXiv:2302.13971.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023b. Llama 2: Open foundation and fine-tuned chat models. Preprint, arXiv:2307.09288.

Zhen Wang, Rameswar Panda, Leonid Karlinsky, Rogerio Feris, Huan Sun, and Yoon Kim. 2023. Multitask prompt tuning enables parameter-efficient transfer learning. Preprint, arXiv:2303.02861.

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. 2022. Finetuned language models are zero-shot learners. International Conference on Learning Representations (ICLR), abs/2109.01652.

Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. 2023. Smoothquant: Accurate and efficient post-training quantization for large language models. In Proceedings of the 40th International Conference on Machine Learning (ICML).

Yuhui Xu, Lingxi Xie, Xiaotao Gu, Xin Chen, Heng Chang, Hengheng Zhang, Zhengsu Chen, Xiaopeng Zhang, and Qi Tian. 2023. Qa-lora: Quantizationaware low-rank adaptation of large language models. Preprint, arXiv:2309.14717.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? In Annual Meeting ofthe Associationfor Computational Linguistics (ACL).

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. 2022. Opt: Open pre-trained transformer language models. Preprint, arXiv:2205.01068.

## Appendix

## A Details on Quantization Scale Learning Procedure

## A.1 Quantization parameter update on QAT

From the conditions and notations in Equation 3, Equations 5 and 6 are derived as follows. First, the derivative of s is presented as follows.

$$
\frac { \partial W _ { q } } { \partial s } = \frac { \partial } { \partial s } ( \tilde { w } \times s + b ) = s \frac { \partial } { \partial s } ( \tilde { w } ) + \tilde { w } = s \frac { \partial } { \partial s } ( \mathrm { r o u n d } \cdot \mathrm { c l a m p } ( w ) ) + \tilde { w }\tag{16}
$$

By applying the STE, the rounding function is considered as an identity function. Therefore, the rounding function, combined with a clamp function ˜r := round  clamp, and its derivative is induced as follows. Note that $\begin{array} { r } { w = \frac { W - b } { s } } \end{array}$

$$
\tilde { \mathbf { r } } ( w ) = \left\{ \begin{array} { l l } { Q n , } & { i f w < Q n } \\ { w , } & { i f Q _ { N } \le w \le Q _ { P } } \\ { Q p , } & { i f w > Q _ { P } } \end{array} \right. \quad \frac { \partial } { \partial w } \tilde { \mathbf { r } } ( w ) = \left\{ \begin{array} { l l } { 1 , } & { i f Q _ { N } \le w \le Q _ { P } } \\ { 0 , } & { o t h e r w i s e } \end{array} \right.\tag{17}
$$

By applying the chain rule, the derivation of term $\widetilde { \mathbf { r } } ( w ) = \widetilde { \mathbf { r } } ( ( W - b ) / s )$ is expressed as below.

$$
\frac { \partial } { \partial s } ( \tilde { \mathbf { r } } ( w ) ) = \frac { \partial \tilde { \mathbf { r } } } { \partial w } \frac { \partial w } { \partial s } = \frac { \partial \tilde { \mathbf { r } } } { \partial w } \frac { \partial } { \partial s } ( \frac { W - b } { s } ) = \frac { \partial \tilde { \mathbf { r } } } { \partial w } ( - \frac { W - b } { s ^ { 2 } } )\tag{18}
$$

Therefore, Equation 16 can be represented with a value w and quantized value $\tilde { w }$ as follows.

$$
\frac { \partial W _ { q } } { \partial s } = s \frac { \partial \widetilde { \mathbf { r } } } { \partial w } ( - \frac { W - b } { s ^ { 2 } } ) + \widetilde { w } = \frac { \partial \widetilde { \mathbf { r } } } { \partial w } ( - \frac { W - b } { s } ) + \widetilde { w } = \left\{ \begin{array} { l l } { Q n , } & { i f \ w < Q n } \\ { - w + \tilde { w } , } & { i f \ Q _ { N } \le w \le Q _ { P } } \\ { Q p , } & { i f \ w > Q _ { P } } \end{array} \right.\tag{19}
$$

Secondly, with a similar context above, the derivative of b is presented as follows.

$$
\frac { \partial W _ { q } } { \partial b } = \frac { \partial } { \partial b } ( \tilde { w } \times s + b ) = s \frac { \partial } { \partial b } ( \tilde { \bf r } ( w ) ) + 1 = s \frac { \partial \tilde { \bf r } } { \partial w } ( \frac { \partial } { \partial b } ( \frac { W - b } { s } ) ) + 1\tag{20}
$$

$$
\frac { \partial  { \widetilde { \mathbf { r } } } } { \partial w } ( - 1 ) + 1 = \left\{ 0 , \ : \ : i f \ : \ : Q _ { N } \leq w \leq Q _ { P } \right.\tag{21}
$$

We also note that the gradient of $W _ { q }$ is presented as follows.

$$
\frac { \partial L } { \partial W _ { q } } = \frac { \partial L } { \partial Y } X ^ { \top }\tag{22}
$$

As a result, the updates on the quantization scale and bias are calculated as multiplication of Equation 22 with Equation 16 and with Equation 20, respectively. This update helps calibrate the quantization function, effectively reducing quantization errors.

## A.2 Quantization parameter and LoRA Parameter update on L4Q

In L4Q, as described in Equation 10, the quantized weight $W _ { q }$ is obtained as follows. First, the pre-trained model weight $W _ { 0 }$ and LoRA parameters are integrated to $W _ { c o m b } = W _ { 0 } + \alpha B A$ . Next, the integrated weight is quantized by the quantization parameters s, b.

Here, the LoRA parameters A and B are independent of the quantization parameters, scale s and bias b. Therefore, the derivatives of $s , b$ follow the same process as in Equation A.1, but with the term w, w˜ defined as follows. Note that $W _ { q } = \tilde { w } \times s + b$

$$
w = \frac { W _ { 0 } + \alpha B A - b } { s } , \tilde { w } = \tilde { \mathrm { \bf r } } ( w ) s . t . \tilde { \mathrm { \bf \bf \Gamma } } \tilde { \mathrm { \bf r } } = \mathrm { r o u n d } \cdot \mathrm { c l a m p }\tag{23}
$$

Seen from the L4Q layer that integrates the LoRA parameters and quantization parameters together, A, B are now considered as variables of $W _ { q } .$ Therefore, from the conditions in Equation 10, the derivative of A, B is presented as follows.

$$
\frac { \partial L } { \partial A } = \frac { \partial W _ { q } } { \partial A } \frac { \partial L } { \partial W _ { q } } , \quad \frac { \partial L } { \partial B } = \frac { \partial L } { \partial W _ { q } } \frac { \partial W _ { q } } { \partial B }\tag{24}
$$

The derivatives $\frac { \partial w } { \partial A }$ and $\frac { \partial w } { \partial B }$ are then can be computed by applying the chain rule with $w ,$ as follows:

$$
\frac { \partial W _ { q } } { \partial A } = \frac { \partial w } { \partial A } \frac { \partial W _ { q } } { \partial w } , \quad \frac { \partial W _ { q } } { \partial B } = \frac { \partial W _ { q } } { \partial w } \frac { \partial w } { \partial B }\tag{25}
$$

From Equation 23, the terms $\textstyle { \frac { \partial w } { \partial A } } , { \frac { \partial w } { \partial A } }$ , and $\frac { \partial W } { \partial w }$ can be expressed as follows:

$$
\frac { \partial w } { \partial A } = \frac { \alpha B ^ { \top } } { s } , \quad \frac { \partial w } { \partial B } = \frac { \alpha A ^ { \top } } { s } , \quad \frac { \partial W } { \partial w } = \frac { \partial } { \partial w } ( \widetilde { \mathbf { r } } ( w ) s + b ) = s \frac { \partial \widetilde { \mathbf { r } } } { \partial w }\tag{26}
$$

Therefore, by substitution of Equation 26 and applying STE on $\frac { \partial \tilde { \mathbf { r } } } { \partial w }$ from Equation 17 on Equation 25, the equation is simplified by the crossed-out products between the terms. As a result, the partial derivatives presented in Equation 12 can be derived as follows.

$$
\frac { \partial W _ { q } } { \partial A } = \frac { \partial w } { \partial A } \frac { \partial W _ { q } } { \partial w } = ( \frac { \alpha B ^ { \top } } { s } ) ( \frac { s \partial \tilde { \mathbf { r } } } { \partial w } ) = \left\{ \begin{array} { l l } { \alpha B ^ { \top } , } & { i f \ Q _ { N } \leq w \leq Q _ { P } } \\ { 0 , } & { o t h e r w i s e } \end{array} \right.\tag{27}
$$

$$
\frac { \partial W _ { q } } { \partial B } = \frac { \partial W } { \partial w } \frac { \partial w } { \partial B } = ( s \frac { \partial \tilde { \bf r } } { \partial w } ) ( \frac { \alpha A ^ { \top } } { s } ) = \left\{ \begin{array} { l l } { \alpha A ^ { \top } , } & { i f Q _ { N } \leq w \leq Q _ { P } } \\ { 0 , } & { o t h e r w i s e } \end{array} \right.\tag{28}
$$

Finally, substitution of Equation 27 and 28 to Equation 24 derives the Equation 29 and 30.

$$
\frac { \partial L } { \partial A } = \left\{ \begin{array} { l l } { \alpha B ^ { \top } ( \frac { \partial L } { \partial Y } X ^ { \top } ) , } & { \quad \mathrm { i f } Q _ { N } \leq w \leq Q _ { P } } \\ { 0 , } & { \quad o t h e r w i s e } \end{array} \right.\tag{29}
$$

$$
\frac { \partial L } { \partial B } = \left\{ \begin{array} { l l } { \alpha ( \frac { \partial L } { \partial Y } X ^ { \top } ) A ^ { \top } , } & { \quad \mathrm { i f } Q _ { N } \leq w \leq Q _ { P } } \\ { 0 , } & { \quad o t h e r w i s e } \end{array} \right.\tag{30}
$$

This form closely resembles the original backpropagation structure of the LoRA parameters A, B as shown in Equation 2, where the updates are expressed as $\begin{array} { r } { \frac { \partial L } { \partial A } = \alpha \frac { \partial L } { \partial \tilde { X } } X ^ { \top } = \alpha \big ( \mathbf { \hat { \boldsymbol { B } } } ^ { \top } \frac { \partial L } { \partial Y } \big ) X ^ { \top } } \end{array}$ , and $\begin{array} { r } { \frac { \partial L } { \partial B } = \alpha \frac { \partial L } { \partial Y } \tilde { X } ^ { \top } = \alpha \frac { \partial L } { \partial Y } ( A X ) ^ { \top } } \end{array}$ , respectively. However, in L4Q, this process includes an added gating condition on the quantized weights, which accounts for the integration of quantization into the LoRA parameters. As a result, we conclude that the backward process of the L4Q layer, which integrates both quantization parameter learning and LoRA parameter adaptation, is designed to account for the impact of quantization on the LoRA parameter updates.

## B Quantization initialization

We evaluate various quantization initialization schemes within L4Q, including method introduced in Section 3.3, LSQ+ (Bhalgat et al., 2020), and conventional symmetric and asymmetric quantization parameter initialization. The methods are depicted as $\mathrm { L } 4 \mathrm { Q } _ { \mathrm { i n i t } } , \mathrm { L S Q } + _ { \mathrm { i n i t } }$ , Symm, Asymm, respectively. In specific, each methods can be represented as the equations below, with quantization scale s and bias b and group-wise aligned model weight W with quantization bit-width n and $Q _ { N } = - 2 ^ { n - 1 } , Q _ { P } = 2 ^ { n - 1 } - 1$

$$
\mathrm { L S Q + _ { i n i t } } : s = { \frac { M a x ( | \mu - 3 \sigma ( W ) | , | \mu + 3 \sigma ( W ) | ) } { 2 ^ { n - 1 } } }\tag{31}
$$

$$
b = 0\tag{32}
$$

$$
\mathrm { S y m m } : ~ s = { \frac { M a x ( A b s ( W ) ) } { 2 ^ { n - 1 } } }\tag{33}
$$

$$
b = 0\tag{34}
$$

$$
\mathrm { A s y m m } : \quad s = { \frac { M a x ( W ) - M i n ( W ) } { Q _ { P } - Q _ { N } } }\tag{35}
$$

$$
b = M a x ( W ) - s \times Q _ { P } = M i n ( W ) - s \times Q _ { N }\tag{36}
$$

$$
\mathrm { L } 4 \mathrm { Q } _ { \mathrm { i n i t } } : s = M a x ( | \frac { M i n ( W ) } { Q n } | , | \frac { M a x ( W ) } { Q p } | )\tag{37}
$$

$$
b = 0\tag{38}
$$

We report the detailed model accuracy evaluation results of L4Q fine-tuning across different initialization methods, along with the quantization error and clipping error for each method, measured both at the initialization point and the end of the training.The LLaMA-2 7B model was trained for 12,800 iterations with a batch size of 128, using the same hyperparameters as in the main evaluation.

As shown in Table 4, while the overall quantization error remains relatively consistent across initialization methods and fine-tuned states, the clipping error exhibits significant variation. The clipping error reflects the number of values clipped during quantization, and different initialization methods lead to varying degrees of clipping throughout training. Notably, L4Q achieves the lowest clipping error and the highest model accuracy, demonstrating the effectiveness of its initialization strategy.

Table 4: MMLU 5-shot benchmark and the sum of quantization errors for various quantization parameter initialization methods within L4Q on the LLaMA-2 7B model. Quantization errors are represented in order of $1 0 ^ { 6 }$ and clipping errors are represented in order of $1 0 ^ { 3 }$
<table><tr><td></td><td></td><td></td><td colspan="5">MMLU 5-shot</td><td colspan="2">Initial</td><td colspan="2">Post-train</td></tr><tr><td>Model</td><td>Method</td><td>#Bits</td><td>Human.</td><td>STEM</td><td>Social.</td><td>Others</td><td>Average</td><td> $\operatorname { E } _ { q u a n t }$ </td><td> $\operatorname { E } _ { c l i p }$ </td><td> $\operatorname { E } _ { q u a n t }$ </td><td> $\mathrm { E } _ { c l i p }$ </td></tr><tr><td>LLaMA-2 7B</td><td>LSQ+</td><td>4</td><td>26.7</td><td>26.8</td><td>26.2</td><td>22.9</td><td>25.7</td><td>11.8</td><td>278.0</td><td>11.8</td><td>360.6</td></tr><tr><td></td><td>Symm</td><td>4</td><td>40.8</td><td>35.9</td><td>48.2</td><td>50.1</td><td>43.5</td><td>11.1</td><td>260.0</td><td>11.0</td><td>282.1</td></tr><tr><td></td><td>Asymm</td><td>4</td><td>41.0</td><td>37.1</td><td>49.7</td><td>50.2</td><td>44.2</td><td>10.5</td><td>0.0</td><td>10.5</td><td>64.7</td></tr><tr><td></td><td>L4Q</td><td>4</td><td>42.9</td><td>37.7</td><td>50.5</td><td>51.9</td><td>45.3</td><td>11.4</td><td>0.0</td><td>11.6</td><td>36.1</td></tr></table>

## C Ablative Study on L4Q Hyperparameters

## C.1 LoRA Rank Size

We investigated the effect of LoRA rank size on L4Q training. Using the LLaMA-2 7B model, we conducted training over 12,800 iterations with 128 batches. The remaining training conditions are consistent with the main experiments. The evaluation results for CSQA and MMLU are presented in Table 5 and Table 6, respectively.

Table 5: Commonsense QA benchmark result on LLaMA-2 7B model. The numbers represent accuracy (%) for each task.
<table><tr><td>Model</td><td>Rank</td><td>HellaSwag</td><td>PIQA</td><td>ARC-c</td><td>ARC-e</td><td>Winogrande</td><td>BoolQ</td><td>OBQA</td><td>Average</td></tr><tr><td>LLaMA-2 7B</td><td>1</td><td>57.6</td><td>78.4</td><td>45.5</td><td>77.0</td><td>68.2</td><td>77.4</td><td>34.0</td><td>62.6</td></tr><tr><td></td><td>2</td><td>57.4</td><td>78.5</td><td>45.1</td><td>76.2</td><td>69.3</td><td>77.6</td><td>34.0</td><td>62.6</td></tr><tr><td></td><td>4</td><td>57.5</td><td>78.2</td><td>46.1</td><td>77.1</td><td>68.7</td><td>78.1</td><td>35.4</td><td>63.0</td></tr><tr><td></td><td>8</td><td>56.9</td><td>78.5</td><td>46.3</td><td>78.1</td><td>69.3</td><td>77.8</td><td>34.8</td><td>63.1</td></tr><tr><td></td><td>16</td><td>57.2</td><td>78.1</td><td>46.3</td><td>77.2</td><td>68.7</td><td>78.9</td><td>33.4</td><td>62.8</td></tr><tr><td></td><td>32</td><td>57.8</td><td>78.4</td><td>46.2</td><td>77.1</td><td>68.7</td><td>78.2</td><td>35.8</td><td>63.2</td></tr><tr><td></td><td>64</td><td>57.4</td><td>78.6</td><td>46.1</td><td>77.1</td><td>69.5</td><td>78.5</td><td>34.8</td><td>63.1</td></tr><tr><td></td><td>128</td><td>57.5</td><td>78.1</td><td>46.0</td><td>77.0</td><td>68.8</td><td>78.4</td><td>35.4</td><td>63.0</td></tr></table>

Table 6: MMLU benchmark result on LLaMA-2 7B model. The numbers represent accuracy (%) for each category.
<table><tr><td colspan="2"></td><td colspan="5">0-shot</td><td colspan="5">5-shot</td></tr><tr><td>Model</td><td>Rank</td><td>Hums.</td><td>STEM</td><td>Social</td><td>Others</td><td>Avg.</td><td>Hums.</td><td>STEM</td><td>Social</td><td>Others</td><td>Avg.</td></tr><tr><td>LLaMA-2 7B</td><td>1</td><td>37.4</td><td>33.3</td><td>43.4</td><td>44.0</td><td>39.4</td><td>40.8</td><td>36.8</td><td>48.2</td><td>49.2</td><td>43.5</td></tr><tr><td rowspan="7"></td><td>2</td><td>38.1</td><td>31.6</td><td>41.6</td><td>42.2</td><td>38.4</td><td>42.2</td><td>35.2</td><td>48.6</td><td>49.0</td><td>43.7</td></tr><tr><td>4</td><td>36.3</td><td>33.4</td><td>42.7</td><td>43.2</td><td>38.7</td><td>42.5</td><td>36.6</td><td>50.2</td><td>51.2</td><td>44.9</td></tr><tr><td>8</td><td>39.5</td><td>34.6</td><td>45.0</td><td>45.0</td><td>41.0</td><td>42.7</td><td>36.7</td><td>50.3</td><td>51.7</td><td>45.0</td></tr><tr><td>16</td><td>38.7</td><td>35.8</td><td>45.7</td><td>45.8</td><td>41.3</td><td>42.0</td><td>36.6</td><td>49.4</td><td>49.7</td><td>44.3</td></tr><tr><td>32</td><td>39.4</td><td>35.0</td><td>46.1</td><td>45.6</td><td>41.3</td><td>42.4</td><td>37.1</td><td>49.8</td><td>49.0</td><td>44.4</td></tr><tr><td>64</td><td>39.6</td><td>35.0</td><td>45.8</td><td>47.2</td><td>41.7</td><td>43.6</td><td>37.4</td><td>50.9</td><td>50.9</td><td>45.6</td></tr><tr><td>128</td><td>38.8</td><td>35.4</td><td>44.8</td><td>44.7</td><td>40.7</td><td>42.8</td><td>36.6</td><td>50.3</td><td>50.6</td><td>44.9</td></tr></table>

Increasing the rank beyond 4 does not lead to significant performance improvements, which aligns with the observations in the original LoRA paper (Hu et al., 2022). Therefore, we generally applied a rank size of 4, considering that higher rank sizes introduce memory and computational overhead during training.

## C.2 Quantization Group Size

We investigated the effect of quantization group size on L4Q training. Using the LLaMA-2 7B model, we conducted experiments with group size 32 to 128 with the same training conditions in the main experiments. The evaluation results for Commonsense QA and MMLU are presented in Table 7 and Table 8, respectively.

Table 7: Commonsense QA benchmark result on LLaMA-2 7B model. The numbers represent accuracy (%) for each task.
<table><tr><td>Model</td><td>Group Size</td><td>Hellaswag</td><td>PIQA</td><td>ARC-c</td><td>ARC-e</td><td>Winogrande</td><td>BoolQ</td><td>OBQA</td><td>Average</td></tr><tr><td>LLaMA-2 7B</td><td>128</td><td>57.2</td><td>78.8</td><td>47.1</td><td>76.9</td><td>70.2</td><td>80.4</td><td>34.8</td><td>63.6</td></tr><tr><td></td><td>64</td><td>57.5</td><td>77.5</td><td>46.7</td><td>78.3</td><td>70.2</td><td>80.7</td><td>34.8</td><td>63.7</td></tr><tr><td></td><td>32</td><td>57.6</td><td>77.6</td><td>47.6</td><td>78.2</td><td>70.1</td><td>80.6</td><td>35.4</td><td>63.9</td></tr></table>

Table 8: MMLU benchmark result on LLaMA-2 7B model. The numbers represent accuracy (%) for each category.
<table><tr><td colspan="2"></td><td colspan="5">0-shot</td><td colspan="5">5-shot</td></tr><tr><td>Model</td><td>Group Size</td><td>Hums.</td><td>STEM</td><td>Social</td><td>Others</td><td>Avg.</td><td>Hums.</td><td>STEM</td><td>Social</td><td>Others</td><td>Avg.</td></tr><tr><td>LLaMA-2 7B</td><td>128</td><td>38.7</td><td>33.8</td><td>45.6</td><td>46.4</td><td>40.9</td><td>42.9</td><td>37.7</td><td>50.5</td><td>51.9</td><td>45.5</td></tr><tr><td></td><td>64</td><td>38.2</td><td>35.6</td><td>47.2</td><td>46.5</td><td>41.5</td><td>43.8</td><td>37.0</td><td>52.2</td><td>52.7</td><td>46.3</td></tr><tr><td></td><td>32</td><td>39.5</td><td>35.9</td><td>47.8</td><td>46.4</td><td>42.1</td><td>44.4</td><td>38.2</td><td>52.3</td><td>52.5</td><td>46.7</td></tr></table>

Having a fine-grained quantization group size leads to performance improvements, which aligns with the observations in the conventional group-wise quantization works (Frantar et al., 2023). We applied a quantization with the group size of 128 considering that smaller quantization group sizes introduce memory and computational overhead during inference and training.

## D Experimental Settings

The baselines and L4Q are trained with AdamW optimizer (Loshchilov and Hutter, 2019) with a weight decay of 0.01. For the learning rate scheduler, a cosine decay scheduler with a linear warm-up through 10% of the total training steps. Learning rates are presented in Table 9.

Table 9: Learning rate conditions used to fine-tuning on each models for L4Q and baselines: QLoRA<sup>\*</sup>, QA-LoRA, and QAT-LoRA.
<table><tr><td colspan="4">Methods</td></tr><tr><td>Model</td><td> $\mathbf { Q L o R A } ^ { * }$ </td><td>QA-LoRA</td><td>QAT-LoRA</td></tr><tr><td>OpenLLaMA 3B</td><td> $1 \times 1 0 ^ { - 5 }$ </td><td> $2 \times 1 0 ^ { - 5 }$ </td><td>L4Q  $5 \times 1 0 ^ { - 5 }$   $5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>LLaMA-1 7B</td><td> $1 \times 1 0 ^ { - 5 }$ </td><td> $2 \times 1 0 ^ { - 5 }$   $5 \times 1 0 ^ { - 5 }$ </td><td> $5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>LLaMA-1 13B</td><td> $1 \times 1 0 ^ { - 5 }$   $5 \times 1 0 ^ { - 5 }$ </td><td> $4 \times 1 0 ^ { - 5 }$ </td><td> $4 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>LLaMA-1 33B</td><td> $1 \times 1 0 ^ { - 5 }$ </td><td> $5 \times 1 0 ^ { - 5 }$   $2 \times 1 0 ^ { - 4 }$ </td><td> $2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>LLaMA-2 7B</td><td> $2 \times 1 0 ^ { - 5 }$ </td><td> $2 \times 1 0 ^ { - 5 }$ </td><td> $2 \times 1 0 ^ { - 4 }$   $2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td> $\mathrm { L L a M A } { - } 2 \ 1 3 \mathrm { B }$ </td><td> $2 \times 1 0 ^ { - 5 }$ </td><td> $2 \times 1 0 ^ { - 5 }$ </td><td> $2 \times 1 0 ^ { - 4 }$   $2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td> $\mathrm { M i s t r a l - v 0 . 1 7 B }$ </td><td> $1 \times 1 0 ^ { - 5 }$ </td><td> $5 \times 1 0 ^ { - 6 }$ </td><td> $5 \times 1 0 ^ { - 6 }$ </td></tr></table>

The batch size is set to 128. For baselines that utilize PTQ-based schemes, such as ${ \boldsymbol { \mathrm { Q L o R A } } } ^ { * }$ and QA-LoRA, training is conducted for 50K iterations. For QAT-based methods, such as QAT, QAT-LoRA, and L4Q, training is conducted for 25K iterations. This reduction in training length for QAT-based methods is due to their faster convergence, as illustrated in Figure 5 with an example of LLaMA-2 7B.

Using the same training hyperparameters, including a learning rate of $2 \times 1 0 ^ { - 5 }$ , the joint training of quantization parameters and LoRA weight parameters enables L4Q to converge more quickly. This allows for halving the training length, which also helps mitigate overfitting.

Additionally, the training sequence length is set to match or exceed the maximum sequence length of the dataset, which is 2048. The only exception is the 33B model with L4Q, where the training sequence length is set to 128.

![](images/c438bc2828d779938696396c6e978ea40637bf3297a65e7e606a435621a581e8.jpg)  
Figure 5: Train loss curve of L4Q and QLoRA. With a same training condition, L4Q converges faster than QLoRA.

## E Training Efficiency

## E.1 Memory Usage

We report the memory usage of QAT (LSQ), quantization-aware PEFT baselines (LoftQ, QLoRA, QA-LoRA, QAT-LoRA), and L4Q in Table 10. Note that QLoRA reduces memory usage further by employing paged optimizers, a technique that can also be applied to other fine-tuning methods, including L4Q. We plan to explore this vertical implementation in future work. As quantization-aware PEFT methods utilize pre-quantized model weights, the memory usage of L4Q and quantization-aware PEFT methods differs by the amount of the reduced memory usage of model weights during inference.

<table><tr><td rowspan="2">Methods</td><td rowspan="2">OpenLLaMA 3B</td><td colspan="3">LLaMA</td></tr><tr><td>7B</td><td>13B</td><td>33B</td></tr><tr><td>LoRA</td><td>15.1</td><td>25.1</td><td>43.8</td><td>71.9</td></tr><tr><td>LoftQ</td><td>5.2</td><td>7.9</td><td>19.6</td><td>31.9</td></tr><tr><td>QLoRA</td><td>5.2</td><td>7.9</td><td>19.6</td><td>31.9</td></tr><tr><td>QA-LoRA</td><td>7.8</td><td>14.8</td><td>27.8</td><td>67.2</td></tr><tr><td>LSQ</td><td>44.2</td><td>79.5</td><td>OOM</td><td>00M</td></tr><tr><td>QAT-LoRA</td><td>22.6</td><td>41.9</td><td>70.6</td><td>OOM</td></tr><tr><td>L4Q</td><td>15.3</td><td>25.4</td><td>44.3</td><td>73.2</td></tr></table>

Table 10: Memory cost (GB) for fine-tuning LLMs on a single NVIDIA A100 GPU. (OOM: Out of Memory)

## E.2 Training Time

We report the total training time of L4Q, QAT (LSQ), and the quantization-aware PEFT baselines(QLoRA<sup>\*</sup> and QA-LoRA) in Table 11. While L4Q has a longer training time per step compared to the baselines due to gradient recomputation during the backpropagation stage, the reduced number of training steps enables L4Q to achieve similar overall training time performance. The training time for QAT on 13B and 33B models was measured using 2 and 4 NVIDIA A100 GPUs on a single node.

Table 11: Training time (in hours) spent on fine-tuning on OpenLLaMA and LLaMA-1 models with a A100 GPU.
<table><tr><td colspan="2">OpenLLaMA</td><td colspan="3">LLaMA</td></tr><tr><td>Methods</td><td>3B</td><td>7B</td><td>13B</td><td>33B</td></tr><tr><td>LSQ</td><td>8.6</td><td>17.1</td><td>35.4</td><td>76.9</td></tr><tr><td>QLoRA*</td><td>4.5</td><td>9.9</td><td>18.0</td><td>38.4</td></tr><tr><td>QA-LoRA</td><td>5.0</td><td>11.2 10.1</td><td>19.8</td><td>39.6 37.9</td></tr><tr><td>L4Q</td><td>4.4</td><td></td><td>16.9</td><td></td></tr></table>

## F Throughput and Speedup of fully-quantized models and mixed-precision models

We investigate the throughput and speedup of fully-quantized models and mixed-precision models, demonstrating that, although the number of LoRA parameters is negligible, it causes a noticeable drop in throughput when its forward path is not merged with that of the base linear layer. Using the LLaMA-1 7B, 13B, and 33B models, we conducted experiments to measure throughput (tokens per second) and compare speedup. In both fully-quantized and mixed-precision models, uniform quantization is applied to the linear layers, except for the head layer (lm\_head), using the EXLLaMA2 kernel<sup>4</sup>, which is designed for 4-bit weight-only quantized inference. For fp16 computations in LoRA within mixed-precision models or the baseline, default GEMM kernels are used. We measured the elapsed time for inferencing 512 tokens over 2000 data points with batch sizes ranging from 1 to 64, calculating throughput by dividing the number of tokens, which is set to be 512, by the elapsed time. The results are presented in Table 12.

Table 12: Throughput (tokens/sec) and Speedup for LLaMA models. L4Q represents fully-quantized models, and QLoRA<sup>\*</sup> represents mixed-precision models. ’OOM’ indicates out-of-memory cases.
<table><tr><td colspan="10"></td></tr><tr><td>Model</td><td>Method</td><td>1</td><td>2</td><td>4</td><td>Batch size 8</td><td>16</td><td>32</td><td>64</td><td>Speedup</td></tr><tr><td>LLaMA-1 7B</td><td>L4Q</td><td>38.04</td><td>75.12</td><td>148.63</td><td>216.51</td><td>255.64</td><td>276.43</td><td>318.43</td><td>1.81</td></tr><tr><td rowspan="3"></td><td>QLoRA*</td><td>24.80</td><td>47.88</td><td>96.51</td><td>184.06</td><td>234.79</td><td>247.79</td><td>299.94</td><td>1.33</td></tr><tr><td>None</td><td>17.04</td><td>33.81</td><td>67.27</td><td>124.13</td><td>199.19</td><td>241.31</td><td>00M</td><td>1.00</td></tr><tr><td>L4Q</td><td>30.68</td><td>59.53</td><td>115.78</td><td>144.44</td><td>160.95</td><td>191.20</td><td>O0M</td><td>1.92</td></tr><tr><td rowspan="3">LLaMA-1 13B LLaMA-1 30B</td><td>QLoRA*</td><td>19.71</td><td>38.90</td><td>77.83</td><td>128.67</td><td>150.72</td><td>156.16</td><td>00M</td><td>1.41</td></tr><tr><td>None</td><td>13.67</td><td>26.97</td><td>53.23</td><td>85.47</td><td>124.17</td><td>00M</td><td>00M</td><td>1.00</td></tr><tr><td>L4Q</td><td>20.43</td><td>40.05</td><td>64.00</td><td>73.29</td><td>79.77</td><td>OOM</td><td>OOM</td><td>2.25</td></tr><tr><td rowspan="3"></td><td>QLoRA*</td><td>13.22</td><td>25.48</td><td>50.81</td><td>66.89</td><td>75.11</td><td>OOM</td><td>O0M</td><td>1.44</td></tr><tr><td>None</td><td>9.13</td><td>17.68</td><td>00M</td><td>00M</td><td>00M</td><td>00M</td><td>00M</td><td>1.00</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Fully-quantized models demonstrate a speedup of over 1.8x, while mixed-precision models achieve a maximum speedup of 1.4x, despite using the same quantization scheme and execution kernel, compared to the fp16 baselines. As a result, fully-quantized models achieve a 30% to 50% greater speedup compared to mixed-precision models. This demonstrates that L4Q, which produces fully-quantized models, offers higher inference efficiency and better hardware utilization than conventional quantization-aware PEFT methods, such as QLoRA and LoftQ, which retain unmerged forward paths for LoRA.

## G Detailed Result on Main Evaluations

We present the Commonsense QA and MMLU benchmark results with averaged accuracy score on Section 4. We present the detailed results of each benchmarks composed of several categories of tasks in Table 13a to Table 14b below. Through evaluation, we demonstrate that L4Q generally achieves higher accuracy in low-bit quantized models compared to both PTQ methods and PTQ-based fine-tuning methods. Notably, L4Q surpasses the pre-trained models on the Commonsense QA benchmarks and on the MMLU benchmarks with LLaMA-1 7B and 33B models. In contrast, PTQ-based fine-tuning methods, including those that incorporate high-precision LoRA weights, show lower performance compared to both L4Q and the pre-trained models. These results emphasize the challenges of recovering from quantization errors with PTQ alone and highlight the effectiveness of L4Q’s joint quantization and fine-tuning scheme.

Table 13a: Commonsense QA benchmark result. The numbers represent accuracy (%) of each task.
<table><tr><td>Model</td><td>Method</td><td>#Bits</td><td>Hella.</td><td>PIQA</td><td>ARC-c 33.9</td><td>ARC-e</td><td>Winogr.</td><td>BoolQ</td><td>OBQA</td><td>Avg.</td></tr><tr><td rowspan="3">OpenLLaMA 3B</td><td>None LoRA GPTQ OmniQ</td><td>16 16 44</td><td>48.8 49.8 47.9 48.2</td><td>75.0 75.6 75.1 73.8</td><td>37.0 31.0 33.1</td><td>69.2 70.2 58.8 69.5</td><td>61.6 63.1 60.5 60.1</td><td>66.9 68.0 57.9 67.5</td><td>28.2 27.2 23.6 26.6</td><td>54.8 55.9 50.7 54.1</td></tr><tr><td>LoftQ QLoRA* QA-LoRA QAT-LoRA L4Q GPTQ</td><td>4&amp;16 4&amp;16 4 4&amp;16 4</td><td>48.0 48.4 48.8 48.8 49.1 46.3</td><td>74.5 74.3 74.9 74.5 74.9 72.6</td><td>34.0 33.0 33.8 35.0 35.2 31.8</td><td>68.6 69.4 69.2 70.1 69.8 64.7</td><td>60.9 61.5 61.9 61.9 61.1 58.1</td><td>67.2 67.1 66.7 65.2 67.7 66.5</td><td>26.0 26.8 26.2 27.0 27.4 25.6</td><td>54.2 54.4 54.5 54.6 55.0</td></tr><tr><td>OmniQ LoftQ QLoRA* QA-LoRA QAT-LoRA L4Q</td><td>33 3&amp;16 3&amp;16 3</td><td>46.5 27.9 45.6 46.3 46.7 47.2</td><td>74.4 57.3 72.6 72.6 74.1 75.0</td><td>30.5 19.5 29.3 28.9 32.3</td><td>56.6 37.3 61.6 66.0 68.3</td><td>59.0 51.0 59.7 59.5 60.5 60.9</td><td>59.8 61.9 64.2 63.4 64.1</td><td>23.0 12.0 24.4 23.8 26.4</td><td>52.2 50.0 38.1 51.0 51.5 53.2</td></tr><tr><td rowspan="3">LLaMA-3 8B</td><td>None LoRA GPTQ</td><td>16 44</td><td>60.2 60.6 56.8 59.2</td><td>39.3 69.3 78.6</td><td>53.4 31.8 49.6</td><td>80.1 82.7 62.6 79.7</td><td>34.3 73.1 72.5</td><td>67.2 2.4 78.5 81.1</td><td>27.0 36.8 33.0 33.4</td><td>54.0 85:.2 57.9 64.9</td></tr><tr><td>OmniQ LoftQ QLoRA* QA-LoRA QAT-LoRA L4Q</td><td>4&amp;16 4&amp;16 4 4&amp;16 4</td><td>57.9 56.7 56.3 59.1 60.5</td><td>78.7 69.5 69.3 79.8 80.4</td><td>46.9 33.3 32.3 50.1 52.7</td><td>78.4 66.3 65.7 80.2 81.6</td><td>70.3 72.9 72.3 74.0 73.6</td><td>77.5 78.4 78.6 83.0 83.6</td><td>33.6 33.0 31.6 34.2 35.0</td><td>63.3 58.6 58.0 65.7 66.8</td></tr><tr><td>GPTQ OmniQ LoftQ QLoRA* QA-LoRA QAT-LoRA L4Q</td><td>33 3&amp;16 3&amp;16</td><td>51.8 55.0 35.9 53.4 52.7 56.6</td><td>68.8 76.7 68.8 73 78.2</td><td>30.2 39.2 29.8 31.8 36.4 47.4</td><td>58.3 69.2 57.7 62.9 67.9 77.8</td><td>67.7 69.2 59.0 69.2 67.6 68.0</td><td>70.1 72.6 67.9 71.5 71.7 80.6</td><td>27.4 28.8 20.8 28.8 26.6 33.4</td><td>53.5 58.7 48.6 55.7 56.6</td></tr><tr><td rowspan="4">LLaMA-1 7B</td><td>NORA GPTQ</td><td>3 16 44</td><td>56.5 58.3 53.9</td><td>78.1 78.8 77.7</td><td>47.8 45.9 40.3</td><td>78.6 375.3</td><td>69.2 90.%</td><td></td><td>82.0 78.3 72.9</td><td>32.2 35.4 30.0</td><td>63.1 63.5 03.74 59.4</td></tr><tr><td>OmniQ LoftQ QLoRA* QA-LoRA QAT-LoRA L4Q</td><td>4&amp;16 4&amp;16 4 4&amp;16 4</td><td>55.7 57.8 56.7 57.2 57.7 57.8</td><td>77.7 79.2 78.9 78.9 78.9 79.1</td><td>38.8 43.1 41.8 41.2 44.7 45.3</td><td>73.5 67.5 76.9 75.2 74.9 75.3 76.0</td><td>67.9 65.3 69.8 70.0 70.6 68.9 69.5</td><td>72.5 75.8 74.6 73.6 75.8</td><td>29.2 35.4 32.2 35 76.1</td><td>34.8</td><td>58.1 62.6 61.3 61.3 62.4 62.7</td></tr><tr><td>GPTQ OmniQ LoftQ QLoRA*</td><td>33 3&amp;16 3&amp;16</td><td>46.6 54.0 43.4 53.9</td><td>71.9 77.1 68.9 76.2</td><td>32.4 35.6 33.0 39.3</td><td>65.4 64.9 65.5</td><td>64.7 56.5</td><td>65.0</td><td>68.0 71.2 58.5</td><td>24.6 28.0 23.0</td><td>53.4 56.5 49.8</td></tr><tr><td>QA-LoRA QAT-LoRA L4Q</td><td>3</td><td>55.4 56.1 55.9</td><td>194 77.6</td><td>42.1</td><td>39.8 41.6</td><td>71.5 28 74.1</td><td>68.9 69.5 68.0 68.9</td><td>72.8 67.1 76.0 76.8</td><td>31.0 30.6 33.0</td><td>59.1 58.7 60.7 61.2</td></tr></table>

Table 13b: Commonsense QA benchmark result. The numbers represent accuracy (%) of each task.
<table><tr><td rowspan=1 colspan=5>Model         Method    #Bits</td><td rowspan=1 colspan=3>Hella. PIQA ARC-c ARC-e Winogr. BoolQ OBQA Avg.</td></tr><tr><td rowspan=1 colspan=2>LLaMA-1 13B</td><td rowspan=1 colspan=3>NORA     18</td><td rowspan=1 colspan=3>508  39.3  38.3  78.   72.8   88.2   83.8</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=3>GPTQ     44OmniQ</td><td rowspan=1 colspan=3>58.80 79.3  43.3  73.3   70.3  78.5  3 8.8 60.0.13.</td></tr><tr><td rowspan=2 colspan=5>QLoRA*   4&amp;16QAT-LoRAL4Q</td><td rowspan=1 colspan=1>LoftQ*4&16</td><td rowspan=2 colspan=2>2 79.0 G  77.7   32272.  76.0   35.0 79.270R29 93879.00. 078.579.2</td></tr><tr><td rowspan=1 colspan=3>OL-LARA 4&16</td><td></td></tr><tr><td rowspan=1 colspan=5>OmniQLoftQ      3&amp;16</td><td rowspan=1 colspan=1>GPTO3</td><td rowspan=2 colspan=2>2  2 39.9   73.0   1 1 103237E74.632 31.4 1.72.737.670.865.54.08675586.07.877.843923R2RR272161.378.445.55.0032</td></tr><tr><td rowspan=1 colspan=5>QLoRA*QA-LoRA</td><td></td></tr><tr><td rowspan=1 colspan=2>LLaMA-1 33B</td><td rowspan=1 colspan=3>NORA01     18</td><td rowspan=1 colspan=3>03.3  81.3  33.9  80.4   75.3   2.0  39.0 68.33.1.i</td></tr><tr><td rowspan=4 colspan=5>LoftQ      4&amp;16QLoRA*    4&amp;16QA-LoRAQAT-LORAL4Q        4</td><td rowspan=1 colspan=1>GPTO 4</td><td rowspan=2 colspan=2>1   80.5   49.1            06952  82.  31.3  65.780.0162.980.382.967.4</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>LoftQ*4&16</td><td></td></tr><tr><td rowspan=2 colspan=3>2                   aaede62.2020 28aS224.953.075.0</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=3>GPTQOmniQ</td><td rowspan=3 colspan=3>60.9 78.7  46.7  7.2  8 2e82. 65.146.07471.29.62.346.174.938.973.963.28.54.8AEG78.746.064.3298882234.00 3381.63.381.02.867.4</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>LoftQ</td><td rowspan=1 colspan=1>3&16</td></tr><tr><td rowspan=1 colspan=5>QLoRAQA-LoRA</td></tr><tr><td rowspan=1 colspan=2>LLaMA-2 7B</td><td rowspan=1 colspan=3>NORAO     18</td><td rowspan=1 colspan=1>37.6  78.6  48.0</td><td rowspan=1 colspan=2>39.3   90.3   35.8  34.4 03.3</td></tr><tr><td rowspan=2 colspan=2></td><td rowspan=1 colspan=2>GPTQOmniQ</td><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1>56.00 77.5  41356.0</td><td rowspan=1 colspan=2>75.0  2  73.34  30. 229.0 59.3</td></tr><tr><td rowspan=1 colspan=2>LoftQ</td><td rowspan=1 colspan=1>4&amp;16</td><td rowspan=1 colspan=1>57.0  78.0</td><td rowspan=1 colspan=2>76.3                              61.7</td></tr><tr><td rowspan=2 colspan=4>QAT-LoRAL4Q</td><td rowspan=1 colspan=1>OL-LoRA</td><td rowspan=1 colspan=1>4&amp;164</td><td rowspan=1 colspan=1>56.656.4</td><td rowspan=2 colspan=1>3 683m33 226例 5571 869.8531.43. 24</td></tr><tr><td rowspan=1 colspan=1>4&16</td><td rowspan=1 colspan=1>57.6   78.7    43.1</td><td></td></tr><tr><td rowspan=2 colspan=2></td><td rowspan=1 colspan=2>GPTQOmniQ</td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>    76.  35.76.4</td><td rowspan=1 colspan=2>709.3   67.7   7.4  12 7.866.171.4</td></tr><tr><td rowspan=1 colspan=2>LoftQ</td><td rowspan=1 colspan=1>3&amp;16</td><td rowspan=1 colspan=1>55.7   19.0</td><td rowspan=1 colspan=2>31.1    48.8    48.1           34.7</td></tr><tr><td rowspan=1 colspan=5>QLoRAQA-LoRAQAT-LoRAL4Q</td><td rowspan=1 colspan=1>&16</td><td rowspan=1 colspan=1>56.477.832.3</td><td rowspan=1 colspan=1>7 212 9.9   580  4135 24823327.656.52373.868.869.57.33S8</td></tr><tr><td rowspan=1 colspan=2>LLaMA-2 13B</td><td rowspan=1 colspan=3>NORA     16</td><td rowspan=1 colspan=1>09.2  79.4  33.5</td><td rowspan=1 colspan=2>79.4   3322   80.4  33. 65.</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>GPTQOmniQ</td><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1>59.5  78.3  43.378.312</td><td rowspan=1 colspan=2>79.3   8.17   80.9   32.034.2 9..66.6</td></tr><tr><td rowspan=3 colspan=4>QLoRA*    4QA-LoRAQAT-LoRAL4Q</td><td rowspan=1 colspan=1>LoftQ*</td><td rowspan=1 colspan=1>4&amp;16</td><td rowspan=1 colspan=1>60.0  79.3   48.1</td><td rowspan=1 colspan=1>79.7    71.9</td></tr><tr><td rowspan=1 colspan=1>&amp;16</td><td rowspan=1 colspan=1>0599 5</td><td rowspan=2 colspan=2>229  0  2 29121433.080.734.4</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>60.5 80.1 51.2 </td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>GPTQOmniQ</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>78.2  42.5  72.3  68. 00  74.          59.761.7</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>LoftQ      3</td><td rowspan=1 colspan=1>&amp;16</td><td rowspan=2 colspan=3>                                             3282250.755.115.302076.170.070.132.6P5 1.78..97.637.85.5p°64.3</td></tr><tr><td rowspan=1 colspan=5>QLoRA*QA-LoRA</td><td rowspan=1 colspan=1>3&16</td></tr><tr><td rowspan=1 colspan=5>Mistral-v0.1 7BNORA     16</td><td rowspan=1 colspan=3>1122  80. 10.  50. 303  80.90   73.0   83.9  32.62: 66.4.220.4</td></tr><tr><td rowspan=5 colspan=2></td><td rowspan=1 colspan=2>GPTQ     4OmniQ</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>50.0. 89.3   46.947.1</td><td rowspan=2 colspan=2>78.            8..4  31. 5 65..665.377.6             75.1           60.7</td></tr><tr><td rowspan=1 colspan=2>LoftQ      4&</td><td rowspan=1 colspan=1>amp;16</td><td rowspan=1 colspan=1>78.0   44.5</td></tr><tr><td rowspan=1 colspan=2>QLoRA</td><td rowspan=1 colspan=1>4&amp;16</td><td rowspan=3 colspan=3>GRRAAG          280221229 212 80.781.650.966.1</td></tr><tr><td rowspan=2 colspan=2>QA-LoRAQAT-LoRAL4Q</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>60.681.749.0</td></tr><tr><td rowspan=1 colspan=1>4&amp;164</td></tr><tr><td rowspan=3 colspan=4>LoftQQLoRA*QA-LoRAQAT-LoRAL4Q</td><td rowspan=1 colspan=1>GPTO</td><td rowspan=1 colspan=1></td><td rowspan=3 colspan=2>79.5  43.5355  75.8   70.  RR 37.8  1.11.3GORR672.74.740.972.4102232858.52 380.546222 26.02.79.770.347.763.1</td></tr><tr><td rowspan=1 colspan=2>LoftQ*</td><td rowspan=1 colspan=1>3&amp;16</td><td></td></tr><tr><td rowspan=1 colspan=1>&16</td><td></td></tr></table>

Table 14a: MMLU benchmark result. The numbers represent accuracy(%) of each task.
<table><tr><td colspan="2">Model LLaMA-1 7B</td><td rowspan="2">#Bits Human.</td><td colspan="3">0-shot</td><td rowspan="2"></td><td colspan="2">Human. STEM</td><td colspan="3">5-shot</td></tr><tr><td>Method None LoRA</td><td>16</td><td>STEM 36.9 26.9 31.5</td><td>Social 32.1 36.9</td><td>Others 37.3</td><td> $\mathbf { A v g } .$  32.5 36.3</td><td>33.9 34.4</td><td>Social 30.3 39.2</td><td>Others 38.3</td><td>Avg. 35.1 36.7</td></tr><tr><td rowspan="2"></td><td>GPTQ 44 OmniQ LoftQ QLoRA QA-LoRA 4 QAT-LoRA</td><td>4&amp;16 4&amp;16 4&amp;16 </td><td>28.4 31.1 32.3 2521</td><td>27.1 27.0 26.7 29.8 30.0 32.7 2 212</td><td>30.5 30.4 37.3 37 39..4</td><td>28.3 30.9 33.0 212 34.9</td><td>31.55 33.6 32.3 3 34.2</td><td>30.4 29.8 30.7 29 32.2</td><td>33.5 37.2 2436</td><td>37.5 37.5 39.0 38.0 39.0 39.9</td><td>32.7 33.3 35.1 33.6 2909</td></tr><tr><td>L4Q GPTQ OmniQ  ${ \mathrm { L o f t } } { \mathrm { Q } } ^ { * }$   $\boldsymbol { \mathrm { Q L o R A } } ^ { * }$  QA-LoRA QAT-LoRA</td><td>4 33 3&amp;16 3&amp;16</td><td>25.8 24.3 42 225 22</td><td>32.1 29. 21.7 2</td><td>22.0 26.8 22.4 462</td><td>3.5 23.7 29.0 24.5 29.1 27.7</td><td>25.9 31.6 23.4 23.4 30.5 28.0</td><td>30.7 25.7 32.1 23.2 28.6 1</td><td>38. 21.9</td><td>39.8 29.7 29.7 23.5 34.9</td><td>37. 23.1 31.5</td></tr><tr><td rowspan="3">LLaMA-1 13B</td><td>L4Q None LoRA GPTQ</td><td>16</td><td>41.0 42.4 39. 33.5</td><td>27.8 36.5 49.3 34.0 34.5 44.9</td><td>293 51. 44.9</td><td>30.6 30.6 43.3 40.1</td><td>293 43.8 45.0 43.1</td><td>29. 31.0 35.3 36.4 35.9</td><td>52.7 54.1</td><td>316 30.4 54.2 53.1</td><td>2918 46.3 47.0</td></tr><tr><td>OmniQ LoftQ QLoRA* QA-LoRA QAT-LoRA L4Q</td><td>44 4&amp;16 4&amp;16 4</td><td>39.0 212 35.3</td><td>35.1 48.6 65 34.8 1 2 35.3 49.5</td><td>48.1 48.5 3</td><td>42.6 42.4 42.1 42.4 42.0 43.2</td><td>43.1 43.4 43.8 43.0 43.3</td><td>35.7 34.3 35.3 34.6 35.0 36.1</td><td>52.3 52. 3 52.0 5323</td><td>512. 52.1 202</td><td>45.7 45.7 45.4 45.9 </td></tr><tr><td>GPTQ OmniQ  $\mathrm { L o f t Q } ^ { * }$   $\mathrm { Q L o \tilde { R } A } ^ { * }$  QA-LoRA QAT-LoRA L4Q</td><td>33 3&amp;16 3&amp;16</td><td>32.6 25.1 532 2 37.6</td><td>28.2 27. 36.3 37.1 24.7 24.0 424 22 28.4.4</td><td>3 6.8 26.1 2 2 43.</td><td>33.1 34.8 25.0 12</td><td>36.0 39.1 24.9 36 34.8</td><td>29.8 33.1 27.4 52 </td><td>47.5 42. 24.1 45.6 44.5</td><td>45. 25.0 47.0 282</td><td>46.0 31.2 25.3 38.8.4 40.9</td></tr><tr><td rowspan="2">LLaMA-1 33B</td><td>None LoRA GPTQ OmniQ  $\mathrm { L o f t Q } ^ { * }$  QLoRA* 4&amp;16 QA-LoRA</td><td>16 44 4&amp;16</td><td>59.2 48.5 49.2 39.0</td><td>44.7 40.1 0 .2 41.3 30. 3 9 1 40.2 60.8</td><td>58.4 58.1 59.1 58.0</td><td>40.7 53. 51.4 52.0 51.8 51.0</td><td>39.3 55. . 53.5 54.6</td><td>34.0 44.7 46.1 45.1 44.8 44.5</td><td>46.6 65.4.4 64. . 65.2 3 .34</td><td>61. 3 61..2 61.5 60.5</td><td>41.8 59.. 55.8 56.4</td></tr><tr><td>QAT-LoRA L4Q GPTQ OmniQ LoftQ QLoRA* QA-LoRA QAT-LoRA L4Q</td><td>4 33 3&amp;16 16</td><td>21 SO 47.0 24.7 418</td><td>21 39.8 39.0 40.0 24.0 31.</td><td>06 21 2 59.4 57. 57.9 59.0 025408 26.4 55 23</td><td>48.9 3533 50.0 50.2 24.6 45.6 46.1</td><td>2 21 22 46. 24.3 2 45.</td><td>43.8 23.2 40.9 39.9</td><td>65.1 66.1 59.2 60.0 22.9 29620 57</td><td>1 88 57. . 25.6 55.</td><td>51.9 52.4 24.0 50.1 48.7 50.6</td></tr><tr><td rowspan="4">LLaMA-2 7B</td><td>None LoRA GPTQ OmniQ</td><td>3 16 44</td><td>466 39.3 37.9 37.7</td><td>38.9 41.0 34.0 34.6 30.1</td><td>57. 41.</td><td>57. 50.00 45.1</td><td>50.1 50.5 41.6 43.9 37.1 41.0</td><td>50.4 43. 40.9</td><td>42 37.0 5. 37.0 33. 48.9</td><td>57.5 59.1 52.2 52.2 41.</td><td>53.1</td><td>45. 45..</td></tr><tr><td> $\mathrm { L o f t } \mathrm { Q } ^ { \mathrm { s } }$   $\mathrm { Q L o R A } ^ { * }$  QA-LoRA QAT-LoRA L4Q GPTQ</td><td>4&amp;16 4&amp;16 4</td><td>36.7 37.3 25 37.5</td><td>31.3 31.5 228 28.5</td><td>42.9 2</td><td>43.6 42.7 43.0 46..</td><td>38.5 38.6 38.9 37.9 40.9 31.3</td><td>42.4 41.5 42.1 42.0 41.8 42.9 36.0</td><td>34.6 35.9 29 37.7 31.7</td><td>51.1 49.5 50.2 20</td><td>49.8 51.1 23015</td><td>43.7 44.6 242</td></tr><tr><td>OmniQ LoftQ QLoRA QA-LoRA QAT-LoRA L4Q</td><td>33 3&amp;16 处6 3</td><td>28.9 33.1 24.1 2326</td><td>30.4 21.3 2</td><td>35.3 21.8 20 36.0</td><td>33.5 33. 23.8 2323 33.8 39.2</td><td>34.3 22.9 32.5 23 34.9</td><td>34.1 23.7 35.4 36.0 34.3 4</td><td>32.4 26.1 32.5 292</td><td>31.3 22.4 2402 42.2</td><td>43.5 24.9 2 44.9</td><td>375 24.2 212 38.0</td></tr></table>

Table 14b: MMLU benchmark result. The numbers represent accuracy(%) of each task.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td rowspan="2">#Bits</td><td colspan="4">0-shot</td><td rowspan="2"></td><td colspan="2">Human.</td><td rowspan="2">5-shot</td><td rowspan="2">Others</td><td rowspan="2">Avg.</td></tr><tr><td>Human. 47.8</td><td>STEM 42.3</td><td>Social Others 60.5</td><td>Avg. 52.1</td><td>STEM</td><td>Social 63.0</td></tr><tr><td rowspan="3">LLaMA-2 13B</td><td>None LoRA GPTQ OmniQ LoftQ</td><td>16 44</td><td>48.8 46.5 47.8</td><td>42.4 40.2 41.9</td><td>60.9 57.7 60.1</td><td>59.4 59.2 56.8 58.9</td><td>52.5 50.0 51.8</td><td>52.0 54.4 52.3 53.0</td><td>43.8 44.3 43.1 43.0</td><td>63.4 62.7 62.5</td><td>61.2 60.8 61.5 60.5</td><td>54.8 55.7 54.7 54.7</td></tr><tr><td> $\mathrm { Q L o { \tilde { R } } { \tilde { A } } } ^ { * }$   $\scriptstyle \mathrm { \hat { Q } A - L o R A }$  QAT-LoRA L4Q</td><td>4&amp;16 4&amp;16 4 4&amp;16 4</td><td>47.2 101 48.4</td><td>42.0 40.9 40.8 41.0 41.8</td><td>60.4 2 60.4</td><td>58.9 57.6 57.4 56.8 58.4</td><td>51.7 50.7 50.4 50.7 51.9</td><td>52.6 51.3 51.6 50.3 53.6</td><td>43.2 23</td><td>62.8 2期</td><td>60.1 60.8 60.7 60.7 60.5</td><td>54.5 54.2 54.1 53.8 55.2</td></tr><tr><td>GPTQ OmniQ LoftQ  $\mathrm { Q L o R A } ^ { * }$   $\mathrm { \partial { \delta \mathrm { { A - L o R A } } } }$   $\scriptstyle \partial \mathrm { A T - L o R A }$  L4Q</td><td>33 3&amp;16</td><td>43.5 42.3 24.2 43.9 43.4 43.7</td><td>37.3 38.9 21.7 38.3 37.4 37.3</td><td>53.6 54.5 23.8 53.9 53.7 53.0</td><td>51.3 23.7 52.2 52.1 52.1</td><td>46.3 46.3 23.5 46.8 46.4 45.9</td><td>46.3 43.4 24.6 48.5 47.5 44.8</td><td>42.7 43.0 28.5 41.1 41.5 40.5</td><td>57.3 58.8 24.0 57.7 56.3 56.3</td><td>56.2 56.5 27.4 55.9 55.3 55.7</td><td>50.4 50.2 26.0 50.6 49.9 48.9</td></tr><tr><td rowspan="5">Mistral-v0.1 7B</td><td>None LoRA GPTQ</td><td>16 16</td><td>54.1 54.5</td><td>39.0 51.2 51.4</td><td>54.4 70.5 70.9</td><td>52.2 67.8 68.2</td><td>47.1 60.2 60.6</td><td>46.6 56.5 56.8</td><td>39.8 52.6 52.9</td><td>58.4 73.5 73.9</td><td>56.7 70.4 70.8</td><td>50.0 62.6 62.9</td></tr><tr><td>OmniQ LoftQ QLoRA* QA-LoRA QAT-LoRA</td><td>44 4&amp;16 4&amp;16 4</td><td>51.8 38.4 52.6 51.4</td><td>47.4 40.8 49.2 46.4</td><td>68.1 69.0 53.5 69.6 66.0</td><td>65.5 51.1 66.0 64.1</td><td>57.6 58.4 45.2 58.7 56.5</td><td>55.9 55.1 39.9 55.7 56.1</td><td>50.4 41.6 51.7 51.4</td><td>71.2 53.4 71.2 72.0</td><td>68.1 68.5 50.8 68.0 67.6</td><td>61.0 61.0 45.7 61.1 61.2</td></tr><tr><td>L4Q GPTQ OmniQ LoftQ</td><td>4&amp;16 4 33</td><td>52.1 52.6 45.0 49.4 32.2</td><td>48.8 48.9 42.4 44.9 33.3</td><td>69.5 69.9 59.7 63.4</td><td>67.7 67.1 57.2 61.6</td><td>58.8 59.0 50.5 54.3</td><td>54.1 56.3 43.0 50.0</td><td>50.5 51.1 43.0 47.0</td><td>71.0 71.7 57.4 66.1</td><td>68.0 68.8 57.8 63.0</td><td>60.2 61.4 49.6 55.9</td></tr><tr><td>QLoRA* QA-LoRA QAT-LoRA</td><td>3&amp;16 3&amp;16</td><td>46.9 45.8 47.6</td><td>43.4 42.9 43.0</td><td>38.0 60.6 60.8 61.3</td><td>41.3 60.2 54.5 59.9</td><td>35.8 52.2 50.5</td><td>34.6 47.6 48.8</td><td>34.1 44.8 43.6</td><td>39.6 61.6 59.2</td><td>41.1 62.9</td><td>37.1 53.6</td></tr><tr><td>L4Q</td><td></td><td>49.9</td><td>43.8</td><td>63.0</td><td>63.0</td><td>52.4 54.5</td><td>48.5 51.1</td><td>46.3 46.1</td><td>63.6 66.4</td><td>56.4 60.2 62.5</td><td>51.7 54.0 56.2</td></tr></table>