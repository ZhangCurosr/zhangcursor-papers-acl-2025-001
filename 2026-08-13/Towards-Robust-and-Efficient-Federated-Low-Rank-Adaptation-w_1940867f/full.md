# Towards Robust and Efficient Federated Low-Rank Adaptation with Heterogeneous Clients

Jabin Koo\*<sup>1</sup>, Minwoo Jang\*<sup>2</sup>, Jungseul Ok<sup>†1,</sup> <sup>2</sup>

<sup>1</sup>Department of Computer Science and Engineering, POSTECH, South Korea <sup>2</sup>Graduate School of Artificial Intelligence, POSTECH, South Korea {jbkoo, minwoo, jungseul}@postech.ac.kr

## Abstract

Federated fine-tuning for Large Language Models (LLMs) faces significant challenges due to the heavy communication overhead of transmitting large model updates. Although Low Rank Adaptation (LoRA) has been proposed as a solution, yet its application in federated learning is complicated by discordance in aggregation. Existing methods addressing this discordance often suffer from performance degradation at low ranks in heterogeneous data settings. In response, we introduce LoRA-A<sup>2</sup> (Low Rank Adaptation with Alternating freeze and Adaptive rank selection), which demonstrates robustness in challenging settings with low ranks and high data heterogeneity. Our experimental findings reveal that LoRA-A<sup>2</sup> maintains performance even under extreme heterogeneity and low rank conditions, achieving up to a significant reduction in uploaded parameters compared to full fine-tuning without compromising performance. This adaptive mechanism increases robustness and communication efficiency in federated fine-tuning, enabling the practical deployment of LLMs in resourceconstrained environments.

## 1 Introduction

Large Language Models (LLMs), exemplified by ChatGPT (OpenAI, 2023), Llama (Dubey et al., 2024) and others, represent a hallmark of the current era. These models are being widely applied in real-world scenarios by fine-tuning them on various task-specific datasets (Dodge et al., 2020). With the expansion of edge devices, the potential to leverage rich, privacy-sensitive data for fine-tuning LLMs has shifted the focus toward federated fine-tuning. Despite its potential, this is often infeasible due to the large size of LLMs, which require extensive computational and communication resources from local devices.

Parameter-Efficient Fine-Tuning (PEFT) methods (Lester et al., 2021; Liu et al., 2022) are increasingly being explored in the context of federated fine-tuning. Among these, Low-Rank Adaptation (LoRA) (Hu et al., 2022) is particularly noteworthy for its significant reduction in number of communicated parameters. However, naive application of LoRA in Federated Learning (FL) (McMahan et al., 2017) environment comes with several challenges such as aggregation discordance. Although several solutions have been proposed, they often remain vulnerable to high heterogeneity and low ranks due to a limited parameter space, making it difficult to reduce rank size for communication efficiency in realistic FL scenarios.

To address this, we introduce $\mathrm { L o R A { - } A ^ { 2 } }$ (Low Rank Adaptation with Alternating freeze and Adaptive rank selection), which is robust to both high heterogeneity and low ranks. $\mathrm { L o R A  – A ^ { 2 } }$ incorporates two main strategies: (1) alternating freeze, which switches between freezing LoRA modules B and A in each round, and (2) adaptive rank selection, which identifies and updates only important ranks in LoRA modules. We conduct experiments across various rank sizes and heterogeneity levels, comparing our algorithm with multiple baselines. Through the experiments, we reveal the vulnerabilities of existing methods and highlight the robustness of LoR $\mathsf { A } { - } \mathsf { A } ^ { 2 }$ in challenging conditions, providing analyses of the reasons for its robustness. Additionally, we empirically demonstrate that our approach achieves performance comparable to or exceeding that of full fine-tuning, while uploading less than 0.2% of parameters to the server.

Our contributions can be summarized as follows:

• We address the vulnerabilities of previous federated LoRA methods in high heterogeneity and low-rank settings, and propose a novel algorithm, LoRA-A<sup>2</sup>, which demonstrates robustness in these challenging conditions.

![](images/c0f1af333b6e21b4e462ea433fb22429d50605efc2d6d5fb9f01ae5af8d1f25d.jpg)  
Figure 1: An overview of the proposed method, $\mathrm { L o R A  – A ^ { 2 } }$ . It alternately trains B and A of the LoRA adapters, with each client training only a subset of the downloaded parameters. LoRA-A<sup>2</sup> is free from several issues for using LoRA in FL, which are discussed in Section 3. A detailed explanation of the method is provided in Section 4.

• Our algorithm effectively reduces communication costs, achieving a significant reduction in uploaded parameters compared to full fine-tuning, while maintaining its performance.

• We provide visualization on adaptive rank selection process and a thorough empirical exploration on how important ranks are efficiently trained and transmitted.

## 2 Related Works

LoRA with adaptive rank selection LoRA (Hu et al., 2022) is a widely used PEFT method for LLMs. It tries to approximate the updated part of the pre-trained model with two smaller matrices. This approach is inspired by previous studies (Li et al., 2018; Aghajanyan et al., 2021), which suggest that newly learned parameters for adaptation lie within a low dimensional subspace.

AdaLoRA (Zhang et al., 2023) assumes a scenario where the total parameter budget is limited. It adaptively selects the rank for each LoRA adapter under this constraint, with a criterion for rank selection based on singular values of the updated part.

ALoRA (Liu et al., 2024) utilizes a router for each LoRA adapter. The router determines which part of each LoRA adapter should be either turned on or off, enabling efficient fine-tuning via pruning. Similarly, DoRA (Mao et al., 2024) re-splits LoRA into smaller groups of LoRAs. During the training session, it estimates the importance of each small LoRA, allowing the parts with less contribution across the whole LoRA to be pruned. Our research extends these adaptive rank selection methods in centralized learning to the FL setting so that each client adaptively selects different ranks suitable for their own dataset.

Federated learning with LoRA As training LLMs on mobile devices becomes feasible, finetuning LLMs via FL has recently gained attention. In line with this trend, using LoRA for federated fine-tuning (Babakniya et al., 2023; Kuo et al., 2024; Wang et al., 2024), is also being considered. However, simply adopting LoRA for FL presents several obstacles, which are discussed in Section 3.

HetLoRA (Cho et al., 2023) assumes that each client may have different computational power, which is a common scenario in FL. Based on this assumption, it allows each client to use a LoRA adapter with a different rank. Zero-padding is then applied to align the dimensions of the clientspecific adapters for aggregation.

Sun et al. (2024) point out that aggregating the two matrices of a LoRA adapter separately cannot fully approximate the original LoRA adapter. Based on this finding, they propose FFA-LoRA, which addresses this issue by freezing half of each LoRA throughout the entire fine-tuning session.

FlexLoRA (Bai et al., 2024) aggregates the product of two matrices comprising each LoRA adapter and then decomposes the aggregated parameters back into two smaller matrices via Singular Value Decomposition (SVD). This approach allows FlexLoRA to overcome the challenges addressed by HetLoRA and FFA-LoRA, respectively, though at the expense of increased computational cost on the server-side for the decomposition process.

## 3 Problem Formulation

Low rank adaptation Because LLMs have billions of parameters, fine-tuning them for specific domains demands significant computational power, which may be infeasible in many situations. To address this issue, PEFT techniques such as LoRA (Hu et al., 2022) have recently gained attention, as they can effectively reduce the number of parameters that need to be trained. Specifically, when finetuning a pre-trained weight matrix $\dot { W _ { 0 } } \in \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } }$ to obtain W, LoRA achieves this by decomposing $\Delta W$ , the update of the weight matrix, into smaller matrices $B \in \mathbb { R } ^ { d _ { 1 } \times r }$ and $A \in \mathbb { R } ^ { r \times d _ { 2 } }$

$$
W = W _ { 0 } + \Delta W = W _ { 0 } + B A ,\tag{1}
$$

where $r \ll \{ d _ { 1 } , d _ { 2 } \}$ denotes the rank of LoRA. With this approximation, the number of trainable parameters is reduced from $d _ { 1 } \cdot d _ { 2 } \tan r \cdot ( d _ { 1 } + d _ { 2 } )$

Federated LoRA and discordance problem Consider a global pre-trained model $W _ { 0 }$ and a set of clients $\{ 1 , 2 , \cdots , K \}$ . The objective in federated fine-tuning is to update $W _ { 0 }$ to obtain a model W that is suitable for all local datasets $\{ \mathcal { D } _ { k } \} _ { k = 1 } ^ { K }$ However, fine-tuning LLMs is very expensive for local devices in terms of both computation and communication, as billions of parameters must be trained and transmitted in each round.

LoRA presents a promising approach in FL for reducing communication costs, as only low rank matrices B and A are trained and transmitted, allowing the number of communicated parameters to be linearly reduced by the rank r of LoRA modules. However, the straightforward application of LoRA in FL introduces a significant issue known as discordance (Sun et al., 2024), primarily due to aggregation algorithms. In methods like FedAvg (McMahan et al., 2017), where each weight is aggregated individually, discordance occurs between the actual and aggregated parameters. That is,

$$
\begin{array} { c } { { \displaystyle \sum _ { k = 1 } ^ { K } w _ { k } \Delta W _ { k } = \sum _ { k = 1 } ^ { K } w _ { k } B _ { k } A _ { k } } } \\ { { \displaystyle \neq \left( \sum _ { k = 1 } ^ { K } w _ { k } B _ { k } \right) \left( \sum _ { k = 1 } ^ { K } w _ { k } A _ { k } \right) } } \end{array}\tag{2}
$$

![](images/4d37678695d853e412de22944a613d06a597d0a6942400405874b7822f59f8d3.jpg)  
(a) Dir(0.1)

![](images/0c6bb25b6157fa472fde59acd67b355eda5945ddfc16bad74feb811e3fd228a1.jpg)  
(b) Dir(0.01)  
Figure 2: Accuracy of previous Federated LoRA methods across different rank sizes in heterogeneous data settings.

in general, where $\textstyle \sum _ { k = 1 } ^ { K } w _ { k } \ = \ 1$ with $w _ { k } ~ \geq ~ 0$ for all $k \in [ K ]$ . One might consider aggregating $\Delta W _ { k } = B _ { k } A _ { k }$ directly to eliminate the discordance, but this approach involves decomposing $\begin{array} { r } { \Delta W = \sum _ { k = 1 } ^ { K } w _ { k } \Delta W _ { k } } \end{array}$ back into B and A for the next round, which is computationally unstable.

Limited parameter space in low rank and high data heterogeneity This discrepancy can be effectively addressed by either freezing the LoRA module A, as suggested by Sun et al. (2024), or employing SVD decomposition, as outlined by Bai et al. (2024). However, Figure 2 illustrates that the accuracy of these approaches decreases significantly at lower ranks under high heterogeneity. We attribute this decline primarily to the restricted parameter space imposed by LoRA. A limited training parameter space constrains the optimization capabilities for complex FL tasks, and a restricted aggregation parameter space exacerbates conflicts among clients. A detailed analysis of this limited parameter space is provided in Appendix C.

## 4 Proposed Method

To tackle the identified challenges, we propose a novel framework called Low Rank Adaptation with Alternating freeze and Adaptive rank selection for federated learning, or $\mathrm { L o R A { - } A ^ { 2 } }$ , for communication efficient FL with LoRA. LoRA- $\cdot \mathrm { A } ^ { 2 }$ adaptively selects LoRA ranks for local training and transmits only the selected part of each adapter in an alternating way.

## 4.1 Alternating Freeze

$\mathrm { L o R A { – } A { ^ 2 } }$ efficiently addresses the issue of discordance by employing a simple alternating freeze technique to train the LoRA modules B and A. Instead of solely training module B while keeping module A frozen permanently, as suggested by FFA-LoRA (Sun et al., 2024), LoR $\mathsf { A } { - } \mathsf { A } ^ { 2 }$ alternates between the two: LoRA module A is frozen during even rounds, while module B is frozen during odd rounds. This method preserves the optimization space while effectively resolving discordance. Specifically, when freezing A, we have

$$
\begin{array} { l } { \displaystyle \Delta W = \sum _ { k = 1 } ^ { K } \left( w _ { k } B _ { k } \right) A } \\ { \displaystyle = \sum _ { k = 1 } ^ { K } \left( w _ { k } B _ { k } A _ { k } \right) = \sum _ { k = 1 } ^ { K } \left( w _ { k } \Delta W _ { k } \right) , } \end{array}\tag{3}
$$

and the same applies when freezing B. In this way, $\mathrm { L o R A { - } A ^ { 2 } }$ trains both B and A, ensuring that A does not remain the same as its initial value.

To further enhance the effect of alternating optimization, we adopt different learning rates for B and A, inspired by LoRA+ (Hayou et al., 2024). Figure 6 demonstrates the effectiveness of alternating freeze and learning rate adjustment.

## 4.2 Adaptive Rank Selection

Furthermore, we propose an adaptive rank selection method designed to reduce the number of transmitted parameters while preserving the training and aggregation parameter space. This approach selects important LoRA ranks to match local communication rank budget $r _ { i }$ out of global LoRA adapter with rank $r _ { G }$ adaptively based on the local dataset of each clients. We mainly focus on communication cost for uploading parameters to the server as it is known that upload bandwidth is generally much slower than download bandwidth and is the major part of communication cost (Konecnˇ y et al. \` , 2016; Suresh et al., 2017; Kairouz et al., 2021).

The adaptive rank selection process provides two key benefits: (1) it minimizes client conflicts by allowing each client to select different LoRA ranks in high heterogeneity, and (2) it reallocates rank resources from less important LoRA modules to modules that require more fine-tuning, which is especially effective when the communication rank budget is small.

To quantify which ranks are more important, we introduce our original criterion $S _ { m , i }$ for each rank i within module m as follows:

$$
\begin{array} { r l } & { S _ { m , i } ^ { B _ { k } } = \Vert \Delta B _ { k [ : , i ] } A _ { [ i , : ] } \Vert _ { F } } \\ & { S _ { m , i } ^ { A _ { k } } = \Vert { B _ { [ : , i ] } } \Delta { A _ { k [ i , : ] } } \Vert _ { F } } \end{array} .\tag{4}
$$

We define the change in $\Delta W$ for each rank i and module m as contribution $( C _ { m , i } )$ , represented as $\begin{array} { r } { \sum C _ { m , i } = \Delta W _ { k } ^ { t + 1 } - \Delta W _ { k } ^ { t } = \sum ( \Delta B _ { k [ : , i ] } A _ { [ i , : ] } ) . } \end{array}$ And define our criterion $S _ { m , i }$ as the Frobenius norm of contribution $( C _ { m , i } )$ . This criterion captures the impact of each rank on model updates based on local gradients. This approach is better suited for LoRA modules than simpler gradient magnitude-based criteria, $| | \Delta B _ { k [ : , i ] } | |$ or $| | \Delta A _ { k [ i , : ] } | |$ , as our criterion explicitly accounts for the interplay between module A and B. The ablation study in Table 9 empirically supports the superiority of this criterion. At each round, participating clients run local training for 1 epoch to obtain $\Delta W$ for calculating the contribution.

After computing $S _ { m , i } ^ { B _ { k } }$ or $S _ { m , i } ^ { A _ { k } }$ for each module $m ,$ , we select top- $. ( r _ { i } \cdot N )$ LoRA ranks from a total of $r _ { G } \cdot N$ based on the scores across the entire model, where N denotes the number of target modules across all the layers of the base model. We refer to the set of selected ranks of client k as $\mathcal { R } _ { k }$

Once the ranks are selected, each client defines LoRA module mask $\boldsymbol { M } _ { k } ^ { ( m ) }$ for the module m to be

$$
\begin{array} { r } { \boldsymbol { M } _ { \boldsymbol { k } _ { [ \cdot , i ] } } ^ { \mathrm { ~ } ( m ) } = \left\{ \begin{array} { l l } { \mathbf { 1 } _ { d _ { 1 } } ^ { T } } & { \mathrm { i f ~ } i \in \mathcal { R } _ { k } } \\ { \mathbf { 0 } _ { d _ { 1 } } ^ { T } } & { \mathrm { o t h e r w i s e } } \end{array} \right. , } \\ { \boldsymbol { M } _ { \boldsymbol { k } _ { [ i , \cdot ] } ^ { ( m ) } } ^ { \mathrm { ~ } ( m ) } = \left\{ \begin{array} { l l } { \mathbf { 1 } _ { d _ { 2 } } } & { \mathrm { i f ~ } i \in \mathcal { R } _ { k } } \\ { \mathbf { 0 } _ { d _ { 2 } } } & { \mathrm { o t h e r w i s e } } \end{array} \right. , } \end{array}\tag{5}
$$

which is producted element-wise to the updated part of $B _ { k }$ (or $A _ { k } )$ . That is, before each backpropagation step, LoRA- $\cdot \mathbf { A } ^ { 2 }$ calculates

$$
\begin{array} { l } { \Delta { B _ { k } } ^ { ( m ) }  \Delta { B _ { k } } ^ { ( m ) } \odot { M _ { k } } ^ { ( m ) } } \\ { \Delta { A _ { k } } ^ { ( m ) }  \Delta { A _ { k } } ^ { ( m ) } \odot { M _ { k } } ^ { ( m ) } } \end{array}\tag{6}
$$

for each $B _ { k } \left( \operatorname { o r } A _ { k } \right)$ , where the notation stands for the Hadamard product. After each local training, each client uploads $B _ { k } \odot M _ { k }$ (or $A _ { k } \odot M _ { k } )$ resulting in sparsification and reducing the number of uploaded parameters. Then, the server aggregates the uploaded ones, which are again added to the $B _ { k } \left( \operatorname { o r } A _ { k } \right)$ saved two rounds before. Algorithm 1 and 2 provides the detailed pseudocode of our $\mathrm { L o R A – A ^ { 2 } }$ algorithm.

## 4.3 Theoretical Insights

In this section, we provide a brief theoretical analysis of the parameter spaces associated with previous methods and our proposed $\mathrm { L o R A { - } A ^ { 2 } }$ framework. To substantiate our approach, we introduce the following proposition:

Algorithm 1 $\mathrm { L o R A – A ^ { 2 } }$ Algorithm 2 LocalTraining   
Initialize $\Delta W = B A$ with $B \in \mathbb { R } ^ { d _ { 1 } \times r _ { G } }$ and [Rank Selection]   
$A \in \mathbb { R } ^ { r _ { G } \times d _ { 2 } }$ for each LoRA adapter Calculate importance scores following (4)   
for $t = 1 , 2 , \cdots , T$ do Define the mask $M _ { k }$ following (5)   
Sample participants $\mathcal { K } ^ { \left( t \right) } \subseteq \left[ K \right]$ for round t [Local Training]   
$\begin{array} { r } { w _ { k } = | \mathcal { D } _ { k } | / \left( \sum _ { k = 1 } ^ { K } | \mathcal { D } _ { k } | \right) } \end{array}$ $\mathbf { i f } t \% 2 = 1$ $B _ { k } ^ { ( t ; e - 1 ) } = B ^ { ( t ) }$ then   
if $t \% 2 = 1$ then for $e = 1 , 2 . \cdots , E$ do   
for $k = 1 , 2 , \cdots , K$ in parallel do $\Delta B _ { k } ^ { ( t ; e ) } = B _ { k } ^ { ( t ; e ) } - B _ { k } ^ { ( t ; e - 1 ) }$   
$\Delta B _ { k } ^ { ( t + 1 ) } =$ LocalTraining $( B ^ { ( t ) } , t )$ $\Delta B _ { k } ^ { ( t ; e ) } = \Delta \bar { B } _ { k } ^ { ( t ; e ) } \odot \bar { M } _ { k }$   
$\begin{array} { r } { B _ { \cdot } ^ { ( t + 1 ) } = B _ { \cdot } ^ { ( t ) } + \sum _ { k = 1 } ^ { K } w _ { k } \Delta B _ { k } ^ { ( t + 1 ) } } \end{array}$   
$A ^ { ( t + 1 ) } = A ^ { ( t ) }$ Backpropagate $\Delta B _ { k } ^ { ( t ; e ) }$   
end for   
end for Return: $B _ { k } ^ { ( t ; E ) } - B ^ { ( t ) }$   
else   
else   
for $k = 1 , 2 , \cdots , K$ in parallel do $\boldsymbol { A } _ { k } ^ { ( t ; e - 1 ) } = \boldsymbol { A } ^ { ( t ) }$   
$\Delta A _ { k } ^ { ( t + 1 ) } = 1$ LocalTraining $\cdot A ^ { ( t ) } , t )$   
for $e = 1 , 2 . \cdots , E$ do   
$\begin{array} { r } { \dot { A } ^ { ( t + 1 ) } = \dot { A } ^ { ( t ) } + \sum _ { k = 1 } ^ { K } w _ { k } \Delta A _ { k } ^ { ( t + 1 ) } } \end{array}$ $\Delta A _ { k } ^ { ( t ; e ) } = A _ { k } ^ { ( t ; e ) } - A _ { k } ^ { ( t ; e - 1 ) }$   
$B ^ { ( t + 1 ) } = B ^ { ( t ) }$   
end for $\Delta \ r { A } _ { k } ^ { ( t ; e ) } = \Delta \ r { \ddot { A } } _ { k } ^ { ( t ; e ) } \odot \ddot { M } _ { k }$   
end if Backpropagate $\Delta A _ { k } ^ { ( t ; e ) }$   
end for end for   
Return: $A _ { k } ^ { ( t ; E ) } - A ^ { ( t ) }$   
end if

Proposition 1. For a model W, consider LoRAbased FL algorithms which update r rank parameters per round. Let $\Omega _ { \mathcal { A } }$ denote the space of all possible parameter values that an algorithm in $\mathcal { A } \in$ FFA-LoRA, FL+LoRA, FlexLoRA, LoRA-A<sup>2</sup> can make. Then, we have the following:

$$
\Omega _ { \mathrm { F F A - L o R A } } \subsetneq \Omega _ { \mathrm { F L + L o R A } } = \Omega _ { \mathrm { F l e x L o R A } } \subset \Omega _ { \mathrm { L o R A - A ^ { 2 } } } .
$$

The proof of the proposition is provided in $\mathsf { A p - }$ pendix D.

Our algorithm is designed to adaptively select the relevant training and aggregation parameter spaces while concurrently reducing the number of parameters that are updated.

## 5 Experiments

In this section, we evaluate the performance of our algorithm against existing FL methods combined with LoRA across various heterogeneity settings and datasets. We assess performance based on accuracy and the total number of uploaded parameters.

## 5.1 Experimental Settings

We mainly adopt pre-trained RoBERTa-base (Liu et al., 2019) as the base model for fine-tuning. The base model has approximately 125M parameters, all of which are frozen during the fine-tuning phase. And a frozen classifier is added upon the model, following Sun et al. (2024). For Table 2 and 3, we adopt RoBERTa-large and DistilBERT(Sanh et al., 2019), respectively. RoBERTa-large has approximately 355M parameters, and DistilBERT has approximately 82M parameters. For fine-tuning, we choose BANKING77 (Casanueva et al., 2020) and 20 Newsgroups (Lang, 1995) datasets. These datasets are chosen for their ability to simulate a controlled level of data heterogeneity using Dirichlet distribution (Hsu et al., 2019). Dataset statistics are reported in Appendix A.

Unless otherwise stated, we trained 30 local clients under full participation, i.e., $\mathcal { K } ^ { \left( t \right) } = \left[ K \right]$ for all $t \in [ T ]$ . The clients were trained for 50 rounds with 5 local epochs. Detailed hyperparameters for experiments are specified in Appendix B.

For baselines, we adopt four methods that utilize LoRA for federated fine-tuning: FL + LoRA, FFA-LoRA (Sun et al., 2024), FlexLoRA (Bai et al., 2024), and HetLoRA (Cho et al., 2023), where FL + LoRA stands for the naive implementation of LoRA in FedAvg (McMahan et al., 2017).

## 5.2 Main Results

We compare our algorithm with the baseline methods under various data heterogeneity settings in BANKING77 and 20 Newsgroups datasets to

<table><tr><td rowspan="2">Method</td><td colspan="3">BANKING77 Dataset</td><td colspan="3">20 Newsgroups Dataset</td><td rowspan="2">Communicated Parameters*</td></tr><tr><td> $D i r ( 0 . 5 )$ </td><td> $D i r ( 0 . 1 )$ </td><td> $D i r ( 0 . 0 1 )$ </td><td> $D i r ( 0 . 5 )$ </td><td> $D i r ( 0 . 1 )$ </td><td> $D i r ( 0 . 0 1 )$ </td></tr><tr><td> $\mathrm { { F L } ( w / o \mathrm { { L o R A } ) } }$ </td><td> $9 2 . 7 6 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $9 0 . 2 9 _ { \pm 0 . 7 3 }$ </td><td> $6 7 . 5 8 _ { \pm 0 . 4 4 }$ </td><td> $7 0 . 9 3 _ { \pm 1 . 0 4 }$ </td><td> $6 8 . 8 2 _ { \pm 0 . 6 9 }$ </td><td> $6 4 . 4 1 _ { \pm 0 . 3 0 }$ </td><td>186B</td></tr><tr><td> $\mathrm { F L } + \mathrm { L o R A } _ { \mathrm { ( R a n k = 8 ) } }$ </td><td> $9 2 . 8 0 _ { \pm 0 . 2 4 }$ </td><td> $9 0 . 4 7 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $6 0 . 9 6 _ { \pm 1 . 4 7 }$ </td><td> $7 0 . 4 4 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $6 7 . 3 3 \substack { \pm 0 . 1 8 }$ </td><td> $4 3 . 9 0 _ { \pm 1 . 0 8 }$ </td><td>1.99B</td></tr><tr><td> $\mathrm { F F A - L o R A _ { ( R a n k = 8 ) } }$ </td><td> $8 7 . 2 0 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $7 7 . 4 4 _ { \pm 1 . 2 8 }$ </td><td> $4 0 . 8 8 _ { \pm 1 . 0 4 }$ </td><td> $6 7 . 0 0 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $6 1 . 2 7 _ { \pm 0 . 7 1 }$ </td><td> $3 7 . 3 4 _ { \pm 0 . 3 0 }$ </td><td>0.991B</td></tr><tr><td> $\mathrm { F l e x L o R A _ { ( R a n k = 8 ) } }$ </td><td> $9 3 . 3 5 _ { \pm 0 . 2 4 }$ </td><td> $\mathbf { 9 2 . 1 4 _ { \pm 0 . 2 5 } }$ </td><td> $6 9 . 8 4 _ { \pm 0 . 6 5 }$ </td><td> ${ \bf 7 0 . 5 9 } _ { \pm 0 . 2 2 }$ </td><td> ${ \bf 6 8 . 1 0 _ { \pm 0 . 3 8 } }$ </td><td> ${ \bf 6 0 . 4 1 { _ { \pm 1 . 5 4 } } }$ </td><td>1.99B</td></tr><tr><td> $\mathrm { O u r s } _ { ( \mathrm { R a n k } = 8 ) }$ </td><td> $9 3 . 2 4 _ { \pm 0 . 2 7 }$ </td><td> $9 1 . 6 1 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $\mathbf { 7 0 . 1 3 _ { \pm 1 . 2 2 } }$ </td><td> $7 0 . 2 6 _ { \pm 0 . 2 1 }$ </td><td> $6 7 . 1 2 _ { \pm 0 . 2 2 }$ </td><td> $5 4 . 5 0 { \scriptstyle \pm 1 . 4 4 }$ </td><td>1.31B</td></tr><tr><td> $\mathrm { F L } + \mathrm { L o R A } _ { \mathrm { ( R a n k = 4 ) } }$ </td><td> $9 2 . 8 6 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $8 8 . 1 1 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $5 4 . 9 9 _ { \pm 0 . 5 9 }$ </td><td> $7 0 . 3 3 \mathrm { \pm 0 . 1 2 }$ </td><td> $6 7 . 2 9 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $4 3 . 1 2 _ { \pm 2 . 6 7 }$ </td><td>0.991B</td></tr><tr><td>FFA-LoR  $\mathbf { A } _ { ( \mathbf { R a n k } = 4 ) }$ </td><td> $8 6 . 9 0 { \scriptstyle \pm 1 . 1 4 }$ </td><td> $7 6 . 3 8 \mathrm { \scriptstyle \pm 0 . 6 1 }$ </td><td> $3 7 . 6 3 _ { \pm 0 . 8 0 }$ </td><td> $6 7 . 7 5 _ { \pm 0 . 4 5 }$ </td><td> $6 1 . 2 5 _ { \pm 0 . 2 6 }$ </td><td> $3 6 . 0 4 _ { \pm 0 . 8 0 }$ </td><td>0.497B</td></tr><tr><td> $\mathrm { F l e x L o R A } _ { \mathrm { ( R a n k = 4 ) } }$ </td><td> $9 2 . 7 1 _ { \pm 0 . 3 1 }$ </td><td> $\underline { { 9 0 . 5 3 } } \pm 0 . 7 0$ </td><td> $5 7 . 3 8 { \scriptstyle \pm 1 . 3 0 }$ </td><td> $7 0 . 0 5 _ { \pm 0 . 1 4 }$ </td><td> ${ \bf 6 8 . 0 0 _ { \pm 0 . 3 3 } }$ </td><td> $\underline { { 5 0 . 5 0 } } \underline { { \pm 2 . 0 9 } }$ </td><td>0.991B</td></tr><tr><td> $\mathrm { O u r s } _ { ( \mathrm { R a n k } = 4 ) }$ </td><td> $9 3 . 2 2 _ { \pm 0 . 2 4 }$ </td><td> $\mathbf { 9 1 . 4 3 _ { \pm 0 . 6 3 } }$ </td><td> ${ \bf 6 9 . 6 3 _ { \pm 1 . 5 2 } }$ </td><td> $\mathbf { 7 0 . 2 8 _ { \pm 0 . 3 2 } }$ </td><td> $6 7 . 1 2 _ { \pm 0 . 6 0 }$ </td><td> ${ \pm 3 . 0 4 } _ { \pm 1 . 6 8 }$ </td><td>0.888B</td></tr><tr><td> $\mathrm { F L } + \mathrm { L o R A } _ { \mathrm { ( R a n k = 2 ) } }$ </td><td> $9 1 . 9 7 _ { \pm 0 . 4 3 }$ </td><td> $8 5 . 5 9 _ { \pm 1 . 1 3 }$ </td><td> $4 9 . 0 8 _ { \pm 0 . 5 6 }$ </td><td> $\mathbf { 7 0 . 1 4 _ { \pm 0 . 1 3 } }$ </td><td> $6 5 . 4 0 _ { \pm 0 . 3 1 }$ </td><td> $3 9 . 0 7 _ { \pm 2 . 2 3 }$ </td><td>0.497B</td></tr><tr><td> $\mathrm { F F A - L o R A _ { ( R a n k = 2 ) } }$   $\mathrm { F l e x L o R A _ { ( R a n k = 2 ) } }$ </td><td> $8 4 . 6 5 { \scriptstyle \pm 1 . 0 5 }$ </td><td> $7 3 . 4 4 _ { \pm 0 . 8 8 }$ </td><td> $3 4 . 4 4 { \scriptstyle \pm 2 . 1 5 }$ </td><td> $6 8 . 1 2 _ { \pm 0 . 4 7 }$ </td><td> $6 1 . 5 7 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $3 6 . 6 5 { \scriptstyle \pm 0 . 5 2 }$ </td><td>0.249B</td></tr><tr><td> $\mathrm { O u r s } _ { ( \mathrm { R a n k } = 2 ) }$ </td><td> $9 2 . 2 2 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $8 7 . 3 1 _ { \pm 0 . 2 7 }$ </td><td> $5 5 . 2 4 { \scriptstyle \pm 2 . 1 9 }$ </td><td> $7 0 . 0 3 _ { \pm 0 . 3 1 }$ </td><td> $6 6 . 1 7 _ { \pm 1 . 7 0 }$ </td><td> $4 8 . 2 3 _ { \pm 1 . 7 3 }$ </td><td>0.497B</td></tr><tr><td></td><td> $\mathbf { 9 3 . 1 0 } _ { \pm 0 . 0 7 }$ </td><td> $\mathbf { 9 } 2 . \mathbf { 0 } 2 _ { \pm 0 . 3 6 }$ </td><td> ${ \bf 6 9 . 4 0 { _ { \pm 0 . 4 8 } } }$ </td><td> $\underline { { 7 0 . 1 2 } } \pm 0 . 1 8$ </td><td> ${ \bf 6 7 . 0 2 _ { \pm 0 . 2 6 } }$ </td><td> ${ \pm 2 . 9 9 } _ { \pm 2 . 5 6 }$ </td><td>0.528B</td></tr><tr><td> $\mathrm { F L } + \mathrm { L o R A } _ { \mathrm { ( R a n k = 1 ) } }$   $\mathrm { F F A - L o R A _ { ( R a n k = 1 ) } }$ </td><td> $9 0 . 6 1 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $\underline { { 8 2 . 2 4 } } \underline { { \pm 1 . 6 8 } }$ </td><td> $4 5 . 7 8 { \scriptstyle \pm 1 . 0 4 }$ </td><td> $6 9 . 4 0 _ { \pm 0 . 3 3 }$ </td><td> $6 3 . 1 6 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $3 6 . 5 8 { \scriptstyle \pm 0 . 9 8 }$ </td><td>0.249B</td></tr><tr><td> $\mathrm { F l e x L o R A } _ { \mathrm { ( R a n k = 1 ) } }$ </td><td> $8 2 . 5 1 { \scriptstyle \pm 0 . 5 3 }$   $9 0 . 4 0 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $7 2 . 9 6 { \scriptstyle \pm 0 . 5 4 }$   $8 2 . 2 0 { \scriptstyle \pm 0 . 7 4 }$ </td><td> $3 3 . 6 8 _ { \pm 0 . 2 0 }$   $4 2 . 7 5 _ { \pm 0 . 8 9 }$ </td><td> $6 7 . 7 3 _ { \pm 0 . 3 0 }$   $6 9 . 5 3 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $6 1 . 3 5 _ { \pm 0 . 2 2 }$ </td><td> $3 4 . 4 4 _ { \pm 0 . 6 8 }$   $3 5 . 5 4 { \scriptstyle \pm 0 . 6 8 }$ </td><td>0.124B</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td> $6 2 . 9 8 { \scriptstyle \pm 1 . 1 2 }$ </td><td></td><td>0.249B</td></tr><tr><td> $\mathrm { O u r s } _ { ( \mathrm { R a n k } = 1 ) }$ </td><td> ${ \bf 9 3 . 2 1 _ { \pm 0 . 1 3 } }$ </td><td> ${ \bf 9 1 . 8 7 _ { \pm 0 . 3 3 } }$ </td><td> ${ \bf 6 8 . 8 8 _ { \pm 1 . 1 5 } }$ </td><td> ${ \bf 7 0 . 3 1 { \scriptstyle \pm 0 . 2 4 } }$ </td><td> ${ \bf 6 6 . 9 5 _ { \pm 0 . 0 7 } }$ </td><td> ${ \pm 4 . 8 4 } _ { \pm 1 . 1 5 }$ </td><td>0.270B</td></tr></table>

Table 1: Results with RoBERTa-base on BANKING77 and 20 Newsgroups datasets. Smaller α for $D i r ( \alpha )$ implies that the simulated setting is more heterogeneous. The best results on each dataset are shown in bold and second best is shown by underline. ∗ This column reports the total number of uploaded parameters, averaged across rows.

demonstrate that our algorithm outperforms previous federated LoRA fine-tuning methods across different non-IID settings and LoRA ranks.

fectively solves the significant communication cost challenges of federated fine-tuning on LLMs.

## 5.3 Analysis on Adaptive Rank Selection

Robustness of $\mathbf { L o R A { - } A ^ { 2 } }$ in low ranks and high heterogeneity Table 1 highlights the vulnerability of previous methods under conditions of high heterogeneity and low ranks. The accuracy of baseline methods declines significantly as rank decreases, whereas our algorithm maintains its performance, achieving up to a 23% accuracy advantage. This suggests that reducing LoRA ranks is challenging for previous methods under realistic heterogeneous data conditions. Also, our algorithm consistently achieves the highest performance or remains within a 1% margin of the best-performing baselines at rank 8 and 4, while showing a large performance gap in low ranks.

In this section, we visualize the process of our adaptive rank selection, and explore how efficiently $\mathrm { L o R A { - } A ^ { 2 } }$ trains and sends important ranks, highlighting the robustness of our algorithm in heterogeneous and low rank environments. To simulate extreme cases of both identical and different client distributions, we test our algorithm on a pathological dataset using the 20 Newsgroups dataset. In this setup, 20 clients each holds data from only two classes, with consecutive pairs sharing the same classes, while others do not. For instance, clients 0 and 1 have classes "medical" and "space," whereas clients 2 and 3 have "motorcycle" and "religions". Detailed settings are shown in Appendix C.

Communication cost reduction by $\mathbf { L o R A { - } A ^ { 2 } }$ Decreasing LoRA ranks in federated LoRA methods reduces the communication cost linearly. Our algorithm achieves performance comparable to or better than fully fine-tuned models even at rank 1, allowing for up to a 99.8% reduction in communicated parameters with minimal performance degradation. This demonstrates that LoRA-A<sup>2</sup> ef-

Robustness to low-rank by adaptive module selection In this experiment, our algorithm selects $2 \cdot N ^ { ( m ) }$ ranks from a total of $1 6 \cdot N ^ { ( m ) }$ across the whole RoBERTa model, guided by our importance criterion, and visualizes the adaptive selection of modules. Figure 3 illustrates the number of ranks selected for each module in the model during the training. The figure shows that most modules are allocated zero ranks, indicating either no need for fine-tuning or the insignificance of updates on those modules. This suggests that our adaptive rank selection automatically prunes out modules that do not require additional fine-tuning.

![](images/66ffd356e4e039dcbec5790f72ba042ef879574e26058f4b0fba8ba720e3b9c5.jpg)  
(a) client 0

![](images/d163f8127be041256abc0c731af408f27be3a7dd9071ff91c0e58f9b46219c7a.jpg)  
(b) client 1

![](images/ee1b2bf2a49c20d986583693eca7eae6894ac44873af5518fc1c47ec87490da8.jpg)  
(c) client 2

Figure 3: Visualization on number of selected rank per module. The x-axis shows RoBERTa module types, while the y-axis indicates layer numbers. Experimented on the 20 Newsgroups dataset with a pathological data distribution. Average 2 ranks were selected out of 16 ranks by our adaptive rank selection algorithm.  
![](images/0d5c1102ef01737b61827ef3493e4bf7dd587c8a5eda2999f2656a35571806ab.jpg)  
(a) Selected layers

![](images/6693d19e3cd58a0128cb2aef4b79a1d08b75afc47e80a16181cbcf5f0675a430.jpg)  
(b) Selected modules  
Figure 4: Ablation analysis on the performance of model when solely fine-tuned on selected layers or types of modules. Experimented on 20 Newsgroups dataset with Dir(0.1) heterogeneity.

To further justify that our adaptive rank selection successfully selects important modules, we conduct an ablation experiment on module selection, following the approach of AdaLoRA (Zhang et al., 2023) but in a federated setting. Figure 4 displays the model’s performance when only specific modules or layers are fine-tuned and other layers are frozen. The ablation experiment demonstrated that last layer in layer experiment and intermediate or output dense modules in module experiment led to the best performance, highlighting their importance for fine-tuning. This aligns with our findings, where the last layers and intermediate / output dense modules are automatically selected through adaptive rank selection, demonstrating the effectiveness of our algorithm in prioritizing essential modules for fine-tuning.

Robustness to data heterogeneity by client clustering Another effect of rank selection is the implicit clustering of clients to minimize conflicts among clients with dissimilar datasets and to enhance cooperation among those with similar ones.

Figure 5 (a) illustrates how much local rank parameters are shared among different clients. The figure shows that clients with similar data distributions tend to share more rank parameters, while those with dissimilar data share fewer. This trend is also evident at the module level in Figure 3, where clients 0 and 1 select a similar number of ranks for each module, differing from client 2, while retaining the tendency to choose more ranks from the last layers or intermediate and output dense modules. These findings suggest that clients with similar datasets converge on the same ranks, facilitating cooperative training, whereas clients with dissimilar datasets select more distinct ranks, resulting in independent parameter updates.

Figure 5 (b) further supports this by visualizing the cosine similarity between clients’ model updates, which approaches 1 for clients with the same classes and remains close to 0 for those without data overlap. This underscores the cooperative nature of updates from similar clients while maintaining independence from dissimilar ones, thereby contributing to the robustness of our algorithm against data heterogeneity.

![](images/481027fac3d3e516325ec6ed0817ce0a4450a5fbddad0a252566f3fafc9c1d4c.jpg)

(a) Rank selection similarity  
![](images/7256d62cf7fec528031ce451909d60295836dd5fbf07a4fc7a9e1ca9d2b173b7.jpg)  
(b) Cosine similarity of local updates  
Figure 5: Visualization of similarity between clients. the x and y axes represent individual clients trained on 20 Newsgroups dataset with pathologic data distribution.

## 5.4 Ablation Studies

The following ablation studies provide empirical evidence supporting our design choices for aggregation tactics and rank selection criteria.

Efficacy of alternating freeze To address the discordance problem in federated LoRA aggregation, we employ a strategy that alternately freezes LoRA modules B and A, rather than freezing module A only as in FFA-LoRA (Cho et al., 2023). Furthermore, we set the learning rate of module B, η<sub>B</sub>, to be five times that of module $A , \eta _ { A } .$ , inspired by LoRA+ (Hayou et al., 2024). This configuration further enhances overall performance and robustness, particularly in highly heterogeneous environments. Figure 6 compares these approaches, showing that solely freezing A is less effective under high data heterogeneity, whereas achieves consistently better performance.

![](images/42e5ef3a5444fb2bcaf86f8d805219ae3fe901228dac66f769d40706714c09dd.jpg)

![](images/c908b133d4b7059cdf4a19d0140259467cf9ec5e26c0b99b389b67a6b035d865.jpg)  
(a) BANKING77  
(b) 20 Newsgroups

Figure 6: Ablation analysis for the effect of alternating freeze and learning rate adjustment under varying levels of heterogeneity.
<table><tr><td># of</td><td colspan="3">RoBERTa-Large</td></tr><tr><td>Ranks</td><td> $_ \mathrm { F L + L o R A }$ </td><td> $\mathrm { F F A - L o R A }$ </td><td>FlexLoRA*</td><td>Ours</td></tr><tr><td>8</td><td> $\underline { { 8 0 . 1 5 } } \pm 0 . 5 8$ </td><td> $6 2 . 9 8 \mathrm { \pm 0 . 6 1 }$ </td><td></td><td> $\mathbf { 8 5 . 9 8 _ { \pm 0 . 8 2 } }$ </td></tr><tr><td>4</td><td> $\underline { { 7 8 . 9 7 } } \pm 0 . 5 2$ </td><td> $6 2 . 4 5 _ { \pm 0 . 3 3 }$ </td><td></td><td> $\mathbf { 8 4 . 6 2 _ { \pm 0 . 3 7 } }$ </td></tr><tr><td>2</td><td> $7 5 . 0 9 _ { \pm 1 . 2 0 }$ </td><td> $6 1 . 5 5 _ { \pm 1 . 0 5 }$ </td><td>-</td><td> $\mathbf { 8 3 . 4 0 _ { \pm 0 . 5 5 } }$ </td></tr><tr><td>1</td><td> $\underline { { 7 3 . 7 5 } } { \pm 1 . 5 3 }$ </td><td> $5 8 . 0 6 _ { \pm 1 . 9 0 }$ </td><td>-</td><td> $\mathbf { 8 5 . 6 6 } \mathbf { \sigma } _ { \pm 0 . 3 6 }$ </td></tr></table>

Table 2: Experimental results on RoBERTa-Large model. The level of heterogeneity is Dir(0.01).

\* FlexLoRA results could not be reported due to an ill-conditioned matrix issue in SVD decomposition.
<table><tr><td rowspan="2"># of Ranks</td><td colspan="4">DistilBERT</td></tr><tr><td> $_ \mathrm { F L + L o R A }$ </td><td> $\mathrm { F F A - L o R A }$ </td><td>FlexLoRA</td><td>Ours</td></tr><tr><td>8</td><td> $3 2 . 5 8 _ { \pm 0 . 3 4 }$ </td><td> $1 8 . 8 2 _ { \pm 0 . 5 7 }$ </td><td> $5 1 . 2 1 _ { \pm 0 . 5 1 }$ </td><td> ${ \pm 2 . 9 7 } _ { \pm 0 . 3 2 }$ </td></tr><tr><td>4</td><td> $3 6 . 9 2 _ { \pm 0 . 3 7 }$ </td><td> $1 6 . 7 3 _ { \pm 0 . 5 2 }$ </td><td> $4 1 . 2 6 _ { \pm 0 . 4 7 }$ </td><td> ${ \bf 5 1 . } 2 4 _ { \pm 0 . 4 4 }$ </td></tr><tr><td>2</td><td> $2 7 . 1 4 _ { \pm 0 . 9 2 }$ </td><td> $1 5 . 4 9 _ { \pm 1 . 2 4 }$ </td><td> $\underline { { 3 4 . 0 5 } } \pm 0 . 8 2 $ </td><td> ${ \bf 4 9 . 9 7 { \scriptstyle \pm 0 . 3 3 } }$ </td></tr><tr><td>1</td><td> $2 1 . 5 9 { \scriptstyle \pm 1 . 1 2 }$ </td><td> $1 4 . 2 9 _ { \pm 1 . 3 4 }$ </td><td> $2 1 . 0 1 { \scriptstyle \pm 1 . 2 3 }$ </td><td> ${ \bf 4 8 . 8 9 2 0 . 4 1 }$ </td></tr></table>

Table 3: Experimental results on DistilBERT model. The level of heterogeneity is Dir(0.01).

Scalability and generalizability on model structures To evaluate the scalability and generalizability of our algorithm across various model structures, we present the experimental results on RoBERTa-large (Liu et al., 2019) and DistilBERT (Sanh et al., 2019) models in Table 2 and Table 3, respectively. These tables illustrate the performance of our model when applied to diverse architectures and parameter configurations. The results show that our algorithm achieves superior performance, even on models with a larger number of parameters or different architectures. This highlights the robust scalability and generalizability of our approach across different model structures.

## 5.5 Additional Experiments

Differential privacy According to Sun et al. (2024), discordance problem of federated LoRA intesnsified when Differential Privacy (DP) is applied (Dwork et al., 2006; Abadi et al., 2016), due to the added noise amplifying errors. Specifically, if $\xi _ { B }$ and $\xi _ { A }$ stand for the noise added to B and A, respectively, we have $\Delta W = ( B + \xi _ { B } ) ( A + \xi _ { A } )$ $B A + B \xi _ { A } + \xi _ { B } A + \xi _ { B } \xi _ { A }$

<table><tr><td>€</td><td></td><td>FL+LoRA FFA-LoRA FlexLoRA|</td><td></td><td>Ours</td></tr><tr><td>∞</td><td> $4 9 . 0 8 _ { \pm 0 . 5 6 }$ </td><td> $3 4 . 4 4 _ { \pm 2 . 1 5 }$ </td><td> $5 5 . 2 4 { \scriptstyle \pm 2 . 1 9 }$ </td><td> ${ \bf 6 9 . 4 0 { \scriptstyle \pm 0 . 4 8 } }$ </td></tr><tr><td>6</td><td> $4 7 . 9 7 _ { \pm 0 . 7 2 }$ </td><td> $3 5 . 3 5 _ { \pm 0 . 9 4 }$ </td><td> $5 0 . 2 2 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $\mathbf { 7 0 . 4 4 } _ { \pm 1 . 8 8 }$ </td></tr><tr><td>3</td><td> $4 4 . 0 1 _ { \pm 0 . 3 8 }$ </td><td> $3 1 . 9 0 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $4 9 . 6 2 { \scriptstyle \pm 0 . 7 6 }$ </td><td> ${ \bf 6 8 . 6 2 } _ { \pm 1 . 6 1 }$ </td></tr><tr><td>1</td><td> $4 1 . 0 5 _ { \pm 1 . 1 1 }$ </td><td> $3 3 . 7 8 _ { \pm 0 . 7 5 }$ </td><td> $\underline { { 4 9 . 3 9 } } \pm 1 . 7 6$ </td><td> $\mathbf { 6 8 . 7 0 _ { \pm 0 . 2 2 } }$ </td></tr></table>

Table 4: Experiments with differential privacy.

Table 4 represents experiments on BANKING77 dataset with DP. Following Ryu et al. (2022), Laplace mechanism is adopted. The level of heterogeneity is $D i r ( 0 . 0 1 )$ and the rank is set to 2 for each method. The clipping constant C is set to either 2 or 5, whichever yields better performance, for each method.

The tables demonstrates that FFA-LoRA (Sun et al., 2024), FlexLoRA (Bai et al., 2024) and Our algorithm effectively mitigate the discordance problem, While FL with LoRA suffers from performance degradation. Moreover our algorithm shows the highest robustness under conditions of severe noise, such as ϵ = 1 and ϵ = 3, outperforming other baseline methods.

Computational overhead Regarding computational overhead, our analysis shows that LoRA-A exhibits a 1.17x increase in computation time compared to standard FL+LoRA, slightly higher than FFA-LoRA (0.93x) and FlexLoRA (1x). This is due to gradient computation for local rank selection. However, we note that communication time, often the dominant bottleneck in federated learning, is significantly reduced by $\mathrm { L o R A { - } A ^ { 2 } }$ , outweighing the modest increase in computation time.

Other experiments We also include further experiments addressing resource heterogeneity settings, pathological distributions, as well as investigations into convergence speed in Appendix C.

## 6 Conclusion

In this work, we tackle the vulnerability of previous methods in high heterogeneity and low ranks by proposing a novel algorithm, $\mathrm { L o R A { - } A ^ { 2 } }$ , which shows robustness in these challenging conditions with alternating freeze and adaptive rank selection. Our approach offers significant improvements in communication efficiency without compromising performance, as demonstrated by a reduction of 99.8% in parameter uploads compared to full fine-tuning. Through extensive experiments, we establish LoR $\mathbf { A } { - } \mathbf { A } ^ { 2 }$ as a superior alternative, providing a practical pathway for efficient and effective federated fine-tuning in diverse and resourceconstrained environments.

## 7 Acknowledgments

This work was supported by Institute of Information $\&$ communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2019-II191906, Artificial Intelligence Graduate School Program (POSTECH); RS-2021-II210739, Development of Distributed/Cooperative AI based 5G+ Network Data Analytics Functions and Control Technology; RS-2024-00457882, AI Research Hub Project; RS-2024-00509258, Global AI Frontier Lab).

## 8 Limitations

$\mathrm { L o R A { – } A { ^ 2 } }$ shows promising results and we plan to distribute the implementation code with detailed instructions for reproducibility. However, several areas remain open for future exploration.

First, our work mainly focuses on classification tasks, primarily due to computational constraints and the use of Dirichlet distribution to simulate non-IID conditions. However, extending $\mathrm { L o R A { – } A { ^ 2 } }$ to more complex tasks, such as natural language generation, could offer additional perspectives. Future work with more resources could explore these broader applications.

Second, our experiments are primarily conducted on comparatively smaller language models, such as RoBERTa-base and RoBERTa-large, due to limited computation resources. Applying LoRA-$\mathrm { A ^ { 2 } }$ to larger models, such as LLaMA or GPT-style architectures, could provide an opportunity to test its scalability. Investigating how well the method handles the increased parameter space of these state-of-the-art models could further demonstrate its efficiency.

Finally, due to the limited access to real world datasets, our current results are mainly based on simulated settings. Extensive research on real world dataset, which typically exhibit more diverse types of noise and heterogeneity would help understand performance and robustness of $\mathrm { L o R A – A ^ { 2 } }$ in practical, dynamic environments.

## References

Martin Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. 2016. Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security, CCS ’16, page 308–318, New York, NY, USA. Association for Computing Machinery.

Armen Aghajanyan, Sonal Gupta, and Luke Zettlemoyer. 2021. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. In Proceedings of the 59th Annual Meeting of the Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 7319–7328, Online. Association for Computational Linguistics.

Sara Babakniya, Ahmed Elkordy, Yahya Ezzeldin, Qingfeng Liu, Kee-Bong Song, MOSTAFA EL-Khamy, and Salman Avestimehr. 2023. SLoRA: Federated parameter efficient fine-tuning of language models. In International Workshop on Federated Learning in the Age of Foundation Models in Conjunction with NeurIPS 2023.

Jiamu Bai, Daoyuan Chen, Bingchen Qian, Liuyi Yao, and Yaliang Li. 2024. Federated fine-tuning of large language models under heterogeneous tasks and client resources. arXiv preprint arXiv:2402.11505.

Daniel J Beutel, Taner Topal, Akhil Mathur, Xinchi Qiu, Javier Fernandez-Marques, Yan Gao, Lorenzo Sani, Hei Li Kwing, Titouan Parcollet, Pedro PB de Gusmão, and Nicholas D Lane. 2020. Flower: A friendly federated learning research framework. arXiv preprint arXiv:2007.14390.

Iñigo Casanueva, Tadas Temcinas, Daniela Gerz,ˇ Matthew Henderson, and Ivan Vulic. 2020. Efficient´ intent detection with dual sentence encoders. In Proceedings ofthe 2nd Workshop on Natural Language Processingfor Conversational AI, pages 38–45, Online. Association for Computational Linguistics.

Daoyuan Chen, Liuyi Yao, Dawei Gao, Bolin Ding, and Yaliang Li. 2023. Efficient personalized federated learning via sparse model-adaptation. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 5234–5256. PMLR.

Yae Jee Cho, Luyang Liu, Zheng Xu, Aldi Fahrezi, Matt Barnes, and Gauri Joshi. 2023. Heterogeneous loRA for federated fine-tuning of on-device foundation models. In International Workshop on Federated Learning in the Age of Foundation Models in Conjunction with NeurIPS 2023.

Jesse Dodge, Gabriel Ilharco, Roy Schwartz, Ali Farhadi, Hannaneh Hajishirzi, and Noah Smith. 2020. Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping. arXiv preprint arXiv:2002.06305.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. 2006. Calibrating noise to sensitivity in private data analysis. In Theory of Cryptography, pages 265–284, Berlin, Heidelberg. Springer Berlin Heidelberg.

Soufiane Hayou, Nikhil Ghosh, and Bin Yu. 2024. Lora+: Efficient low rank adaptation of large models. arXiv 2402.12354.

Tzu-Ming Harry Hsu, Hang Qi, and Matthew Brown. 2019. Measuring the effects of non-identical data distribution for federated visual classification. arXiv preprint arXiv:1909.06335.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al. 2021. Advances and open problems in federated learning. Foundations and trends® in machine learning, 14(1–2):1–210.

Jakub Konecnˇ y, H Brendan McMahan, Felix X Yu, Pe-\` ter Richtárik, Ananda Theertha Suresh, and Dave Bacon. 2016. Federated learning: Strategies for improving communication efficiency. arXiv preprint arXiv:1610.05492.

Kevin Kuo, Arian Raje, Kousik Rajesh, and Virginia Smith. 2024. Federated lora with sparse communication. arXiv preprint arXiv:2406.05233.

Ken Lang. 1995. Newsweeder: Learning to filter netnews. In Proceedings of the Twelfth International Conference on Machine Learning, pages 331–339.

Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045–3059, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Chunyuan Li, Heerad Farkhoor, Rosanne Liu, and Jason Yosinski. 2018. Measuring the intrinsic dimension of objective landscapes. arXiv preprint arXiv:1804.08838.

Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin A Raffel. 2022. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. Advances in Neural Information Processing Systems, 35:1950–1965.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Zequan Liu, Jiawen Lyn, Wei Zhu, Xing Tian, and Yvette Graham. 2024. ALoRA: Allocating low-rank adaptation for fine-tuning large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 622–641, Mexico City, Mexico. Association for Computational Linguistics.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In International Conference on Learning Representations.

Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, and Benjamin Bossan. 2022. Peft: State-of-the-art parameter-efficient finetuning methods.

Yulong Mao, Kaiyu Huang, Changhao Guan, Ganglin Bao, Fengran Mo, and Jinan Xu. 2024. DoRA: Enhancing parameter-efficient fine-tuning with dynamic rank distribution. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11662– 11675, Bangkok, Thailand. Association for Computational Linguistics.

Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. 2017. Communication-Efficient Learning of Deep Networks from Decentralized Data. In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, volume 54 of Proceedings of Machine Learning Research, pages 1273–1282. PMLR.

OpenAI. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Minseok Ryu, Youngdae Kim, Kibaek Kim, and Ravi K Madduri. 2022. Appfl: open-source software framework for privacy-preserving federated learning. In 2022 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW), pages 1074–1083. IEEE.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.

Youbang Sun, Zitao Li, Yaliang Li, and Bolin Ding. 2024. Improving loRA in privacy-preserving federated learning. In The Twelfth International Conference on Learning Representations.

Ananda Theertha Suresh, X Yu Felix, Sanjiv Kumar, and H Brendan McMahan. 2017. Distributed mean estimation with limited communication. In International

conference on machine learning, pages 3329–3337. PMLR.

Ziyao Wang, Zheyu Shen, Yexiao He, Guoheng Sun, Hongyi Wang, Lingjuan Lyu, and Ang Li. 2024. Flora: Federated fine-tuning large language models with heterogeneous low-rank adaptations. arXiv preprint arXiv:2409.05976.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. 2023. Adaptive budget allocation for parameter-efficient fine-tuning. In The Eleventh International Conference on Learning Representations.

<table><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=2>Dir(0.01)</td><td rowspan=1 colspan=2>Dir(0.1)</td><td rowspan=1 colspan=2>Dir(0.5)</td></tr><tr><td rowspan=1 colspan=1>Train</td><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>Train</td><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>Train</td><td rowspan=1 colspan=1>Test</td></tr><tr><td rowspan=1 colspan=1>max $\lceil \mathcal { D } _ { k } \} \rvert _ { k \in [ K ] }$ </td><td rowspan=1 colspan=1>1317</td><td rowspan=1 colspan=1>877</td><td rowspan=1 colspan=1>911</td><td rowspan=1 colspan=1>606</td><td rowspan=1 colspan=1>576</td><td rowspan=1 colspan=1>383</td></tr><tr><td rowspan=1 colspan=1>min $\left| \left\{ \mathcal { D } _ { k } \right\} \right| _ { k \in [ K ] }$ </td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>58</td><td rowspan=1 colspan=1>37</td><td rowspan=1 colspan=1>151</td><td rowspan=1 colspan=1>100</td></tr><tr><td rowspan=1 colspan=1>max $\textstyle { \overline { { \{ { \mathcal { C } } _ { k } \} | _ { k \in [ K ] } } } }$ </td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>14</td></tr><tr><td rowspan=1 colspan=1>min $\textstyle \prod C _ { k } \} | _ { k \in [ K ] }$ </td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=1>Number of classes</td><td rowspan=1 colspan=6>20</td></tr><tr><td rowspan=1 colspan=1>Number of clients</td><td rowspan=1 colspan=6>30</td></tr></table>

Table 5: Statistics of 20 Newsgroups datasets.
<table><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=2>Dir(0.01)</td><td rowspan=1 colspan=2> $\overline { { D i r ( 0 . 1 ) } }$ </td><td rowspan=1 colspan=2>Dir(0.5)</td></tr><tr><td rowspan=1 colspan=1>Train</td><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>Train</td><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>Train</td><td rowspan=1 colspan=1>Test</td></tr><tr><td rowspan=1 colspan=1>max $\| \langle \mathcal { D } _ { k } \rangle | _ { k \in [ K ] }$ </td><td rowspan=1 colspan=1>639</td><td rowspan=1 colspan=1>212</td><td rowspan=1 colspan=1>672</td><td rowspan=1 colspan=1>185</td><td rowspan=1 colspan=1>473</td><td rowspan=1 colspan=1>133</td></tr><tr><td rowspan=1 colspan=1>min $\overline { { \{ \mathcal { D } _ { k } \} | _ { k \in [ K ] } } }$ </td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>139</td><td rowspan=1 colspan=1>43</td><td rowspan=1 colspan=1>248</td><td rowspan=1 colspan=1>75</td></tr><tr><td rowspan=1 colspan=1>max $\overline { { \{ \mathcal { C } _ { k } \} | _ { k \in [ K ] } } }$ </td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>34</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>65</td><td rowspan=1 colspan=1>52</td></tr><tr><td rowspan=1 colspan=1>min |{Ck}|k∈[K]</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>37</td><td rowspan=1 colspan=1>31</td></tr><tr><td rowspan=1 colspan=1>Number of intents</td><td rowspan=1 colspan=6>77</td></tr><tr><td rowspan=1 colspan=1>Number of clients</td><td rowspan=1 colspan=6>30</td></tr></table>

Table 6: Statistics of BANKING77 dataset.

## A Dataset Statistics

BANKING77 (Casanueva et al., 2020) is an intent classification dataset with 77 fine-grained intents related to the banking domain, comprising 10,003 training samples and 3,080 test samples. 20 Newsgroups (Lang, 1995) is a widely used text classification dataset with 20 classes, each representing a unique topic. It contains 11,314 training samples and 7,532 test samples.

We provide the statistics of two datasets in Table 5 and Table 6, respectively. $\mathcal { D } _ { k }$ and $| \mathcal { C } _ { k } |$ denotes the local dataset of k and the number of unique classes in $\mathcal { D } _ { k } .$ , respectively. Figure 7 shows the distribution of a local dataset for varying α simulating the Dirichlet distribution.

## B Reproducibility

Hyperparameters When training, we use AdamW (Loshchilov and Hutter, 2019) optimizer with a learning rate of $\eta = 0 . 0 0 0 5$ . For LoR $\mathsf { A } { - } \mathsf { A } ^ { 2 }$ since B and A of each LoRA module are optimized separately, we use different learning rates for them. Specifically, $\eta _ { A } = \eta$ is used for A and $\eta _ { B } = 5 \cdot \eta _ { A }$ is used for B, which is inspired by LoRA+ (Hayou et al., 2024). For HetLoRA, $\gamma \ = \ 0 . 9 9$ is used for the decaying factor as suggested by Cho et al. (2023). When evaluating, we merge the LoRA adapter ∆W with the pre-trained model $W _ { 0 }$ using a scaling factor, so that $\begin{array} { r } { W _ { f t } = W _ { 0 } + \frac { 1 6 } { r } \Delta W } \end{array}$

Implementation details We simulate our FL setup using Flower (Beutel et al., 2020), and utilize HuggingFace PEFT (Mangrulkar et al., 2022)

![](images/329301514e0e1d048fc4fdd7c327c826d1258bee0dccb776786516e6cf74eec2.jpg)  
(a) α = 0.01

![](images/3c6377fd35005e29f01e744b84435b251d09b60762b0055e322fe0062dbc6cf6.jpg)  
(b) α = 0.1

![](images/b269ec8c2c94dbf10801ec944ec57600c347c0dd0a4c25e8b7d1987b683b03cd.jpg)

![](images/12281c132cda66b8973376416db57ba8a8d4dfefe7941e99c2c1afe335caa418.jpg)  
(c) $\alpha = 0 . 5$  
(d) α = 1

Figure 7: Local distribution of client 0 for different Dir(α) on 20 Newsgroup dataset experiments.
<table><tr><td></td><td>Rank | FL+LoRA</td><td> $\mathrm { F F A - L o R A }$ </td><td>FlexLoRA|</td><td>Ours</td></tr><tr><td>8</td><td> $5 3 . 8 0 _ { \pm 1 . 4 4 }$ </td><td> $5 2 . 6 0 { \scriptstyle \pm 0 . 9 6 }$ </td><td> ${ \bf 6 0 . 3 6 _ { \pm 1 . 1 5 } }$ </td><td> $5 8 . 7 4 { \scriptstyle \pm 0 . 9 5 }$ </td></tr><tr><td>4</td><td> $5 5 . 0 3 _ { \pm 0 . 4 3 }$ </td><td> $5 0 . 5 7 { \scriptstyle \pm 1 . 5 8 }$ </td><td> ${ \bf 5 9 . 1 2 _ { \pm 0 . 9 8 } }$ </td><td> $5 8 . 6 2 { \scriptstyle \pm 1 . 5 1 }$ </td></tr><tr><td>2</td><td> $5 0 . 4 0 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $4 8 . 3 6 _ { \pm 0 . 8 6 }$ </td><td> $5 5 . 4 6 _ { \pm 0 . 9 9 }$ </td><td> ${ \bf 5 9 . 6 3 _ { \pm 0 . 5 9 } }$ </td></tr><tr><td>1</td><td> $5 1 . 2 4 { \scriptstyle \pm 3 . 1 2 }$ </td><td> $4 6 . 9 2 _ { \pm 1 . 3 0 }$ </td><td> $5 1 . 0 5 _ { \pm 0 . 6 9 }$ </td><td> ${ \bf 5 9 . 1 1 _ { \pm 0 . 8 8 } }$ </td></tr></table>

Table 7: Experiments on pathologic settings.

library to train base models with LoRA. The base models are loaded using HuggingFace Transformers (Wolf et al., 2020) library. All experiments are conducted three times to ensure reproducibility, and the code will be released soon to promote transparency and support future research.

## C Additional Experiments

Pathologic setting Table 7 provides experiments on pathologic setting, which is also used to generate Figure 5 in Section 5.3, to show the efficacy of adaptive rank selection. In this setting, we have K = 20 clients. And client (2k  1) and client (2k) exclusively possess half of class (2k  1) and (2k) of 20 Newsgroups datasets, respectively, for $k = 1 , 2 , \cdots , 1 0$

Convergence speed analysis Figure 8 shows the convergence curves of our algorithm and baselines. The figure demonstrates that our algorithm shows similar convergence speed compared to baseline methods in various levels of heterogeneity.

Resource heterogeneity In this experiment, we analyze the efficacy of our algorithm under varying resource constraints across clients. Specifically, we assume that each client has a different communication cost budget (Chen et al., 2023) and has different maximum LoRA rank for its adapter. Following Bai et al. (2024), we simulate three types of resource heterogeneity, as illustrated in Figure 9.

![](images/2da1a8a35ee77bd955c432abe79ba95048bd7e6d8789ee3ff513bba071453763.jpg)  
(a) Dir(0.5)

![](images/3c1a472d28197567060de907b22352c7fc1100563741ab7d394e9cda75f439e4.jpg)  
(b) Dir(0.1)

![](images/46c9cbee2b316be4a9c1f46b90cf61c045ca0e1836d4338229f64b4e32d67aef.jpg)  
(c) Dir(0.01)

Figure 8: Convergence curve of baseline methods in various levels of heterogeneity. Experimented on BANKING77 dataset, with local ranks all set to 2.  
![](images/04aea2e3bcdfe24f4a18a35c8efe034b3218fde4d0f23557bb2c52f59c62a548.jpg)  
Figure 9: Three types of simulated client rank distributions used to evaluate performance under resource heterogeneity settings.

<table><tr><td rowspan="2">Distribution</td><td rowspan="2">Method</td><td colspan="2">BANKING77 Dataset</td><td rowspan="2">Communicated Parameters</td></tr><tr><td>Dir(0.1)</td><td> $D i r ( 0 . 0 1 )$ </td></tr><tr><td rowspan="3">Uniform</td><td>HetLoRA</td><td> $\underline { { 8 6 . 9 1 } } \pm 0 . 4 3$ </td><td> $6 8 . 5 3 \substack { \pm 2 . 1 4 }$ </td><td>3.09B</td></tr><tr><td>FlexLoRA</td><td> $7 3 . 0 1 _ { \pm 0 . 6 9 }$ </td><td> $4 5 . 4 1 _ { \pm 1 . 6 0 }$ </td><td>3.09B</td></tr><tr><td>Ours</td><td> $\mathbf { 9 2 . 0 2 _ { \pm 0 . 1 6 } }$ </td><td> ${ \bf 7 0 . 6 7 { \scriptstyle \pm 0 . 7 6 } }$ </td><td>1.97B</td></tr><tr><td rowspan="3">Heavy Tail</td><td>HetLoRA</td><td> $8 5 . 8 2 \substack { \pm 0 . 5 4 }$ </td><td> $6 9 . 5 7 { \scriptstyle \pm 1 . 1 3 }$ </td><td>1.06B</td></tr><tr><td>FlexLoRA</td><td> $8 2 . 6 9 _ { \pm 0 . 8 6 }$ </td><td> $5 2 . 4 6 _ { \pm 1 . 7 2 }$ </td><td>1.06B</td></tr><tr><td>Ours</td><td> $\mathbf { 9 1 . 7 2 _ { \pm 0 . 0 7 } }$ </td><td> ${ \bf 6 9 . 9 5 } _ { \pm 2 . 2 3 }$ </td><td>0.942B</td></tr><tr><td rowspan="3">Normal</td><td>HetLoRA</td><td> $8 4 . 5 7 { \scriptstyle \pm 0 . 5 5 }$ </td><td> ${ \bf 7 0 . 3 4 _ { \pm 0 . 1 5 } }$ </td><td>1.34B</td></tr><tr><td>FlexLoRA</td><td> $7 7 . 0 8 _ { \pm 0 . 6 8 }$ </td><td> $5 3 . 3 7 { \scriptstyle \pm 3 . 4 9 }$ </td><td>1.34B</td></tr><tr><td>Ours</td><td> $\mathbf { 9 2 . 0 8 _ { \pm 0 . 1 8 } }$ </td><td> $\underline { { 6 9 . 0 4 } } \underline { { \pm 0 . 6 4 } }$ </td><td>0.932B</td></tr></table>

Table 8: Experimental results under resource heterogeneity settings.

In Table 8, we compare our method against Het-LoRA and FlexLoRA, two previous LoRA methods that can handle resource heterogeneity in FL. The experimental results demonstrates that our algorithm shows slightly better or similar performance compared to HetLoRA with less number of communicated parameters.

Efficacy of importance criterion As mentioned in Section 4.2, other criteria such as magnitudebased or importance-based scoring functions can be used for selecting ranks. Table 9 shows that our criterion outperforms others, with less communication than the magnitude-based criterion.

<table><tr><td></td><td>BANKING77 Dataset  $D i r ( 0 . 1 )$ </td><td> $D i r ( 0 . 0 1 )$ </td><td>Communicated Parameters</td></tr><tr><td>Importance</td><td> $9 1 . 2 9 _ { \pm 0 . 7 6 }$ </td><td> $6 6 . 9 2 { \scriptstyle \pm 1 . 5 8 }$ </td><td>0.215B</td></tr><tr><td>Magnitude</td><td> $\underline { { 9 1 . 7 1 } } \underline { { + 0 . 2 3 } }$ </td><td> $6 8 . 0 0 { \scriptstyle \pm 0 . 5 7 }$ </td><td>0.651B</td></tr><tr><td>Ours</td><td> $\mathbf { 9 } 2 . \mathbf { 0 } 2 _ { \pm 0 . 3 6 }$ </td><td> ${ \bf 6 9 . 4 0 { \scriptstyle \pm 0 . 4 8 } }$ </td><td>0.507B</td></tr></table>

Table 9: Ablation study on scoring functions.

![](images/92a233f3afcdcbe3e350ff173c14542e82120227a6f32b2e02ee12cf0bf5733d.jpg)  
Figure 10: Average Gradient Similarity on various level of heterogeneity. Experimented on 20 Newsgroups dataset and the ranks were all set to 2.

Client drift experiment To thoroughly analyze the impact of data heterogeneity within constrained parameter spaces, we conducted additional experiments that illustrate the local client drift observed in baseline methods operating under these limitations. We quantified the degree of client drift by calculating the "Average Gradient Similarity," defined as follows:

AverageGradientSimilarity =

$$
\frac { 1 } { n ^ { 2 } } \sum _ { i } ^ { n } \sum _ { i } ^ { n } \frac { ( \Delta W _ { i } ^ { t } - \Delta W _ { i } ^ { t - 1 } ) \cdot ( \Delta W _ { j } ^ { t } - \Delta W _ { j } ^ { t - 1 } ) } { | | \Delta W _ { i } ^ { t } - \Delta W _ { i } ^ { t - 1 } | | \cdot | | \Delta W _ { j } ^ { t } - \Delta W _ { j } ^ { t - 1 } | | }\tag{7}
$$

The experimental results presented in Figure 10 indicate a rapid decline in average gradient similarity as the level of heterogeneity increases. In contrast, our method demonstrates greater robustness, exhibiting lower client drift even in rounds where only the LoRA module A is updated. These findings are consistent with the results shown in Figure 2 and Table 1, which illustrate that FFA-LoRA experiences the most significant performance decline between the directional settings of 0.1 and 0.01, while our algorithm maintains its effectiveness in heterogeneous environments.

## D Theoretical Proofs

Here’s brief proof for the proposition made in section 4.3:

Proof) First, since FFA-LoRA freezes all the $A _ { i } ^ { \prime } \mathbf { s }$ permanently, $\Omega _ { \mathrm { F F A - L o R A } } ~ = ~ \{ B _ { i } \} _ { i = 1 } ^ { N } .$ Next, since FL + LoRA and FlexLoRA update $B _ { i } { ' } \mathrm { s }$ and $A _ { i } ^ { \prime } \mathbf { s }$ simultaneously, Ω<sub>FL + LoRA</sub> = $\{ ( B _ { i } , A _ { i } ) \} _ { i = 1 } ^ { N } = \Omega _ { \mathrm { F l e x L o R A } }$ . Finally, $\Omega _ { \mathrm { L o R A - A ^ { 2 } } } =$ $\left\{ \left( \bar { B } _ { i } , \bar { A } _ { i } \right) \right\} _ { i = 1 } ^ { N }$ , where its subspace $\{ B _ { i } \} _ { i = 1 } ^ { N }$ or $\{ A _ { i } \} _ { i = 1 } ^ { N }$ is optimized according to the Alternating freeze and Adaptive rank selection algorithm. Therefore, noting that $r \quad \leq \quad r _ { G }$ , we have $\Omega _ { \mathrm { F F A - L o R A } } \subsetneq \Omega _ { \mathrm { F L + L o R A } } = \Omega _ { \mathrm { F l e x L o R A } } \subset$ $\Omega _ { \mathrm { L o R A - A ^ { 2 } } } . \square$