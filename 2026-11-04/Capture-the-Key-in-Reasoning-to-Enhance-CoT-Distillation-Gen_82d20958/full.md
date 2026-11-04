# Capture the Key in Reasoning to Enhance CoT Distillation Generalization

Chengwei Dai<sup>1,2</sup>, Kun Li<sup>1</sup>∗, Wei Zhou<sup>1</sup>, Songlin Hu<sup>1</sup> <sup>1</sup>Institute of Information Engineering, Chinese Academy of Sciences <sup>2</sup>School of Cyber Security, University of Chinese Academy of Sciences {daichengwei, likun2, zhouwei, husonglin}@iie.ac.cn

## Abstract

As Large Language Models (LLMs) scale up and gain powerful Chain-of-Thoughts (CoTs) reasoning abilities, practical resource constraints drive efforts to distill these capabilities into more compact Smaller Language Models (SLMs). We find that CoTs consist mainly of simple reasoning forms, with a small proportion ( 4.7%) of key reasoning steps that truly impact conclusions. However, previous distillation methods typically involve supervised fine-tuning student SLMs only on correct CoTs data produced by teacher LLMs, resulting in students struggling to learn the key, instead imitating the teacher’s reasoning forms and making errors or omissions in reasoning. To address these issues, drawing an analogy to human learning, where analyzing mistakes according to correct solutions often reveals the crucial steps leading to successes or failures, we propose mistakE-Driven key reasonIng step distillaTion (EDIT), a novel method that further aids SLMs learning key reasoning steps rather than mere simple fine-tuning. Firstly, to expose the crucial steps in CoTs, we carefully design specific prompts to generate dual CoTs data with similar reasoning paths but divergent conclusions. Then, we apply the minimum edit distance algorithm on the dual CoTs data to locate these key steps and optimize the likelihood on these tokens. Extensive experiments and analysis validate the effectiveness of EDIT across both in-domain(IND) and out-ofdomain(OOD) benchmark reasoning datasets<sup>1</sup>.

## 1 Introduction

With the rapid growth in model size and pretraining data, LLMs have demonstrated impressive CoT reasoning performance in natural language processing (NLP) (Brown et al., 2020; Hoffmann et al., 2022; Chowdhery et al., 2023; OpenAI,

2023b, 2024). However, due to the giant model architecture and massive parameters (e.g. GPT-3 (Brown et al., 2020) with 175 billion parameters), the deployment of LLMs in resource-constrained environments becomes challenging.

To address this, researchers (Xu et al., 2023; Jiang et al., 2023b) have explored distilling knowledge from LLMs into smaller language models (SLMs) via instruction-tuning, as seen in LMs like Alpaca (Taori et al., 2023) and Vicuna (Chiang et al., 2023). Despite progress, these distilled models often struggle with complex causal reasoning. To enhance this capability, some studies (Magister et al., 2023; Ho et al., 2023; Fu et al., 2023; Chen et al., 2024a; Zhou and Ai, 2024) explore distilling the CoT reasoning ability from LLMs of over 100B parameters (Wei et al., 2022a,b) by fine-tuning on CoTs data annotated by teacher LLMs, known as standard CoTs distillation. Besides, other studies (Hsieh et al., 2023; Li et al., 2022; Liu et al., 2023; Chen et al., 2024b) propose distilling CoTs within a multi-task learning framework by incorporating additional objectives. However, CoTs usually consist mainly of simple reasoning forms, with a small proportion ( 4.7%<sup>2</sup>) of key reasoning steps that are pivotal moments in reasoning that significantly influence subsequent thought processes and conclusions. The essence of the above methods is the simple Supervised Fine-Tuning (SFT) paradigm, where the student model is trained solely on the teacher’s correct reasoning data. This paradigm may result in students struggling to learn the key reasoning steps, instead imitating the teacher’s reasoning forms and making errors or omissions on these steps, as illustrated in Figure 1.

Drawing an analogy to human learning, where analyzing mistakes according to correct solutions often reveals the key reasoning steps leading to successes or failures, we propose a mistakE-Driven key reasonIng step distillaTion (EDIT). This approach focuses on dual CoTs data, encompassing both positive and negative examples of teachers reasoning. By examining dual CoTs, students can identify and learn from the crucial reasoning steps, thereby improving their CoTs. Specifically, we first retain all CoTs data annotated by the teacher, irrespective of correctness. Subsequently, based on the powerful in-context learning ability of LLMs, we design two comprehensive prompts to instruct teachers to produce dual CoTs that share similar intermediate reasoning steps but lead to divergent conclusions. Finally, we utilize the minimum edit distance algorithm to locate key reasoning steps in dual CoTs, as shown in Figure 3, and then utilize a fine-grained token level loss function to optimize the likelihood on these tokens.

![](images/3563e765f27c15221d01e1d4dc4b29ca58effc061e0d5d446fceaef4712b074d.jpg)  
Figure 1: Examples of CoTs generated by teacher LLMs and student SLMs on our test dataset. Simply SFT leads to an "unthinking" student who imitates the teacher’s reasoning style but makes errors and omissions on key steps, where the imitated contents are highlighted in red, and the key steps are marked with boxes .

Extensive experiments show that SLMs distilled by EDIT exhibit higher performance and generalization than the baselines on both IND and OOD benchmark reasoning datasets. Further analyses indicate that EDIT can generate higher-quality CoTs with more correct key reasoning steps by auto evaluation and case studies. Notably, we also show EDIT can benefit more from logical mistake patterns than knowledge or mathematical calculation errors in dual CoTs, potentially paving the way for future research on the efficient use of mistakes.

Our contributions can be summarized as follows:

• We reveal a shortfall in the popular distillation methods, where the simple SFT paradigm may result in students mimicking the teacher’s reasoning forms but making errors or omissions on key reasoning steps, thus diminishing the

versatility of CoTs.

• We propose mistake-driven key reasoning step distillation, which allows students to learn key reasoning steps from our specifically designed dual CoTs data, further improving reasoning.

• Extensive experiments validate the effectiveness of our method across both IND and OOD datasets, showing that EDIT can improve the reasoning generalization of student models.

## 2 Related Works

CoT Reasoning. The emergent ability appears in LLMs across a wide range of NLP tasks (Chowdhery et al., 2023; Wei et al., 2022a). One such ability is CoT reasoning, which involves generating a series of intermediate reasoning steps. This ability has been further explored recently with the release of OpenAI’s o1 model (OpenAI, 2024). While CoT prompting techniques (Wei et al., 2022b) significantly enhance the problem-solving capabilities of models (Kojima et al., 2022; Wang et al., 2023b; Huang et al., 2023), it has little effect on smaller models (Wei et al., 2022a). Chung et al. (2022) suggest that CoT reasoning can be induced in SLMs via instruction tuning on CoTs data. Our work show that the CoT reasoning capabilities of SLMs can be further improved by learning from key reasoning steps in dual CoTs data.

Knowledge Distillation from LLMs. There has been a lot of work dedicated to distilling knowledge (Hinton et al., 2015) from powerful proprietary LLMs, e.g. ChatGPT (OpenAI, 2023a) in a blackbox setting. However, most of these works primarily focus on the general ability distillation by instruction tuning on large and diverse datasets (Peng et al., 2023; Jiang et al., 2023b; Li et al., 2024). In contrast, we aim to distill the CoT reasoning capabilities from LLMs same as the standard CoTs distillation (Magister et al., 2023; Ho et al., 2023). Besides, some studies (Li et al., 2022; Hsieh et al., 2023; Liu et al., 2023) employ LLM’s rationale or self-evaluation output to enhance SLM’s reasoning in a multi-task learning framework. Fu et al. (2023) fine-tune SLMs on four types of reasoning data to ensure out-of-distribution generalization. Wang et al. (2023c) distill SLMs by learning from self-reflection and feedback from LLMs in an interactive multi-round paradigm. Chen et al. (2023) uses the teacher model to generate multiple correct CoTs for each question and maintains consistency by minimizing the bidirectional KL divergence between the answer distributions of different CoTs. Chen et al. (2024a) maxmize the mutual relationship of the two tasks from the Information Bottleneck perspective. Ranaldi and Freitas (2024) use in-family and out-family teachers to generate more CoTs for SFT. Different from the above works, we assist CoTs distillation with teachers’ mistakes to alleviate the style imitation of teachers’ reasoning.

Learning from Mistakes. Recent studies use mistake data to enhance the performance of LMs. Shinn et al. (2023) propose Reflexion that allows the LLM agent to self-reflect from its mistakes. Wang and Li (2023) introduce a study assistant that collects and retrieves LLMs’ training mistakes to guide future inferences. Li et al. (2023) propose CoK that corrects potential mistakes in the rationale by retrieving knowledge to avoid error propagation. However, both of the above methods require the models to be large enough to have basic CoT reasoning or instruction-following capabilities, which is almost impossible to occur in vanilla SLMs. Wang et al. (2023a) propose finetuning on counterfactual data to ensure the faithful reasoning of the student model. An et al. (2023) propose LEMA that fine-tunes language models on corrected mistake data, where the mistakes are collected from various LLMs e.g. LLaMA2-70B (Touvron et al., 2023), WizardLM-70B (Xu et al., 2023), and corrected by GPT-4 (OpenAI, 2023b). Additionally, Sun et al. (2024) propose Retrieved In-Context Principles, which retrieve mistakes to provide customized guidance and improve model performance during inference. In contrast, we collect the teachers’ mistakes to create a dual CoTs dataset for further key reasoning steps learning.

## 3 Methodology

We present the overview of our proposed method in Figure 2. Concretely, (1) unlike prior works (Magister et al., 2023; Hsieh et al., 2023; Chen et al., 2024b) that only focus on correct CoTs annotated by teacher LLMs, we first retain all CoTs reasoning data, regardless of its correctness. (2) Then based on the previously retained correct and wrong CoTs, we construct dual CoTs datasets consisting of positive-negative CoT pairs that follow similar intermediate reasoning steps but lead to divergent conclusions. Specifically, we design two comprehensive contextual prompts to instruct teacher LLMs to rectify the originally wrong CoTs and corrupt originally correct CoTs. (3) Finally, we distill the student SLMs by training on the teacher’s correct CoTs reasoning data and further Key Reasoning Steps Learning on the dual CoTs datasets.

## 3.1 CoTs Annotated by LLMs

We utilize CoT Prompting (Wei et al., 2022b) to extract CoTs for a raw dataset $\mathcal { D } = \{ ( q , a ) \}$ from LLMs, where $q$ is the question and a is the golden answer. Specifically, we first create a CoTs Extraction Prompt CEP that contains several humancurated question-CoTs pair examples and the task description, which can be found in Appendix C.1. For each $q \in \mathcal { D }$ , we extract CoTs as:

$$
C o T \sim L L M \left( \mathrm { C E P } \oplus q \right)\tag{1}
$$

where means concatenation. Then, following Zelikman et al. (2022), we classify the CoTs annotated dataset into two datasets according to the final answer’s correctness. One is the CoTs-original correct dataset ${ \mathcal D } ^ { + } = \{ ( q , C o T ^ { + } ) ~ | ~ \forall ( q , \bar { a } ) ~ \in ~$ $\mathcal { D } , \hat { a } ~ = ~ a ~ \& ~ \hat { a } \in C o T ^ { + } \}$ and the other is CoTs-original wrong dataset $\mathcal { D } ^ { - } = \{ ( q , C o T ^ { - } )$ \_ $\forall ( q , a ) \in \mathcal { D } , \hat { a } \neq a \ \& \ \hat { a } \in C o T ^ { - } \}$

## 3.2 Dual CoTs Generation

We define dual CoTs data as contrasting CoTs that follow similar reasoning steps but reach divergent conclusions compared to the original. To provide a deeper understanding, we also present several examples of dual CoTs in Appendix B. In the following, we will introduce how to generate dual CoTs datasets including $\mathcal { D } ^ { + - }$ contrasting to $\mathcal { D } ^ { + }$ and $\mathcal { D } ^ { - + }$ contrasting to $\mathcal { D }$ −.

![](images/f7961362e57324f2b64e74bea4ccd2f23e6c2fb3936dfaa6c35b83e62e89728e.jpg)  
Figure 2: Overview of our method EDIT. (1) We first extract all CoTs data annotated by teacher LLMs (2) and ask teacher LLMs to generate dual CoTs data using our designed two comprehensive prompts. (3) Then we fine-tune student SLMs on both original correct and rectified-after CoTs data. Finally, we apply key reasoning step learning on the pre-tuned student SLMs by identifying the minor difference between the dual CoTs.

Rectify Wrong CoTs. To generate correct CoTs contrasting with the originally wrong CoTs, inspired by Rationalization (Zelikman et al., 2022), we design an Answer Hint Prompt AHP that shares the same examples with CEP but with different organizational structures. The template of AHP can be found in Appendix C.2. Each example in the context and the final provided question will be inserted with a hint that tells LLMs the answer first before CoTs. Thus, due to the same incontext examples and hint answers, teacher LLM can rectify its original wrong CoTs data with similar reasoning steps but correct answers. For each $q \in \mathcal { D } ^ { - }$ , we rectify CoTs as follows and then have the Rectified CoTs dataset $\mathcal { D } ^ { - + } = \{ ( q , C o T ^ { - + } ) \}$

$$
C o T ^ { - + } \sim L L M \left( \bar { \mathrm { A H P } } \oplus q \oplus a \right)\tag{2}
$$

Corrupt Correct CoTs. To generate incorrect CoTs contrasting with the originally correct CoTs, a straightforward approach is to use AHP with incorrect hint answers to prompt LLMs to produce wrong CoTs. However, in practice, we find that LLMs rarely follow the incorrect hints and still generate correct CoTs. This may be due to the simplicity of the questions, which fall within the LLMs’ knowledge range. Additionally, LLMs, having undergone Reinforcement Learning from Human Feedback (RLHF) (Ouyang et al., 2022), may resist providing unhelpful answers. Therefore, we design a Contrastive CoTs Prompt CCP to entice LLMs to generate incorrect CoTs, leveraging their strong in-context learning capabilities. The prompt template can be found in Appendix C.3. Specifically, to ensure that the synthesis of incorrect CoTs with special data properties, we randomly sample negative examples from $\mathcal { D } ^ { - }$ and positive examples from $\mathcal { D } ^ { - + }$ , pair them, and place them into the CCP as curated joint in-context examples. For each $q \in \mathcal { D } ^ { + }$ , we corrupt CoTs as follows and then have the corrupted CoTs dataset $\mathcal { D } ^ { + - } = \{ ( q , C o T ^ { + - } ) \}$ :

$$
C o T ^ { + - } \sim L L M \left( \mathrm { C C P } \oplus q \oplus C o T ^ { + } \right)\tag{3}
$$

![](images/d428af59d8e9dfc6cb7015632f99e2db062db12864a24b23c0e72876bc058ca7.jpg)  
Figure 3: Examples of locating key reasoning steps in dual CoTs, where the correct CoT and the wrong CoT are dual to each other. The identified key steps in correct reasoning and wrong reasoning are respectively marked in green and red.

## 3.3 Training Student with CoTs

Surpervised Fine-tuning on Correct CoTs. After preparing the dual $\mathrm { C o T s } ^ { 3 }$ , we first fine-tune student models on the teachers’ original correct CoTs dataset $\mathcal { D } ^ { + }$ and rectified CoTs dataset $\mathcal { D } ^ { - + }$ The training objective is as follows:

$$
\pi _ { s f t } = \arg \operatorname* { m a x } _ { \pi } \mathbb { E } _ { q , C o T \sim \mathcal { D } _ { m e r g e } ^ { + } } \log \pi ( C o T \mid q )\tag{4}
$$

where the merged correct CoTs dataset $\mathcal { D } _ { m e r g e } ^ { + } =$ $\mathcal { D } ^ { + } \cup \mathcal { D } ^ { - + }$ , and $\pi _ { s f t }$ denotes the student with the base inference ability after the initial fine-tuning.

Key Reasoning Steps Learning. Inspired by (Guo et al., 2024) who leverage fine-grained quality signals to align human preference, we propose a key reasoning steps learning (KRSL) method to further encourage students to comprehend the reasons behind both correct and wrong CoTs.

Step1. We pair the teacher’s original correct CoTs dataset $\mathcal { D } ^ { + }$ with its corrupted CoTs dataset $\mathcal { D } ^ { + - }$ , creating an originally correct dual CoTs dataset ${ \mathcal { D } } _ { d u a l } ^ { + } ~ = ~ \{ ( q , C o T ^ { + } , C o T ^ { + - } ) \}$ , where $C o T ^ { + }$ and $C o T ^ { + - }$ are dual to each other; similarly, the teacher’s inherently wrong dual CoTs dataset $\mathcal { D } _ { d u a l } ^ { - } = \{ ( q , C o T ^ { - + } , C o T ^ { - } ) \}$ . By merging them, we obtain the ultimate dual CoTs datasets $\mathcal { D } _ { d u a l } = \mathcal { D } _ { d u a l } ^ { + } \cup \mathcal { D } _ { d u a l } ^ { - }$ , which is prepared for the subsequent learning of key reasoning steps.

Step2. Then we employ the minimum edit distance to identify the key steps in both correct reasoning and wrong reasoning, as shown in Figure 3. In this way, students can identify less frequent text segments that are inserted or replaced in wrong CoTs compared to correct CoTs, and vice versa. These text segments are considered key reasoning steps. After that, we assign token-level weights to facilitate fine-grained learning for correct CoTs and wrong CoTs in $D _ { d u a l }$ respectively<sup>4</sup>:

$$
\begin{array} { r l } & { \omega _ { t } ^ { + } = \left\{ \begin{array} { l l } { \alpha , } & { \mathrm { i f } \ C o T _ { t } ^ { + } \mathrm { ~ i s ~ i n s e r t e d ~ o r ~ r e p l a c e d } } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} \right. , } \\ & { \omega _ { t } ^ { - } = \left\{ \begin{array} { l l } { \beta , } & { \mathrm { i f } \ C o T _ { t } ^ { - } \mathrm { ~ i s ~ d e l e t e d ~ o r ~ r e p l a c e d } } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} \right. } \end{array}\tag{5}
$$

where $\alpha \ge 0 , \beta \ge 0$ and $\omega _ { t } ^ { + }$ represents the weight of t-th token in the correct CoTs (semantically same with $\omega _ { t } ^ { - } )$ . We set the weights to zero to ignore the impact of identical tokens in the dual CoTs.

Step3. Finally, to ensure that the student makes correct decisions on key steps in correct reasoning, we optimize the student model on these tokens with weighted negative log-likelihood. Conversely, to prevent the student from making key steps present in wrong reasoning, we optimize the student model on these steps with weighted positive log-likelihood. The sum of both is taken as the final loss. The optimization objective is as follows:

$$
\begin{array} { r l r } {  { \operatorname* { m a x } _ { \pi _ { s f t } } \mathbb { E } _ { q , C o T ^ { + } , C o T ^ { - } \sim \mathcal { D } _ { d u a l } } } } \\ & { } & { \mathcal { L } ( \pi _ { s f t } , q , C o T ^ { + } , \omega ^ { + } ) - \mathcal { L } ( \pi _ { s f t } , q , C o T ^ { - } , \omega ^ { - } ) } \\ & { } & { ( 6 ) } \end{array}
$$

$$
\begin{array} { l l } { \mathrm { w h e r e } } & { { \mathcal { L } } \left( \pi , q , C o T , \omega \right) = } \\ & { - \displaystyle \sum _ { C o T _ { t } \in C o T } \omega _ { t } \log \pi ( C o T _ { t } \mid q , C o T _ { < t } ) } \end{array}\tag{7}
$$

## 4 Experiments

## 4.1 Experimental Setup

In-domain (IND) Dataset: BIG-Bench Hard (BBH) (Suzgun et al., 2023) consists of 27 challenging tasks that span arithmetic, symbolic reasoning, etc. This collection is mainly composed of multiple-choice questions, along with a minority of open-ended questions. To underscore the superiority of our method, we divide the BBH dataset for each subtask into a training set (BBH-train) for distillation and a test set (BBH-test) for in-domain evaluation, following a 4:1 ratio.

Out-of-domain (OOD) Dataset: (1) BIG-Bench Sub (BB-sub) is derived from the BIG-Bench (BB) (Guo et al., 2023), which includes 203 tasks covering linguistics, mathematics, common-sense reasoning, etc. To simplify our evaluation, we refine the selection of tasks from BB by identifying those associated with keywords such as "multiplechoice" and "reasoning." Additionally, we exclude any tasks that are part of the BBH dataset, narrowing our pool to 61 distinct subtasks. For each of these subtasks, we randomly sample up to 100 instances, culminating in the BB-sub dataset. (2) AGIEval (Zhong et al., 2023) is a benchmark that assesses LMs on reasoning capabilities using human exams across various fields, including English, Math, Law, and Logic. We focused on the English multiple-choice questions within this benchmark for evaluation. (3) AI2 Reasoning Challenge (ARC) (Clark et al., 2018) comprises ARC-Easy and ARC-Challenge from middle and high school science exams. ARC-E features simpler questions, while ARC-C includes more challenging ones. We use their test sets for evaluation. Detailed statistics for all mentioned benchmarks are provided in Appendix A.9.1. BigBench, AGIEval, and ARC are standard benchmarks for evaluating LLMs reasoning performance. Specifically, BigBench and AGIEval have been employed in related works (Fu et al., 2023; Jiang et al., 2023b), and ARC is frequently used in technical reports for LLaMA3 (AI@Meta, 2024) and GPT-4 (OpenAI, 2023b).

Models & Implementation Details. We employ the widely-used open-source language model,

LLaMA2-7B (Touvron et al., 2023), as our student SLM. For the teacher model, given its performance and cost-effectiveness, we employ OpenAI’s advanced black-box LLM, ChatGPT, specifically using the "gpt-3.5-turbo-0613" variant for extracting CoTs with the same manual prompt that is used in (Suzgun et al., 2023). We employ LoRA (Hu et al., 2022) for parameter-efficient fine-tuning of the student SLMs. We empirically set α in KRSL as 1.0 and $\beta$ as 0.025. We also conducted experiments on the impact of hyperparameters in the Appendix A.2. Our experiments leverage a mixedprecision training strategy, carried out on 4 A100 GPUs. We employ vLLM (Kwon et al., 2023) to enhance inference speed, using a greedy decoding method for text generation on a single A100 GPU. More training details and hyperparameter settings can be found in Appendix A.9.2.

Baselines. We compare EDIT with the following baselines: (1) Teacher & Vanilla Student under various settings, e.g., Zero-shot (+ CoT) or Fewshot (+ CoT). (2) Std-CoT (Magister et al., 2023), which is a standard CoTs distillation method that directly fine-tunes student SLMs on CoTs data. (3) MT-CoT (Li et al., 2022) is a multi-task CoTs distillation strategy that aims to optimize both the prediction of answers and the learning of CoTs concurrently. (4) SCOTT (Wang et al., 2023a) aims to bolster the reasoning consistency in the student SLMs by integrating counterfactual data into its training regimen. (5) SBS (Hsieh et al., 2023) propose to distill rationales and answers separately. (6) On this basis, SBS-MI (Chen et al., 2024b) add the mutual information learning objectives into distillation. We also compare different variants of EDIT by removing training stages and data components. (7) w/o RWC + KRSL on $D _ { d u a l } ^ { + }$ excludes RWC<sup>5</sup> in the first step and only uses $D _ { d u a l } ^ { + }$ in the second step. (8) w/o RWC + KRSL on $D _ { d u a l }$ excludes RWC in the first step and uses all dual datasets in the second step. (9) w/ RWC + w/o KRSL uses RWC in the first step and skips the second step.

## 4.2 Main Results

We compare EDIT with the baselines across both IND and OOD datasets in Table 1 and the results of more commonly used reasoning subtasks can be found in Appendix A.1. We illustrate the results by answering the following research questions.

<table><tr><td>Method</td><td>Distill?</td><td>BBH-test</td><td>BB-sub</td><td>AGIEval</td><td>ARC-E</td><td>ARC-C</td><td rowspan="2">AVG</td></tr><tr><td>In-domain?</td><td></td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td colspan="7">Teacher: ChatGPT (gpt-3.5-turbo)</td><td rowspan="2">61.9</td></tr><tr><td>Zero-shot-CoT</td><td>X</td><td>42.7</td><td>44.1</td><td>49.5</td><td>91.9</td><td>81.1</td></tr><tr><td>Few-shot-CoT</td><td>X</td><td>73.1</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="8">Student: LLaMA2-7B</td></tr><tr><td>Zero-shot</td><td>X</td><td>14.8</td><td>15.5</td><td>6.9</td><td>18.2</td><td>13.9</td><td>13.9</td></tr><tr><td>Few-shot</td><td>X</td><td>15.1</td><td>28.5</td><td>25.5</td><td>25.5</td><td>25.4</td><td>24.0</td></tr><tr><td>Zero-shot-CoT</td><td>X</td><td>10.6</td><td>7.7</td><td>7.1</td><td>18.4</td><td>14.8</td><td>11.7</td></tr><tr><td>Few-shot-CoT</td><td>X</td><td>16.3</td><td>25.3</td><td>9.9</td><td>17.2</td><td>17.2</td><td>17.2</td></tr><tr><td>MT-CoT (Li et al., 2022)</td><td>√</td><td>56.8</td><td>30.3</td><td>22.0</td><td>49.4</td><td>38.2</td><td>39.3</td></tr><tr><td>SCOTT (Wang et al., 2023a)</td><td>√</td><td>42.4</td><td>18.8</td><td>13.0</td><td>45.7</td><td>34.1</td><td>30.8</td></tr><tr><td>Std-CoT (Magister et al., 2023)</td><td>√</td><td>54.2</td><td>28.7</td><td>21.6</td><td>59.6</td><td>45.1</td><td>41.8</td></tr><tr><td>SBS (Hsieh et al., 2023)</td><td>√</td><td>42.4</td><td>27.7</td><td>28.8</td><td>68.5</td><td>48.6</td><td>43.2</td></tr><tr><td>SBS-MI (Chen et al., 2024b)</td><td>V</td><td>42.9</td><td>24.3</td><td>29.2</td><td>68.4</td><td>49.3</td><td>42.8</td></tr><tr><td>w/o RWC + w/ KRSL on  $D _ { d u a l } ^ { + }$ </td><td>√</td><td>55.1</td><td>30.1</td><td>24.1</td><td>60.3</td><td>44.1</td><td>42.7</td></tr><tr><td>w/o RWC + w/ KRSL on  $D _ { d u a l }$ </td><td>√</td><td>55.4</td><td>30.1</td><td>24.2</td><td>63.6</td><td>48.3</td><td>44.3</td></tr><tr><td>w/ RWC + w/o KRSL</td><td>√</td><td>59.7</td><td>30.0</td><td>24.5</td><td>61.9</td><td>45.5</td><td>44.3</td></tr><tr><td>EDIT (ours, w/ RWC + w/ KRSL on  $D _ { d u a l } )$ </td><td>√</td><td>60.9</td><td>31.1</td><td>25.9</td><td>64.1</td><td>50.5</td><td>46.5</td></tr></table>

Table 1: Results (Accuracy, %) of the main experiment.  
![](images/57cace4e59e8b47edcfa5c3f044049a9f21f18ecb82eae2780cf1f92cb2cb404.jpg)  
Figure 4: Ablation results on model size for four OOD datasets. The dotted line indicates the performance of the teacher LLM under the Zero-shot-CoT setting. We also present the results on the IND dataset in Appendix A.3.

Can CoT distillation improve the performance of students? From the table, it is evident that the student SLMs with distillation outperform those that are not distilled. This demonstrates that the reasoning ability of LLMs can be effectively transferred to SLMs by distilling CoTs.

Can EDIT further enhance the performance of students compared to other distillation methods? It can be observed that our proposed method EDIT outperforms the popular and common distillation baseline Std-CoT on both IND and OOD datasets, achieving an average improvement of 4.7 %, which demonstrates the effectiveness and generalizability of EDIT. However, EDIT performs worse on AGIEval and ARC-E compared to SBS, likely due to a strong correlation between questions and answers in these datasets. SBS allows the model to directly predict answers, benefiting from the special properties of these datasets. In addition, SBS has obvious disadvantages because the rationale it generates is inconsistent with the answer logic (Dai et al., 2024).

How significant are the improvements in EDIT attributed to the rectified wrong CoTs and the key steps learning, respectively? Ablation results in the table show that removing the rectified wrong CoTs (w/o RWC) and removing key reasoning steps learning (w/o KRSL) result in performance degradation on almost all IND and OOD, emphasizing the importance of both components. On the one hand, the rectified teachers’ mistakes aid the students in learning diverse ways of thinking. On the other hand, KRSL directs the student’s attention to crucial steps in the dual CoTs, thereby improving the reasoning ability of the students. Additionally, we note that although KRSL and DPO (Rafailov et al., 2023) share very similar learning principles, DPO performed unexpectedly poorly in this scenario. Detailed experiments and analyses are provided in Appendix A.8.

## 4.3 Ablation Study

EDIT is universally applicable to SLMs of various sizes. To better adapt to the community’s varying computational resource requirements, we conduct experiments on models of different sizes, including TinyLLaMA-1.1B (Zhang et al., 2024), LLaMA2-7B and 13B. The results in Figure 4 show that EDIT outperforms the baselines across different model sizes. While smaller models like the 1.1B variant show more modest gains on simpler benchmarks (e.g., ARC-E and ARC-C), we observe significant improvements on more challenging benchmarks like BB-sub and AGIEval across all model sizes. We attribute this phenomenon to two key factors: (1) smaller models’ limited capacity constrains complex reasoning acquisition, and (2) simpler benchmarks inherently offer less improvement potential. This suggests that the more challenging a task is, the more it requires genuine reasoning rather than mere imitation, highlighting the benefits that EDIT brings to students.

EDIT is universally applicable to SLMs with various architectures. To cater to the community’s diverse model preferences, we conduct experiments on models of different architectures, including CodeLLaMA-7B (Touvron et al., 2023), LLaMA3-8B (AI@Meta, 2024), and Mistral-7Bv0.2 (Jiang et al., 2023a). As shown in Figure 5, EDIT consistently outperforms its variant w/o KRSL and the baseline Std-CoT across all model architectures. Notably, the performance gap is significantly larger for the stronger model, Mistral, indicating that our method provides greater benefits with more powerful base models.

Correct key reasoning steps have a greater impact than incorrect ones. We conduct an ablation study on the key reasoning steps in KRSL where students learn exclusively from either the correct or wrong reasoning steps (referred to §3.3, we set $\alpha = 0 \mathrm { o r } \beta = 0$ , respectively). The results shown in Figure 6 indicate that learning key reasoning steps solely from either correct or wrong CoTs leads to a decline in performance. This demonstrates that joint learning from both correct and wrong key reasoning steps is more beneficial for enhancing reasoning. Furthermore, we observe a greater performance drop in the absence of key steps in correct CoTs (w/o Correct) compared to the absence of key steps in wrong CoTs (w/o Wrong), suggesting that key steps from correct CoTs have a more significant impact on students’ learning.

![](images/c7180240745530856127228996de9f564c8f01a6bdaffc81aa747328d7fae279.jpg)  
Figure 5: Ablation results on different student models for the IND and OOD datasets. We compare EDIT with its variants w/o KRSL and Std-CoT. The results are reported by IND-AVG and OOD-AVG, which respectively denote average accuracy on IND and OOD datasets.

![](images/f3a96a17e217382be876127863ccccd8a816afb97d505ec64230b3354c5f6896.jpg)  
Figure 6: Ablation results on key reasoning steps for the IND (BBH-test) and OOD (others) datasets. w/o Correct represents that students only learn key reasoning steps in wrong CoTs, and w/o Wrong represents that students only learn key reasoning steps in correct CoTs.

Challenging dual CoTs data is important. We explore which component of the dual CoTs dataset in KRSL plays a more significant role: the originally correct dual CoTs $\mathcal { D } _ { d u a l } ^ { + }$ or the inherently wrong dual CoTs $\mathcal { D } _ { d u a l } ^ { - } .$ From the Table 2, compared to using $\mathcal { D } _ { d u a l } ^ { + } ,$ employing $\mathcal { D } _ { d u a l } ^ { - }$ resulted in superior performance, even with less data, which demonstrates that the dual CoTs constructed from the inherent wrong CoTs of teachers are more challenging compared to $\mathcal { D } _ { d u a l } ^ { + }$ and more effectively

highlight the key steps in reasoning.
<table><tr><td>Dataset</td><td> $\mathcal { D } _ { d u a l } ^ { + }$   $( \# = 3 8 0 5 )$ </td><td> $\mathcal { D } _ { d u a l } ^ { - }$   $( \nVDash 1 4 0 2 )$ </td><td> $\mathcal { D } _ { d u a l }$   $( \# = 5 2 0 7 )$ </td></tr><tr><td>BBH-test</td><td>61.3</td><td>60.9</td><td>60.9</td></tr><tr><td>BB-sub</td><td>31.2</td><td>30.8</td><td>31.1</td></tr><tr><td>AGIEval</td><td>24.4</td><td>26.0</td><td>25.9</td></tr><tr><td>ARC-E</td><td>64.6</td><td>63.8</td><td>64.1</td></tr><tr><td>ARC-C</td><td>48.9</td><td>50.5</td><td>50.5</td></tr><tr><td>AVG</td><td>46.1</td><td>46.4</td><td>46.5</td></tr></table>

Table 2: Results across dual CoTs datasets in KRSL.

## 5 Analysis

## 5.1 Quality of Generated CoTs

Beyond reasoning accuracy, the quality of CoTs is crucial for interpretable AI. Thus, we use the sota LLM, GPT-4, to score the quality of CoTs generated by Std-CoT, EDIT, and teacher LLMs. The evaluation focuses on which CoT best reflects the key reasoning steps in the problem-solving process, with the prompt template detailed in Appendix C.4. The distribution of evaluation scores is shown in Figure 7, where we observe that the score distribution for CoTs generated by EDIT is closer to that of the teacher compared to Std-CoT. This demonstrates that EDIT is more effective in learning key reasoning steps, producing higher-quality CoTs.

## 5.2 Other Analysis

Considering the differences in training data sizes due to dual CoTs, we conduct a Cost Analysis in Appendix A.4 to enable a fairer comparison. To better illustrate the quality of key reasoning steps in the generated CoTs, we conduct a Case Study in Appendix A.5. Additionally, since our method is mistake-driven, we also explore the impact of different Mistake Patterns on the method’s performance in Appendix C.5.

## 6 Conclusion

In this paper, we propose a mistake-driven key reasoning step distillation method to alleviate student imitation of teachers’ reasoning forms. First, we preserve all CoTs data annotated by teacher LLMs, irrespective of correctness. Using these data, we design two comprehensive prompts to guide teachers in generating dual CoTs data. Finally, we utilize the minimum edit distance algorithm to identify the key reasoning steps and employ a fine-grained loss function for guided learning. Extensive experiments demonstrate EDIT’s effectiveness in enhancing student SLMs’ reasoning capabilities. We hope our work can make the community attach the importance of learning key reasoning steps in dual CoTs, collectively advancing the efficiency of CoT reasoning distillation.

![](images/a9a5f548e5888ea28fd844b8d266938ea2c66d18fa0a787f1615f3c3bc09cd8b.jpg)  
Figure 7: Score distribution evaluated by GPT-4 on BBH-test. Kernel density estimation is used to visualize the distribution of CoTs quality scores.

## Limitations

Currently, most assessments of CoT distillation focus primarily on accuracy (Magister et al., 2023; Ho et al., 2023; Shridhar et al., 2023; Wang et al., 2023c), which is insufficient because safe LLMs rely heavily on trustworthy CoTs. We hope the community to develop standards for evaluating the quality of CoTs, rather than relying solely on automatic assessments by GPT-4.

## Acknowledgements

This work is supported by the National Natural Science Foundation of China (No. U24A20335).

## References

AI@Meta. 2024. Llama 3 model card.

Shengnan An, Zexiong Ma, Zeqi Lin, Nanning Zheng, Jian-Guang Lou, and Weizhu Chen. 2023. Learning from mistakes makes LLM better reasoner. CoRR, abs/2310.20689.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei.

2020. Language models are few-shot learners. In NeurIPS.

Hongzhan Chen, Siyue Wu, Xiaojun Quan, Rui Wang, Ming Yan, and Ji Zhang. 2023. Mcc-kd: Multi-cot consistent knowledge distillation. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 6805–6820.

Xin Chen, Hanxian Huang, Yanjun Gao, Yi Wang, Jishen Zhao, and Ke Ding. 2024a. Learning to maximize mutual information for chain-of-thought distillation. In Findings of the Association for Computational Linguistics: ACL 2024.

Xin Chen, Hanxian Huang, Yanjun Gao, Yi Wang, Jishen Zhao, and Ke Ding. 2024b. Learning to maximize mutual information for chain-of-thought distillation. In ACL (Findings), pages 6857–6868. Association for Computational Linguistics.

Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. 2023. Vicuna: An opensource chatbot impressing gpt-4 with 90%\* chatgpt quality.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. 2023. Palm: Scaling language modeling with pathways. J. Mach. Learn. Res., 24:240:1– 240:113.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Y. Zhao, Yanping Huang, Andrew M. Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. 2022. Scaling instruction-finetuned language models. CoRR, abs/2210.11416.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the AI2 reasoning challenge. CoRR, abs/1803.05457.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. CoRR, abs/2110.14168.

Chengwei Dai, Kun Li, Wei Zhou, and Songlin Hu. 2024. Improve student’s reasoning generalizability through cascading decomposed cots distillation. In EMNLP, pages 15623–15643. Association for Computational Linguistics.

Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, and Tushar Khot. 2023. Specializing smaller language models towards multi-step reasoning. In ICML, volume 202 of Proceedings of Machine Learning Research, pages 10421–10430. PMLR.

Geyang Guo, Ranchi Zhao, Tianyi Tang, Wayne Xin Zhao, and Ji-Rong Wen. 2023. Beyond imitation: Leveraging fine-grained quality signals for alignment. CoRR, abs/2311.04072.

Geyang Guo, Ranchi Zhao, Tianyi Tang, Xin Zhao, and Ji-Rong Wen. 2024. Beyond imitation: Leveraging fine-grained quality signals for alignment. In ICLR. OpenReview.net.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the MATH dataset. In NeurIPS Datasets and Benchmarks.

Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean. 2015. Distilling the knowledge in a neural network. CoRR, abs/1503.02531.

Namgyu Ho, Laura Schmid, and Se-Young Yun. 2023. Large language models are reasoning teachers. In ACL (1), pages 14852–14882. Association for Computational Linguistics.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. 2022. Training compute-optimal large language models. CoRR, abs/2203.15556.

Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alex Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. 2023. Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. In ACL (Findings), pages 8003–8017. Association for Computational Linguistics.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. Lora: Low-rank adaptation of large language models. In ICLR. OpenReview.net.

Jiaxin Huang, Shixiang Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. 2023. Large language models can self-improve. In EMNLP, pages 1051–1068. Association for Computational Linguistics.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023a. Mistral 7b. CoRR, abs/2310.06825.

Yuxin Jiang, Chunkit Chan, Mingyang Chen, and Wei Wang. 2023b. Lion: Adversarial distillation of proprietary large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 3134–3154. Association for Computational Linguistics.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. In NeurIPS.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In SOSP, pages 611–626. ACM.

Ming Li, Lichang Chen, Jiuhai Chen, Shwai He, Jiuxiang Gu, and Tianyi Zhou. 2024. Selective reflectiontuning: Student-selected data recycling for LLM instruction-tuning. CoRR, abs/2402.10110.

Shiyang Li, Jianshu Chen, Yelong Shen, Zhiyu Chen, Xinlu Zhang, Zekun Li, Hong Wang, Jing Qian, Baolin Peng, Yi Mao, Wenhu Chen, and Xifeng Yan. 2022. Explanations from large language models make small reasoners better. CoRR, abs/2210.06726.

Xingxuan Li, Ruochen Zhao, Yew Ken Chia, Bosheng Ding, Shafiq Joty, Soujanya Poria, and Lidong Bing. 2023. Chain-of-knowledge: Grounding large language models via dynamic knowledge adapting over heterogeneous sources. In The Twelfth International Conference on Learning Representations.

Weize Liu, Guocong Li, Kai Zhang, Bang Du, Qiyuan Chen, Xuming Hu, Hongxia Xu, Jintai Chen, and Jian Wu. 2023. Mind’s mirror: Distilling self-evaluation capability and comprehensive thinking from large language models. CoRR, abs/2311.09214.

Lucie Charlotte Magister, Jonathan Mallinson, Jakub Adámek, Eric Malmi, and Aliaksei Severyn. 2023.

Teaching small language models to reason. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 1773–1781. Association for Computational Linguistics.

OpenAI. 2023a. Chatgpt (June 13 version). https: //chat.openai.com.

OpenAI. 2023b. Gpt-4 technical report. https:// cdn.openai.com/papers/gpt-4.pdf. Accessed: [insert date here].

OpenAI. 2024. Learning to reason with llms. https://openai.com/index/ learning-to-reason-with-llms/. Accessed: [insert date here].

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In NeurIPS.

Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. 2023. Instruction tuning with GPT-4. CoRR, abs/2304.03277.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In NeurIPS.

Leonardo Ranaldi and André Freitas. 2024. Aligning large and small language models via chain-of-thought reasoning. In EACL (1), pages 1812–1827. Association for Computational Linguistics.

Noah Shinn, Beck Labash, and Ashwin Gopinath. 2023. Reflexion: an autonomous agent with dynamic memory and self-reflection. CoRR, abs/2303.11366.

Kumar Shridhar, Alessandro Stolfo, and Mrinmaya Sachan. 2023. Distilling reasoning capabilities into smaller language models. In ACL (Findings), pages 7059–7073. Association for Computational Linguistics.

Hao Sun, Yong Jiang, Bo Wang, Yingyan Hou, Yan Zhang, Pengjun Xie, and Fei Huang. 2024. Retrieved in-context principles from previous mistakes. CoRR, abs/2407.05682.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed Chi, Denny Zhou, and Jason Wei. 2023. Challenging big-bench tasks and whether chain-of-thought can solve them. In ACL (Findings), pages 13003–13051. Association for Computational Linguistics.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/ stanford\_alpaca.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open foundation and finetuned chat models. CoRR, abs/2307.09288.

Danqing Wang and Lei Li. 2023. Learning from mistakes via cooperative study assistant for large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 10667–10685.

Peifeng Wang, Zhengyang Wang, Zheng Li, Yifan Gao, Bing Yin, and Xiang Ren. 2023a. SCOTT: selfconsistent chain-of-thought distillation. In ACL (1), pages 5546–5558. Association for Computational Linguistics.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023b. Self-consistency improves chain of thought reasoning in language models. In ICLR. OpenReview.net.

Zhaoyang Wang, Shaohan Huang, Yuxuan Liu, Jiahai Wang, Minghui Song, Zihan Zhang, Haizhen Huang, Furu Wei, Weiwei Deng, Feng Sun, and Qi Zhang. 2023c. Democratizing reasoning ability: Tailored learning from large language model. In EMNLP, pages 1948–1966. Association for Computational Linguistics.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. 2022a. Emergent abilities of large language models. Trans. Mach. Learn. Res., 2022.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022b. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS.

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. 2023. Wizardlm: Empowering large language models to follow complex instructions. CoRR, abs/2304.12244.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. 2022. Star: Bootstrapping reasoning with reasoning. In NeurIPS.

Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. 2024. Tinyllama: An open-source small language model. CoRR, abs/2401.02385.

Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. 2023. Agieval: A human-centric benchmark for evaluating foundation models. CoRR, abs/2304.06364.

Yuhang Zhou and Wei Ai. 2024. Teaching-assistantin-the-loop: Improving knowledge distillation from imperfect teacher models in low-budget scenarios.

## A Additional Experiment

## A.1 Detailed Performance on Reasoning Subtasks

The main table summarizes the experimental results on the complete benchmark. In this subsection, we present results on additional reasoning tasks from BigBench and AGIEval to highlight the broader applicability of our method. As shown in Table 3, our approach consistently surpasses the baseline models on nearly all subtasks, including key mathematical reasoning benchmarks such as AQuA, SAT-MATH, GSM8K (Cobbe et al., 2021), and MATH (Hendrycks et al., 2021). Notably, this performance is achieved despite our training dataset containing only 200 simple math reasoning examples out of 5207 total samples. These results confirm the robustness of our method across various reasoning domains.

## A.2 Impact of Hyperparameters

In this section, we explore the impact of hyperparameters on EDIT performance through grid search, with the results shown in the Table 4. Increasing α from 0 to 1 (comparing Group A to C or B to D) leads to significant performance improvements across most benchmarks. However, increasing β beyond 0.025 results in a noticeable performance drop, indicating that the two loss terms in Eq.6 need to be balanced for optimal performance. Excessive dominance of either term negatively impacts model training, showing a collaborative yet adversarial relationship between the two terms.

<table><tr><td>Subtasks / Method</td><td>Source</td><td>In-domain</td><td>MT-CoT</td><td>SCOTT</td><td>Std-CoT</td><td>Std-CoT w/ Repeat Sampling</td><td>Std-CoT w/ Dual CoTs</td><td>EDIT (Ours)</td></tr><tr><td>Date Understanding</td><td>BBH</td><td>√</td><td>74.0</td><td>54.0</td><td>82.0</td><td>76.0</td><td>74.0</td><td>80.0</td></tr><tr><td>Temporal Sequences</td><td>BBH</td><td>√</td><td>94.0</td><td>66.0</td><td>94.0</td><td>98.0</td><td>86.0</td><td>98.0</td></tr><tr><td>Multi-Step Arithmetic</td><td>BBH</td><td>√</td><td>6.0</td><td>0.0</td><td>8.0</td><td>14.0</td><td>18.0</td><td>18.0</td></tr><tr><td>Sports Understanding</td><td>BBH</td><td>√</td><td>90.0</td><td>74.0</td><td>90.0</td><td>86.0</td><td>86.0</td><td>90.0</td></tr><tr><td>Elementary Math QA</td><td>BigBench</td><td>X</td><td>10.0</td><td>13.0</td><td>11.0</td><td>14.0</td><td>17.0</td><td>20.0</td></tr><tr><td>Identify Math Theorems</td><td>BigBench</td><td>X</td><td>9.4</td><td>9.4</td><td>20.8</td><td>18.9</td><td>24.5</td><td>26.4</td></tr><tr><td>StrategyQA</td><td>BigBench</td><td>x</td><td>50.0</td><td>31.0</td><td>57.0</td><td>50.0</td><td>49.0</td><td>59.0</td></tr><tr><td>AQuA-RAT</td><td>AGIEval</td><td>X</td><td>15.4</td><td>14.6</td><td>17.3</td><td>23.2</td><td>22.8</td><td>24.4</td></tr><tr><td>SAT-Math</td><td>AGIEval</td><td>X</td><td>15.5</td><td>21.4</td><td>20.9</td><td>23.6</td><td>20.0</td><td>24.5</td></tr><tr><td>GSM8K</td><td>GSM8K</td><td>X</td><td>15.3</td><td>17.1</td><td>15.4</td><td>10.9</td><td>14.7</td><td>17.5</td></tr><tr><td>MATH</td><td>MATH</td><td>X</td><td>4.3</td><td>4.1</td><td>5.1</td><td>5.0</td><td>5.0</td><td>5.6</td></tr><tr><td>AVG</td><td></td><td></td><td>34.9</td><td>27.7</td><td>38.3</td><td>38.1</td><td>38.8</td><td>42.1</td></tr></table>

Table 3: Results on commonly used reasoning subtasks.

## A.3 Ablation Study on Model Size for In-domain Dataset

The results of the model size ablation study on IND datasets are presented in Figure 8. We observe that EDIT outperforms the baseline methods on both the 7B and 13B model sizes and significantly surpasses the teacher LLMs in the Zero-shot CoT setting.

![](images/5f673a6ef9a35a58e725a2d250d3bbe2ead4ca4aac5590af57e1ff0ceb482dbf.jpg)  
Figure 8: Ablation study on model size for the IND dataset (BBH-test). The dotted line indicates the performance of the teacher LLM under the Zero-shot-CoT setting.

## A.4 Cost Analysis

Considering that our method utilizes dual CoTs data, which results in twice the amount of training data compared to the baselines, we implement two additional baseline settings to ensure a fair comparison and ablate the impact of the increased data size due to dual CoTs: (1) Std-CoT w/ Repeat Sampling. We perform random repeat sampling on the baseline’s original training data until the volume matches that of EDIT; (2) Std-CoT w/ Dual CoTs. We train the Std-CoT using all data included in EDIT, adding the marker "[Counterfactual Reasoning]" before the negative sample’s question to differentiate it from positive reasoning. Results in Table 5 show that while Std-CoT benefits from additional data, it underperforms compared to EDIT across most tasks. EDIT’s superiority stems from its method of learning key reasoning steps beyond mere imitation, allowing students to learn from mistakes. Additionally, Std-CoT with Dual CoTs outperforms that with Repeat Sampling in OOD tasks by incorporating counterfactual reasoning, reducing overfitting and better generalizing the reasoning. This supports our view that simple fine-tuning with correct teacher data is insufficient for true reasoning learning.

## A.5 Case Study

We present 5 cases sampled from BBH, AGIEval, and ARC in Table 20, 21, 22, 23 and 24 to clearly compare the CoT generated by EDIT with the teacher LLM and the standard CoTs distillation (Std-CoT). We utilize ✓ and ✗ to denote whether the CoT is correct or incorrect, respectively. From Tables 20 and 21, we observe that both the teacher and Std-CoT models make mistakes at the same positions in their reasoning processes, even though the nature of their mistakes differs. These positions can be considered key reasoning steps. In contrast, the EDIT CoT exhibits a changed way of thinking and demonstrates correct reasoning at these corresponding positions (highlighted in green), leading to the correct answers. Especially for the case in Table 24, while the Std-CoT and teacher models both adopt a logic of enumerating and analyzing each option, EDIT raises issues or questions for each option and then answers them. This suggests that EDIT, through learning key reasoning steps, avoids overfitting to the teacher CoT’s reasoning steps and instead adapts its reasoning logic to solve the problem effectively. Table 22 reveals nearly identical reasoning among the three CoTs, yet in the critical reasoning steps 7 and 8, Std-CoT fails to make the correct decisions, whereas EDIT correctly executes stack operations. Cases from OOD benchmarks, shown in Tables 23 and 24, indicate that EDIT can accurately analyze problems and provide more logical reasoning.

<table><tr><td>Group</td><td>α</td><td> $\beta$ </td><td>BBH-test</td><td>BB-sub</td><td>AGIEval</td><td>ARC-E</td><td>ARC-C</td><td>AVG</td></tr><tr><td>A</td><td>0</td><td>0</td><td>59.7</td><td>30.0</td><td>24.5</td><td>61.9</td><td>45.5</td><td>44.32</td></tr><tr><td>B</td><td>0</td><td>0.025</td><td>59.0</td><td>30.2</td><td>24.1</td><td>62.1</td><td>45.9</td><td>44.26</td></tr><tr><td>C</td><td>1</td><td>0</td><td>60.2</td><td>30.5</td><td>23.4</td><td>62.7</td><td>48.0</td><td>44.96</td></tr><tr><td>D</td><td>1</td><td>0.025</td><td>60.9</td><td>31.1</td><td>25.9</td><td>64.1</td><td>50.5</td><td>46.50</td></tr><tr><td>E</td><td>1</td><td>0.05</td><td>59.7</td><td>30.0</td><td>24.7</td><td>61.9</td><td>45.5</td><td>44.36</td></tr></table>

Table 4: Results of ranging hyperparameters.
<table><tr><td>Method</td><td>Training Data Size</td><td>BBH-test</td><td>BB-sub</td><td>AGIEval</td><td>ARC-E</td><td>ARC-C</td><td>AVG</td></tr><tr><td>Std-CoT w/ Repeat Sampling</td><td>10414</td><td>59.4</td><td>30.3</td><td>24.0</td><td>58.0</td><td>42.1</td><td>42.8</td></tr><tr><td>Std-CoT w/ Dual CoTs</td><td>10414</td><td>54.8</td><td>32.9</td><td>25.1</td><td>62.2</td><td>44.1</td><td>43.8</td></tr><tr><td>EDIT (ours)</td><td>10414</td><td>60.9</td><td>31.1</td><td>25.9</td><td>64.1</td><td>50.5</td><td>46.5</td></tr></table>

Table 5: Results (Accuracy, %) of the cost analysis.

## A.6 Mistake Pattern Mining

In this subsection, we delve into the influence of various mistake patterns on the EDIT. Based on the observation of mistake data, we utilize gpt-3.5-turbo-0613 to categorize all the teacher’s wrong CoTs into four types, including Logical Errors (LEs), Knowledge Errors (KEs), Mathematical Calculation Errors (MCEs) and Other Errors (OEs). The statistic result for mistake pattern data can be found in Table 6. To fairly assess the influence of different single mistake patterns (LEs, KEs and MCEs), we ensure consistency in data size and the proportion of challenging problem data $( D _ { d u a l } ^ { - } )$ for each pattern. Since the available data for MCEs is the smallest, we randomly select 356 instances from $D _ { d u a l } ^ { + }$ and 56 instances from $D _ { d u a l } ^ { - }$ , creating three dual CoT datasets— $\cdot D _ { L E s } , D _ { K E s } ,$ and $D _ { M C E s }$ —each with 412 samples. Then we conduct experiments using these datasets in KRSL and the results of EDIT trained on these mistake patterns are shown in Table 7.

From the table, we can see that KRSL on $D _ { L E s }$ consistently outperforms other mistake patterns, with KEs and MCEs having a relatively smaller impact. This suggests that LEs provide a broader range of reasoning patterns that are relevant for mathematical, commonsense, and symbolic reasoning. As for KEs and MCEs, since these types of mistakes are more specific compared to LEs, it is not easy for the model to learn a general reasoning solution from these mistakes. Therefore, learning the key reasoning steps from logical reasoning errors is the most effective way among them.

## A.7 Integration with Self-Consistency

In this subsection, we explore the integration of our method with the widely-used CoT reasoning technique, Self-Consistency (SC). SC improves reasoning performance by generating multiple reasoning paths and selecting the most consistent answer through majority voting. For SC, we apply majority voting with 8 sampled reasoning paths, using temperature=0.7 and topp=0.95 for decoding. As shown in Table 8, nearly all CoT distillation methods, including our method EDIT, show significant performance improvements when combined with SC. This demonstrates that EDIT can be effectively integrated with CoT reasoning techniques, providing both flexibility and scalability.

## A.8 KRSL v.s. DPO

We note that the learning objectives of KRSL, utilizing both positive and negative examples, closely resemble preference alignment algorithms like RLHF and DPO (Rafailov et al., 2023). Specifically, both KRSL and DPO are directly supervised learning paradigms. However, there are key differences:

1. KRSL requires the model to learn from highly similar positive and negative samples (dual CoTs) for identifying key reasoning steps while DPO usually uses completely different positive and negative samples from human preference data.

<table><tr><td>Mistake Patterns &amp; Dataset</td><td>LEs</td><td>KEs</td><td>MCEs</td><td>OEs</td><td> $\mathrm { L E s + K E s }$ </td><td> $\begin{array} { c } { \mathrm { L E s + } } \\ { \mathrm { M C E s } } \end{array}$ </td><td> $\mathrm { K E s + M C E s }$ </td><td> $\mathrm { L E s } + \mathrm { K E s } + \mathrm { M C E s }$ </td><td>Total</td></tr><tr><td> $\mathcal { D } _ { d u a l } ^ { + }$ </td><td>2618</td><td>452</td><td>356</td><td>51</td><td>255</td><td>45</td><td>26</td><td>2</td><td>3805</td></tr><tr><td></td><td>1077</td><td>77</td><td>56</td><td>62</td><td>105</td><td>22</td><td>3</td><td>0</td><td>1402</td></tr><tr><td> $\begin{array} { c } { { D _ { d u a l } ^ { - } } } \\ { { D _ { d u a l } ^ { - } } } \end{array}$ </td><td>3695</td><td>529</td><td>412</td><td>113</td><td>360</td><td>67</td><td>29</td><td>2</td><td>5207</td></tr></table>

Table 6: Classification statistics of mistake data patterns.
<table><tr><td>Dataset</td><td>BBH-test</td><td>BB-sub</td><td>AGIEval</td><td>ARC-E</td><td>ARC-C</td><td>AVG</td></tr><tr><td> $D _ { L E s }$ </td><td>60.1</td><td>31.0</td><td>24.6</td><td>63.0</td><td>45.8</td><td>44.9</td></tr><tr><td> $D _ { K E s }$ </td><td>60.0</td><td>30.6</td><td>24.2</td><td>62.0</td><td>46.1</td><td>44.6</td></tr><tr><td> $D _ { M C E s }$ </td><td>59.4</td><td>30.4</td><td>24.4</td><td>62.3</td><td>45.8</td><td>44.5</td></tr></table>

Table 7: Performance (Accuracy, %) comparison across mistake pattern datasets used in KRSL. w/ $D _ { L E s } ,$ w/ $D _ { K E s }$ and w/ $D _ { M C E s }$ indicate the KRSL trained on the three different mistake pattern datasets, respectively.
<table><tr><td>Method + Self-consistency</td><td>BBH-test</td><td>BB-sub</td><td>AGIEval</td><td>ARC-E</td><td>ARC-C</td><td>AVG</td></tr><tr><td>MT-CoT</td><td>56.4</td><td>32.2</td><td>22.3</td><td>68.5</td><td>52.8</td><td>46.4</td></tr><tr><td>SCOTT</td><td>41.1</td><td>22.0</td><td>16.7</td><td>56.1</td><td>40.6</td><td>35.5</td></tr><tr><td>Std-CoT</td><td>56.3</td><td>31.2</td><td>25.2</td><td>66.2</td><td>50.0</td><td>45.8</td></tr><tr><td>Std-CoT w/ Repeat Sampling</td><td>60.4</td><td>33.3</td><td>24.1</td><td>64.4</td><td>47.1</td><td>45.9</td></tr><tr><td>Std-CoT w/ Dual CoTs</td><td>58.4</td><td>33.6</td><td>26.8</td><td>64.4</td><td>48.2</td><td>46.3</td></tr><tr><td>EDIT(ours)</td><td>62.0</td><td>32.0</td><td>27.2</td><td>70.4</td><td>54.1</td><td>49.1</td></tr></table>

Table 8: Results of Integration with Self-consistency (Accuracy, major vote@8).
<table><tr><td>Method</td><td>BBH-test</td><td>BB-sub</td><td>AGIEval</td><td>ARC-E</td><td>ARC-C</td><td>AVG</td></tr><tr><td>w/ DPO</td><td>10.2</td><td>15.4</td><td>4.8</td><td>5.1</td><td>4.9</td><td>8.1</td></tr><tr><td>w/ KRSL</td><td>60.9</td><td>31.1</td><td>25.9</td><td>64.1</td><td>50.5</td><td>46.5</td></tr></table>

Table 9: Performance (Accuracy, %) comparison between DPO and KRSL implementation in EDIT.

2. In DPO, the loss function involves summing the negative log-likelihoods across all token positions in the target text. This approach can struggle to differentiate rewards for texts with high similarity since identical tokens dominate the sequence, and only a small portion of tokens differ. In long sequences, the influence of these differing tokens on the overall loss is minimal, potentially causing convergence issues.

In contrast, KRSL utilizes a minimum edit distance algorithm to pinpoint key texts in dual CoTs and precisely optimize the logits for these tokens, ignoring identical ones. This makes KRSL more suitable for learning from dual CoTs compared to DPO. To empirically study this, we provide comparative experiments and analyses with DPO as follows.

We compare KRSL with DPO by implementing DPO in the EDIT and training LLaMA2- 7B on complete dual CoTs data using the dpo\_trainer implemented in the TRL <sup>6</sup>, with the following settings: learning rate of 1e-5, a cosine learning rate scheduler, a warmup ratio of 0.3, DPO beta of 0.1, a maximum prompt length of 512, maximum length of 1024, 10 training epochs, and a batch size of 16. The results (Table 9) show significant performance degradation with DPO. Thus, we check the model’s generation results in Table 10 and find that the output pattern almost completely collapses, outputting only the answer without the intermediate reasoning process. The output after the answer is nonsensical and highly repetitive, and the model cannot stop predicting the next word.

<table><tr><td>## Input</td><td></td></tr><tr><td>Question</td><td>Task Description: Select the humorous edit that &#x27;ruins&#x27; the input movie or musical artist name. Q: Which of the following is a humorous edit of this artist or movie name: &#x27;little shop of horrors&#x27;? Options:</td></tr><tr><td></td><td>(A) little shoy of horrors (B) little hop of horrors</td></tr><tr><td></td><td></td></tr><tr><td></td><td>(C) little wshop of horrors</td></tr><tr><td></td><td>(D) ulittle shop of horrors</td></tr><tr><td></td><td>A: little shop of horrors</td></tr><tr><td></td><td></td></tr><tr><td></td><td>Q:Which of the following is a humorous edit of this artist or movie name: &#x27;little shop of horrors&#x27;?</td></tr><tr><td></td><td>Options:</td></tr><tr><td></td><td>(A) little shoy of horrors</td></tr><tr><td></td><td>(B) little hop of horrors</td></tr><tr><td>EDIT w/ DPO</td><td>(C) little wshop of horrors</td></tr><tr><td></td><td>(D) ulittle shop of horrors</td></tr><tr><td></td><td>A:little shop of horrors</td></tr><tr><td></td><td>Q: Which of the following is a humorous edit of this artist or movie name: &#x27;little shop of horrors&#x27;? Options:...</td></tr></table>

Table 10: A failure case in EDIT w/ DPO from BIG-Bench Hard.

## A.9 Details of Experimental Settings

## A.9.1 Dataset Statistics

Table 11, 12, 14 and 15 show the data statistics of AGIEval, ARC, BIG-Bench Hard (BBH) and BIG-Bench Sub (BB-sub), respectively.

<table><tr><td rowspan=1 colspan=1>No.</td><td rowspan=1 colspan=2>Task</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Size</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=2>AQuA-RAT</td><td rowspan=1 colspan=1>254</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=2>LogiQA-EN</td><td rowspan=1 colspan=1>651</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=2>LSAT-AR</td><td rowspan=1 colspan=1>230</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=2>LSAT-LR</td><td rowspan=1 colspan=1>510</td><td rowspan=2 colspan=1>55</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=2>LSAT-RC</td><td rowspan=1 colspan=1>269</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=2>SAT-Math</td><td rowspan=1 colspan=1>220</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=2>SAT-EN</td><td rowspan=1 colspan=1>206</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=2>SAT-EN (w/o Psg.)</td><td rowspan=1 colspan=1>206</td><td rowspan=1 colspan=1>4</td></tr></table>

Table 11: Statistics of AGIEval dataset.

<table><tr><td>Task</td><td>Size</td><td># Choices</td></tr><tr><td>ARC-E</td><td>2376</td><td>4-5</td></tr><tr><td>ARC-C</td><td>1172</td><td>4-5</td></tr></table>

Table 12: Statistics of ARC test dataset.

<table><tr><td>Arguments</td><td>Student</td><td>Teacher</td></tr><tr><td>do sample</td><td>False</td><td>True</td></tr><tr><td>temperature</td><td></td><td>0.2</td></tr><tr><td>top-p</td><td>1.0</td><td>1.0</td></tr><tr><td>top-k</td><td></td><td></td></tr><tr><td>max new tokens</td><td>1024</td><td>2048</td></tr><tr><td># return sequences</td><td>1</td><td>1</td></tr></table>

Table 13: Generation configs of students and teachers.

<table><tr><td>No.</td><td>Task</td><td>Size</td><td># Choices</td></tr><tr><td>1</td><td>Boolean Expressions</td><td>250</td><td>2</td></tr><tr><td>2</td><td>Causal Judgement</td><td>187</td><td>2</td></tr><tr><td>3</td><td>Date Understanding</td><td>250</td><td>6</td></tr><tr><td>4</td><td>Disambiguation QA</td><td>250</td><td>4</td></tr><tr><td>5</td><td>Dyck Languages</td><td>250</td><td>-</td></tr><tr><td>6</td><td>Formal Fallacies Syllogisms Negation</td><td>250</td><td>2</td></tr><tr><td>7</td><td>Geometric Shapes</td><td>250</td><td>11</td></tr><tr><td>8</td><td>Hyperbaton (Adjective Ordering)</td><td>250</td><td>2</td></tr><tr><td>9</td><td>Logical Deduction (3 objects)</td><td>250</td><td>3</td></tr><tr><td>10</td><td>Logical Deduction (5 objects)</td><td>250</td><td>5</td></tr><tr><td>11</td><td>Logical Deduction (7 objects)</td><td>250</td><td>7</td></tr><tr><td>12</td><td>Movie Recommendation</td><td>250</td><td>5</td></tr><tr><td>13</td><td>Multi-Step Arithmetic</td><td>250</td><td>-</td></tr><tr><td>14</td><td>Navigate</td><td>250</td><td>2</td></tr><tr><td>15</td><td>Object Counting</td><td>250</td><td>-</td></tr><tr><td>16</td><td>Penguins in a Table</td><td>146</td><td>5</td></tr><tr><td>No.</td><td>Task</td><td>Size</td><td># Choices</td></tr><tr><td>17</td><td>Reasoning about Colored Objects</td><td>250</td><td>18</td></tr><tr><td>18</td><td>Ruin Names</td><td>250</td><td>11</td></tr><tr><td>19</td><td>Salient Translation Error Detection</td><td>250</td><td>6</td></tr><tr><td>20</td><td>Snarks</td><td>178</td><td>2</td></tr><tr><td>21</td><td>Sports Understanding</td><td>250</td><td>2</td></tr><tr><td>22</td><td>Temporal Sequences</td><td>250</td><td>4</td></tr><tr><td>23</td><td>Tracking Shuffled Objects (3 objects)</td><td>250</td><td>3</td></tr><tr><td>24</td><td>Tracking Shuffled Objects (5 objects)</td><td>250</td><td>5</td></tr><tr><td>25</td><td>Tracking Shuffled Objects (7 objects)</td><td>250</td><td>7</td></tr><tr><td>26</td><td>Web of Lies</td><td>250</td><td>2</td></tr><tr><td>27</td><td>Word Sorting</td><td>250</td><td></td></tr><tr><td></td><td>Sum</td><td>6511</td><td>一</td></tr><tr><td>No. | Task</td><td></td><td>Size</td><td># Choices</td></tr><tr><td>1</td><td>abstract_narrative_understanding</td><td>100</td><td>5</td></tr><tr><td>2</td><td>anachronisms</td><td>100</td><td>2</td></tr><tr><td>3</td><td>analogical_similarity</td><td>100</td><td>7</td></tr><tr><td>4</td><td></td><td>70</td><td>2</td></tr><tr><td></td><td>analytic_entailment</td><td></td><td>2</td></tr><tr><td>5</td><td>cause_and_effect</td><td>100</td><td>26</td></tr><tr><td>6</td><td>checkmate_in_one</td><td>100</td><td>10</td></tr><tr><td>7 8</td><td>cifar10_classification</td><td>100</td><td></td></tr><tr><td>9</td><td>code_line_description</td><td>60</td><td>4</td></tr><tr><td>10</td><td>conceptual_combinations</td><td>100</td><td>4</td></tr><tr><td>11</td><td>crass_ai</td><td>44</td><td>4</td></tr><tr><td>12</td><td>elementary_math_qa</td><td>100</td><td>5</td></tr><tr><td>13</td><td>emoji_movie</td><td>100</td><td>5</td></tr><tr><td>14</td><td>empirical_judgments</td><td>99</td><td>3</td></tr><tr><td>15</td><td>english_russian_proverbs</td><td>80</td><td>4</td></tr><tr><td>16</td><td>entailed_polarity</td><td>100</td><td>2</td></tr><tr><td>17</td><td>entailed_polarity_hindi</td><td>100</td><td>2</td></tr><tr><td>18</td><td>epistemic_reasoning</td><td>100</td><td>25</td></tr><tr><td>19</td><td>evaluating_information_essentiality</td><td>68 100</td><td>2</td></tr><tr><td>20</td><td>fantasy_reasoning</td><td>59</td><td>10</td></tr><tr><td>21</td><td>figure_of_speech_detection</td><td>100</td><td>4</td></tr><tr><td>22</td><td>goal_step_wikihow</td><td>31</td><td>5</td></tr><tr><td>23</td><td>gre_reading_comprehension</td><td>42</td><td>4</td></tr><tr><td>24</td><td>human_organs_senses</td><td>53</td><td>4</td></tr><tr><td>25</td><td>identify_math_theorems</td><td>47</td><td>5</td></tr><tr><td>26</td><td>identify_odd_metaphor</td><td></td><td>2</td></tr><tr><td>27</td><td>implicatures</td><td>100 82</td><td>25</td></tr><tr><td>28</td><td>implicit_relations</td><td>100</td><td>2</td></tr><tr><td>29</td><td>indic_cause_and_effect</td><td>100</td><td>26</td></tr><tr><td>30</td><td>intersect_geometry kanji_ascii</td><td>100</td><td>5</td></tr><tr><td>31</td><td>kannada</td><td>100</td><td>4</td></tr><tr><td>No.</td><td>Task</td><td>Size</td><td># Choices</td></tr><tr><td></td><td></td><td></td><td>2</td></tr><tr><td>32</td><td>key_value_maps</td><td>100 100</td><td>3</td></tr><tr><td>33 34</td><td>logic_grid_puzzle logical_args</td><td>32</td><td>5</td></tr><tr><td>35</td><td>logical_fallacy_detection</td><td>100</td><td>2</td></tr><tr><td>36</td><td>metaphor_boolean</td><td>100</td><td>2</td></tr><tr><td>37</td><td>metaphor_understanding</td><td>100</td><td>4</td></tr><tr><td>38</td><td>minute_mysteries_qa</td><td>100</td><td>4</td></tr><tr><td>39</td><td>mnist_ascii</td><td>100</td><td>10</td></tr><tr><td>40</td><td>moral_permissibility</td><td>100</td><td>2</td></tr><tr><td>41</td><td>movie_dialog_same_or_different</td><td>100</td><td>2</td></tr><tr><td>42</td><td>nonsense_words_grammar</td><td>50</td><td>4</td></tr><tr><td>43</td><td>odd_one_out</td><td>86</td><td>5</td></tr><tr><td>44</td><td>parsinlu_qa</td><td>100</td><td>4</td></tr><tr><td>45</td><td>physical_intuition</td><td>81</td><td>4</td></tr><tr><td>46</td><td>play_dialog_same_or_different</td><td>100</td><td>2</td></tr><tr><td>47</td><td>presuppositions_as_nli</td><td>100</td><td>3</td></tr><tr><td>48</td><td>riddle_sense</td><td>49</td><td>5</td></tr><tr><td>49</td><td>similarities_abstraction</td><td>76</td><td>4</td></tr><tr><td>50</td><td>simple_ethical_questions</td><td>100</td><td>4</td></tr><tr><td>51</td><td>social_iqa</td><td>100</td><td>3</td></tr><tr><td>52</td><td>strange_stories</td><td>100</td><td>2</td></tr><tr><td>53</td><td>strategyqa</td><td>100</td><td>2</td></tr><tr><td>54</td><td>swahili_english_proverbs</td><td>100</td><td>4</td></tr><tr><td>55</td><td>swedish_to_german_proverbs</td><td>72</td><td>4</td></tr><tr><td>56</td><td>symbol_interpretation</td><td>100</td><td>5</td></tr><tr><td>57</td><td>timedial</td><td>100</td><td>3</td></tr><tr><td>58</td><td></td><td>100</td><td>5</td></tr><tr><td></td><td>undo_permutation</td><td></td><td>5</td></tr><tr><td>59 60</td><td>unit_interpretation vitaminc_fact_verification</td><td>100 100</td><td>3</td></tr><tr><td>61</td><td>winowhy</td><td>100</td><td>2</td></tr><tr><td></td><td>Sum</td><td>5384</td><td></td></tr></table>

Table 14: Statistics of BIG-Bench Hard dataset.

Table 15: Statistics of BIG-Bench sub dataset. We filter the original dataset by retrieving tasks with keywords "multiple choice" and randomly sample up to 100 examples per task. Note, the task in BBH will not be involved in BB-sub.

## A.9.2 Hyperparameters Settings

In our study, we ensure consistency in the hyperparameter settings across all baselines, including our proposed EDIT approach, to maintain the fairness of our comparative analysis. Here, we detail the hyperparameter configurations employed in our experiments.

Training Steps and Batch Size. The number of training steps is determined based on the size of the training dataset, the batch size, and the number of gradient accumulation steps required. We maintain a consistent batch size across all baselines to eliminate any performance discrepancies that could arise from varying batch sizes.

Learning Rate. Our initial exploratory experiments focused on the standard CoTs distillation method using the LLaMA-2 model. We found that while the batch size had minimal impact on performance, the learning rate was a critical factor. We tested learning rates of 1e-4, 2e-4, and 3e-4, observing optimal performance at 2e-4 across the standard CoT and other distillation baselines, as well as our EDIT approach. Consequently, we set the learning rate to 2e-4 for all methods involved in our study.

Epochs and Evaluation Strategy. Throughout our training process, we monitored the training loss curve and noted that it generally plateaued by the 15th epoch, indicating that the models had achieved convergence. Therefore, we set the number of epochs to 15 for 7B models. The process of determining the number of epochs for other model sizes followed a similar pattern. To mitigate the potential risk of overfitting and to ensure our evaluation reflects the most effective model configuration, we systematically selected checkpoints from the epoch that demonstrated the best performance on the IND task. These checkpoints were then used to evaluate performance on OOD tasks.

The hyperparameters in training and inference can be found in Table 16 and Table 13 respectively. In the KRSL, the second phase training in EDIT, the learning rate is empirically set as 5e-6.

## A.9.3 Computation Budget

Our experimental code is based on modifications of Meta’s open-source llama-recipes<sup>7</sup>, utilizing the FSDP framework and training the model in parallel on four 80GB A100 GPUs. In our experimental setup, training a 7B model during the SFT stage takes approximately 40 minutes per epoch. For KRSL, each epoch takes around 90 minutes. With the same settings, training the Mistral model will see about a 10% increase in training time. We will release our code in the future.

<table><tr><td>Hyperparameter</td><td>TinyLLaMA-1.1B</td><td>LLaMA2-13B</td><td>LLaMA2-7B / CodeLLaMA-7B /LLaMA3-8B / Mistral-7B-v0.2</td></tr><tr><td>gradient accumulation steps</td><td>4</td><td>8</td><td>4</td></tr><tr><td>per device batch size</td><td>16</td><td>8</td><td>16</td></tr><tr><td>learning rate</td><td>2e-4</td><td>2e-4</td><td>2e-4</td></tr><tr><td>epoches</td><td>20</td><td>15</td><td>10</td></tr><tr><td>max length</td><td>1024</td><td>1024</td><td>1024</td></tr><tr><td>β of AdamW</td><td>(0.9,0.999)</td><td>(0.9,0.999)</td><td>(0.9,0.999)</td></tr><tr><td>€ of AdamW</td><td>1e-8</td><td>1e-8</td><td>1e-8</td></tr><tr><td>γ of Scheduler</td><td>0.95</td><td>0.95</td><td>0.95</td></tr><tr><td>weight decay</td><td>0</td><td>0</td><td>0</td></tr><tr><td>warmup ratio</td><td>0</td><td>0</td><td>0</td></tr><tr><td>rank of LoRA</td><td>64</td><td>64</td><td>64</td></tr><tr><td>α of LoRA</td><td>32</td><td>32</td><td>32</td></tr><tr><td>target modules</td><td>q_proj, v_proj</td><td>q_proj, v_proj</td><td>q_proj, v_proj</td></tr><tr><td>drop out of LoRA</td><td>0.05</td><td>0.05</td><td>0.05</td></tr></table>

Table 16: Training hyperparameters.

## B Example of Dual CoTs

We provide dual CoTs examples with three different mistake patterns including logical errors, knowledge errors and mathematical calculation errors in Table 17, 18, 19 and mark the correct/wrong key reasoning steps in different colors. We observe that our carefully crafted prompts for generating correct CoT and wrong CoT effectively ensure the desired dual CoT characteristics: similar reasoning steps leading to different conclusions. For instance, subordinating conjunctions in Table 17 like "however," "despite," and "even though," as well as certain verb and noun phrases, significantly influence the reasoning process and the conclusion. These elements represent the key reasoning steps that we aim for the model to learn.

## C Prompt Templates

## C.1 CoTs Extraction Prompt

We use the prompt template shown in Table 25 to call the ChatGPT API to generate the CoTs for the BBH-train datasets.

## C.2 Answer Hint Prompt

We list the Answer Hint Prompt templates in Table 26, which imply the teacher LLMs to generate the CoTs based on the given answers following the in-context examples.

## C.3 Contrastive CoTs Prompt

We list the Contrastive CoTs Prompt templates in Table 27, which query the teacher LLMs to generate the CoTs with similar rationales to the original ones but divergent answers by following the few examples provided with contrastive CoT pairs.

## C.4 Evaluation Prompt of CoTs Quality

We list the evaluation prompt templates of CoTs quality in Table 28.

## C.5 Mistake Pattern Mining Prompt

For mistake pattern mining, we employ the prompt template delineated in Table 29, which includes the definitions of the four distinct mistake patterns.

<table><tr><td>school, Lena is unhappy in her marriage and recently lost her job. Over the course of the day, George and Lena get into a number of heated arguments. Later in the afternoon they play a game of darts. They split the first two games, and the third game is close until the end. Who will win comes down to George&#x27;s last shot. If he hits a high Question point region, he wins; if he hits a low point region, Lena wins. George doesn&#x27;t care that Lena is having a difficult</td><td>## Input Task Description: Answer questions about causal attribution. Q: How would a typical person answer each of the following questions about causation? George and his sister Lena reunite at their parentshouse for Thanksgiving. Whereas George just got into medical</td></tr><tr><td>Correct CoT</td><td>of the dart board. As he sets up his shot, however, he loses his balance. The dart slips out of his hand and wobbles toward the board. The dart lands in the high point region. George triumphs in his victory while Lena is sad. Did George hit the high point region intentionally? Options: - Yes - No A: Here in this question, we are told that &quot;George tries to aim at the high point region of the dart board.&quot; However, due to losing his balance, the dart slips out of his hand and wobbles toward the board. A typical person would</td></tr><tr><td></td><td>assume that George did not intentionally hit the high point region, as his intention was to aim at the high point region but his actions resulted in an unintended outcome. Therefore, the answer is No. Here in this question, we are told that George tries to aim at the high point region of the dart board. Despite losing</td></tr><tr><td>Wrong CoT</td><td>his balance, the dart lands in the high point region. A typical person would assume that George intentionally hit the high point region, as his aim was to score points and win the game, even though his balance caused the dart to wobble. Therefore, the answer is Yes.</td></tr></table>

Table 17: A casual judgment dual CoTs example from BIG-Bench Hard where the wrong CoT shows a logical error.
<table><tr><td rowspan="6">Question</td><td>## Input Task Description: Recommend movies similar to the given list of movies.</td></tr><tr><td>Q: Find a movie similar to Pulp Fiction, Schindler&#x27;s List, Apollo 13, Dances with Wolves:</td></tr><tr><td>Options:</td></tr><tr><td>(A) Iron Eagle IV</td></tr><tr><td>(B) The Shawshank Redemption</td></tr><tr><td>(C) Knockin&#x27; on Heaven&#x27;s Door (D) Aguirre The Wrath of God</td></tr><tr><td rowspan="6">Correct CoT</td><td>A: - Pulp Fiction (crime, drama; 1994)</td></tr><tr><td>- Schindler&#x27;s List (biography, drama, history; 1993) - Apollo 13 (adventure, drama, history; 1995)</td></tr><tr><td>- Dances with Wolves (adventure, drama, western; 1990)</td></tr><tr><td>These are all famous movies produced in the 1990s. Amongst all the options, the only movie similar to these ones seems to be The Shawshank Redemption (drama; 1994), as it is also a critically acclaimed film with a compelling</td></tr><tr><td>storyline. Therefore, the answer is (B).</td></tr><tr><td></td></tr><tr><td rowspan="4">Wrong CoT</td><td>- Pulp Fiction (crime, drama; 1994)</td></tr><tr><td>- Schindler&#x27;s List (biography, drama, history; 1993)</td></tr><tr><td>- Apollo 13 (adventure, drama, history; 1995)</td></tr><tr><td>These are all famous movies produced in the 1990s. Amongst all the options, the only movie similar to these ones</td></tr><tr><td></td><td>- Dances with Wolves (adventure, drama, western; 1990) seems to be Aguirre The Wrath of God (adventure, biography, drama; 1972), as it is also a historical drama film. Therefore, the answer is (D).</td></tr></table>

Table 18: A movie recommendation example from BIG-Bench Hard where the wrong CoT shows a knowledge-based error.
<table><tr><td>Question</td><td>## Input Task Description: Solve multi-step arithmetic problems. Q  $( ( - 1 + \overleftarrow { 6 \mathrm { ~ - ~ } 4 } + - 6 ) \cdot ( - 2 \ast - 5 \mathrm { ~ - ~ } 1 \ast 1 ) ) =$  A:</td></tr><tr><td>Correct CoT</td><td>Let's recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). So, remember to always compute the expressions inside parentheses or brackets first. This equation can be written as Å - B, where A = (-1 + -6 - -4 + -6) and  $\mathrm { \bf B } = ( - 2 \mathrm { \bf \ ^ { * } \mathrm { \bf ~ - } } 5 \mathrm { \bf \mathrm { ~ - } } 1 \mathrm { \bf \ ^ { * } \mathrm { \bf ~ 1 } } ) .$  Let's calculate  $\mathbf { A } = ( - 1 + 6 \ - \ 4 \ + \ 6 ) = ( - 1 + ( - 6 ) \ - ( - 4 ) + ( - 6 ) ) = ( - 1 \ - 6 \ + 4 \ - \ 6 ) = ( - 7 \ + 4 \ - \ 6 ) = ( - 3 \ - \ 6 ) = - 9 .$  Let's calculate  $\mathrm { B } = ( - 2 \ast - 5 - - 1 \ ^ { \ast } \ 1 ) = ( 1 0 \ - \ ( - 1 ) ) = ( 1 0 + 1 ) = 1 1 .$  Then, the final equation is  $\mathbf { A } - \mathbf { B } = - 9 - 1 1 = - 2 0 .$  Therefore, the answer is -20.</td></tr><tr><td>Wrong CoT</td><td>Let's recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). So, remember to always compute the expressions inside parentheses or brackets first. This equation can be written as Ä - B; where A = (-1 + -6 - -4 + -6) and  $\mathrm { \bf B } = ( - 2 \mathrm { \bf \ ^ { * } \mathrm { \bf ~ - } } 5 \mathrm { \bf \mathrm { ~ - } } 1 \mathrm { \bf \ ^ { * } \mathrm { \bf ~ 1 } } ) .$  Let's calculate  $\mathbf { A } = ( - 1 + - 6 \ - \ 4 \ + \ 6 ) = ( - 1 \ + - 6 \ - 4 \ + 6 ) = ( - 1 \ - 6 \ - 4 \ - 6 ) = ( - 7 \ - 6 \ - 4 \ - 6 ) = ( - 1 3 \ - \ 4 \ - 6 ) = - 2 3 .$  Let's calculate  $\mathrm { B } = ( - 2 \ast - 5 - - 1 \ ^ { \ast } \ 1 ) = ( 1 0 \ - \ ( - 1 ) ) = ( 1 0 + 1 ) = 1 1 .$  Then, the final equation is A - B = -23 - 11 = -34. Therefore, the answer is -34.</td></tr><tr><td>Question:</td><td>On the table, there are three pink scrunchiephone chargers, two orange crayons, and two black scrunchiephone chargers. If I remove all the black items from the table, how many scrunchiephone chargers remain on it? Options:(A) zero(B) one (C) two (D) three (E) four (F) five (G) six (H) seven (I) eight (J) nine (K) ten (L) eleven (M) twelve (N) thirteen (O) fourteen (P) fifteen (Q) sixteen</td></tr><tr><td>Teacher</td><td>According to this question, there are three pink scrunchiephone chargers, two orange crayons, and two black scrunchiephone chargers on the table. If we remove all the black items from the table, we are left with three pink scrunchiephone chargers and two orange crayons. Therefore, the number of scrunchiephone chargers that remain on the table is five. X Therefore, the answer is (F).</td></tr><tr><td>Std-CoT</td><td>According to this question, there are three pink scrunchiephone chargers, two orange crayons, and two black scrunchiephone chargers on the table. If we remove all the black items from the table, we are left with three pink scrunchiephone chargers, two orange crayons, and two black scrunchiephone chargers. The number of scrunchiephone chargers remaining on the table is seven. Therefore, the answer is (H). X</td></tr><tr><td>EDIT</td><td>According to this question, there are a total of 5 scrunchiephone chargers on the table: 3 pink scrunchiephone chargers and 2 black scrunchiephone chargers. If we remove all the black items, we are left with 3 pink scrunchiephone chargers. Therefore, the answer is (D). L</td></tr></table>

Table 19: A multistep arithmetic dual CoTs example from BIG-Bench Hard where the wrong CoT shows a mathematical calculation error.

Table 20: A reasoning about colored objects case from BIG-Bench Hard.

<table><tr><td>Question:</td><td>Find a movie similar to Jurassic Park, Independence Day, Forrest Gump, Batman. Options: (A) The Firm (B) Roger Dodger (C) Innocent Man (D) An (E) The Silence of the Lambs</td></tr><tr><td>Teacher</td><td>- Jurassic Park (action, adventure, science-fiction; 1993) - Independence Day (action, science-fiction, drama; 1996) - Forrest Gump (comedy, drama, romance; 1994) - Batman (action, adventure; 1989) These are all famous Hollywood movies produced in the 1980s and 1990s. Amongst all the options, the only movie similar to these ones seems to be The Firm (drama, mystery, thriller; 1993), as it is also a popular movie from the same era. X Therefore, the answer is (A).</td></tr><tr><td>Std-CoT</td><td>Jurassic Park is an action and adventure movie. Independence Day is an action, science-fiction, and adventure movie. Forrest Gump is a comedy, drama, and romance movie. Batman is an action, adventure, and crime movie. The Firm is a crime, drama, and thriller movie. Roger Dodger is a comedy and drama movie. Innocent Man is a crime, drama, and thriller movie. The Silence of the Lambs is a crime, drama, and thriller movie. Amongst all the options, the movie similar to Jurassic Park seems to be The Firm. X Therefore, the answer is (A).</td></tr><tr><td>EDIT</td><td>Issue 1: Is this movie similar to Jurassic Park? Above all the options, the only movie similar to Jurassic Park seems to be The Silence of the Lambs due to its suspenseful and thrilling nature. Issue 2: Is this movie similar to Independence Day? Above all the options, the only movie similar to Independence Day seems to be Roger Dodger due to its sci-fi and action-packed nature. Issue 3: Is this movie similar to Forrest Gump? Above all the options, the only movie similar to For- rest Gump seems to be The Firm due to its drama and emotional depth. Issue 4: Is this movie similar to Batman? Above all the options, the only movie similar to Batman seems to be An due to its superhero and action-packed nature.</td></tr><tr><td>Question: properly.</td><td>Correctly close a Dyck-n word. Complete the rest of the sequence, making sure that the parentheses are closed Input: &lt; (( &lt; &gt; )</td></tr><tr><td rowspan="9">Teacher</td><td>We should process each input one by one and keep track of the stack configuration.</td></tr><tr><td>0: empty stack</td></tr><tr><td>1: &lt; ; stack: &lt;</td></tr><tr><td>2: (; stack: &lt; (</td></tr><tr><td>3: (; stack: &lt; ((</td></tr><tr><td>4: { ; stack: &lt; (( {</td></tr><tr><td>5: &lt; ; stack: &lt; (( { &lt;</td></tr><tr><td>6: &gt; ; stack: &lt; (( {</td></tr><tr><td>7: } ; stack: &lt; (({</td></tr><tr><td rowspan="3"></td><td>8: ) ; stack: &lt; ((</td></tr><tr><td>Now, we have reached the end. The final stack is "&lt; ((".</td></tr><tr><td>We will need to pop out "(", "(" one by one in that order.</td></tr><tr><td rowspan="14">Std-CoT</td><td>So, we need ",". Therefore, the answer is ) ). X</td></tr><tr><td>We should process each input one by one and keep track of the stack configuration.</td></tr><tr><td>0: empty stack 7</td></tr><tr><td>1: &lt; ; stack: &lt;</td></tr><tr><td>2: (; stack: &lt; (</td></tr><tr><td>3: (; stack: &lt; ((</td></tr><tr><td>4: { ; stack: &lt; (( { 5: &lt; ; stack: &lt; (( { &lt;</td></tr><tr><td>6: &gt; ; stack: &lt; (( {</td></tr><tr><td>7: } ; stack: &lt; (({</td></tr><tr><td>8: ) ; stack: &lt; (( {</td></tr><tr><td>Now, we have reached the end. The final stack is "&lt; ( ( {".</td></tr><tr><td>We will need to pop out "{", "(", "&lt;" one by one in that order.</td></tr><tr><td>So, we need "I", ")", "&gt;". Therefore, the answer is }) &gt;. X</td></tr><tr><td>We should process each input one by one and keep track of the stack configuration.</td></tr><tr><td rowspan="9">0: empty stack EDIT</td><td></td></tr><tr><td>1: &lt; ; stack: &lt;</td></tr><tr><td></td></tr><tr><td>2: (; stack: &lt; (</td></tr><tr><td>3: (; stack: &lt; ((</td></tr><tr><td>4: { ; stack: &lt; (( {</td></tr><tr><td>5: &lt; ; stack: &lt; (( { &lt;</td></tr><tr><td>6: &gt; ; stack: &lt; (( {</td></tr><tr><td>7: } ; stack: &lt; ((</td></tr><tr><td>8: ) ; stack: &lt; (</td><td></td></tr><tr><td></td><td>Now, we have reached the end. The final stack is "&lt; (".</td></tr><tr><td></td><td>We will need to pop out "(", "&lt;" one by one in that order.</td></tr><tr><td></td><td>So, we need ")", "&gt;". Therefore, the answer is ) &gt;.</td></tr><tr><td>Question:</td><td>Anna has 4 books. If she decide to arrange the 4 books in every possible combination and moved just one book every minute, how long would it taken by her ? Options: (A)22 minutes (B)30 minutes (C)15 minutes (D)24 minutes (E)35 minutes</td></tr><tr><td>Teacher</td><td>First, Anna has 4 books and she wants to arrange them in every possible combination. To find the number of possible combinations, we can use the formula for permutations, which is n! (n factorial), where n is the number of items. In this case, n = 4, so the number of possible combinations is 4! = 4 x 3 x 2 x 1 = 24. Next, we know that Anna can move just one book every minute. Since there are 24 possible combina- tions, it will take her 24 minutes to go through all of them. 1</td></tr><tr><td>Std-CoT</td><td>Therefore, the answer is (D) 24 minutes. This question focuses on the time it takes Anna to arrange the 4 books in every possible combination. We know that Anna moved just one book every minute. Plugging in the minutes per move: 1 minute There are 4 books: A, B, C, D There are (4!) = 4 different combinations: A, B, C, D; B, A, C, D; B, C, A, D; C, A, B, D; D, A, B, C The total number of minutes it would take Anna to arrange the books in every possible combination is the sum of the minutes needed for each combination: (4 minutes) + (4 minutes) + (4 minutes) + (4 minutes) + (4 minutes) = 15 minutes. X</td></tr><tr><td>EDIT</td><td>This question asks for the total amount of time it would take Anna to arrange the books in every possible combination and moved just one book every minute. We first need to calculate the number of possible combinations: 4! = 24. Then, we need to multiply the number of possible combinations by the number of minutes per combination: 24 * 1 minute = 24 minutes. Therefore, the answer is (D) 24 minutes. L</td></tr></table>

Table 21: A movie recommendation case from BIG-Bench Hard.

Table 22: A dyck languages case from BIG-Bench Hard.

Table 23: A AQuA-RAT case from AGIEval.

<table><tr><td>Question:</td><td>Which method is the safest way to watch an eclipse of the Sun? Options: (A) Turn away after two or three minutes. (B) Look at the Sun through a long telescope. (C) Cast an image through a pinhole onto a screen. (D) Blink often until your eyes get used to the light.</td></tr><tr><td rowspan="4">Teacher</td><td>First, it is important to note that directly looking at the Sun during an eclipse can be extremely dangerous and can cause permanent damage to the eyes.</td></tr><tr><td>Option (A) suggests turning away after two or three minutes. However, even a brief exposure to the Sun during an eclipse can be harmful, so this method is not safe.</td></tr><tr><td>Option (B) suggests looking at the Sun through a long telescope. This method is also unsafe because the intense light from the Sun can damage the telescope and cause harm to the eyes.</td></tr><tr><td>Option (C) suggests casting an image through a pinhole onto a screen. This method is a safe and rec- ommended way to watch an eclipse. By creating a pinhole camera, the image of the Sun is projected onto a screen, allowing for safe viewing without directly looking at the Sun.</td></tr><tr><td>Std-CoT</td><td>Therefore, the answer is (C) Cast an image through a pinhole onto a screen. L We know that looking directly at the Sun can cause permanent damage to the eyes. Therefore, the sanswer is (C) L</td></tr><tr><td rowspan="2">EDIT</td><td>casting an image through a pinhole onto a screen. We are told that this question is about the safest way to watch an eclipse of the Sun. Looking at the Sun through a long telescope is not recommended as it can cause permanent damage to the eyes. Blinking often until your eyes</td></tr><tr><td>get used to the light is also not recommended as it can cause eye strain and discomfort. The only remaining option is to cast an image through a pinhole onto a screen. This method involves creating a small hole in a card or paper and projecting the Sun&#x27;s image onto a surface, such as a wall or a piece of paper. This method is safe as it does not require direct eye contact with the Sun.</td></tr></table>

Table 24: A case from AI2 Reasoning Challenge.

{Task Description}. Your response should conclude with the format "Therefore, the   
answer is".   
Q: {Task Example Question No.1}   
A: Let’s think step by step. {Human-Curated-CoTs No.1}.   
Q: {Task Example Question No.2}   
A: Let’s think step by step. {Human-Curated-CoTs No.2}.   
Q: {Task Example Question No.2}   
A: Let’s think step by step. {Human-Curated-CoTs No.3}.   
Q: {QUESTION}   
A: Let’s think step by step.

Table 25: CoTs extraction prompt template of gpt-3.5-turbo for generating the CoTs data.

{Task Description}. Your response should conclude with the format "Therefore, the   
answer is".   
Q: {Task Example Question No.1}   
H: {The correct answer is [HINT ANSWER No.1]}   
A: Let’s think step by step. {Human-Curated-CoTs No.1}.   
Q: {Task Example Question No.2}   
H: {The correct answer is [HINT ANSWER No.2]}   
A: Let’s think step by step. {Human-Curated-CoTs No.2}.   
Q: {Task Example Question No.3}   
H: {The correct answer is [HINT ANSWER No.3]}   
A: Let’s think step by step. {Human-Curated-CoTs No.3}.   
Q: {QUESTION}   
H: {The correct answer is [HINT ANSWER]}   
A: Let’s think step by step.

Table 26: Answer Hint Prompt templates for rectifying the wrong CoTs data based on the hint answers.

{Task Description}. You need to complete the [Wrong Response] which requires you   
to give the   
most likely incorrect answer to the [Question] and the rationale for the incorrect   
answer.   
The incorrect answer and rationale in the [Wrong Response] must be different from   
the correct   
answer and rationale in the [Right Response].   
[Question]: {Task Example Question No.1}   
[Right Response]: {Corrected CoT No.1}   
[Wrong Response]: {Wrong CoT No.1}   
[Question]: {Task Example Question No.2}   
[Right Response]: {Corrected CoT No.2}   
[Wrong Response]: {Wrong CoT No.2}   
[Question]: {Task Example Question No.3}   
[Right Response]: {Corrected CoT No.3}   
[Wrong Response]: {Wrong CoT No.3}   
[Question]: {USER\_QUESTION}   
[Right Response]: {Corrected CoT}   
[Wrong Response]:

Table 27: Contrastive CoTs Prompt templates for mistaken the correct CoTs data. The examples are sampled from the teachers’ original wrong CoTs data and its corrected CoTs. In this way, teacher LLMs can expose the reasoning flaws in problems that were originally solved correctly.

```ini
[System] You are a helpful and precise assistant for assessing the quality of the
response.
[Question]: {QUESTION}
[Reference Answer]: {ANSWER}
[AI Assistant 1’s Answer Start]
{ASSISTANT1}
[AI Assistant 1’s Answer End]
[AI Assistant 2’s Answer Start]
{ASSISTANT2}
[AI Assistant 2’s Answer End]
[AI Assistant 3’s Answer Start]
{ASSISTANT3}
[AI Assistant 3’s Answer End]
[System] We would like to request your feedback, in the form of scoring, on which
of the
responses from AI Assistant 1, 2 and 3 effectively demonstrates the key reasoning
steps in
solving this question. Key Reasoning Steps refer to certain crucial steps in the
process of
logical reasoning or problem-solving. These steps play a significant role in the
thinking
process and have a notable impact on subsequent reasoning. Each student will
receive an
overall score on a scale of 1 to 10, where a higher score signifies that the
assistant’s
response is more effectively demonstrates the key reasoning steps for the
question.
Please provide a comprehensive explanation, avoiding any potential bias and
ensuring that
the order in which the responses were presented does not affect your judgment.
And then
output three lines indicating the scores for AI Assistant 1, 2 and 3,
respectively.
Output with the following format:
Evaluation evidence: <your evaluation explanation here>
Score of AI Assistant 1: <score>
Score of AI Assistant 2: <score>
Score of AI Assistant 3: <score>
```  
Table 28: Prompt template of GPT-4 for assessing CoTs quality. In the analysis, we use this template to eval the quality of CoTs generated by Std-CoT, EDIT and the teacher LLM respectively.

```ini
[System] You are a helpful assistant who is good at identifying types of reasoning
mistakes.
There are now three types of inference errors, as follows:
(a). Logical reasoning errors. This type of error involves the logical structure
of reasoning,
including assumptions, reasoning rules, argument chains, etc. Among logical
errors, students
may make errors such as invalid reasoning, insufficient or incorrect assumptions,
and jumps in
reasoning. Students may make errors in selecting reasoning strategies or methods.
The chosen
method may not be suitable for a specific problem, or may lead to misleading
reasoning.
(b). Knowledge errors in reasoning. This type of error involves misunderstanding
or incomplete
understanding of facts, concepts or knowledge, conceptual confusion, and cognitive
biases.
(c). Numerical calculation errors. This type of error involves mathematical
calculation errors,
which may include incorrect calculations, conversions or errors in the processing
of numerical
values.
(d). Other errors. All other errors that do not belong to the above three
categories.
I will give you a dictionary with the following fields and meanings:
{
"input": reasoning question.
"right_output": the correct answer.
"wrong_output": the wrong answer.
}
You need to first form your own opinion about the problem based on the reasoning
questions and the
correct answers, and then analyze the reasons for the mistakes in the wrong
answers in "Rationale:".
Then give your classification results in "Category:", e.g., (a), (b) or (c), etc.
If an answer
involves errors in multiple categories, you should point them out and connect them
with ’+’ sign
in the category. For example, if an answer involves logical errors and
mathematical calculation
errors, then the category should be a+c.
You must output with the following format:
Rationale: <your analysis process and explanation of the final classification
results>
Category: <only fill in with a or b or c or a+b or a+c or b+c or a+b+c or d.>
```  
Table 29: Prompt templates of GPT-3.5 for classifying the mistakes. In the analysis, we use this template to classify the mistake data used in EDIT.