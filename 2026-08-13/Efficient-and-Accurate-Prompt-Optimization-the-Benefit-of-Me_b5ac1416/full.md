# Efficient and Accurate Prompt Optimization: the Benefit of Memory in Exemplar-Guided Reflection

Cilin Yan<sup>1</sup>\*<sup>†</sup>, Jingyun Wang<sup>1</sup>\*, Lin Zhang<sup>2</sup>\*, Ruihui Zhao<sup>2</sup>, Xiaopu Wu<sup>2</sup>, Kai Xiong<sup>2</sup>, Qingsong Liu<sup>2</sup>, Guoliang Kang <sup>1‡</sup>, Yangyang Kang<sup>3,2‡</sup>

<sup>1</sup>Beihang University, <sup>2</sup>ByteDance, <sup>3</sup>Zhejiang University

{clyanhh, wangjingyun0730, kgl.prml}@gmail.com, zhanglin.hb@bytedance.com {zhaoruihui, wuxiaopu, xiongkai.kx, liuqingsong, yangyangkang}@bytedance.com

## Abstract

Automatic prompt engineering aims to enhance the generation quality of large language models (LLMs). Recent works utilize feedbacks generated from erroneous cases to guide the prompt optimization. During inference, they may further retrieve several semantically related exemplars and concatenate them to the optimized prompts to improve the performance. However, those works only utilize the feedbacks at the current step, ignoring historical and unseleccted feedbacks which are potentially beneficial. Moreover, the selection of exemplars only considers the general semantic relationship and may not be optimal in terms of task performance and matching with the optimized prompt. In this work, we propose an Exemplar-Guided Reflection with Memory mechanism (ERM) to realize more efficient and accurate prompt optimization. Specifically, we design an exemplar-guided reflection mechanism where the feedback generation is additionally guided by the generated exemplars. We further build two kinds of memory to fully utilize the historical feedback information and support more effective exemplar retrieval. Empirical evaluations show our method surpasses previous state-of-the-arts with less optimization steps, i.e., improving F1 score by 10.1 on LIAR dataset, and reducing half of the optimization steps on ProTeGi.

## 1 Introduction

Prompt optimization is crucial for enhancing the performance of Large Language Models (LLMs). Even a subtle adjustment to the prompt may lead to an obvious improvement or decline in performance, thereby highlighting the critical role of prompt engineering for LLMs. Manual prompt engineering demands significant human effort and expert knowledge, while traditional fine-tuning methods (Lester et al., 2021; Shin et al., 2020) heavily rely on substantial computational resources and powerful GPUs. Therefore, it is necessary to explore automatic prompt engineering, which is compatible with black-box APIs (e.g., GPT-4) and does not require extensive resources.

Recently, feedback-based methods (Ye et al., 2024; Juneja et al., 2024) exhibit promising performance for automatic prompt engineering, which generally leverage feedbacks generated from failure cases to facilitate the prompt optimization process. Previous feedback-based methods (Pryzant et al., 2023; Ye et al., 2024) have two main drawbacks. Firstly, they throw unselected and historical feedbacks which may benefit the prompt optimization, resulting in more optimization steps to achieve satisfactory performance. Secondly, during inference, previous methods (Hu et al., 2023; Juneja et al., 2024) may retrieve several semantically related exemplars and concatenate them to the optimized prompt to improve the performance. However, the retrieved exemplars are not optimal without evaluating their influence on the task performance. Those drawbacks largely constrain both the efficiency and accuracy of the prompt optimization process.

In this work, we introduce an Exemplar-Guided Reflection with Memory mechanism (ERM) to achieve efficient and accurate prompt optimization. Firstly, we propose an exemplar-guided reflection mechanism. As shown in Figure 1(a), we manually design an instructive meta-prompt. Unlike previous meta-prompts which simply guide LLMs to reflect on the current case, our instructive metaprompt further directs LLMs to generate exemplars by selecting typical wrong samples and providing detailed solution processes for them. Thanks to the detailed solution processes within exemplars, LLMs therefore yield more informative feedbacks.

![](images/9341943b8110c4c7a7e3008b85d5e2e2744081baa851131eaf9054c8044e3384.jpg)  
Figure 1: Feedback-based automatic prompt engineering methods commonly employ a meta-prompt , which guides LLMs to evaluate the current case, provide feedbacks , and generate refined prompts . In this work, we design an instructive meta-prompt to select exemplars with detailed solution processes, and generate feedbacks for the current case. These feedbacks are stored in Feedback Memory and periodically retrieved to efficiently guide the optimization of prompts . Additionally, these exemplars are stored and assessed in an Exemplar Factory to enhance prediction accuracy.

We then propose Feedback Memory to store all feedbacks and assign a priority score to each of them, as shown in Figure 1(b). During the optimization process, we retrieve a group of feedbacks with the highest priority scores and instruct LLMs to generate a new prompt for the feedbacks. After evaluating the refined prompts, we update the priority scores of the associated feedbacks accordingly, i.e., we increase the score for improved performance and decrease it if no gain. Consequently, feedbacks with valuable insights will be consistently selected rather than ignored throughout the optimization process. As demonstrated in Figure 1(c), we store all exemplars in Exemplar Factory and assign a prior score to each piece. At the inference stage, we retrieve a set of exemplars with the highest priority scores, and concatenate the exemplars to our refined prompt to further improve the performance.

We conduct an extensive evaluation on seven tasks to compare ERM with the latest prompt optimization approaches. Our results demonstrate substantial improvements over state-of-the-art methods, notably achieving a 10.1 F1 score improvement on the LIAR dataset. Furthermore, the optimization speed of ERM is roughly twice as fast as ProTeGi (Pryzant et al., 2023).

Our contributions are summarized as follows:

1) We design an instructive meta-prompt, which guides LLMs to select exemplars and therefore yield more informative feedbacks.

2) We propose a Feedback Memory to store historical feedbacks by their priority scores, enabling effective retrieval and utilization of feedbacks for prompt optimization.

3) We propose an Exemplar Factory to store and evaluate exemplars. By retrieving exemplars and concatenating them to our refined prompt at the inference stage, we further enhance the performance of LLMs.

4) We conduct extensive experiments on various tasks and show superior performance of our method to previous state-of-the-arts. Additionally, our optimization steps can be largely reduced, e.g., the steps of our method are approximately half of that in ProTeGi.

## 2 Related Work

## 2.1 Automatic Prompt Optimization

Prompt engineering (Zhou et al., 2022) aims to identify suitable prompts as inputs for large language models (LLMs) to perform various tasks. To minimize human effort, researchers have explored automatic prompt optimization (Lester et al., 2021; Shin et al., 2020; Li and Liang, 2021). Previous works adopt various strategies for automatic prompt optimization, such as evolutionary-based methods, trajectory-based methods, and feedbackbased methods. Evolutionary-based methods (Guo et al., 2024; Fernando et al., 2024) utilize LLMs to rewrite a set of prompts with evolutionary algorithms (Holland, 1992; Storn and Price, 1997), which select the best prompts on a validation set to simulate the natural selection process for optimizing prompts. Trajectory-based methods (Yang et al., 2024; Tang et al., 2024) employ an LLM prompt optimizer to generate new prompts based on historical prompts, scores, or error examples. Feedbackbased methods (Pryzant et al., 2023; Juneja et al., 2024) use LLMs to summarize feedbacks on erroneous cases, leveraging the feedbacks to optimize and create new prompts. In this work, we primarily focus on feedback-based methods, with the aim of writing stronger feedbacks and efficiently utilizing them for optimization.

![](images/2ee726849e16263943054e4e56fbc5d340d780778ea2e45efd8b1499dd8de727.jpg)  
Figure 2: Pipeline of ERM. In wrong prediction samples, the instructive reflective meta-prompt is employed to select exemplars with detailed answer processes, which are subsequently followed by feedback generation. The feedbacks are stored in feedback memory storage, and the exemplars are stored in exemplar memory storage. These stored feedbacks are periodically retrieved to efficiently guide prompt optimization, with selective forgetting based on their effectiveness in enhancing optimization. Additionally, these exemplars are assessed to enhance prediction accuracy.

## 2.2 Long-Term Memory Mechanisms

Existing automatic prompt optimization methods (Pryzant et al., 2023; Juneja et al., 2024) face challenges in maintaining a robust long-term memory function, limiting their ability to retain and utilize valuable feedbacks for prompt optimization. MemoryBank (Zhong et al., 2024) solves the challenge of maintaining a robust long-term memory conversation history in previous LLMs (Touvron et al., 2023; Zeng et al., 2022; Taori et al., 2023) by introducing a mechanism that enhances their ability to store and recall relevant information over time. This approach mimics human memory dynamics through a selective retention strategy inspired by the Ebbinghaus Forgetting Curve (Ebbinghaus,

2013). Our work builds on these advancements by using memory storage to implement feedbacks and exemplars in long-term memory. We implement a forgetting strategy for feedbacks and exemplars that are retrieved but deemed unvaluable, thereby enhancing the efficiency and accuracy of long-term memory retention in prompt optimization.

## 3 Method

In this section, we propose ERM, a novel method designed to achieve efficient and accurate prompt optimization. As shown in Figure 2, ERM is composed of three core components: (1) Exemplar-Guided Reflection, employing an instructive metaprompt (Section 3.2), guides prompt optimizer to first generate exemplars by identifying typical wrong samples and providing detailed solution processes, followed by generating feedback. (2) We then propose a Feedback Memory (Section 3.3) to store all feedbacks and assign a priority score to each piece of them. These feedbacks can then be retrieved and utilized during optimization efficiently. After evaluating the refined prompts, we update the priority scores of the associated feedbacks. (3) Finally, we utilize an Exemplar Factory (Section 3.4) to store and evaluate exemplars, which serve as additional resources during prediction. By incorporating the retrieved exemplars into our refined prompt, task model are further guided to achieve improved accuracy.

## 3.1 Preliminary

Given a training set $\mathcal { D } _ { t r a i n } = \{ ( q _ { i } , a _ { i } ) \} _ { i = 1 } ^ { n _ { t } }$ (q<sub>i</sub> represents the question, $a _ { i }$ is the paired answer, and $n _ { t }$ is the total number of training samples) and a test set $\mathcal { D } _ { t e s t }$ drawn from a specific task, along with a score function $s ( \cdot )$ for this task, we aim to perform the task using a black-box task model $M _ { s } \left( e . g . \right.$ , ChatGPT), which combines the prompt $p$ with questions from $\mathcal { D } _ { t e s t }$ as input to generate responses. These responses are then evaluated by the score function to calculate an average score over $\mathcal { D } _ { t e s t }$ . The goal of prompt optimization is to find an optimal prompt $p ^ { * }$ drawn from the natural language space that maximizes the expectation of the average score over $\mathcal { D } _ { t e s t }$

$$
p ^ { * } = \arg \operatorname* { m a x } _ { p } \mathbb { E } _ { ( q _ { i } , a _ { i } ) \sim \mathcal { D } _ { t e s t } } [ s ( M _ { s } ( q _ { i } ; p ) , a _ { i } ) ] ,\tag{1}
$$

where $p = [ p _ { I } , p _ { R } ( q _ { i } ) ]$ might be composed of two parts: one includes the invariant content $p _ { I }$ , which remains independent of the question and may include task descriptions and general solution steps, and the other is the variable content $p _ { R } ( q _ { i } )$ , which is question-specific. We leverage a more powerful prompt optimizer $M _ { e } ~ ( e . g . , \mathrm { G P T } \cdot 4 )$ compared with the task model $M _ { s }$ to summarize feedbacks and optimize the prompt.

Previous work typically divides prompt optimization into three steps: prompt initialization, new prompt proposal, and prompt search.

1) Prompt initialization. Prompt initialization can be achieved by both manual initialization and induction initialization. Following Pro-TeGi (Pryzant et al., 2023), we initialize the original prompt $p ^ { 0 }$ manually.

2) New prompt proposal. Commonly, previous methods use task model $M _ { s }$ to evaluate on a subset of $\mathcal { D } _ { t r a i n }$ , and then use prompt optimizer $M _ { e }$ to summarize errors from $n _ { b }$ wrong samples $B ~ = ~ \{ ( q _ { i } , a _ { i } ) \} _ { i = 1 } ^ { n _ { b } }$ , where the response of task model $M _ { s } ( q _ { i } , p ^ { t } )$ is different from $a _ { i }$ . Feedbacks is then generated as $\mathcal { F } ^ { t } = M _ { e } ( p ^ { t } , \boldsymbol { B } ; p _ { r e f } ^ { m e t a } )$ with $p _ { r e f } ^ { m e t a }$ serving as the meta-prompt that guides the prompt optimizer in generating feedback. The prompt optimizer then optimizes and refines the prompt $p ^ { t }$ based on the feedbacks to obtain refined prompts $p ^ { t + 1 } = M _ { e } ( p ^ { t } , \boldsymbol { B } , f ^ { t } ; p _ { o p t } ^ { m e t a } )$ ), where $f ^ { t } \in \mathcal { F } ^ { t }$ , and $p _ { o p t } ^ { m e t a }$ is the meta-prompt guiding the prompt optimizer to propose refined prompt.

3) Prompt search. Following ProTeGi, we employ a beam search strategy to further select the refined prompts. Among several candidate prompts $\mathcal { P } ^ { t + 1 }$ , we select k prompts which perform best on the validation set, which is the subset of the training set. These k prompts are then used for the next optimization step.

## 3.2 Exemplar-Guided Reflection

To encourage the prompt optimizer generate more informative feedbacks, we propose an Exemplar-Guided Reflection in Figure 2(a), which utilizes an instructive meta-prompt to select typical wrong samples with detailed solution processes as exemplars and generate feedbacks for them. Detailedly, we first utilize the instructive meta-prompt $p _ { r e f * } ^ { m e t a }$ which guides the prompt optimizer to select $n _ { e }$ diverse and significantly representative wrong samples from the wrong samples $\boldsymbol { B }$ as exemplars $\mathcal { E } ^ { t }$ and provide detailed solution processes for them:

$$
\mathcal { E } ^ { t } = M _ { e } ( p ^ { t } , B ; p _ { r e f * } ^ { m e t a } ) ,\tag{2}
$$

where $\mathcal { E } ^ { t } = \{ e _ { i } \} _ { i = 1 } ^ { n _ { e } } = \{ ( q _ { i } , a _ { i } , \mathrm { c o t } _ { i } ) \} _ { i = 1 } ^ { n _ { e } }$ is a set of exemplars $e _ { i }$ with detailed solution processes cot<sub>i</sub>. Then, the prompt optimizer generates $n _ { f }$ feedbacks $\mathcal { F } ^ { t } = \bar { \{ f _ { i } ^ { t } \} } _ { i = 1 } ^ { n _ { f } }$ , which offer insights on example predictions and suggestions on modification of the prompt:

$$
\mathcal { F } ^ { t } = M _ { e } ( p ^ { t } , \boldsymbol { B } , \mathcal { E } ^ { t } ; p _ { r e f * } ^ { m e t a } ) ,\tag{3}
$$

Based on the wrong samples $\boldsymbol { B }$ and each item in the generated feedbacks $f ^ { t } \in \mathcal { F } ^ { t }$ , the model finally produce a refined prompt $p ^ { t + 1 }$ for each feedback:

$$
p ^ { t + 1 } = M _ { e } ( p ^ { t } , \mathcal { B } , f ^ { t } ; p _ { o p t } ^ { m e t a } ) .\tag{4}
$$

## 3.3 Feedback Memory

Aiming to accelerate the convergence of prompt optimization process, we propose a Feedback Memory in Figure 2(b). We store the feedbacks with priority scores via a long-term memory mechanism and retrieve them efficiently for optimization. By evaluating the generated prompts, we selectively forget the feedbacks to ensure that all stored feedbacks remain beneficial for prompt optimization.

Feedback Memory Storage In Feedback Memory, we store the valuable feedbacks during the optimization process and assign a priority score to each piece of them, which serves as a basic foundation for Feedback Forgetting Updating. To effectively store useful feedbacks and prevent adverse impacts on prompt optimization, we employ a feedback filtering strategy: (1) We evaluate the refined prompts generated based on the feedbacks, and only store the informative feedbacks whose corresponding prompts bring improvements on the validation set. Such strategy ensures that only valuable feedbacks are stored and retrieved. (2) Additionally, we employ the BGE-M3 model (Chen et al., 2024a) to calculate the semantic similarity between newly generated feedbacks and the stored ones. We ignore the feedbacks of high similarity with the previous ones to avoid redundant information.

Feedback Retrieval During the optimization process, we periodically select historical feedbacks from the memory based on their priority scores. Specifically, we calculate the selection probability for each feedback as follows:

$$
P _ { f } = \operatorname { s o f t m a x } \left( \left\{ e ^ { \frac { s _ { p } ( f _ { i } ) } { \tau _ { f } } } \right\} _ { i = 1 } ^ { | \tilde { \mathcal { F } } | } \right) ,\tag{5}
$$

where $\tau _ { f }$ is the temperature, controlling the tendency to select high-scoring feedbacks, and $\tilde { \mathcal { F } }$ denotes all feedbacks stored in the memory. We then randomly select $n _ { \hat { f } }$ feedbacks according to their selection probabilities:

$$
\hat { \mathcal { F } } = \{ f _ { i } \} _ { i = 1 } ^ { n _ { \hat { f } } } = \mathrm { s a m p l e } ( \tilde { \mathcal { F } } , P _ { f } ) .\tag{6}
$$

Feedback Forgetting Updating The selected feedbacks $\hat { \mathcal { F } }$ guide prompt optimizer generate new prompts $p ^ { t + 1 } = M _ { e } ( p ^ { t } , \mathcal { B } , \hat { \mathcal { F } } ; p _ { o p t * } ^ { m e t a } )$ , where $p _ { o p t * } ^ { m e t a }$ is the meta-prompt that efficiently utilizes the feedback group to generate a refined prompt. We then update their priority scores by evaluating the generated prompt: we increase the priority score if the performance is improved but decrease it if no gain.

$$
s _ { p } ^ { t } ( f ) = ( 1 - \beta ) s _ { p } ( f ) ^ { t - 1 } + \beta \mathbb { I } ( f ) ,\tag{7}
$$

where $\mathbb { I } ( f )$ represents whether sufficient performance gain is achieved and $\beta$ is a hyper-parameter to control the speed of updating. Besides, the feedback will be removed from the storage once its priority score falls below a certain threshold θ:

$$
\tilde { \mathcal { F } } ^ { t } = \{ f \mid f \in \tilde { \mathcal { F } } ^ { t - 1 } , s _ { p } ^ { t } ( f ) \geq \theta \} .\tag{8}
$$

With such Forgetting Updating mechanism, we ensure that the most valuable feedbacks are continuously utilized, which efficiently accelerate the convergence of our optimization process.

## 3.4 Exemplar Factory

As shown in Figure 2(c), we store the exemplars along with a priority score to each piece of them, similar to that in Feedback Memory. These exemplars are stored in memory and retrieved for prediction, allowing us to assess their impact on the task. We selectively forget exemplars, ensuring that the valuable ones will be retrieved to enhance the prediction performance.

Exemplar Memory Storage The exemplar memory storage retains valuable exemplars. We introduce an exemplar filtering strategy to ensure stored exemplars benefit prediction: (1) We verify that the detailed solution process of the exemplar generated by prompt optimizer matches to the ground truth label. (2) When a new generated exemplar is identical to the stored ones, we replace the stored exemplars with probability $p$ and reject the new exemplar with probability $1 - p$ to avoid redundant storage.

Exemplar Retrieval Each exemplar $e _ { i }$ is assigned a priority score $s _ { p } ( e _ { i } )$ . During the prompt optimization process for question $q _ { j }$ , we calculate the selection probability for each exemplar as follows:

$$
P _ { e } ^ { r } = \mathrm { s o f t m a x } \left( \left\{ e ^ { \frac { s _ { p } ( e _ { i } ) \cdot s _ { s } ^ { j } ( e _ { i } ) } { \tau _ { e } } } \right\} _ { i = 1 } ^ { | \tilde { \mathcal { E } } | } \right) ,\tag{9}
$$

where $s _ { p } ( e _ { i } )$ is the priority score of exemplar $e _ { i }$ and $s _ { s } ^ { j } ( e _ { i } )$ is its semantic similarity to the question $q _ { j } , \tilde { \mathcal { E } }$ represents the stored exemplars, and $\tau _ { e }$ is the temperature. We then randomly sample five exemplars as variable content $p _ { R } ( q _ { j } )$ of prompt. During the inference stage, we select the five exemplars with the highest $s _ { p } ( e _ { i } ) \cdot s _ { s } ^ { j } ( e _ { i } )$ as variable content $p _ { R } ( q _ { j } )$ of prompt for more accurate predictions.

Exemplar Forgetting Updating We adjust the priority scores of exemplars based on whether incorporating them as the variable content $p _ { R } ( q _ { j } )$ in the prompt leads to improvements. Exemplars with low priority scores are promptly removed to ensure that only valuable ones are stored.

## 4 Experiments

Datasets. We perform evaluation on 7 standard datasets : WSC (Levesque et al., 2012), Ethos (Mollas et al., 2022), ArSarcasm (Farha and Magdy, 2020), Liar (Wang, 2017), BBH-navigate (Suzgun et al., 2022), GSM8k (Cobbe et al., 2021), WebNLG (Gardent et al., 2017). Among these, Ar-Sarcasm, Ethos, and Liar, and BBH-navigate contain true/false questions, WSC contains multiplechoice questions, GSM8K contains questions with integer answers, and WebNLG contains questions requiring natural language generation.

<table><tr><td rowspan="2">Method</td><td colspan="4">True / False</td><td colspan="2">Generative</td><td>Multiple-choice</td></tr><tr><td>LIAR</td><td></td><td></td><td></td><td>BBH ETHOS ArSarcasm WebNLG GSM8K</td><td></td><td>WSC</td></tr><tr><td></td><td>(F1)</td><td>(F1)</td><td>(F1)</td><td>(F1)</td><td>(Rouge-L)</td><td>(Acc.)</td><td>(Acc.)</td></tr><tr><td>Empty</td><td>46.4</td><td>69.4</td><td>93.0</td><td>83.7</td><td>49.4</td><td>89.0</td><td>77.3</td></tr><tr><td>CoT (Kojima et al., 2022)</td><td>46.0</td><td>81.9</td><td>84.5</td><td>83.7</td><td>49.3</td><td>89.0</td><td>81.3</td></tr><tr><td>APE (Zhou et al., 2022)</td><td>47.7</td><td>72.9</td><td>94.0</td><td>83.8</td><td>51.3</td><td>91.3</td><td>79.3</td></tr><tr><td>ProTeGi (Pryzant et al., 2023)</td><td>58.5</td><td>73.6</td><td>96.5</td><td>84.1</td><td>55.7</td><td>91.0</td><td>80.0</td></tr><tr><td>OPRO (Yang et al., 2024)</td><td>47.9</td><td>75.7</td><td>93.5</td><td>84.5</td><td>51.9</td><td>90.7</td><td>83.3</td></tr><tr><td>Promptbreeder (Fernando et al., 2024)</td><td>47.1</td><td>74.3</td><td>94.5 93.0</td><td>83.8</td><td>51.0</td><td>91.7</td><td>80.0</td></tr><tr><td>EvoPrompt (Guo et al., 2024)</td><td>47.9 54.7</td><td>75.0 70.8</td><td>94.0</td><td>83.8 83.6</td><td>50.2 51.8</td><td>90.7 90.3</td><td>78.8</td></tr><tr><td>GPO (Tang et al., 2024)</td><td></td><td></td><td></td><td></td><td></td><td></td><td>84.0</td></tr><tr><td>ERM</td><td>68.6</td><td>86.1</td><td>98.0</td><td>85.1</td><td>59.6</td><td>93.3</td><td>86.0</td></tr><tr><td>∆</td><td>+10.1</td><td>+4.2</td><td>+1.5</td><td>+0.6</td><td>+3.9</td><td>+1.6</td><td>+2.0</td></tr></table>

Table 1: Comparisons of our method with existing LLM-based prompt optimizers under zero-shot setting.
<table><tr><td>Method</td><td>Prompt</td><td>Rouge-L</td></tr><tr><td>Empty ProTeGi</td><td>Write the following triples as fluent English text.</td><td>49.4 55.7</td></tr><tr><td></td><td>You are given a set of triples that need to be converted into coherent and fluent English sentences. Each triple consists of a subject, predicate, and object. Your task is to accurately convey the information from these triples into well-formed sentences. Ensure the sentences are complete, grammatically correct, and clearly express the relationships provided in the triples.</td><td></td></tr><tr><td>OPRO</td><td>Convert the following sets of triples into coherent, natural, and fluent English sentences.</td><td>51.9</td></tr><tr><td>PromptBreeder</td><td>Transform these triples into smooth and stylish English sentences, and make them shine!</td><td>50.9</td></tr><tr><td>EvoPrompt</td><td>Turn the provided triples into smooth, flowing English sentences that will impress everyone!</td><td>50.2</td></tr><tr><td>GPO ERM</td><td>Rewrite these triples into fluent and natural English sentences. Convert the following triples into coherent and fluent English sentences. Ensure that all</td><td>51.8 59.6</td></tr><tr><td></td><td>relationships and attributes are accurately conveyed. When multiple associations or attributes are involved, break down the information into smaller, logical sentences to maintain clarity.</td><td></td></tr></table>

Table 2: Prompts optimized by different methods on the WebNLG dataset.

Baselines. We compare several representative methods, including existing LLM-based prompt optimizers: APE (Zhou et al., 2022), Pro-TeGi (Pryzant et al., 2023), OPRO (Yang et al., 2024), Promptbreeder (Fernando et al., 2024), Evo-Prompt (Guo et al., 2024), and GPO (Tang et al., 2024). In addition, we consider the baseline using manually written simple prompts (“Manual”), which we provide in the appendix, and the instruction “Let’s think step by step.” from chain-ofthought prompting (“CoT”) (Kojima et al., 2022) for performance comparison.

Evaluation Metrics. We report the F1 score on Ethos, ArSarcasm, Liar and BBH-navigate following (Pryzant et al., 2023), accuracy on WSC and GSM8k following (Tang et al., 2024; Juneja et al., 2024) and ROUGE-L on WebNLG following (Tang et al., 2024).

Implementation Details. For the task model, we use Doubao-Pro (ByteDance, 2024). For the prompt optimizer, we use GPT-4o (OpenAI, 2024).

We repeat all the experiments three times and report the average of the results. Other details are presented in appendix.

## 4.1 Main Results

Comparison under Zero-shot Setting. Table 1 presents the results of different methods for prompt optimization across true/false questions, generative questions, and multiple-choice questions.

For true/false questions, our method demonstrates a significant improvement over previous works. Specifically, our method outperforms trajectory-based methods (OPRO and GPO) by 13.9. Trajectory-based methods utilize an LLM prompt optimizer to generate new prompts based on historical prompts, scores, or error examples, but may struggle to identify “better prompts”, limiting their performance. Our method also outperforms ProTeGi (feedback-based method) by 10.1, which can be attributed to our exemplar-guided reflection, feedback memory and example factory.

![](images/15eef15fb0530620b39b68da8fabe05fde55a17144e639f5f54765fee803315c.jpg)

Figure 3: The efficiency of our approach ERM. The size of the circles represents performance, with larger circles indicating better performance. The vertical axis shows the optimization steps needed for different methods to achieve peak performance across datasets.
<table><tr><td rowspan="2">Method</td><td colspan="4">True / False</td><td colspan="2">Generative</td><td>Multiple-choice</td></tr><tr><td>LIAR (F1)</td><td>(F1)</td><td>(F1)</td><td>(F1)</td><td>BBH ETHOS ArSarcasm WebNLG GSM8K (Rouge-L)</td><td>(Acc.)</td><td>WSC (Acc.)</td></tr><tr><td>APE (Zhou et al., 2022)</td><td>51.2</td><td>74.3</td><td>93.2</td><td>84.3</td><td>53.1</td><td>91.8</td><td>80.3</td></tr><tr><td>ProTeGi (Pryzant et al., 2023)</td><td>60.3</td><td>73.6</td><td>97.0</td><td>84.1</td><td>56.3</td><td>91.0</td><td>81.0</td></tr><tr><td>OPRO (Yang et al., 2024)</td><td>52.1</td><td>75.0</td><td>94.8</td><td>84.7</td><td>52.4</td><td>90.8</td><td>85.0</td></tr><tr><td>Promptbreeder (Fernando et al., 2024)</td><td>51.8</td><td>75.7</td><td>95.7</td><td>84.5</td><td>52.7</td><td>91.7</td><td>81.5</td></tr><tr><td>EvoPrompt (Guo et al., 2024)</td><td>52.3</td><td>76.4</td><td>94.3</td><td>83.9</td><td>51.8</td><td>90.9</td><td>80.4</td></tr><tr><td>GPO (Tang et al., 2024)</td><td>56.6</td><td>75.0</td><td>95.5</td><td>83.8</td><td>53.4</td><td>90.5</td><td>84.9</td></tr><tr><td>ERM</td><td>68.6</td><td>86.1</td><td>98.0</td><td>85.1</td><td>59.6</td><td>93.3</td><td>86.0</td></tr></table>

Table 3: Comparisons of our method with existing LLM-based prompt optimizers under few-shot setting.
<table><tr><td>Exemplar-Guided Reflection</td><td>Feedback Memory Exemplar Factory</td><td></td><td>LIAR BBH (F1) (F1)</td><td>(F1)</td><td></td><td>(F1)</td><td>ETHOS ArSarcasm WebNLG GSM8K (Rouge-L)</td><td>(Acc.)</td><td>WSC (Acc.)</td></tr><tr><td></td><td></td><td></td><td>58.5</td><td>73.6</td><td>96.5</td><td>84.1</td><td>55.7</td><td>91.0</td><td>80.0</td></tr><tr><td>√</td><td></td><td></td><td>62.9</td><td>75.7</td><td>97.0</td><td>84.2</td><td>56.9</td><td>92.7</td><td>82.0</td></tr><tr><td>√</td><td></td><td>√</td><td>67.2</td><td>84.7</td><td>97.0</td><td>84.9</td><td>58.6</td><td>93.0</td><td>84.0</td></tr><tr><td>√</td><td>√</td><td></td><td>66.6</td><td>82.6</td><td>97.5</td><td>84.8</td><td>58.8</td><td>93.0</td><td>85.0</td></tr><tr><td>√</td><td>√</td><td>√</td><td>68.6</td><td>86.1</td><td>98.0</td><td>85.1</td><td>59.6</td><td>93.3</td><td>86.0</td></tr></table>

Table 4: Effect of each component in our method.

For generative questions and multiple-choice questions, our method also significantly outperforms previous methods. Specifically, on the WebNLG dataset, our approach surpasses previous methods by 3.9 in Rouge-L score. Table 2 visualizes the optimized prompts on the WebNLG dataset, demonstrating that our method’s optimized prompts are more effective at capturing the critical information needed to enhance task performance. The exemplar factory boosts the F1 score by 3.7 on the LIAR dataset, while the feedback memory provides an improvement of 2.0.

Efficiency of Our Method. Our approach introduces a memory mechanism to efficiently store and utilize feedbacks. We show the optimization steps needed for different methods to achieve peak performance across datasets in Figure 3, which highlights the superior efficiency of our method. Specifically, according to Figure 3(a), on the LIAR dataset, our method reaches an F1 score of 68.6 by the 7th step, while ProTeGi only achieves 58.5 by the 13th step, demonstrating that our method nearly doubles the optimization speed.

Comparison under Few-shot Setting. Table 3 presents a comparison between our method and others under few-shot settings. For each approach, we dynamically select five relevant examples through k-nearest neighbors (kNN) clustering in the embedding space. According to the results, ERM consistently outperforms the previous methods. Notably, on the LIAR dataset, our approach achieves an 8.3 F1 score improvement over previous methods, demonstrating the effectiveness of selecting valuable wrong examples as exemplars and equipping them with chain-of-thought-like solution processes.

<table><tr><td>Retrieval</td><td>Exemplar Filtering</td><td>Selective Forget.</td><td>LIAR (F1)</td><td>BBH (F1)</td><td>WebNLG (Rouge-L)</td></tr><tr><td></td><td></td><td></td><td>62.9</td><td>75.7</td><td>56.9</td></tr><tr><td>√</td><td></td><td></td><td>62.3</td><td>75.0</td><td>57.0</td></tr><tr><td>√</td><td>√</td><td></td><td>65.7</td><td>81.3</td><td>58.4</td></tr><tr><td>√</td><td>√</td><td>√</td><td>66.6</td><td>82.6</td><td>58.8</td></tr></table>

Table 5: Effect of each component in Exemplar Factory.
<table><tr><td>Retrieval</td><td>Feedback Filtering</td><td>Selective Forget.</td><td>LIAR (F1)</td><td>BBH (F1)</td><td>WebNLG (Rouge-L)</td></tr><tr><td></td><td></td><td></td><td>66.6</td><td>82.6</td><td>58.8</td></tr><tr><td>√</td><td></td><td></td><td>66.4</td><td>81.9</td><td>58.8</td></tr><tr><td>√</td><td>√</td><td></td><td>67.5</td><td>82.6</td><td>59.2</td></tr><tr><td>√</td><td>√</td><td>√</td><td>68.6</td><td>86.1</td><td>59.6</td></tr></table>

Table 6: Effect of each component in Feedback Memory.

## 4.2 Ablation Study

Effect of Each Component. In Table 4, we conduct experiments to verify the effectiveness of each key component in our method.

We adopt a strategy which dentify exemplars, contemplate the corresponding chain of thought and then complete feedbacks, and observe that ERM improves the F1 score by 4.4 on the LIAR dataset compared with the approaches without the instructive meta-prompt, which validates the effectiveness of the instructive meta-prompt. Additionally, the introduction of the memory mechanism for feedback memory and exemplar factory brought a further 5.7 improvement on the LIAR dataset, confirming the effectiveness of the memory mechanism.

Effect of Exemplar Factory. As shown in Table 5, incorporating exemplar filtering when storing exemplars does not enhance performance. This is because the behavior of the prompt optimizer is unpredictable and may generate incorrect or unconventional questions. Retrieving such examples does not enhance prediction performance. However, filtering out erroneously generated exemplars and redundant ones already in storage resulted in a 3.4 improvement, highlighting the importance of exemplar filtering. The introduction of a selective forgetting further improved the F1 score by 0.9 on the LIAR dataset, as it removes exemplars that do not aid in prediction, thereby enhancing performance.

Effect of Feedback Memory. As shown in Table 6, directly storing feedbacks for periodic optimization without the feedback filtering strategy does not improve performance. Introducing the filtering strategy increased the F1 score on the LIAR dataset by 0.9 compared to not using stored feedbacks. Additionally, incorporating selective forgetting, which discards suboptimal feedback promptly, further enhanced the F1 score by an additional 0.9.

## 5 Conclusion

In this paper, we introduce Exemplar-Guided Reflection with Memory mechanism (ERM), a novel approach to achieve efficient and accurate prompt optimization. Using a instructive reflection meta-prompt, ERM instructs LLMs to select exemplars with detailed solution processes and generate stronger feedback. We then propose Feedback Memory mechanism to efficiently exploit potentially valuable feedback. Additionally, Exemplar Factory is introduced to further enhance the accuracy of prediction by pre-assessing the impact on the task. ERM refines prompts authored by human experts and outperforms established automatic prompt engineering baselines across various scenarios, with optimization steps approximately half of that in previous work.

## 6 Limitations

In this work, we effectively utilize feedbacks and exemplars using a long-term memory mechanism. However, in real-world applications, we encounter additional challenges: some questions continue to be incorrectly answered during the optimization process, and prompt optimization doesn’t always align with human expectations. When the model struggles to optimize, introducing human intervention might aid in enhancing prompt optimization. This paper lacks exploration on how humans could assist in the optimization process. For instance, with persistent incorrect answers, human input could offer crafted solutions, helping the expert model generate improved feedback. Additionally, due to computational and budget constraints, our experiments are limited to representative tasks.

## References

Derek Austin and Elliott Chartock. 2024. Gradsum: Leveraging gradient summarization for optimal prompt engineering. arXiv preprint arXiv:2407.12865.

ByteDance. 2024. Doubao.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024a. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216.

Yongchao Chen, Jacob Arkin, Yilun Hao, Yang Zhang, Nicholas Roy, and Chuchu Fan. 2024b. Prompt optimization in multi-step tasks (promst): Integrating human feedback and heuristic-based sampling. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 3859–3920.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, and 1 others. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Nicholas Crispino, Kyle Montgomery, Fankun Zeng, Dawn Song, and Chenguang Wang. 2023. Agent instructs large language models to be general zeroshot reasoners. arXiv preprint arXiv:2310.03710.

Mingkai Deng, Jianyu Wang, Cheng-Ping Hsieh, Yihan Wang, Han Guo, Tianmin Shu, Meng Song, Eric P Xing, and Zhiting Hu. 2022. Rlprompt: Optimizing discrete text prompts with reinforcement learning. arXiv preprint arXiv:2205.12548.

Hermann Ebbinghaus. 2013. Memory: A contribution to experimental psychology. Annals ofneurosciences, 20(4):155.

Ling Fan, Harry Jiannan Wang, Kunpeng Zhang, Zilong Pei, and Anjun Li. 2023. Towards an automatic prompt optimization framework for ai image generation. In International Conference on Human-Computer Interaction, pages 405–410. Springer.

Ibrahim Abu Farha and Walid Magdy. 2020. From arabic sentiment analysis to sarcasm detection: The arsarcasm dataset. In Proceedings of the 4th Workshop on Open-Source Arabic Corpora and Processing Tools, with a Shared Task on Offensive Language Detection, pages 32–39.

Chrisantha Fernando, Dylan Sunil Banarse, Henryk Michalewski, Simon Osindero, and Tim Rocktäschel. 2024. Promptbreeder: Self-referential selfimprovement via prompt evolution. In Forty-first International Conference on Machine Learning.

Claire Gardent, Anastasia Shimorina, Shashi Narayan, and Laura Perez-Beltrachini. 2017. Creating training corpora for nlg micro-planning. In 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, pages 179–188. Association for Computational Linguistics (ACL).

Qingyan Guo, Rui Wang, Junliang Guo, Bei Li, Kaitao Song, Xu Tan, Guoqing Liu, Jiang Bian, and Yujiu Yang. 2024. Connecting large language models with evolutionary algorithms yields powerful prompt optimizers. In The Twelfth International Conference on Learning Representations.

Yaru Hao, Zewen Chi, Li Dong, and Furu Wei. 2024. Optimizing prompts for text-to-image generation. Advances in Neural Information Processing Systems, 36.

John H Holland. 1992. Genetic algorithms. Scientific american, 267(1):66–73.

Xinyu Hu, Pengfei Tang, Simiao Zuo, Zihan Wang, Bowen Song, Qiang Lou, Jian Jiao, and Denis Charles. 2023. Evoke: Evoking critical thinking abilities in llms via reviewer-author prompt editing. arXiv preprint arXiv:2310.13855.

Gurusha Juneja, Nagarajan Natarajan, Hua Li, Jian Jiao, and Amit Sharma. 2024. Task facet learning: A structured approach to prompt optimization. arXiv preprint arXiv:2406.10504.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199– 22213.

Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691.

Hector Levesque, Ernest Davis, and Leora Morgenstern. 2012. The winograd schema challenge. In Thirteenth international conference on the principles of knowledge representation and reasoning.

WeiJie Li, Jin Wang, and Xuejie Zhang. 2024. Promptist: Automated prompt optimization for text-toimage synthesis. In CCF International Conference on Natural Language Processing and Chinese Computing, pages 295–306. Springer.

Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190.

Yujian Betterest Li and Kai Wu. 2023. Spell: Semantic prompt evolution based on a llm. arXiv preprint arXiv:2310.01260.

Shihong Liu, Samuel Yu, Zhiqiu Lin, Deepak Pathak, and Deva Ramanan. 2024. Language models as black-box optimizers for vision-language models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12687– 12697.

Ruotian Ma, Xiaolei Wang, Xin Zhou, Jian Li, Nan Du, Tao Gui, Qi Zhang, and Xuanjing Huang. 2024. Are large language models good prompt optimizers? arXiv preprint arXiv:2402.02101.

Oscar Mañas, Pietro Astolfi, Melissa Hall, Candace Ross, Jack Urbanek, Adina Williams, Aishwarya Agrawal, Adriana Romero-Soriano, and Michal Drozdzal. 2024. Improving text-to-image consistency via automatic prompt optimization. arXiv preprint arXiv:2403.17804.

Wenyi Mo, Tianyu Zhang, Yalong Bai, Bing Su, Ji-Rong Wen, and Qing Yang. 2024. Dynamic prompt optimizing for text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26627–26636.

Ioannis Mollas, Zoe Chrysopoulou, Stamatis Karlos, and Grigorios Tsoumakas. 2022. Ethos: a multi-label hate speech detection dataset. Complex & Intelligent Systems, 8(6):4663–4678.

OpenAI. 2022. Chatgpt.

OpenAI. 2024. Gpt-4o.

Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, and Michael Zeng. 2023. Automatic prompt optimization with" gradient descent" and beam search. In The 2023 Conference on Empirical Methods in Natural Language Processing.

Taylor Shin, Yasaman Razeghi, Robert L Logan IV, Eric Wallace, and Sameer Singh. 2020. Autoprompt: Eliciting knowledge from language models with automatically generated prompts. arXiv preprint arXiv:2010.15980.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, and 1 others. 2022. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615.

Rainer Storn and Kenneth Price. 1997. Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces. Journal of global optimization, 11:341–359.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, and 1 others. 2022. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261.

Xinyu Tang, Xiaolei Wang, Wayne Xin Zhao, Siyuan Lu, Yaliang Li, and Ji-Rong Wen. 2024. Unleashing the potential of large language models as prompt optimizers: An analogical analysis with gradient-based model optimizers. arXiv preprint arXiv:2402.17564.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. 2023. Stanford alpaca: An instruction-following llama model.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, and 1 others. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Soobin Um and Jong Chul Ye. 2024. Minorityprompt: Text to minority image generation via prompt optimization. arXiv preprint arXiv:2410.07838.

William Yang Wang. 2017. " liar, liar pants on fire": A new benchmark dataset for fake news detection. arXiv preprint arXiv:1705.00648.

Zongyu Wu, Hongcheng Gao, Yueze Wang, Xiang Zhang, and Suhang Wang. 2024. Universal prompt optimizer for safe text-to-image generation. arXiv preprint arXiv:2402.10882.

Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, and Xinyun Chen. 2024. Large language models as optimizers. Preprint, arXiv:2309.03409.

Qinyuan Ye, Maxamed Axmed, Reid Pryzant, and Fereshte Khani. 2024. Prompt engineering a prompt engineer. In Findings ofthe Associationfor Computational Linguistics: ACL 2024.

Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, and 1 others. 2022. Glm-130b: An open bilingual pre-trained model. arXiv preprint arXiv:2210.02414.

Tianjun Zhang, Xuezhi Wang, Denny Zhou, Dale Schuurmans, and Joseph E Gonzalez. 2022. Tempera: Test-time prompting via reinforcement learning. arXiv preprint arXiv:2211.11890.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2024. Memorybank: Enhancing large language models with long-term memory. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pages 19724–19731.

Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. 2022. Large language models are human-level prompt engineers. arXiv preprint arXiv:2211.01910.

Dongsheng Zhu, Daniel Tang, Weidong Han, Jinghui Lu, Yukun Zhao, Guoliang Xing, Junfeng Wang, and Dawei Yin. 2024. Vislinginstruct: Elevating zeroshot learning in multi-modal language models with autonomous instruction optimization. In Proceedings ofthe 2024 Conference ofthe North American Chapter of the Association for Computational Linguistics, pages 2122–2135.

<table><tr><td>Dataset Name</td><td>Task</td><td>Train &amp; Dev</td><td>Test</td></tr><tr><td>LIAR (Wang, 2017)</td><td>True/False</td><td>3681</td><td>461</td></tr><tr><td>BBH-Navigate (Suzgun et al., 2022)</td><td>True/False</td><td>96</td><td>144</td></tr><tr><td>ETHOS (Mollas et al., 2022)</td><td>True/False</td><td>440</td><td>200</td></tr><tr><td>ArSarcasm (Farha and Magdy, 2020)</td><td>True/False</td><td>8437</td><td>2110</td></tr><tr><td>WebNLG (Gardent et al., 2017)</td><td>Language Generation</td><td>200</td><td>300</td></tr><tr><td>GSM8K (Cobbe et al., 2021)</td><td>Integer Generation</td><td>200</td><td>300</td></tr><tr><td>WSC (Levesque et al., 2012)</td><td>Multiple-Choice</td><td>100</td><td>150</td></tr></table>

Table 7: Dataset sizes and data splits.
<table><tr><td>Dataset</td><td>License</td><td>Source</td></tr><tr><td>LIAR (Wang, 2017)</td><td>Unknown</td><td>https://www.cs.ucsb.edu/~cwilliam/data/liar_dataset.zip</td></tr><tr><td>BIG-bench Hard (Suzgun et al., 2022) Apache-2.0</td><td></td><td>https://github.com/google/BIG-bench(original) https://github.com/suzgunmirac/BIG-Bench-Hard(reformatted)</td></tr><tr><td>ETHOS (Mollas et al., 2022)</td><td></td><td>GNU GPLv3 https://huggingface.co/datasets/iamollas/ethos</td></tr><tr><td>ArSarcasm (Farha and Magdy, 2020)</td><td>MIT</td><td>https://github.com/iabufarha/ArSarcasm</td></tr><tr><td>WebNLG (Gardent et al., 2017)</td><td>CC BY 4.0</td><td>https://github.com/fuzihaofzh/webnlg-dataset</td></tr><tr><td>GSM8K (Cobbe et al., 2021)</td><td>MIT</td><td>https://github.com/openai/grade-school-math</td></tr><tr><td>WSC (Levesque et al., 2012)</td><td>CC BY 4.0</td><td>https://huggingface.co/datasets/ErnestSDavis/winograd_wsc</td></tr></table>

Table 8: License and Source of the datasets used in this study.

## A Additional Details for the Setup

## A.1 Tasks and Data Details

We present a summary of the dataset sizes and data split information in Table 7. Table 8 provides details on the sources and licensing information of the datasets. To the best of our knowledge, our usage of these datasets aligns with their intended purposes, and the data we utilize do not contain any personal or sensitive information.

LIAR (Wang, 2017) is an English fake news detection corpus comprising 4,000 statements, each accompanied by context and lie labels. For our experiments, we adopt the same dataset split as ProTeGi (Pryzant et al., 2023), utilizing 3,681 instances for training and 461 instances for testing.

BIG-bench Hard dataset (Suzgun et al., 2022) is a subset of the BIG Bench dataset (Srivastava et al., 2022), comprising 23 tasks that present significant challenges for current language models. For our experiments, we select the navigation task, which requires determining whether an agent, following a series of navigation steps, returns to its initial starting point. Consistent with the dataset split used by GPO (Tang et al., 2024), we employ 96 instances for training and 144 instances for testing. ETHOS (Mollas et al., 2022) is an English hate speech detection dataset consisting of 997 online comments, each annotated with hate speech labels. In accordance with previous research, we utilize the same dataset split, employing 440 instances for training and 200 instances for testing.

ArSarcasm dataset (Farha and Magdy, 2020) is an Arabic sarcasm detection corpus containing 10,000 online comments, each labeled for sarcasm. We utilize the original dataset split, with 8,437 instances designated for training and 2,110 instances for testing.

WebNLG corpus consists of sets of triplets that describe facts—entities and the relations between them—paired with their corresponding expressions in natural language text. Following the dataset split used by GPO (Tang et al., 2024), we utilize 200 instances for training and 300 instances for testing. GSM8K (Cobbe et al., 2021) comprises 8.5K highquality linguistically diverse grade school math word problems, crafted by human problem writers. Following the dataset split used by GPO (Tang et al., 2024), we utilize 200 instances for training and 300 instances for testing.

WSC was introduced both as an alternative to the Turing Test and as a measure of a system’s ability to perform commonsense reasoning. Following the approach used by GPO (Tang et al., 2024), we sample 100 examples for the training set and 150 for the test set.

## A.2 Implementation Details

We select Doubao-pro (ByteDance, 2024) as the task model and set its temperature to 0, ensuring deterministic outputs following the GPO (Tang et al., 2024) and AgentInstruct (Crispino et al., 2023). For the prompt optimizer, we utilize gpt-4o-2024- 05-13, the underlying model of GPT-4o (OpenAI,

2024). Its temperature is set to 1.0 to promote diverse generation. The initial prompts for different tasks can be found in Section F. In each step, the optimizer generates 8 candidate task prompts. Following GPO (Tang et al., 2024) and OPRO (Yang et al., 2024), the best-performing one is selected as the task prompt for the next iteration. All experiments are conducted three times, and we report the average results.

## B More Related Work

Prompt engineering aims to identify suitable prompts as inputs for large language models (LLMs) to perform various tasks. To reduce human effort, researchers have explored automatic prompt optimization (Lester et al., 2021; Shin et al., 2020; Li and Liang, 2021).

Continuous approaches (Lester et al., 2021; Shin et al., 2020; Li and Liang, 2021) optimize within the embedding space of LLMs and update based on backpropagating gradients. Prefix tuning (Li and Liang, 2021) introduces new learnable tokens that can be considered as prompts in continuous space, which are learned for specific tasks. However, since these tokens are defined in continuous space, they are not easily interpretable, and these methods require access to model weights, making them unsuitable for use with closed-source LLMs like ChatGPT (OpenAI, 2022).

Discrete methods (Deng et al., 2022; Zhang et al., 2022) directly optimize natural language prompts. Several strategies have been developed for this purpose. Some approaches (Pryzant et al., 2023; Juneja et al., 2024) optimize prompts based on error feedback, while others (Yang et al., 2024; Tang et al., 2024) utilize multiple prompts and their respective scores to enable the model to identify superior prompts. Additionally, certain methods (Guo et al., 2024; Fernando et al., 2024; Li and Wu, 2023) employ genetic algorithms to rewrite prompts through processes of variation and natural selection. Furthermore, some methods (Ye et al., 2024; Ma et al., 2024) enhance the controllability of feedback generation and prompt optimization by modifying meta-prompts. To improve the accuracy of error summaries, some works (Juneja et al., 2024; Austin and Chartock, 2024) cluster similar erroneous samples instead of using randomly selected ones.

Recently, automatic prompt optimization has also been explored in the context of multi-step tasks (Chen et al., 2024b) and multi-modality tasks (Zhu et al., 2024; Fan et al., 2023; Hao et al., 2024; Um and Ye, 2024; Mo et al., 2024; Wu et al., 2024; Li et al., 2024). PROMST (Chen et al., 2024b) optimizes multi-step tasks by introducing human-designed feedbacks and a score prediction model. VisLingInstruct (Zhu et al., 2024) autonomously evaluates and optimizes instructional texts through in-context learning, improving the synergy between visual perception and linguistic expression in multi-modal language models. Liu (Liu et al., 2024) enables the language model to rewrite refined prompts based on the scores of historical prompts on image classification datasets. OPT2I (Mañas et al., 2024) uses a method similar to OPRO, where the language model rewrites better prompts based on the history of prompts and their generated image consistency objective scores.

<table><tr><td>Method</td><td>APE</td><td>OPRO</td><td>GPO</td><td>ProTeGi</td><td>ERM</td></tr><tr><td>Feedback-based</td><td>x</td><td>x</td><td>x</td><td>√</td><td>√</td></tr><tr><td>F1</td><td>47.7</td><td>47.9</td><td>54.7</td><td>58.5</td><td>68.6</td></tr><tr><td>Total. Tokens</td><td>7415</td><td>8652</td><td>7360</td><td>12836</td><td>7503</td></tr></table>

Table 9: Total consumed tokens comparison of our method with existing LLM-based prompt optimizers.

## C More Experiments

Total Consumed Tokens Comparison. As shown in Table 9, we compare the total number of tokens consumed for different methods in the LIAR dataset. Our method belongs to the feedback-based prompt optimization category. Compared with Pro-TeGi, our method can efficiently exploit feedbacks with the Feedback Factory, allowing us to optimize with fewer steps and significantly reducing the total number of consumed tokens. Non-feedbackbased methods do not require feedback, so the average number of tokens per step is about half that in feedback-based methods. However, since the Feedback Factory can efficiently use feedbacks to reduce optimization steps significantly, the total number of consumed tokens of our method is comparable to those non-feedback-based methods.

Effect of Exemplars’ Solutions. As shown in Table 10, we compared direct retrieval for prediction on the training set and found that using exemplars yields better results. This is because (1) the Exemplar Factory pre-assesses exemplars for their effectiveness on the task, filtering useful ones, and (2) the prompt optimizer crafts chain-of-thought answers tailored to the questions, enhancing prediction accuracy.

<table><tr><td>Method</td><td>LIAR</td><td>BBH</td><td>WebNLG</td></tr><tr><td>Zero-shot</td><td>62.9</td><td>75.7</td><td>56.9</td></tr><tr><td>Five relevant examples</td><td>65.7</td><td>78.5</td><td>57.4</td></tr><tr><td>Ours</td><td>66.6</td><td>82.6</td><td>58.8</td></tr></table>

Table 10: Comparison of our method and dynamically selecting five relevant examples using k-nearest neighbors (kNN) clustering in the embedding space.

<table><tr><td rowspan="2">Stage</td><td colspan="2">Prompt optimizing</td><td colspan="2">Inference</td></tr><tr><td>LLM API</td><td>Memory Mechanisms</td><td>LLM API</td><td>Exemplar Retrieval</td></tr><tr><td>time (s)</td><td>28.161</td><td>0.062</td><td>1.201</td><td>0.029</td></tr></table>

Table 11: Computational overhead introduced by the memory mechanisms and exemplar retrieval.

Computational Overhead of Each Component in ERM. (1) During the prompt optimization phase, the optimization time mainly comes from the LLM API calls and the memory mechanisms’ overhead. We measure the average time distribution for each prompt optimization. As shown in Table 11, the average time for each optimization LLM API call is 28.161 seconds, while the execution time for the memory mechanisms is 0.062 seconds (we run BGE-M3 on A800 GPU). Since memory mechanisms primarily involve the extraction and retrieval of BGE-M3 features, the cost is low, and the execution time overhead is less than 1%, which can be considered negligible. (2) During the inference stage, the cost time mainly comes from the LLM API calls and the exemplar retrieval overhead. We measure the average time distribution for each prediction stage. As shown in Table 11, the average time for LLM API calls is 1.201 seconds, and the average time for exemplar retrieval is 0.029 seconds. The time overhead for exemplar retrieval is also negligible.

Case Studies. In Tables 12, 13, and 14, we present examples of prompt optimization with Feedback Memory on the LIAR, BBH, and GSM8K datasets, respectively. Feedback memory effectively combines multiple feedbacks and utilizes them to optimize prompts, preventing the loss of valuable feedbacks during the optimization process. This approach results in a stronger optimized prompt.

## D More Discussion

Potential Benefit of Memory Mechanism. In practical applications, prompt optimization might not always generate a prompt that meets human expectations. Thanks to our proposed memory mechanism, we can easily incorporate human intervention. Specifically, we have explored the following two aspects:

1) When the model struggles with a particular exemplar and fails to resolve it successfully, we can trigger human intervention to verify whether the answer for that exemplar is correct, thereby correcting noisy labels or considering the addition of a chain of thought for that exemplar.

2) When the prompt fails to achieve the desired effect, we can introduce human feedbacks to improve prompt optimization.

Analysis about Performance Improvements. The performance improvement of our method stems from both indirect feedbacks and direct exemplars.

1) Exemplar-guided Reflection enables the prompt optimizer to generate strong feedbacks. The Feedback Memory collects potentially valuable feedbacks, avoiding the issue of feedback forgetting that may occur with sequential optimization of multiple feedbacks. By considering multiple feedbacks simultaneously, it enhances the language model’s ability to retain information, leading to the generation of better prompts and therefore indirectly boosting performance.

2) Our proposed Exemplar Factory identifies exemplars that are more targeted to specific questions, therefore directly enhancing the prediction performance.

## E Meta-Prompt

Here are the meta-prompts we used in Section 3.

## F Additional Result

Here, we present the initial prompt, the ProTeGioptimized prompt, and ERM-optimized prompt across different tasks.

![](images/d34f2dee2d6909c4a9e2b8f6e1556ba2ade072400368fa8f2de18c89cf21ed69.jpg)  
Table 12: Intermediate prompt optimized by Feedback Memory on the LIAR dataset.

![](images/4580e752397ff27f81c2caa52983da1a4856174cc5d562fe94a74f3781c82709.jpg)  
Table 13: Intermediate prompt optimized by Feedback Memory on the BBH dataset.

![](images/690eed918c0561a8390d36f1197ad1d6ba977af000b0a511763a0b45201097ca.jpg)  
Table 14: Intermediate prompt optimized by Feedback Memory on the GSM8K dataset.

![](images/d4436aee0440e1924f1a3877a92ed8f1d09aa20062ec216c11a47e795c164e04.jpg)  
Figure 4: Intructive reflection meta-prompt.

![](images/f966f523900e31fd3f73462d3f4f5ab00f47290cfdd4b87367961a2b72b86241.jpg)  
Figure 5: Optimization meta-prompt.

![](images/d6c6c884618c9be71febeff524302bb9cfafa500fd198493947e900f86f34af3.jpg)  
Figure 6: Retrieval optimization meta-prompt.

![](images/06220c1d11c5d00c48c6ab8abb96b265a1c455a7a37c146e8bdfc6ebb3e68bbb.jpg)  
Figure 7: Initial prompt of the LIAR dataset.

![](images/5ffda0a73596871c73365fe8a809aac7c3a37efdad3d6afd142fd3012804980c.jpg)  
Figure 8: ProTeGi optimized prompt of the LIAR dataset.

![](images/29cd0b540d242ec57c734c93f0cbcbe7c074b7a37cf68159b8002a03137655d0.jpg)  
Figure 9: ERM optimized prompt of the LIAR dataset.

![](images/044cd275ffef1a95dc89388171669f160f4d742bbfb7a28b9169bc7adbac71fa.jpg)  
Figure 10: Initial prompt of the BBH dataset.

![](images/8ca11bc6aa8063de03a7fcf7c8c1019d9ad1fcc6feefb2a9d78d580a9d977e42.jpg)  
Figure 11: ProTeGi optimized prompt of the BBH dataset.

![](images/8a340281a60f03096c6f2173adcc6ccbb8103337183a9a75b98cdb7d2a0d6dbf.jpg)  
Figure 12: ERM optimized prompt of the BBH dataset.

![](images/392d5d4a579e547125110a300934c845a0cae8b7e521ab432a840ac9c99c8682.jpg)  
Figure 13: Initial prompt of the ETHOS dataset.

![](images/933a555c2c53b75378ea3d3fd21482ca35de1d7ec2e4f3de1c2db1d1278375ef.jpg)  
Figure 14: ProTeGi optimized prompt of the ETHOS dataset.

![](images/0a7f31e1cceeb5bdbee1030a33660cc285743e1727ba72ab4536ef20c6310fd7.jpg)  
Figure 15: ERM optimized prompt of the ETHOS dataset.

![](images/3f1b397722185a5067f388f208c96aab2b4d08e6baab419f01858081805aded6.jpg)  
Figure 16: Initial prompt of the ArSarcasm dataset.

![](images/66a41d617d60658c9c57da9928642ede565a737abd801b7c8af75dda2ccb4cd6.jpg)  
Figure 17: ProTeGi optimized prompt of the ArSarcasm dataset.

![](images/4c3dc8f325df7b5d4844788d70858f6f5099de5c9cbc15434366fcbdc6a5d3f1.jpg)  
Figure 18: ERM optimized prompt of the ArSarcasm dataset.

![](images/e0f60d000e1dcd7c0c0e6b9cb6e3b5cdcb1451420e7870ffab39f197162bc9eb.jpg)  
Figure 19: Initial prompt of the WebNLG dataset.

![](images/dc13d4337df6425392567277cabd6e712dec3dd3933d95802baf43656ab2abda.jpg)  
Figure 20: ProTeGi optimized prompt of the WebNLG dataset.

![](images/63b85855d089766c741915f690d01cb25276e26754ff5c5e13b741cf762f5689.jpg)  
Figure 21: ERM optimized prompt of the WebNLG dataset.

![](images/1210a6e0713d734f3a0593c1959d7d6756acc5139fde6d2c3412e2f9449ad35d.jpg)  
Figure 22: Initial prompt of the GSM8K dataset.

![](images/ce126c78d360b224bd9696c7e72be07778783554ac5c9362bc173f1cccaad40b.jpg)  
Figure 23: ProTeGi optimized prompt of the GSM8K dataset.

![](images/dcf403463b942899a956f3b8cf9aae7d8144d3e59a354edccd0e7692f33a7632.jpg)

Figure 24: ERM optimized prompt of the GSM8K dataset.  
![](images/710549fcdaa6380b256edd2a95aa3c5ef855ef283f18aa21bd73a5496789b29c.jpg)  
Figure 25: Initial prompt of the WSC dataset.

![](images/60cf6f0240bf517088f5caa32141aef58b942ea426c8b21b0d9a3e50c02e9ddc.jpg)  
Figure 26: ProTeGi optimized prompt of the WSC dataset.

![](images/8d46fd871e5246646c470d58b9d8f092a05b370219a492272500eec64a9b1880.jpg)  
Figure 27: ERM optimized prompt of the WSC dataset.