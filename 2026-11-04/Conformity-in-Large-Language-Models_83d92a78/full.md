# Conformity in Large Language Models

Xiaochen Zhu\* Caiqi Zhang\* Tom Stafford Nigel Collier Andreas Vlachos

University of Cambridge University of Sheffield

{xz479, cz391, nhc30, av308}@cam.ac.uk

t.stafford@sheffield.ac.uk

## Abstract

The conformity effect describes the tendency of individuals to align their responses with the majority. Studying this bias in large language models (LLMs) is crucial, as LLMs are increasingly used in various information-seeking and decision-making tasks as conversation partners to improve productivity. Thus, conformity to incorrect responses can compromise their effectiveness. In this paper, we adapt psychological experiments to examine the extent of conformity in popular LLMs. Our findings reveal that all tested models exhibit varying levels of conformity toward the majority, regardless of their initial choice or correctness, across different knowledge domains. Notably, we are the first to show that LLMs are more likely to conform when they are more uncertain in their own prediction. We further explore factors that influence conformity, such as training paradigms and input characteristics, finding that instruction-tuned models are less susceptible to conformity, while increasing the naturalness of majority tones amplifies conformity. Finally, we propose two interventions, Devil’s Advocate and Question Distillation, to mitigate conformity, providing insights into building more robust language models.

## 1 Introduction

Although large language models (LLMs) have rapidly advanced and exhibit increasingly humanlike behavior (Aher et al., 2023; Kasneci et al., 2023; Hu and Collier, 2024), they are often affected by biases present in the data they are trained on (Navigli et al., 2023; Yu et al., 2023; Hu et al., 2025; Zhang et al., 2024a). Most biases studied in LLMs tend to be overt and domain-specific (e.g., gender, race, etc.) making them relatively easier to detect and mitigate (Gallegos et al., 2024; Ranaldi et al., 2024). However, in humans, more subtle, meta-cognitive biases exist across different knowledge domains, such as the Dunning-Kruger Effect (Kruger and Dunning, 1999), confirmation bias (Mercier and Sperber, 2017), and the one we focus on in this study—the conformity effect. Conformity refers to a form of social influence, in which an individual’s beliefs or behaviour shift towards being inline with the majority (Asch, 1955; Sowden et al., 2018), as shown in Figure 1. Extensively studied in psychology, conformity is observed not only in subjective or open-ended contexts but also in situations with a clear right answer (Bernheim, 1994; Crutchfield, 1955). For example, Asch (1955) demonstrated that under peer pressure, individuals often abandon correct answers to align with an incorrect majority, even in simple perceptual tasks.

Studying conformity in LLMs is particularly crucial. The conversational use of LLMs for complex task-solving has been shown to enhance both quality and productivity, offering a promising future for such systems (Dell’Acqua et al., 2023). However, conformity can significantly degrade the performance of language models, especially in multiagent systems that utilize LLM ensembles (LLM-MAs) or involves human interactions (Guo et al., 2024; Hong et al., 2023; Chen et al., 2024a; Feng et al., 2024). When LLMs conform to incorrect answers, it can undermine the effectiveness of these systems, particularly in tasks such as collectiveintelligence collaboration or constructive debate (Zhang et al., 2024e; Patel et al., 2024; Khan et al., 2024), ultimately negating the potential benefits that LLMs offer in these contexts.

Recent work has acknowledged the existence of conformity effect in LLMs. Zhang et al. (2023) observed conformity in general tasks such as chess move validation and multiple-choice question answering, where models aligned with perceived peer pressure. Baltaji et al. (2024) extended this investigation to cross-cultural collaboration and debate, highlighting that conformity remains a persistent issue in more diverse, open-ended discussions. Additionally, recent research has shown that LLMs trained with human preferences exhibit behaviors similar to the conformity effect, such as exploiting human judgments and generating outputs that appeal to evaluators regardless of their correctness—demonstrating patterns of sycophancy (Perez et al., 2023; Sharma et al., 2024). However, these studies primarily focus on identifying the presence of conformity without exploring the underlying factors that drive it. Moreover, there is a lack of detailed analysis on the mechanisms influencing susceptibility to conformity, and no potential mitigation strategies have been proposed.

![](images/d73794c88bc7da601a1acf36231a8e317c4fc74129e41f552e4edcb5239af90e.jpg)  
Figure 1: An example of LLMs conforming to an incorrect majority answer. We asked the model “What is the oldest college in Cambridge?”. Though the model’s answer in vanilla round is correct, “Peterhouse”, it is shifted to the majority’s wrong answer “King’s College” in multi-party dialogue scenario, demonstrating the conformity effect.

In this work, we investigate a range of stateof-the-art LLMs (e.g., Llama-3, Qwen2, Gemma-2, and Mistral-v0.3) on datasets covering a wider range of tasks, subjective vs. objective question answering, estimation vs. memorisation, multiple choice vs. open-ended question answering (MMLU, BigBenchHard, PopQA, CommonsenseQA, Politiscale, OpinionsQA). Our findings reveal that conformity is a universal phenomenon among LLMs and pervasive across knowledge domains. To further investigate the factors that trigger conformity, we conduct evaluations under varying conditions, examining both training paradigms (e.g., pre-training vs. instruction-tuning) and input characteristics (e.g., tones, prompt complexity). Our work is the first to show that models with higher initial confidence in their original predictions for a question are less likely to conform (p < 0.001) when exposed to an incorrect majority. We also propose two simple prompt-based heuristics to mitigate the conformity effect: Devil’s Advocate, and Question Distillation. Devil’s Advocate is inspired by Janis (1972)’s psychology study that introduced extra wrong answers that differ from the unanimous majority. Question Distillation replaces the repeated answer token from the majority that model over attends to with brief summaries.

## 2 Methodology

Asch (1956) defined conformity as the phenomenon where individuals lacks of independence in the face of group pressure. The study defined the individuals as critical subjects, who often shifted their solutions to align with the majority, who are referred to as confederates, regardless of whether the majority’s answers were correct. The Asch conformity experiment (Asch, 1951), involves a simple visual perception task. A single participant, the critical subject, will be asked to give an answer after a wrong answer is given unanimously by an increasing number of confederates. The critical subject’s answer will then be recorded to examine whether it’s correct or conformed to the majority. Allen and Levine (1969) extended the visual perception task to information and opinion items.

In our case, we identify the critical subject as a language model $L M _ { \theta }$ . We replace the visual perception task with Q&A in the form of a dialogue. Given a dataset $Q = \{ q 1 , q 2 , . . . q _ { n } \}$ , we define a prompt function $f ( q , p , c ; L M _ { \theta } )$ that takes the question, q, the number of total participants in the dialogue, $p ,$ and an in-domain distractor answer, c, to generate a dialogue template that probes for the language model’s response. When $p > 1$ , the model is the pth participant to answer the question, with all preceding $p - 1$ confederates unanimously expressing c as their answer. Otherwise, if $p = 1$ , the language model is the only participant in the dialogue and it is not affected by the non-existent distractor answer c. Firstly, we probe the model’s initial answer to the question, $a _ { i } ^ { o } = f ( q _ { i } , 1 , \emptyset ; L M _ { \theta } )$ . Then we define the evaluation set $S = \{ ( q _ { i } , a _ { i } ^ { o } , c _ { i } ) \mid q _ { i } \in Q \}$

Asch (1951) focuses on the number of correct answers and the frequency of conformity by directly comparing the critical subject’s answer with respect to its original answer. Similarly, we define and monitor the level of conformity $C L _ { p } ,$ and level of resistance $R L _ { p }$ of the critical subject model $L M _ { \theta }$ with respect to participant number p and the augmented evaluation set S from question dataset $Q$ as follows:

$$
C L _ { p } ( S , p ; L M _ { \theta } ) = \frac { \sum _ { i = 1 } ^ { | S | } 1 \left( \hat { a } _ { i } = c _ { i } \right) } { | S | }\tag{1}
$$

$$
R L _ { p } ( S , p ; L M _ { \theta } ) = \frac { \sum _ { i = 1 } ^ { | S | } 1 \left( \hat { a } _ { i } = a _ { i } ^ { o } \right) } { | S | }\tag{2}
$$

, where $\hat { a } _ { i } = f ( q _ { i } , p , c _ { i } ; L M _ { \theta } )$ . We record the proportion instead of instances for better cross comparison as the size of different Q&A datasets varies.

Objective vs. Subjective Questions. Allen and Levine (1969) reported different patterns of conformity on information and opinion items from human participants. Building on this, we examine conformity of models on both objective and subjective question. Under our definition, objective questions have clear, fact-based answers that can be verified as either correct or incorrect, typically in areas like mathematics, factual knowledge, or natural sciences. In contrast, subjective questions don’t have a single correct answer and often depend on personal opinions, interpretations, or perspectives. They are more common in areas like literature, ethics, or social sciences, where answers can vary based on individual reasoning or experiences.

Evaluation Strategy. We apply different evaluation strategies for conformity on objective and subject questions. Zhang et al. (2024d) point out that current models are trained to generate facts even when such facts are missing from their parametric knowledge. For factual questions, when an incorrect answer is provided by the model, it is ambiguous whether the model has memorized an incorrect fact or hallucinated due to missing information. To reduce the effect of such potential confounders, for objective questions, we first allow the LLMs to respond to the original datasets and select only the questions they answer correctly for the subsequent conformity test. That is, given an objective Q&A dataset $Q = \{ ( q _ { 1 } , a _ { 1 } ) , ( q _ { 2 } , a _ { 2 } ) , \dots , ( q _ { n } , a _ { n } ) \}$ , we define $S = \{ ( q _ { i } , a _ { i } ^ { o } , c _ { i } ) \mid ( q _ { i } , a _ { i } ) \in Q \land a _ { i } = a _ { i } ^ { o } \}$ For subjective questions, since there is no single correct answer, we include all questions and examine how the models’ top-ranked initial answer (under greedy decoding) changes when confederates unanimously take an different stance.

<table><tr><td rowspan=1 colspan=1>Tones</td><td rowspan=1 colspan=1>Examples</td></tr><tr><td rowspan=1 colspan=1>Plain</td><td rowspan=1 colspan=1> $\mathrm { " K i n g s " , " K i n g s " , " K i n g s " , . . . , " K i n g s " }$ </td></tr><tr><td rowspan=1 colspan=1>Neutral</td><td rowspan=1 colspan=1>&quot;I think it is Kings&quot;, &quot;My answer is Kings&quot;,&quot;Kings, in my opinion&quot;, ..., &quot;It&#x27;s Kings&#x27;</td></tr><tr><td rowspan=1 colspan=1>Confident</td><td rowspan=1 colspan=1>&quot;I am sure it is Kings&quot;, &quot;Kings, of course&quot;,.., &quot;Sure thing it&#x27;s Kings&quot;</td></tr><tr><td rowspan=1 colspan=1>Uncertain</td><td rowspan=1 colspan=1>&quot;I am not sure if it&#x27;s Kings&quot;, &quot;I guess it&#x27;sKings&quot;, ..., &quot;Kings? perhaps&quot;</td></tr></table>

Table 1: Different tones with Unanimous answers.

Confederate Setting. Regarding the choices in the responses, in Asch’s experiments, confederates are always Unanimous. We extend this by introducing a controlled setting, Diverse, where choices in the responses are selected uniformly at random. Since each answer is randomly chosen, no majority answer exists for the model to conform to. Ideally, there is no conformity effect in the Diverse setting.

Regarding the tones of responses, we include the following: (1) Plain: responses only contain answers with no additional phrasing. (2) Neutral: responses are closer to everyday dialogue utterances. (3) Confident: responses reflect certainty, with confederates expressing confidence in their answers. (4) Uncertain: responses convey hesitation or doubt in the confederates’ answers. Asch (1951) has been criticized for not adequately controlling confounding factors during conversations (e.g., eye contact or other unnecessary language cues between confederates and the critical subject) (Forsyth, 2014). To address this issue, we select the Unanimous-Plain setting as the base condition, as it only provides the confederates’ choices, similar to the improvement introduced in the Crutchfield situation where confederates’ choices were presented on a screen to the critical subject, thus eliminating any unnecessary cues that could confound the conformity effect in conversations (Crutchfield, 1955). Examples of each tone are provided in Table 1, a dialogue illustration is in Figure 1, and dialogue templates are in Appendix C.

![](images/5600e4880ab3432c3361167e3b00837e209b0047132013996c259876724e3e39.jpg)  
Figure 2: Conformity level for Llama-3-8B-Instruct in various question-answering tasks. The stacked bar plots show the proportion of resistance level $R L _ { p }$ (green), conformity level $C L _ { p }$ (red), and other responses (blue) across p ranging from 2 to 10 in four objective datasets (Commonsense QA, MMLU, PopQA, and BBH Object Counting) and two subjective datasets (Politiscale and OpinionsQA). The figure illustrates how conformity behavior exists across different knowledge domains.

## 3 Experiments

## 3.1 Experiment Setup

Models. We use Llama-3-8B (Meta, 2024), Mistral-v0.3-7B (Jiang et al., 2023), Qwen2-7B (Yang et al., 2024a), and Gemma2-9B (Gemma et al., 2024). For each model, we employ both the instruction-tuned and base versions to investigate the effect of instruction tuning. We apply greedy decoding to generate the answers. Details can be found in Appendix A. We use VLLM library to serve all models (Kwon et al., 2023).

Datasets. We evaluate the LLMs on various datasets across different knowledge domains. The objective QA datasets we used are MMLU (Hendrycks et al., 2021), BigBenchHard (Object Counting) (Suzgun et al., 2023), PopQA (Mallen et al., 2023), and CommonsenseQA (Talmor et al., 2019). The Q&A format includes both multiplechoice and open-ended questions. For subjective

![](images/6c45a75e3adcb35916886795c0b02a617f15ee9506ca79802fdf595e5204f803.jpg)  
Figure 3: Performance of Llama-3-8B-Instruct on MMLU in dialogues comprised confederates with Unanimous vs. Diverse incorrect answers.

Q&A datasets, we use Politiscale (Conobi, 2018) and OpinionsQA (Santurkar et al., 2023).

## 3.2 Conformity Effect in LLMs

We first show that the conformity effect is widespread. Figure 2 illustrates that conformity influences performance across diverse tasks and knowledge domains. The resistance level $R L _ { p }$ (shown in green) gradually decreases, while the conformity level $C L _ { p }$ (shown in red) increases as the number of confederates grows. This pattern holds across both subjective and objective datasets. More results across different models are in Figure 14 in Appendix D. For both PopQA and BBH, we also notice a significant number of responses that are neither correct nor conforming (shown in blue). This suggests that the conformity setting may also mislead the model to select other incorrect answers.

Participants  
![](images/fbfd89f2be8b1ed5aae8bcb7d775ccc650f0558b643c4003e0e4a31ce51fa928.jpg)

![](images/8f7d4cc6c2b87aefb575f99cb2d571a081d391a7138dd1083e7f3cf31237e7e3.jpg)

![](images/edd4037e6ed7b57e244db8e41e449c731913b432fc6ae913ad683b5cf98bba26.jpg)

![](images/a4fe017f16fd0bdf345082361cfacaadcb13fc6deb23a591723e68f4bdf520bc.jpg)

Figure 4: Conformity levels across different models and participant numbers with different tones on MMLU.  
![](images/c8ef2955c4a84e7f014439a876d7c421d2675bd368cf32ff45b2695f10e2aed4.jpg)  
Figure 5: Conformity level across pre-trained and instruction-tuned models with Unanimous-Plain on MMLU.

We then demonstrate that the performance decrease is not due to the dialogue setting but unanimous answers. Figure 3 compares the Llama-3-8B-Instruct model under the Unanimous-Plain and Diverse-Plain settings. In the Unanimous-Plain setting, the model’s performance decreases significantly, while in the Diverse-Plain setting, increasing the number of participants does not substantially affect performance. This result suggests that the observed conformity arises from unanimous answers rather than the dialogue setting.

## 3.3 Factors Influencing the Conformity Effect

We identify two key factors that influence the extent of the conformity effect: the tone of confederates and whether the models are instruction-tuned.

Tones of Confederates. Figure 4 compares the model’s conformity levels under the Unanimous setting with different tones. We have the following two findings: (1) Comparing Plain and Neutral, we find that the Neutral setting consistently increases the conformity level. The more natural and conversational tone, closer to real-life dialogue, amplifies the tendency to conform. (2) Comparing Neutral, Uncertain and Confident expressions, we find Confident consistently increases conformity. However, the effect of Uncertain expressions on conformity varies across models. For Gemma2 and Llama3, Uncertain expressions lead to lower conformity, as expected: if the LLM perceives the participants as lacking confidence, it relies more on its own beliefs. This indicates that these models are more sensitive to the second difference. In contrast, for Mistral and Qwen2, Uncertain expressions increase conformity.

Previous psychological experiments on humans also studied the factor of confidence in conformity. Simmons and Nelson (2006) found that individuals who expressed their opinions with high confidence were significantly more likely to influence the decisions or opinions of others. Similarly, Moussaïd et al. (2013) discovered that opinions expressed with high confidence tend to have a greater influence on the final group decision, as confidence can signal competence or authority.

Instruction-tuning. As shown in Figure 5, instruction tuning reduces conformity across all models. For Gemma2 and Llama-3, it significantly lowers the conformity level. However, for Mistral and Qwen2, the effect is more limited, as their initial conformity levels are already low.

The Difficulty of Questions. We observe that question difficulty influences the level of conformity in responses. For instance, in the relatively more challenging BBH Object Counting task (Figure 2), we find a higher conformity level compared to simpler tasks. We further analyze the performance of Llama-3-8B-Instruct across 57 subjects in the

![](images/2f2624430a3e60db3ee67dc9b40f949880fcb6c386730adfb0590f0841435e25.jpg)  
Figure 6: Subject accuracy vs. conformity level of Llama-3-8B-Instruct over 57 subjects in MMLU.

MMLU dataset. By identifying tasks with lower accuracy as more difficult, we confirm our hypothesis that LLMs are more likely to conform when facing harder tasks. Figure 6 supports this by showing a strong negative correlation ( 0.777) between task accuracy and the conformity level. This finding aligns with Morgan et al. (2015), who showed that humans are more likely to adopt others’ solutions in more difficult trials compared to easier ones.

## 4 Confidence in Conformity

In this section, we investigate the underlying patterns of questions that make models prone to conformity. We observe that some questions are never influenced by the confederates (i.e., they are always answered correctly). Inspired by the previous human study that lower confidence level in the original answer may lead to more severe conformity (Baron et al., 1996), we estimate the model’s confidence on these non-conforming questions.

For LLMs, the confidence estimation can be a effective indicator of incorrect answers (Zhang et al., $2 0 2 4 \mathbf { b } , \mathbf { c } ;$ Yang et al., 2024b). We conduct experiments using the MMLU and PopQA datasets. For MMLU, we measure confidence using the log probability of the option, and for PopQA, we apply consistency-based uncertainty estimation via EigV (Lin et al., 2024). First, we select questions that models have never conformed to the majority, regardless of the number of confederates, and compare them to questions where conformity occurs at least once. The confidence distributions for these two groups of questions in the vanilla setting (without confederates) are shown in Figure 7.

Our results indicate that the model’s initial confidence is a key predictor of whether it will conform on a given question. Models with higher initial confidence are less likely to conform, whereas if the initial confidence is low, it is more prone to align with the majority. The p-values are all smaller than 0.001, indicating a significant difference. We observe a similar pattern in the PopQA dataset, which contains open-ended questions (see Figure 13 in Appendix D).

## 5 Eliminating the Conformity Effect

We propose two methods to eliminate the conformity effect: Devil’s Advocate and Question Distillation. Figure 8 shows that both approaches effectively mitigate the conformity effect.

## 5.1 Devil’s Advocate (DA)

Janis (1972) highlighted that assigning someone the role of a devil’s advocate can reduce conformity in decision-making by injecting diversity, thus, encouraging independent thinking and mitigating the suppression of alternative viewpoints. This is also confirmed by group deliberation research (Karadzhov et al., 2024), where diversity in opinions is crucial for improving decision quality. In our experiment, we adopted this strategy by having one extra confederate who provides a different incorrect answer to inject diversity and examine the impact on language model conformity.

In objective questions, the devil’s advocate reduces the conformity effect. As shown in Figure 8 for MMLU, the DA method significantly reduces conformity in models that are more susceptible to conformity biases (e.g., Gemma2 and Llama3). For models that are initially more resistant to conformity (such as Mistral and Qwen2), this effect is less pronounced. Interestingly, even when the devil’s advocate provides an incorrect answer, the mere presence of dissent reduces overall conformity, consistent with the idea that diversity of opinions, even when inaccurate, can lead to more effective deliberation and independent model outputs. This mirrors the broader observation that diverse groups tend to outperform homogeneous ones in decision-making tasks (Karadzhov et al., 2024).

Devil’s Advocate is equally effective for subjective questions. Allen and Levine (1969) found that additional dissent reduces conformity in factual discussions but has less impact on opinion-based items. In contrast, our experiments show that the Devil’s Advocate approach is equally effective for LLMs on subjective question datasets, as demonstrated in Figure 9. This divergence suggests that LLMs may treat subjective questions similarly to objective ones, as opinions are learned during training much like factual information, indicating a potential lack of true subjectivity in their responses.

![](images/54dc6f21047547796bc276dff183141a2f22ead2666616c1a425a9eb166af93f.jpg)  
Figure 7: Confidence distribution on MMLU. The data is normalized so the total area of the histogram equals 1.

![](images/ca2cebb7440ad88e9be26c5b9e687b1f8124aee785d6fa54a83609bee9d55a8c.jpg)  
Figure 8: Resistance levels across different models and participant numbers, showing the impact of Question Distillation (QD) and Devil’s Advocate (DA) in reducing conformity, compared to original MMLU performance.

![](images/f30eb9021ebcbcac0f454cc6ef652b2ed0adb2ee6d5a480a0fbc7bbcbe9b1f97.jpg)  
Table 2: Attention heatmap of LLama-3-8B-Instruct’s answer tokens on a machine learning question in MMLU. Background color reflects token chunks attention scores. Darker color represents higher attention score for the token chunks. (Upper) The model gives a conformed answer with focuses on the repetitive choice. (Lower) The model gives a correct answer with Question Distillation to reduce ill-focused attention.

## 5.2 Question Distillation (QD)

To analyze which parts of the input prompt contribute most to the conformity effect, we examine the model’s attention distribution of the answer tokens over the input prompt, as presented in Table 2. We find that, instead of focusing on the digit representing the number of participants in the dialogue, the model overemphasizes the repeated answers, as shown in the upper part of Table 2, leading to conformity. A natural solution is to remove this misplaced attention during the model’s inference.

Therefore, we propose the Question Distillation (QD) method to address this issue. QD aims to simplify the prompt, making the task clearer for the model. Rather than listing all confederates’ answers individually, we summarize them into shorter prompt (e.g., “All participants before you have chosen...”), as shown in the lower part of Table 6. In QD, the model applies less attention to the majority choice. As demonstrated in Figure 8, Question Distillation results in a substantial decrease in conformity across various settings.

![](images/143dee7b347880200ba3567094b0d9feb1e503edfaa389723bbd9fd025c27bb5.jpg)  
Figure 9: The DA method shows a similar reduction in conformity for objective questions in the MMLU dataset compared to subjective questions in the OpinionsQA dataset for Llama-3-8B-Instruct.

## 5.3 Generalizing to Sycophancy

Sharma et al. (2024) show that LLMs exhibit sycophancy. LLMs tend to align with user-provided beliefs regardless of the model’s own knowledge or the factual correctness of those beliefs. As an extension, we investigate whether our methods for mitigating conformity can generalize to sycophancy.

We follow the evaluation pipeline proposed by Sharma et al. (2024), using the TriviaQA dataset (Joshi et al., 2017). We apply DA method, which is directly applicable in this setting by injecting a dissenting opinion into the user’s input that triggers sycophancy. In contrast, our QD method is less relevant here, as sycophantic prompts generally lack the repetitive content that QD is designed to simplify.

<table><tr><td>Metric</td><td>Base</td><td>Instruct</td></tr><tr><td>Sycophant↓</td><td>63.20%</td><td>25.18%</td></tr><tr><td>Sycophant (DA) ↓</td><td>41.37%</td><td>12.37%</td></tr><tr><td>Correct ↑</td><td>36.80%</td><td>74.82%</td></tr><tr><td>Correct (DA) ↑</td><td>57.82%</td><td>87.62%</td></tr></table>

Table 3: Sycophantic and correct answer rates before and after applying Devil’s Advocate on TriviaQA using Llama-3-8B and its Instruct variant.

We evaluate the DA method on both Llama-3- 8B and its instruction-tuned variant. As shown in Table 3, introducing a single dissenting (incorrect) answer alongside the user’s original (also incorrect) belief significantly reduces sycophantic responses and improves answer accuracy in this QA task. Further details on the experimental setup are provided in Appendix E.

## 6 Discussion

In this section, we explore the connection between our findings and psychological studies, aiming to deepen the understanding of the relation between psychology and NLP. We examine how our study extends to other areas of NLP research and, by comparing our results with psychological insights, raise important questions about the mechanisms driving LLM conformity and the causality between human psychology and LLM behavior.

Applying psychological frameworks to LLMs is an emerging approach as these models begin to display anthropomorphic traits, including preferences and social norms. Stereotypes, biases, and other human-like behaviors are shaped by the vast amounts of training data used in pre-training (Ke et al., 2024; Demszky et al., 2023). Griffin et al. (2023) showed that LLMs are susceptible to external input and exhibit psychological shifts reminiscent of human responses. Moreover, Yiu et al. (2024) proposed that LLMs imitate human cultural and social transmission, suggesting that they replicate human biases, such as the conformity effect.

Deutsch and Gerard (1955) expanded on Asch (1951) by distinguishing two types of conformity pressures: normative (conforming to gain approval) and informational (believing the group’s judgment is more accurate). These pressures explain causes of conformity via the rational informational weighting framework, where individuals balance their own knowledge with external inputs (Sperber et al., 2010; Bernard et al., 2015). We interpret this theory for LLMs by equating personal perception to the parametric knowledge obtained during pre-training, while external beliefs are represented by the input from other sources during inference. In other words, we explore how introducing new knowledge can override the model’s original belief, similar to in-context learning (ICL) (Brown et al., 2020; Dong et al., 2023). This phenomenon can be interpreted as successful knowledge editing (KE) (De Cao et al., 2021), where the external input (confederates’ responses) outweighs the model’s pre-trained knowledge. Zheng et al. (2023) demonstrated that the factual knowledge embedded in LLMs can be efficiently edited through promptbased ICL. Other studies attribute this behavior to human preference-based training, suggesting that models become sycophantic to users’ input (Perez et al., 2023; Sharma et al., 2024).

Our findings align with previous psychological studies, showing that both humans and LLMs conform to unanimous confederates and are more prone to conformity in difficult tasks. However, unlike humans, LLMs do not reduce conformity in response to uncertainty, as observed by Baron et al. (1996). These findings offer insights into how studying conformity in humans can help us understand LLM behavior, but many questions remain. For example, Cialdini and Goldstein (2004) suggested that human conformity stems from the desire for accuracy and social acceptance. A key question for future research is whether LLMs exhibit conformity for similar reasons when exposed to dialogues involving conformity, or if they conform simply due to uncertainty in their responses. Investigating these motivations will help clarify how LLMs process social influences and provide further insight into their underlying decision-making processes.

## 7 Conclusion

Our study reveals that various SOTA LLMs exhibit conformity to majority opinions, a behavior observed across multiple knowledge domains, indicating its pervasive nature. We present two key findings: (1) instruction-tuned models show greater resistance to conformity compared to their base counterparts, and (2) the conformity effect in LLMs is amplified when majority inputs are presented in a more natural and conversational tone. Our analysis highlights that initial confidence in the model’s prediction is a critical factor. LLMs tend to choose the unanimous answer more when their confidence is lower. To address this issue, we propose two effective, prompt-based interventions, Devil’s Advocate and Question Distillation, that reduce conformity without requiring additional model training. Our Devil’s Advocate method also generalize well for sycophancy mitigation. These findings not only underscore parallels between subtle human social biases and LLM behavior but also open new directions for exploring and mitigating such subtle biases, ultimately contributing to the development of more robust and fair language models.

## Limitations

Our study focuses exclusively on text-based, singlemodal interactions. While this isolates conformity effects in language tasks, real-world human-AI interactions often involve multimodal inputs (e.g., visual, auditory cues), which may also influence conformity. For example, classic Asch experiments included non-verbal cues like gestures and facial expressions. Future work should incorporate multimodal frameworks to examine how LLMs conform when exposed to diverse stimuli, offering deeper insights into cross-modal conformity effects.

Our use of controlled, artificial Q&A dialogues may not fully capture the complexity of real-world interactions. Human-AI collaboration involves more nuanced social dynamics, where factors such as conversational context, feedback, and multi-turn exchanges could impact conformity differently. Future studies should explore more realistic, openended scenarios to assess how conformity unfolds in dynamic human-AI interactions.

We advocate for researchers to explore additional methods to mitigate the conformity effect. In this study, we propose Devil’s Advocate and Question Distillation. Future research could further investigate other self-criticism techniques for models, such as self-refine (Madaan et al., 2023) and self-challenge (Chen et al., 2024b). Additionally, identifying and erasing patterns from the training data that may lead to conformity is a promising direction (Zhang et al., 2025).

## Ethics Statement

Our research adheres to strict ethical standards. No human participants were involved in our experiments, and no deception or manipulation was applied—only LLMs were evaluated in the conformity tests. We ensured compliance with the licenses of all datasets and models used. After thorough assessment, we do not anticipate any additional ethical concerns or risks related to our work.

## Acknowledgement

We acknowledge the use of an icon from Flaticon and thank its creators for providing this visually appealing design. Andreas Vlachos is supported by the ERC grant AVeriTeC (GA 865958).

## References

Gati V. Aher, Rosa I. Arriaga, and Adam Tauman Kalai. 2023. Using large language models to simulate multiple humans and replicate human subject studies.

In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings ofMachine Learning Research, pages 337–371. PMLR.

Vernon L Allen and John M Levine. 1969. Consensus and conformity. Journal ofExperimental Social Psychology, 5(4):389–399.

Solomon E Asch. 1951. Effects of group pressure upon the modification and distortion of judgments. Groups, Leadership and Men: Research in Human Relations, page 177.

Solomon E Asch. 1955. Opinions and social pressure. Scientific American, 193(5):31–35.

Solomon E Asch. 1956. Studies of independence and conformity: I. a minority of one against a unanimous majority. Psychological monographs: General and applied, 70(9):1.

Razan Baltaji, Babak Hemmatian, and Lav Varshney. 2024. Conformity, confabulation, and impersonation: Persona inconstancy in multi-agent LLM collaboration. In Proceedings of the 2nd Workshop on Cross-Cultural Considerations in NLP, pages 17–31, Bangkok, Thailand. Association for Computational Linguistics.

Robert S Baron, Joseph A Vandello, and Bethany Brunsman. 1996. The forgotten variable in conformity research: Impact of task importance on social influence. Journal of personality and social psychology, 71(5):915.

Stéphane Bernard, Paul Harris, Nathalie Terrier, and Fabrice Clément. 2015. Children weigh the number of informants and perceptual uncertainty when identifying objects. Journal ofExperimental Child Psychology, 136:70–81.

B Douglas Bernheim. 1994. A theory of conformity. Journal of political Economy, 102(5):841–877.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

Jiaqi Chen, Yuxian Jiang, Jiachen Lu, and Li Zhang. 2024a. S-agents: self-organizing agents in openended environment. ArXiv preprint, abs/2402.04578.

Yulong Chen, Yang Liu, Jianhao Yan, Xuefeng Bai, Ming Zhong, Yinghao Yang, Ziyi Yang, Chenguang Zhu, and Yue Zhang. 2024b. See what LLMs cannot answer: A self-challenge framework for uncovering LLM weaknesses. In First Conference on Language Modeling.

Robert B Cialdini and Noah J Goldstein. 2004. Social influence: Compliance and conformity. Annu. Rev. Psychol., 55(1):591–621.

Conobi. 2018. PolitiScales - About — politiscales.party. https://politiscales.party/about. [Accessed 01-10-2024].

Richard S Crutchfield. 1955. Conformity and character. American psychologist, 10(5):191.

Nicola De Cao, Wilker Aziz, and Ivan Titov. 2021. Editing factual knowledge in language models. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 6491– 6506, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Fabrizio Dell’Acqua, Edward McFowland III, Ethan R Mollick, Hila Lifshitz-Assaf, Katherine Kellogg, Saran Rajendran, Lisa Krayer, François Candelon, and Karim R Lakhani. 2023. Navigating the jagged technological frontier: Field experimental evidence of the effects of ai on knowledge worker productivity and quality. Harvard Business School Technology & Operations Mgt. Unit Working Paper, (24-013).

Dorottya Demszky, Diyi Yang, David S Yeager, Christopher J Bryan, Margarett Clapper, Susannah Chandhok, Johannes C Eichstaedt, Cameron Hecht, Jeremy Jamieson, Meghann Johnson, et al. 2023. Using large language models in psychology. Nature Reviews Psychology, 2(11):688–701.

Morton Deutsch and Harold B Gerard. 1955. A study of normative and informational social influences upon individual judgment. The journal ofabnormal and social psychology, 51(3):629.

Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. 2023. A survey on in-context learning. ArXiv preprint, abs/2301.00234.

Xueyang Feng, Zhi-Yuan Chen, Yujia Qin, Yankai Lin, Xu Chen, Zhiyuan Liu, and Ji-Rong Wen. 2024. Large language model-based human-agent collaboration for complex task solving. ArXiv preprint, abs/2402.12914.

Donelson R Forsyth. 2014. Group dynamics. Wadsworth Cengage Learning.

Isabel O Gallegos, Ryan A Rossi, Joe Barrow, Md Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Ruiyi Zhang, and Nesreen K Ahmed. 2024. Bias and fairness in large language models: A survey. Computational Linguistics, pages 1–79.

Gemma, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, Johan Ferret, Peter Liu, Pouya Tafti, Abe Friesen, Michelle Casbon, Sabela Ramos, Ravin Kumar, Charline Le Lan, Sammy Jerome, Anton Tsitsulin, Nino Vieillard, Piotr Stanczyk, Sertan Girgin, Nikola Momchev, Matt Hoffman, Shantanu Thakoor, Jean-Bastien Grill, Behnam Neyshabur, Olivier Bachem, Alanna Walton, Aliaksei Severyn, Alicia Parrish, Aliya Ahmad, Allen Hutchison, Alvin Abdagic, Amanda Carl, Amy Shen, Andy Brock, Andy Coenen, Anthony Laforge, Antonia Paterson, Ben Bastian, Bilal Piot, Bo Wu, Brandon Royal, Charlie Chen, Chintu Kumar, Chris Perry, Chris Welty, Christopher A. Choquette-Choo, Danila Sinopalnikov, David Weinberger, Dimple Vijaykumar, Dominika Rogoz inska, Dustin Herbison, Elisa Bandy, Emma Wang,´ Eric Noland, Erica Moreira, Evan Senter, Evgenii Eltyshev, Francesco Visin, Gabriel Rasskin, Gary Wei, Glenn Cameron, Gus Martins, Hadi Hashemi, Hanna Klimczak-Plucinska, Harleen Batra, Harsh´ Dhand, Ivan Nardini, Jacinda Mein, Jack Zhou, James Svensson, Jeff Stanway, Jetha Chan, Jin Peng Zhou, Joana Carrasqueira, Joana Iljazi, Jocelyn Becker, Joe Fernandez, Joost van Amersfoort, Josh Gordon, Josh Lipschultz, Josh Newlan, Ju yeong Ji, Kareem Mohamed, Kartikeya Badola, Kat Black, Katie Millican, Keelin McDonell, Kelvin Nguyen, Kiranbir Sodhia, Kish Greene, Lars Lowe Sjoesund, Lauren Usui, Laurent Sifre, Lena Heuermann, Leticia Lago, Lilly McNealus, Livio Baldini Soares, Logan Kilpatrick, Lucas Dixon, Luciano Martins, Machel Reid, Manvinder Singh, Mark Iverson, Martin Görner, Mat Velloso, Mateo Wirth, Matt Davidow, Matt Miller, Matthew Rahtz, Matthew Watson, Meg Risdal, Mehran Kazemi, Michael Moynihan, Ming Zhang, Minsuk Kahng, Minwoo Park, Mofi Rahman, Mohit Khatwani, Natalie Dao, Nenshad Bardoliwalla, Nesh Devanathan, Neta Dumai, Nilay Chauhan, Oscar Wahltinez, Pankil Botarda, Parker Barnes, Paul Barham, Paul Michel, Pengchong Jin, Petko Georgiev, Phil Culliton, Pradeep Kuppala, Ramona Comanescu, Ramona Merhej, Reena Jana, Reza Ardeshir Rokni, Rishabh Agarwal, Ryan Mullins, Samaneh Saadat, Sara Mc Carthy, Sarah Perrin, Sébastien M. R. Arnold, Sebastian Krause, Shengyang Dai, Shruti Garg, Shruti Sheth, Sue Ronstrom, Susan Chan, Timothy Jordan, Ting Yu, Tom Eccles, Tom Hennigan, Tomas Kocisky, Tulsee Doshi, Vihan Jain, Vikas Yadav, Vilobh Meshram, Vishal Dharmadhikari, Warren Barkley, Wei Wei, Wenming Ye, Woohyun Han, Woosuk Kwon, Xiang Xu, Zhe Shen, Zhitao Gong, Zichuan Wei, Victor Cotruta, Phoebe Kirk, Anand Rao, Minh Giang, Ludovic Peran, Tris Warkentin, Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia Hadsell, D. Sculley, Jeanine Banks, Anca Dragan, Slav Petrov, Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray Kavukcuoglu, Clement Farabet, Elena Buchatskaya, Sebastian Borgeaud, Noah Fiedel, Armand Joulin, Kathleen Kenealy, Robert Dadashi, and Alek Andreev. 2024. Gemma 2:

Improving open language models at a practical size. Preprint, arXiv:2408.00118.

Lewis Griffin, Bennett Kleinberg, Maximilian Mozes, Kimberly Mai, Maria Do Mar Vau, Matthew Caldwell, and Augustine Mavor-Parker. 2023. Large language models respond to influence like humans. In Proceedings of the First Workshop on Social Influence in Conversations (SICon 2023), pages 15–24, Toronto, Canada. Association for Computational Linguistics.

Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V. Chawla, Olaf Wiest, and Xiangliang Zhang. 2024. Large language model based multi-agents: A survey of progress and challenges. In Proceedings ofthe Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI 2024, Jeju, South Korea, August 3-9, 2024, pages 8048– 8057. ijcai.org.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, et al. 2023. Metagpt: Meta programming for multi-agent collaborative framework. ArXiv preprint, abs/2308.00352.

Tiancheng Hu and Nigel Collier. 2024. Quantifying the persona effect in LLM simulations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10289–10307, Bangkok, Thailand. Association for Computational Linguistics.

Tiancheng Hu, Yara Kyrychenko, Steve Rathje, Nigel Collier, Sander van der Linden, and Jon Roozenbeek. 2025. Generative language models exhibit social identity biases. Nature Computational Science, 5(1):65–75.

Irving L Janis. 1972. Victims of groupthink: A psychological study of foreign-policy decisions and fiascoes.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7B. Preprint, arXiv:2310.06825.

Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, Vancouver, Canada. Association for Computational Linguistics.

Georgi Karadzhov, Andreas Vlachos, and Tom Stafford. 2024. The effect of diversity on group decisionmaking. ArXiv preprint, abs/2402.01427.

Enkelejda Kasneci, Kathrin Seßler, Stefan Küchemann, Maria Bannert, Daryna Dementieva, Frank Fischer, Urs Gasser, Georg Groh, Stephan Günnemann, Eyke Hüllermeier, et al. 2023. Chatgpt for good? on opportunities and challenges of large language models for education. Learning and individual differences, 103:102274.

Luoma Ke, Song Tong, Peng Cheng, and Kaiping Peng. 2024. Exploring the frontiers of llms in psychological applications: A comprehensive review.

Akbir Khan, John Hughes, Dan Valentine, Laura Ruis, Kshitij Sachan, Ansh Radhakrishnan, Edward Grefenstette, Samuel R. Bowman, Tim Rocktäschel, and Ethan Perez. 2024. Debating with more persuasive llms leads to more truthful answers. In Fortyfirst International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. Open-Review.net.

Justin Kruger and David Dunning. 1999. Unskilled and unaware of it: how difficulties in recognizing one’s own incompetence lead to inflated selfassessments. Journal of personality and social psychology, 77(6):1121.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles.

Zhen Lin, Shubhendu Trivedi, and Jimeng Sun. 2024. Generating with confidence: Uncertainty quantification for black-box large language models. Transactions on Machine Learning Research.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9802–9822, Toronto, Canada. Association for Computational Linguistics.

Hugo Mercier and Dan Sperber. 2017. The enigma of reason. Harvard University Press.

Meta. 2024. Llama 3 model card.

Thomas JH Morgan, Kevin N Laland, and Paul L Harris. 2015. The development of adaptive conformity in young children: effects of uncertainty and consensus. Developmental science, 18(4):511–524.

Mehdi Moussaïd, Juliane E Kämmer, Pantelis P Analytis, and Hansjörg Neth. 2013. Social influence and the collective dynamics of opinion formation. PloS one, 8(11):e78433.

Roberto Navigli, Simone Conia, and Björn Ross. 2023. Biases in large language models: origins, inventory, and discussion. ACM Journal of Data and Information Quality, 15(2):1–21.

Bhrij Patel, Vishnu Sashank Dorbala, and Amrit Singh Bedi. 2024. Embodied question answering via multillm systems. ArXiv preprint, abs/2406.10918.

Ethan Perez, Sam Ringer, Kamile Lukosiute, Karina Nguyen, Edwin Chen, Scott Heiner, Craig Pettit, Catherine Olsson, Sandipan Kundu, Saurav Kadavath, Andy Jones, Anna Chen, Benjamin Mann, Brian Israel, Bryan Seethor, Cameron McKinnon, Christopher Olah, Da Yan, Daniela Amodei, Dario Amodei, Dawn Drain, Dustin Li, Eli Tran-Johnson, Guro Khundadze, Jackson Kernion, James Landis, Jamie Kerr, Jared Mueller, Jeeyoon Hyun, Joshua Landau, Kamal Ndousse, Landon Goldberg, Liane Lovitt, Martin Lucas, Michael Sellitto, Miranda Zhang, Neerav Kingsland, Nelson Elhage, Nicholas Joseph, Noemi Mercado, Nova DasSarma, Oliver Rausch, Robin Larson, Sam McCandlish, Scott Johnston, Shauna Kravec, Sheer El Showk, Tamera Lanham, Timothy Telleen-Lawton, Tom Brown, Tom Henighan, Tristan Hume, Yuntao Bai, Zac Hatfield-Dodds, Jack Clark, Samuel R. Bowman, Amanda Askell, Roger Grosse, Danny Hernandez, Deep Ganguli, Evan Hubinger, Nicholas Schiefer, and Jared Kaplan. 2023. Discovering language model behaviors with model-written evaluations. In Findings of the Associationfor Computational Linguistics: ACL 2023, pages 13387–13434, Toronto, Canada. Association for Computational Linguistics.

Leonardo Ranaldi, Elena Ruzzetti, Davide Venditti, Dario Onorati, and Fabio Zanzotto. 2024. A trip towards fairness: Bias and de-biasing in large language models. In Proceedings of the 13th Joint Conference on Lexical and Computational Semantics (\*SEM 2024), pages 372–384, Mexico City, Mexico. Association for Computational Linguistics.

Shibani Santurkar, Esin Durmus, Faisal Ladhak, Cinoo Lee, Percy Liang, and Tatsunori Hashimoto. 2023. Whose opinions do language models reflect? In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 29971–30004. PMLR.

Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R. Bowman, Esin Durmus, Zac Hatfield-Dodds, Scott R. Johnston, Shauna Kravec, Timothy Maxwell, Sam McCandlish, Kamal Ndousse, Oliver Rausch, Nicholas Schiefer, Da Yan, Miranda Zhang, and Ethan Perez. 2024. Towards understanding sycophancy in language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Joseph P Simmons and Leif D Nelson. 2006. Intuitive confidence: choosing between intuitive and nonintuitive alternatives. Journal of experimental psychology: General, 135(3):409.

Sophie Sowden, Sofia Koletsi, Eva Lymberopoulos, Elisabeta Militaru, Caroline Catmur, and Geoffrey Bird. 2018. Quantifying compliance and acceptance through public and private social conformity. Consciousness and cognition, 65:359–367.

Dan Sperber, Fabrice Clément, Christophe Heintz, Olivier Mascaro, Hugo Mercier, Gloria Origgi, and Deirdre Wilson. 2010. Epistemic vigilance. Mind & language, 25(4):359–393.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, and Jason Wei. 2023. Challenging BIG-bench tasks and whether chain-of-thought can solve them. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, pages 13003–13051, Toronto, Canada. Association for Computational Linguistics.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4149–4158, Minneapolis, Minnesota. Association for Computational Linguistics.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. 2024a. Qwen2 technical report.

Ruihan Yang, Caiqi Zhang, Zhisong Zhang, Xinting Huang, Sen Yang, Nigel Collier, Dong Yu,

and Deqing Yang. 2024b. Logu: Long-form generation with uncertainty expressions. Preprint, arXiv:2410.14309.

Eunice Yiu, Eliza Kosoy, and Alison Gopnik. 2024. Transmission versus truth, imitation versus innovation: What children can do that large language and language-and-vision models cannot (yet). Perspectives on Psychological Science, 19(5):874–883.

Yue Yu, Yuchen Zhuang, Jieyu Zhang, Yu Meng, Alexander J. Ratner, Ranjay Krishna, Jiaming Shen, and Chao Zhang. 2023. Large language model as attributed training data generator: A tale of diversity and bias. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Caiqi Zhang, Zhijiang Guo, and Andreas Vlachos. 2024a. Do we need language-specific fact-checking models? the case of Chinese. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 1899–1914, Miami, Florida, USA. Association for Computational Linguistics.

Caiqi Zhang, Fangyu Liu, Marco Basaldella, and Nigel Collier. 2024b. LUQ: Long-text uncertainty quantification for LLMs. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 5244–5262, Miami, Florida, USA. Association for Computational Linguistics.

Caiqi Zhang, Ruihan Yang, Zhisong Zhang, Xinting Huang, Sen Yang, Dong Yu, and Nigel Collier. 2024c. Atomic calibration of llms in long-form generations. Preprint, arXiv:2410.13246.

Hanning Zhang, Shizhe Diao, Yong Lin, Yi Fung, Qing Lian, Xingyao Wang, Yangyi Chen, Heng Ji, and Tong Zhang. 2024d. R-tuning: Instructing large language models to say ‘I don’t know’. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7113–7139, Mexico City, Mexico. Association for Computational Linguistics.

Jintian Zhang, Xin Xu, and Shumin Deng. 2023. Exploring collaboration mechanisms for llm agents: A social psychology view. ArXiv preprint, abs/2310.02124.

Yang Zhang, Shixin Yang, Chenjia Bai, Fei Wu, Xiu Li, Xuelong Li, and Zhen Wang. 2024e. Towards efficient llm grounding for embodied multi-agent collaboration. ArXiv preprint, abs/2405.14314.

Zhaohan Zhang, Ziquan Liu, and Ioannis Patras. 2025. Get confused cautiously: Textual sequence memorization erasure with selective entropy maximization. In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 10924–10939, Abu Dhabi, UAE. Association for Computational Linguistics.

Ce Zheng, Lei Li, Qingxiu Dong, Yuxuan Fan, Zhiyong Wu, Jingjing Xu, and Baobao Chang. 2023. Can we edit factual knowledge by in-context learning? In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4862–4876, Singapore. Association for Computational Linguistics.

## Appendix

## A Experiment Details

We have used Gemma-9B, Llama-3-8B, Mistrial-7B-v0.3, Qwen2-7B and their instruction-tuned versions. All models are publicly available online. All experiments are run on $1 \mathrm { ~ ^ { * } ~ A l 0 0 }$ (80 GB). The experiments takes around 1 hour per setting per model per dataset. In total, all experiments roughly $1 \ast 8$ models (including both instruction and base) \* 6 datasets \* 1 setting + 7 settings \* 1 dataset $( \mathrm { M M L U } ) ^ { * } 8$ models = 48 + 56 = 104 hours. No hyperparameters is required in our experiments. All models are run with bf16 precision and greedy decoding (temperature=0, top-p=1). We use VLLM (v0.5.4) to serve all models.

## B Experiment Workflow

Our experimentation workflow involves the following three steps:

1. Eliciting Baseline Response: We first query the model with a plain question (without introducing conformity-inducing dialogue) to capture its unprompted response.

2. Simulating Conformity: We incrementally introduce a varying number of simulated participants (ranging from 2 to 10), who unanimously provide incorrect answers in the dialogue. Each participant’s response is randomly sampled to ensure diversity and mitigate any bias from a fixed wrong answer. For instance, if the correct answer to a multiple-choice question is "A," three participants may choose $" \mathrm { C " }$ in one run, whereas seven participants might select $" \mathrm { D } " $ in another.

3. Recording Model’s Output: At each step, we record the model’s responses and analyze whether it aligns with the initial correct answer or conforms to the unanimous incorrect responses introduced in the dialogue.

We employ greedy decoding for all experiments to ensure deterministic outputs, as this provides consistency across runs. Consequently, multiple runs were not conducted, as they would yield the same outputs under this setup.

## C Prompts

![](images/c7ffd666e95a3fbb427d7535c571e73abf6d043914bed8b0efa7ffd4d6383568.jpg)  
Table 4: Prompt templates for vanilla round of MCQA.

## D Additional Results

![](images/a48649c39feed402c0660b8afceef2e19362f6af55b0aaaaf00fccd51cbf0f44.jpg)  
Table 5: Prompt templates for MCQA with confederates.

![](images/f2f935e5478979b70a34fb63969ae774bc260a16c5053a6deefc9b3a0479d286.jpg)  
Table 6: Prompt templates for MCQA with Question Distillation.

![](images/c83e261de79b5c8ba00efcf96dcbae359b857552c3cf2ece1cd0f730e919cdc6.jpg)  
Figure 10: Conformity levels across different language models and participant numbers. The line graphs depict the conformity behavior of four LLMs (Gemma-2-9B, Llama-3-8B, Mistral-7B-v0.3, and Qwen2-7B) in relation to the number of participants and four possible tones (Plain, Neutral, Confident, and Uncertain) in the multi-party conversation on MMLU.

![](images/ed7ac5f99d6f3ba306ff9caa2710a73778344db9adf552acb15576205713433f.jpg)  
Figure 11: Resistance level across different language models and participant numbers. The line graphs illustrates the behaviour of four LLMs (Gemma-2-9B, Llama-3-8B, Mistral-7B-v0.3, and Qwen2-7B) when Question Distillation (QD) and Devils’ Advocate (DA) are applied as a counter measure for conformity effect, comparing with the original performance on MMLU.

![](images/0c68a4ecd8d01103d352bf7080e6b8698421720064931913203d54b20363d67e.jpg)  
Figure 12: Logits of Llama-3-8B-Instruct on MMLU with different settings.

![](images/aca6992374adc5cf40f0e1eff2a93e7ca9449b20c0da5f57217586e4dc83d0b8.jpg)  
Figure 13: Distribution of confidences with Llama3-8B-Instruct on PopQA. Since no question was correctly answered across all 10 confederates, we selected questions that were answered correctly by up to 7 confederates. We used EigV (Lin et al., 2024) for uncertainty estimation.

![](images/ad0ddf9c30566d68b42962a879860356d59f03b1bf53e30a8c8d77fe413e0a6f.jpg)  
Figure 14: Conformity level for all models we tested on MMLU dataset.

## E Case Study: Mitigating Sycophancy

To evaluate the broader applicability of Devil’s Advocate (DA) and Question Distillation (QD) methods, we specifically investigated their effectiveness in reducing sycophancy. For this, we adopted the experimental pipeline outlined in Sharma et al. (2024).

Devil’s Advocate (DA): The DA method involves introducing an additional dissenting opinion (in this case, a wrong answer) to the input. This approach is directly applicable in the sycophancy setup, where the model’s responses tend to align with user preferences or biases rather than objective correctness.

Question Distillation (QD): The QD method focuses on summarizing repetitive or extraneous content in the input. However, this heuristic is less relevant in sycophancy settings, as these inputs typically do not involve repetitive text that could benefit from such simplification.

We applied the DA method to the TriviaQA (Joshi et al., 2017) dataset under sycophancy conditions as described in (Sharma et al., 2024). We evaluate our method on Llama-3-8B and its Instruct variant. Namely, we follow the steps below

1. Retrieve the questions the model answered correctly.

2. Identify the subset of questions affected by sycophancy.

3. Correct the model’s behavior in sycophancy settings by augmenting the question with an extra in-domain wrong answer as DA.

Below is an illustrative example from our experiments using Llama-3-8B-Instruct.

![](images/116d7115c29fd847ecf7de9eee6109e184d64390fea9701bcbd6c9556479fd19.jpg)