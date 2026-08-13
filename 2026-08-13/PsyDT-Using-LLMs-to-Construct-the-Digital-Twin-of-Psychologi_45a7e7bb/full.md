# PsyDT: Using LLMs to Construct the Digital Twin of Psychological Counselor with Personalized Counseling Style for Psychological Counseling

Haojie Xie<sup>1</sup>\*, Yirong Chen<sup>2</sup>∗, Xiaofen Xing<sup>2†</sup>, Jingkai Lin<sup>2</sup>, Xiangmin Xu<sup>1,3†</sup>

<sup>1</sup>School of Future Technology, South China University of Technology

<sup>2</sup>School of Electronic and Information Engineering, South China University of Technology <sup>3</sup>Pazhou Lab, Guangzhou

fthjxie@mail.scut.edu.cn, {xfxing, xmxu}@scut.edu.cn

## Abstract

Currently, large language models (LLMs) have made significant progress in the field of psychological counseling. However, existing mental health LLMs overlook a critical issue where they do not consider the fact that different psy chological counselors exhibit different personal styles, including linguistic styles and thera peutic types, etc. As a result, these LLMs fail to satisfy the individual needs of clients who seek different counseling styles. To help bridge this gap, we propose PsyDT, a novel framework using LLMs to construct the Digital Twin of Psychological counselor with personalized counseling style. Compared to the time consuming and costly approach of collecting a large number of real-world counseling cases to create a specific counselor’s digital twin, our framework offers a faster and more costeffective solution. To construct PsyDT, we utilize dynamic one-shot learning by using GPT-4 to capture counselor’s unique counseling style, mainly focusing on linguistic style and therapy technique. Subsequently, using existing singleturn long-text dialogues with client personality, GPT-4 is guided to synthesize multi-turn dialogues of specific counselor. Finally, we finetune the LLMs on the synthesized dataset, Psy-DTCorpus, to achieve the digital twin of psychological counselor with personalized counseling style. Experimental results indicate that our proposed PsyDT framework can synthesize multi-turn dialogues that closely resemble realworld counseling cases and demonstrate better performance compared to other baselines, thereby show that our framework can effectively construct the digital twin of psychological counselor with a specific counseling style.<sup>1</sup>

## 1 Introduction

In recent years, LLMs have made significant advancements, exemplified by ChatGPT (OpenAI,

![](images/1a763846246f81c056770932a8ae15200f6e509a24ac7c18503fb7d5174c5514.jpg)  
Figure 1: Difference between our proposed PsyDT framework and others. (a) Previous methods mixed multi-turn dialogues with multiple counseling styles to fine-tune LLM. (b) Our proposed PsyDT framework uses LLMs to construct the digital twin of psychological counselor with a specific counseling style.

2022), GPT-4 (OpenAI, 2024), LLaMA (Touvron et al., 2023), Qwen (Bai et al., 2023), ChatGLM (Du et al., 2022), etc. While these LLMs excel in a variety of tasks, they often encounter limitations in specialized fields such as mental health due to a lack of domain-specific expertise. In addition, with the global rise in the prevalence of depression and anxiety (Santomauro et al., 2021), mental health has garnered widespread attention, prompting researchers to explore the application of LLMs in psychological counseling. The value of mental health LLMs lies in their potential to provide emotional support and counseling services to individuals. Currently, a series of mental health LLMs have been proposed, including MeChat (Qiu et al., 2024a), PsyChat (Qiu et al., 2024b), SoulChat (Chen et al., 2023), EmoLLM (EmoLLM, 2024), MindChat (Xin Yan, 2023), and CPsyCoun (Zhang et al., 2024), etc.

Due to ethics policies and privacy protection, real-world multi-turn mental health dialogues datasets are exceedingly rare. Consequently, most recent LLM-based mental health researches rely on synthesizing multi-turn dialogues datasets. For instance, Qiu et al. (2024a) introduced SMILE, a technique for expanding single-turn dialogues into multi-turn dialogues, thereby better simulating realworld interactions between clients and counselors. Similarly, Chen et al. (2023) developed SoulChat-Corpus, a multi-turn empathetic dialogue dataset with more than 2 million samples, in which the input is the multi-turn dialogue context, and the target is empathetic responses that cover expressions such as questioning, comfort, recognition, listening, trust, emotional support, etc. Zhang et al. (2024) later proposed CPsyCoun, a report-based multi-turn dialogue reconstruction and evaluation framework for Chinese psychological counseling.

![](images/ed18f2d201ffce17f2d2ec3683aedd8b0eff50aa7861e61597da811dfc532c07.jpg)  
Figure 2: Illustration of multi-turn dialogues synthesis method of PsyDT framework and PsyDTLLM model.

However, despite the promising results of previous mental health LLMs, they overlook a critical issue where they do not take into account the fact that different psychological counselors exhibit different personal styles, including variations in linguistic style and therapeutic type (Hernawati et al., 2019; She et al., 2023; Pinchot et al., 1975). As a result, these LLMs struggle to meet the individual needs of clients who seek different counseling styles. Moreover, as illustrated in Figure 1, fine-tuning LLMs on multi-turn mental health dialogues datasets that blend multiple counseling styles often lead to instability in the response. To this end, we introduce the concept of psychological counselor’s digital twin. To construct the digital twin, a large volume of real-world counseling cases from a specific psychological counselor are typically collected to fine-tune LLMs. However, the process of collecting counseling cases is not only time-consuming but also costly.

In response to these challenges, we propose PsyDT, a novel framework using LLMs to construct the Digital Twin of Psychological counselor with personalized counseling style. To construct PsyDT, we need to synthesize multi-turn dialogues of a specific psychological counselor. To ensure the quality of the synthetic multi-turn dialogues, we first select 5,000 high-quality single-turn dialogues from the SoulChatCorpus (Chen et al., 2023). Subsequently, in order to ensure the complexity and diversity of clients’ linguistic style in the synthetic multi-turn dialogues, we employ GPT-4 to simulate the Big Five personality traits (Costa and McCrae, 1999) of clients based on the clients’ question from single-turn dialogues. To conduct real-world psychological counselor’s digital twin, we invited a professional psychological counselor to play the role of digital twin. The counselor was asked to have conversations covering 12 counseling topics with 12 different clients. And then GPT-4 is employed to summarize the linguistic style and therapy technique according to counseling topic from above collected real-world counseling cases of the specific counselor. Combining the above simulated clients’ personality, GPT-4 is used as the guidance for multi-turn dialogues synthesis of specific counselor. Finally, based on the synthesized multi-turn dialogues dataset, PsyDTCorpus, we construct the digital twin of psychological counselor with the specific counseling style using the Multi-Turn Instruction Fine-tuning (MIFT) method. Extensive experimental results demonstrate that our proposed PsyDT framework can quickly and cost-effectively construct the digital twin of psychological counselor with a specific counseling style, validating the effectiveness of the PsyDT framework.

In summary, the contributions of this work can be succinctly outlined as follows:

• To the best of our knowledge, our work is the first to use LLMs to construct the digital twin of psychological counselor with personalized counseling style. The proposed multi-turn dialogues synthesis method of PsyDT framework can quickly and cost-effectively synthesize PsyDTCorpus, a high-quality multi-turn mental health dialogues dataset of psychological counselor with specific counseling style.

• We design automatic evaluations and manual evaluations of synthetic dataset and fine-tuned LLM of psychological counselor with specific counseling style to indicate effectiveness and superiority of our PsyDT framework.

• Experimental results indicate that PsyDT can synthesize multi-turn dialogues that closely resemble real-world counseling cases and demonstrate better performance compared to other baselines, thereby demonstrate the strong potential of PsyDT for application in real-world psychological counseling.

## 2 Methodology

This section explores the process of PsyDT framework. We first introduce multi-turn dialogues synthesis method of PsyDT, which consists of three components: Dynamic One-Shot Learning from Counseling Cases, Client Personality Simulation and Multi-Turn Mental Health Dialogues Synthesis, as shown in Figure 2. And then we fine-tune the LLMs on the synthesized dataset, PsyDTCorpus, to achieve the digital twin of psychological counselor with a specific counseling style. The specific process of synthesizing multi-turn dialogues is shown in Algorithm 1 in the Appendix. All the data in our paper are in Chinese.

## 2.1 Single-Turn Dialogues Preparation

To ensure the quality of the synthetic multi-turn dialogues, we select 5,000 single-turn long-text dialogues with rich presentation of clients’ personality traits from the SoulChatCorpus (Chen et al., 2023) by utilizing GPT-4 as the client personality evaluator. An example of the single-turn long-text dialogues is shown in Figure 7 in the appendix. These dialogues primarily consist of four components: Counseling Topic, Title of Client’s Question, Detail of Client’s Question, and Counselor’s Long Text Response. The distribution of counseling topics is shown in Figure 3.

![](images/2f932b964f2c321b1f11b63832d4786d2379bf23361c6894ec5fea5d2c6471f4.jpg)  
Figure 3: Distribution of counseling topics.

## 2.2 Dynamic One-Shot Learning from Counseling Cases

## 2.2.1 Real-World Counseling Cases Collection

To construct psychological counselor’s digital twin, we invited a psychological counselor who meets national professional standards to provide psychological counseling. We recruited volunteers from online community, who were asked to select a counseling topic suitable for themselves. Before the consultation, volunteers were informed of precautions and were required to read and sign an informed consent form. During the formal consultation session, volunteers will engage in a dialogue with the psychological counselor. The entire session will follow the standard psychological consultation model and last for 50 minutes, conducted entirely via textbased communication. After the consultation, the counselor who conducted the session will review and verify the consultation records, correcting any errors and redacting any sensitive information to protect the privacy of the participants. In addition, we employ GPT-4 to perform security checks and data cleaning on these counseling cases. As a result, the real-world counseling cases of the specific psychological counselor on 12 topics were collected, with total number of 12 and total cost of about ¥12000. One of the real-world counseling cases is presented in Figure 8 (Counseling Topic: Relationship) in the appendix.

## 2.2.2 Linguistic Style and Therapy Technique Summarization

In real-world psychotherapy scenarios, different psychological counselors exhibit distinct personal styles, including unique linguistic styles and therapeutic types, etc. To construct above psychological counselor’s digital twin, we first employ GPT-4 to capture the linguistic style from several collected real-world counseling cases. Subsequently, we also employ GPT-4 to summarize the therapeutic type of collected counseling cases of specific counselor. According to statistics, 12 counseling cases all use Rational Emotive Behavior Therapy (REBT) (Dryden, 2005) for psychological counseling. And then the therapeutic type is used to retrieve the corresponding knowledge from the therapy technique knowledge base. More details can be seen in Appendix B.1.

## 2.3 Client Personality Simulation

To ensure both complexity and diversity of clients’ linguistic style in synthetic multi-turn dialogues, we employ GPT-4 to simulate the Big Five personality traits (Costa and McCrae, 1999) of clients based on their question, as detailed in Appendix B.2. The Big Five personality theory encompasses five core dimensions: Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism (OCEAN).

## 2.4 Multi-Turn Mental Health Dialogues Synthesis

In order to prevent excessive homogenization of synthesized multi-turn dialogues, we enable singleturn dialogues to dynamically match real-world counseling cases based on counseling topics. The summarized linguistic style and therapy technique with specific topic and extracted client’s Big Five personality traits polymerized with counseling case with specific topic, emotion guide and response guide, GPT-4 is employed to synthesize single-turn dialogues into multi-turn dialogues. We denote synthesized multi-turn dialogues dataset as PsyDT-Corpus. For more details, please refer to B.3.

## 2.5 PsyDTLLM Model

Based on the synthesized multi-turn dialogues dataset, PsyDTCorpus, we construct the specific psychological counselor’s digital twin, Psy-DTLLM, through multi-turn instruction fine-tuning. The loss $\ell _ { \theta }$ during training is,

$$
\ell _ { \boldsymbol { \theta } } = \sum _ { i = 1 } ^ { n } \ell ( \hat { y _ { i } } , y _ { i } )\tag{1}
$$

$$
y _ { i } = L L M _ { \theta } ( c _ { < i } )\tag{2}
$$

where $\theta , \hat { y _ { i } } , y _ { i } , c _ { < i }$ denote the trainable parameters of the LLM, the i-th target utterance of counselor, the i-th predicted utterance of counselor, dialogue history with less than i turns, respectively.

## 3 Experiments

In this section, we conduct multiple comprehensive analyses of the synthesized multi-turn dialogues dataset PsyDTCorpus and the fine-tuned PsyDTLLM of psychological counselor with specific counseling style. In summary, we address the following research questions (RQs):

• RQ1: How similar is the synthesized multiturn dialogues dataset with real-world counseling cases of specific counselor?

• RQ2: What is the performance of PsyDTCorpus as compared with baseline datasets?

• RQ3: Do the different components in the multi-turn dialogues synthesis method of PsyDT synthesize the corresponding effect of PsyDTCorpus?

• RQ4: What is the performance of Psy-DTLLM as compared with baseline LLMs?

## 3.1 Analysis of PsyDTCorpus Dataset

## 3.1.1 RQ1: Similarity Analysis

To validate the effectiveness of the multi-turn dialogues synthesis method of PsyDT framework, we compare the multi-turn mental health dialogues constructed by three methods, which are PsyDT\_Prompt, SoulChat\_Prompt (Chen et al., 2023), Smile\_Prompt (Qiu et al., 2024a). For each topic, we randomly select 20 single-turn dialogue samples. Then, all the samples are used to synthesize multi-turn dialogues based on the above three methods, respectively. Given the potential of using LLMs for evaluating text generation quality (Chiang and Lee, 2023), we attempt to automatically assess the similarity between three synthetic multiturn dialogues dataset and real-world counseling cases in the matter of linguistic style and therapy technique. We employ two state-of-the-art LLMs as evaluators: GPT-4o<sup>2</sup> and Claude 3.5<sup>3</sup>. We take the average of the similarity scores given by two LLMs as the final result. For more details, please refer to Appendix C.1.

![](images/9048808faaa87e1b0abf8eac4fcaa74102c36b6ca37dc91232f800cb9c7f2bb1.jpg)  
(a) Linguistic Style

![](images/8a0c65c122c0b65f2bc1a6d56f6ff53e6ce06d4c603d666dd63015534f90b4c5.jpg)  
(b) Therapy Technique  
Figure 4: Similarity results for the proposed multi-turn dialogue synthesis method and other baseline methods.

Figure 4 illustrates that the similarity between multi-turn dialogues synthesized with PsyDT\_Prompt and real-world counseling cases is approximately 70%, surpassing multi-turn dialogues synthesized with Smile\_Prompt and SoulChat\_Prompt, especially in therapy technique. It indicates that the multi-turn dialogue synthesis method of PsyDT has excellent alignment performance in the linguistic style and therapy technique of the specific psychological counselor, which can effectively synthesize multi-turn dialogues of counselor with specific counseling style.

## 3.1.2 RQ2: Overall Dataset Comparisons

To validate the superiority of the synthetic multiturn dialogues dataset PsyDTCorpus, we conduct a comprehensive comparison between PsyDTCorpus and the following datasets: SMILECHAT (Qiu et al., 2024a), SoulChatCorpus (Chen et al., 2023), and CPsyCounD (Zhang et al., 2024). Detailed information regarding the employed datasets is provided in Table 1. Open., Size, NoT., LoC., LoP., EmoE., CogE., TheT. respectively represent opensource, dataset size, average number of turns, average length of client’s response, average length of psychological counselor’s response, emotional empathy<sup>4</sup>, cognitive empathy (Preston and De Waal, 2002), therapy technique. We randomly select 50 dialogue samples from each of these four datasets for a manual evaluation. Our evaluation team consists of four senior psychological postgraduate students and an experienced psychotherapist to ensure accuracy and professionalism. The evaluation metrics are listed in Table 5 in the appendix. Based on references Swank and Lambie, 2012 and Association et al., 2012, we summarize four professional assessment dimensions in the field of psychological counseling: Conversation Strategy (Con.), State and Attitude (Sta.), Relationship Building (Rel.), and Application of Therapy Technique (App.). Additionally, we manually evaluate the fluency (Flu.) and safety (Saf.) of the data. The rating scale of Con., Sta., Rel. and App. is (0, 1, ..., 9, 10), while (0,1) for Flu., where higher score means better. Fleiss’ κ (Fleiss, 1971) for Con., Sta., Rel., The. and Inf. are 0.411, 0.403, 0.407, 0.435 and 0.547, indicating moderate annotation agreement.

As shown in Table 1, the PsyDTCorpus dataset significantly outperforms the other datasets across all four professional assessment dimensions. To visually illustrate the stability and score distribution of our multi-turn dialogues dataset compared to other datasets across these four dimensions, we design boxplots of the evaluation results. As shown in Figure 5, it is evident that PsyDTCorpus excels in quality across all four dimensions compared to the other baseline datasets. These results underscore the better superiority and professionalism of PsyDTCorpus over the other baseline datasets, indicating the strong potential of PsyDT for application in real-world psychological counseling.

Table 1: Dataset evaluation results. The best score for each metric is in-bold, while the second best score is underlined.
<table><tr><td rowspan="2">Datasets</td><td colspan="5">Statistics</td><td colspan="3">Abilities</td><td colspan="6">Expert Evaluation</td></tr><tr><td>Open.</td><td>Size</td><td>NoT.</td><td>LoC.</td><td>LoP.</td><td>EmoE.</td><td>CogE.</td><td>TheT.</td><td>Con.</td><td>Sta.</td><td>Rel.</td><td>App.</td><td>Flu.</td><td>Saf.</td></tr><tr><td>SMILECHAT</td><td>√</td><td>56k</td><td>10.4</td><td>26.1</td><td>28.9</td><td>√</td><td></td><td></td><td>5.38</td><td>5.92</td><td>5.65</td><td>4.37</td><td>0.84</td><td>√</td></tr><tr><td>SoulChatCorpus</td><td>√</td><td>258k</td><td>5.9</td><td>41.4</td><td>90.0</td><td>√</td><td></td><td></td><td>5.24</td><td>5.80</td><td>5.62</td><td>4.38</td><td>0.86</td><td>√</td></tr><tr><td>CPsyCounD</td><td>√</td><td>3.1k</td><td>8.0</td><td>32.9</td><td>52.6</td><td>√</td><td></td><td>√</td><td>5.57</td><td>6.02</td><td>5.66</td><td>5.49</td><td>0.72</td><td>√</td></tr><tr><td>PsyDTCorpus</td><td>√</td><td>5k</td><td>18.1</td><td>31.6</td><td>58.1</td><td>√</td><td>√</td><td>√</td><td>8.39</td><td>8.69</td><td>8.29</td><td>8.12</td><td>1.00</td><td>√</td></tr></table>

![](images/0bc945ce1a78313106097dc59616a42328da775718ffb85924fcc444b837f230.jpg)  
(a) Conversation Strategy

![](images/53a0d506388d3ea5cbda3eaf9de19652fb22b4b93d484def977d8a012e7216ae.jpg)  
(b) State and Attitude

![](images/48d774b2d7e0c37c55bbb33b39c76f3eacc5dcb97dc6eec0b13c6661e8282227.jpg)  
(c) Relationship Building

![](images/ef29ff27d50dbeaab8fab06c12ed31e0fc33fb4b91f14e11c48198e4887cc4a5.jpg)  
(d) Application of Therapy Technique  
Figure 5: Results of manual evaluation for PsyDTCorpus and baseline datasets on 4 professional dimensions.

## 3.1.3 RQ3: Ablation Studies

To validate that our synthetic multi-turn dialogues dataset, PsyDTCorpus, integrates linguistic style, therapy technique, and client personality, we design three ablation manual evaluations. Our evaluation team consists of four senior psychological postgraduate students and an experienced psychotherapist to ensure accuracy and professionalism. We randomly select 16 single-turn dialogues. Using multi-turn dialogues synthesis method of PsyDT from Figure 14, we synthesize these 16 sets of dialogues, each time excluding one of the following elements: linguistic style, therapy technique, and client personality. The evaluators are asked to choose the dialogues that best represented the corresponding linguistic style, therapy technique, and client personality between original dialogues and ablated dialogues. We took the optimal answer for each set of dialogue from the five evaluators through voting mechanism and then calculating the agreement for all 16 sets. As shown in Figure 6, our synthetic multi-turn dialogue achieved a fidelity of over 60%. This demonstrates that our multi-turn dialogues synthesis method of PsyDT framework can effectively integrate linguistic style, therapy technique, and client personality.

![](images/b8ff5fb75c61a7ced3c8a115214a5815d05ff67f563cf077478647fba4821756.jpg)  
Figure 6: Results of ablation study on synthetic dialogues and other ablated dialogues.

## 3.2 Analysis of PsyDTLLM model

## 3.2.1 Baselines

We compare PsyDTLLM with the following baselines.

• Closed-source: ChatGPT (OpenAI, 2022; Ouyang et al., 2022); GPT-4 (OpenAI, 2024).

• Open-source: Baichuan2-7B-Chat (Yang et al., 2023); GLM4-9B-Chat (GLM et al., 2024); InternLM2-Chat-7B (Cai et al., 2024); Llama3-8B-Instruct (Touvron et al., 2023); Llama3.1-8B-Instruct (Touvron et al., 2023); Qwen2-7B-Instruct (Yang et al., 2024); Yi-1.5-9B-Chat (Young et al., 2024).

• Domain-specific: MeChat (Qiu et al., 2024a); PsyChat (Qiu et al., 2024b); SoulChat (Chen et al., 2023); EmoLLM (EmoLLM, 2024); MindChat (Xin Yan, 2023); CPsyCoun (Zhang et al., 2024).

To facilitate a more accurate comparison of the capabilities of various models, we choose models of similar magnitudes, such as the 6B/7B/8B/9B model parameter sizes for comparison.

## 3.2.2 Implementation details

The Qwen2-7B-Instruct is used as the backbone. The whole implementation is based on the LLaMA-Factory (Zheng et al., 2024). PsyDTLLM is finetuned for 3 epochs on the proposed PsyDTCorpus with a batch size of 2 per GPU, using 8 NVIDIA-A800-80G GPUs. The cosine-type learning rate scheduler with warmup\_ratio = 0.03 and warmup\_max $\underline { l } r = 1 . 0 e - 5$ is used. 16-bit half-precision floating point numbers is utilized to accelerate training and improves model performance. The random seed is set to 42. In the inference phase, all the LLMs adopt the following configuration: temperature = 0.9, top\_p = 0.75, top $\_ k = 2 0$

## 3.2.3 RQ4: Overall Model Comparisons

To verify the superiority of PsyDTLLM compared to other baseline models, we randomly split the PsyDTCorpus dataset into 4760 sets of training data and 240 sets of testing data. The testing data includes 12 topics, each with 20 sets of data, for a total of 4311 turns to conduct automatic evaluation. For each sample, each model generates an answer for evaluation. We used 5 evaluation metrics as automatic metrics: ROUGE-1, ROUGE-2, ROUGE-L (Lin, 2004), BLEU-4 (Papineni et al., 2002), $F _ { B E R T }$ of BERTSCORE (Zhang\* et al., 2020). Generally, as shown in Table 2, PsyDTLLM outperforms other baseline models in both automatic evaluation metrics, which demonstrates that PsyDTLLM outperforms other baseline models in the semantic understanding ability.

Subsequently, we randomly extracts 20 turns of data from each topic for a total of 240 turns to conduct professional evaluation. We also employ two state-of-the-art LLMs as evaluators: GPT-4o and Claude 3.5. We take the average of the scores given by two LLMs as the final result. The metrics to assess are as follows: Emotional Empathy (EmoE.), Cognitive Empathy (CogE.), Conversation Strategy (Con.), State and Attitude (Sta.), and Safety (Saf.).

The ranting scale of EmoE., CogE., Con., Sta. is (0, 1, 2, 3) and Saf. is (0, 1), where higher score means better. Generally, as shown in Table 2, PsyDTLLM outperforms other baseline models in both professional metrics, demonstrating better performance in the psychological counseling, thereby indicating the strong potential of PsyDT for application in real-world psychological counseling. For more details, please refer to Appendix C.2.

## 4 Related Work

## 4.1 Mental Health LLMs

LLMs typically underperform in domains requiring complex mental acumen and high levels of empathy. Nonetheless, significant advancements have been made by researchers. For instance, Qiu et al. (2024a) introduced SMILE approach, leveraging ChatGPT to transform single-turn dialogues into multi-turn interactions. Chen et al. (2023) developed SoulChatCorpus, a multi-turn empathetic conversation dataset comprising over 2 million samples. By fine-tuning LLM on SoulChatCorpus, they achieved superior performance in empathetic dialogue tasks. Zhang et al. (2024) presented CPsy-Coun, a framework for reconstructing and evaluating multi-turn dialogues in Chinese psychological counseling. To optimize the utilization of counseling reports, they proposed a two-phase method for generating high-quality dialogues. Furthermore, they established a comprehensive evaluation benchmark for automatic assessment of multi-turn psychological counseling. In this paper, we propose PsyDT, a novel framework using LLMs to construct the digital twin of psychological counselor with personalized counseling style.

## 4.2 Personality Traits

Psychological personality traits are fundamental in shaping individual behavior, thoughts, and emotions. These traits represent enduring patterns of thought, feeling, and behavior that constitute an individual’s unique psychological profile (Matthews et al., 2003). The Big Five, also known as the Five-Factor Model (FFM) (Costa and McCrae, 1999), is a widely recognized framework that distills personality into five broad dimensions: Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism (OCEAN). Extensive empirical research supports the robustness of these dimensions across diverse cultures, indicating a universal structure to human personality. The Myers-

Table 2: Model evaluation results.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Models</td><td colspan="5">Automatic.</td><td colspan="5">Professional.</td></tr><tr><td>R-1</td><td>R-2</td><td>R-L</td><td>B-4</td><td>FBERT</td><td>EmoE.</td><td>CogE.</td><td>Con.</td><td>Sta.</td><td>Saf.</td></tr><tr><td rowspan="2">Closed</td><td>ChatGPT</td><td>31.72</td><td>7.77</td><td>24.52</td><td>7.24</td><td>96.69</td><td>1.70</td><td>1.74</td><td>1.88</td><td>1.99</td><td>1.00</td></tr><tr><td>GPT-4</td><td>26.51</td><td>6.79</td><td>18.23</td><td>5.31</td><td>96.59</td><td>1.80</td><td>1.99</td><td>2.06</td><td>1.89</td><td>1.00</td></tr><tr><td rowspan="7">Open</td><td>Baichuan2-7B-Chat</td><td>15.40</td><td>3.69</td><td>11.84</td><td>3.46</td><td>94.14</td><td>1.35</td><td>1.34</td><td>1.44</td><td>1.49</td><td>1.00</td></tr><tr><td>GLM4-9B-Chat</td><td>23.38</td><td>5.45</td><td>14.35</td><td>3.84</td><td>96.58</td><td>1.68</td><td>1.88</td><td>1.94</td><td>1.74</td><td>1.00</td></tr><tr><td>InternLM2-Chat-7B</td><td>27.15</td><td>5.87</td><td>20.38</td><td>5.49</td><td>96.62</td><td>1.87</td><td>1.92</td><td>2.04</td><td>2.05</td><td>1.00</td></tr><tr><td>Llama3-8B-Instruct</td><td>26.31</td><td>5.25</td><td>19.64</td><td>5.11</td><td>95.16</td><td>1.58</td><td>1.72</td><td>1.77</td><td>1.81</td><td>1.00</td></tr><tr><td>Llama3.1-8B-Instruct</td><td>30.20</td><td>5.84</td><td>22.88</td><td>5.96</td><td>96.54</td><td>1.61</td><td>1.70</td><td>1.81</td><td>1.90</td><td>1.00</td></tr><tr><td>Qwen2-7B-Instruct</td><td>23.42</td><td>5.28</td><td>15.42</td><td>4.05</td><td>96.64</td><td>1.81</td><td>2.09</td><td>2.18</td><td>2.12</td><td>1.00</td></tr><tr><td>Yi-1.5-9B-Chat</td><td>29.32</td><td>6.89</td><td>21.85</td><td>7.50</td><td>96.66</td><td>1.75</td><td>1.79</td><td>2.11</td><td>1.93</td><td>1.00</td></tr><tr><td rowspan="6">Domain</td><td>MeChat</td><td>30.71</td><td>7.05</td><td>24.43</td><td>6.73</td><td>96.55</td><td>1.54</td><td>1.58</td><td>1.66</td><td>1.96</td><td>1.00</td></tr><tr><td>PsyChat</td><td>27.96</td><td>5.21</td><td>21.44</td><td>4.83</td><td>96.19</td><td>1.36</td><td>1.40</td><td>1.34</td><td>1.79</td><td>1.00</td></tr><tr><td>SoulChat</td><td>28.93</td><td>5.93</td><td>23.26</td><td>5.49</td><td>96.42</td><td>1.29</td><td>1.36</td><td>1.42</td><td>1.76</td><td>1.00</td></tr><tr><td>MindChat</td><td>22.55</td><td>3.44</td><td>17.75</td><td>3.48</td><td>93.89</td><td>1.13</td><td>1.25</td><td>1.13</td><td>1.54</td><td>1.00</td></tr><tr><td>EmoLLM</td><td>23.26</td><td>4.01</td><td>18.50</td><td>3.74</td><td>91.74</td><td>1.06</td><td>1.18</td><td>1.21</td><td>1.36</td><td>1.00</td></tr><tr><td>CPsyCounX</td><td>23.71</td><td>4.32</td><td>17.59</td><td>3.59</td><td>95.46</td><td>1.28</td><td>1.42</td><td>1.54</td><td>1.60</td><td>1.00</td></tr><tr><td>Our</td><td>PsyDTLLM</td><td>36.03</td><td>10.08</td><td>28.86</td><td>9.99</td><td>96.79</td><td>1.90</td><td>2.13</td><td>2.19</td><td>2.26</td><td>1.00</td></tr></table>

Briggs Type Indicator (MBTI) (Hammer, 1996) is another influential model, categorizing individuals into 16 distinct personality types based on four dichotomous scales: Extraversion-Introversion, Sensing-Intuition, Thinking-Feeling, and Judging-Perceiving. Raymond Cattell’s theory (Cattell, 1946), derived from factor analysis, identified 16 primary factors that describe human personality. Cattell’s model has significantly influenced the development of various personality assessment tools, including the Sixteen Personality Factor Questionnaire (16PF).

## 4.3 Therapy Techniques

Psychological therapy techniques significantly contribute to individual mental well-being and overall quality of life. These methods support individuals in identifying, addressing, and managing psychological challenges and conflicts (Meier, 2010). Rational Emotive Behavior Therapy (REBT) (Dryden, 2005), developed by psychologist Albert Ellis in the 1950s, is a prominent form of psychotherapy that highlights the role of irrational beliefs in causing emotional distress and dysfunctional behavior. REBT posits that psychological disturbances are not caused by events themselves but by individuals’ interpretations and reactions to these events. Cognitive Behavioral Therapy (CBT) (Beck, 1979), primarily developed by Aaron Beck in the 1960s, is a widely practiced and highly effective form of psychotherapy that focuses on the interplay between thoughts, emotions, and behaviors. CBT is recognized as one of the most prominent and well-researched approaches in the field of psychology. Dialectical Behavior Therapy (DBT) (Linehan, 2014), created by psychologist Marsha M. Linehan in the late 1980s, is a comprehensive psychotherapeutic approach initially designed for individuals with borderline personality disorder (BPD). It has since been adapted to treat various other mental health conditions, including depression, anxiety disorders, substance abuse, and eating disorders.

## 5 Conclusion

In this paper, we propose PsyDT, a novel framework using LLMs to construct the digital twin of psychological counselor with personalized counseling style. The proposed multi-turn dialogues synthesis method of PsyDT framework can quickly and cost-effectively synthesize PsyDTCorpus, a high-quality multi-turn mental health dialogues dataset of psychological counselor with specific counseling style, which closely resemble realworld counseling cases. This indicates the strong potential of PsyDT for application in real-world psychological counseling.

## Limitation and Future Work

Although the experimental results demonstrate the effectiveness of PsyDT, some limitations need to be considered. Psychological counseling is complex. Our framework only constructs digital twin of psychological counselor with specific counseling style, which satisfies the individual needs of clients who seek specific counseling style, but can not guarantee to solve their psychological problems and meet counseling needs of all clients. In addition, relying solely on one psychological counselor is somewhat arbitrary, we hope to continue exploring the application of multiple psychological counselors in the joint diagnosis in the field of mental health LLMs in the future.

## Ethical Statement

• Data Privacy: We implemented a rigorous data cleaning protocol to synthesize our dataset and ensure privacy protection (Hovy and Spruit, 2016). These measures included rule-based cleaning, manual rewriting, and human proofreading to guarantee the absence of sensitive or private content. For instance, the initial data collection contained private information from psychologists, such as personal details, contact information, addresses, and workplaces. Following the cleaning process, all such sensitive information was entirely removed, ensuring the protection of personal data. Additionally, any conversations with potential harm to clients, others, or society were thoroughly expunged from our dataset.

• Potential Risks of the Model: During the human evaluation phase, we conducted a focused safety assessment of the model’s outputs. Given the absence of human feedback during the model fine-tuning phase, some responses may inevitably pose potential harm to users. If there is no noticeable improvement in user interactions with the PsyDTLLM model, we strongly recommend seeking immediate assistance from a professional counselor or psychiatrist. It is crucial to remember that virtual conversational agents cannot replace real-world therapy. Furthermore, when implementing this model in downstream applications, responses generated by the AI should be used solely as references.

• Annotator Compensation: We engaged experts in psychology to conduct a manual evaluation of the model’s output. Each annotator spent approximately 3 minutes assessing each sample, for which they received a compensation of ¥3. This corresponds to an hourly wage of ¥6, which is higher than the current Chinese minimum wage of ¥20 per hour.

## Acknowledgements

This work was supported by the Guangdong Basic and Applied Basic Research Foundation (2025A1515011203) and Guangdong Provincial Key Laboratory of Human Digital Twin (2022B1212010004).

## References

American Psychological Association et al. 2012. Competency benchmarks in professional psychology.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, et al. 2023. Qwen technical report. Preprint, arXiv:2309.16609.

Aaron T Beck. 1979. Cognitive therapy ofdepression. Guilford press.

Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, et al. 2024. Internlm2 technical report. Preprint, arXiv:2403.17297.

Raymond B Cattell. 1946. The description andmeasurement of personality. Yonkers-on-Hudson, NY: World.

Yirong Chen, Xiaofen Xing, Jingkai Lin, Huimin Zheng, Zhenyu Wang, Qi Liu, and Xiangmin Xu. 2023. SoulChat: Improving LLMs’ empathy, listening, and comfort abilities through fine-tuning with multi-turn empathy conversations. In Findings ofthe Association for Computational Linguistics: EMNLP 2023, pages 1170–1183, Singapore. Association for Computational Linguistics.

Cheng-Han Chiang and Hung-yi Lee. 2023. Can large language models be an alternative to human evaluations? In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), Toronto, Canada. Association for Computational Linguistics.

Paul T Costa and Robert R McCrae. 1999. A fivefactor theory of personality. The five-factor model of personality: Theoretical perspectives, 2:51–87.

Windy Dryden. 2005. Rational emotive behavior therapy. Encyclopedia of cognitive behavior therapy, pages 321–324.

Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. 2022. GLM:

General language model pretraining with autoregressive blank infilling. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 320–335, Dublin, Ireland. Association for Computational Linguistics.

EmoLLM. 2024. Emollm.

Joseph L Fleiss. 1971. Measuring nominal scale agreement among many raters. Psychological Bulletin, 76(5):378–382.

Team GLM, :, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, et al. 2024. Chatglm: A family of large language models from glm-130b to glm-4 all tools. Preprint, arXiv:2406.12793.

Allen L Hammer. 1996. MBTI applications: A decade ofresearch on the Myers-Briggs Type Indicator. Consulting Psychologists Press.

Lucia Hernawati, Bagus Wismanto, and Rachmad Djati Winarno. 2019. The differences of counseling satisfaction between counselees with counseling style matches with their personalities and those who are not. Open Universiteit www. ou. nl Cyberjaya University College of Medical Science De La Salle University-Dasmarinas, page 54.

Dirk Hovy and Shannon L. Spruit. 2016. The social impact of natural language processing. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 591–598, Berlin, Germany. Association for Computational Linguistics.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Marsha Linehan. 2014. DBT? Skills training manual. Guilford Publications.

Gerald Matthews, Ian J Deary, and Martha C Whiteman. 2003. Personality traits. Cambridge University Press.

Augustine Meier. 2010. Counselling and therapy techniques: Theory & practice.

OpenAI. 2022. Introducing chatgpt.

OpenAI. 2024. Gpt-4 technical report. Preprint, arXiv:2303.08774.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, et al. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730–27744. Curran Associates, Inc.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings ofthe 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.

Nancy Pinchot, Anthony C Riccio, and Herman J Peters. 1975. Elementary school students’ and their parents preferences for counselors. Counselor Education and Supervision, 15(1):28–33.

Stephanie D Preston and Frans BM De Waal. 2002. Empathy: Its ultimate and proximate bases. Behavioral and brain sciences, 25(1):1–20.

Huachuan Qiu, Hongliang He, Shuai Zhang, Anqi Li, and Zhenzhong Lan. 2024a. SMILE: Single-turn to multi-turn inclusive language expansion via ChatGPT for mental health support. In Findings of the Associationfor Computational Linguistics: EMNLP 2024, pages 615–636, Miami, Florida, USA. Association for Computational Linguistics.

Huachuan Qiu, Anqi Li, Lizhi Ma, and Zhenzhong Lan. 2024b. Psychat: A client-centric dialogue system for mental health support. In 2024 27th International Conference on Computer Supported Cooperative Work in Design (CSCWD), pages 2979–2984.

Damian F Santomauro, Ana M Mantilla Herrera, Jamileh Shadid, Peng Zheng, Charlie Ashbaugh, et al. 2021. Global prevalence and burden of depressive and anxiety disorders in 204 countries and territories in 2020 due to the covid-19 pandemic. The Lancet, 398(10312):1700–1712.

Zhuang She, Tomáš Rihá<sup>ˇ</sup> cek, Jun Xu, Wenxian Yang,ˇ Dan Xu, Ningning Zhou, Weidong Ji, and Juzhe Xi. 2023. Psychometric evaluation of the cooper– norcross inventory of preferences–therapist version. Assessment, 30(5):1651–1661.

Jacqueline M Swank and Glenn W Lambie. 2012. The assessment of cacrep core curricular areas and student learning outcomes using the counseling competencies scale. Counseling Outcome Research and Evaluation, 3(2):116–127.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models. Preprint, arXiv:2302.13971.

Dong Xue\* Xin Yan. 2023. Mindchat: Psychological large language model. https://github.com/ X-D-Lab/MindChat.

Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, et al. 2023. Baichuan 2: Open large-scale language models. Preprint, arXiv:2309.10305.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, et al. 2024. Qwen2 technical report. Preprint, arXiv:2407.10671.

Alex Young, Bei Chen, Chao Li, Chengen Huang, et al. 2024. Yi: Open foundation models by 01.ai. Preprint, arXiv:2403.04652.

Chenhao Zhang, Renhao Li, Minghuan Tan, Min Yang, Jingwei Zhu, Di Yang, Jiahao Zhao, Guancheng Ye, Chengming Li, and Xiping Hu. 2024. CPsyCoun: A report-based multi-turn dialogue reconstruction and evaluation framework for Chinese psychological counseling. In Findings of the Association for Computational Linguistics ACL 2024, pages 13947– 13966, Bangkok, Thailand and virtual meeting. Association for Computational Linguistics.

Tianyi Zhang\*, Varsha Kishore\*, Felix Wu\*, Kilian Q. Weinberger, and Yoav Artzi. 2020. bertbertscore: Evaluating text generation with bert. In International Conference on Learning Representations.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. 2024. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand. Association for Computational Linguistics.

## A Reproducibility Checklist

• System Hardware: We train the PsyDTLLM on the Ubuntu 20.04.6 LTS server that has 2 CPUs called "Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz", 8 NVIDIA A800- SXM4-80GB GPUs, and 1,024GB memory.

• Driver Version: The version of Nvidia driver is "525.105.17". The version of CUDA is "24.1.2"

• Package version: python=3.8, torch<sup>5</sup>=2.3.0, transformers<sup>6</sup>=4.43.0, deepspeed<sup>7</sup>=0.14.0, datasets=2.18.0 and jieba=0.42.1 are recommended.

## B Details of PsyDT Framework

## B.1 Linguistic Style and Therapy Technique Summarization

The prompt of the summarizing linguistic style and one example of the summarized linguistic style from counseling case of Figure 8 are illustrated in Figure 9, 10. The prompt of the summarizing therapeutic type is shown in Figure 11. Some therapeutic types and details of the knowledge base are presented in Table 3, 4.

## B.2 Client Personality Simulation

The prompt of the simulating client personality and an example of simulated client personality traits from single-turn dialogue of Figure 7 are illustrated in Figure 12, 13.

## B.3 Multi-Turn Mental Health Dialogues Synthesis

The prompt used for synthesizing multi-turn dialogues and an example of PsyDTCorpus from single-turn dialogue of Figure 7 are illustrated in Figure 14, 15.

## C Details of Experiments

## C.1 Similarity Analysis of PsyDTCorpus Dataset

The detailed content of the prompt of similarity analysis is shown in Figure 16, 17.

## C.2 Model Comparisons of PsyDTLLM Model

The detailed content of the prompt of model comparisons is shown in Figure 18, 19, 20, 21, and 22.

## D Case Study

In this section, we present an close look for examples of clients seeking emotional support from ChatGPT, Qwen2-7B-Instruct, SoulChat, CPsy-CounX, and PsyDTLLM via case studies. as shown in Figure 23, 24, 25, 26, and 27, respectively. ChatGPT and Qwen2-7B-Instruct tend to provide advice rather than asking questions or listening. SoulChat fails to satisfy the individual need of client who seek specific counseling styles. CPsy-CounX tries to explicitly use some professional therapy terms in its replies, which may cause confusion for clients who are not in the field of psychology. Our PsyDTLLM, like a real-world psychological counselor, implicitly expresses emotional empathy and cognitive empathy to meet the needs of clients who seek specific counseling style.

Algorithm 1 The multi-turn dialogues synthesis method of PsyDT framework.   
Initialize: topic\_rc: real-world counseling case of current counseling topic;   
rc\_list: 12 real-world counseling cases list with 12 counseling topic;   
topic\_ls: linguistic style of counseling case of current counseling topic;   
ls\_list: linguistic styles list of counseling cases with 12 counseling topics;   
topic\_ttype: therapeutic type of counseling case of current counseling topic;   
ttype\_list: therapeutic types list of counseling cases with 12 counseling topics;   
tt\_list: therapeutic techniques list of counseling cases with 12 counseling topics;   
st\_dial: current single-turn dialogue, including counseling topic, client’s question, and counselor’s long  
text response;   
st\_dial\_list: given 5000 single-turn dialogues list;   
st\_dial\_cp: simulated client’s Big Five personality traits, including Openness, Conscientiousness, Ex  
traversion, Agreeableness, and Neuroticism (OCEAN);   
psy\_ls: linguistic style of specific psychological counselor of current counseling topic;   
psy\_tt: specific knowledge of therapy technique of specific psychological counselor of current counseling   
topic;   
mt\_dial: multi-turn dialogue synthesized from the current single-turn dialogue st\_dial;   
emo\_gui: emotion guide;   
res\_gui: response guide;   
Output: mt\_dial\_list: synthetic multi-turn dialogues list. 1: for topic\_rc in range rc\_list do 2: topic\_ls GPT-4.LingStyPromptGen(topic\_rc); 3: ls\_list.append(topic\_ls); 4: topic\_ttype GPT-4.TheTypePromptGen(topic\_rc); 5: ttype\_list.append(topic\_ttype); 6: end for 7: tt\_list KB\_Retrieve(ttype\_list); 8: for st\_dial in range st\_dial\_list do 9: st\_dial\_cp GPT-4.CliPerPromptGen(st\_dial.client\_problem);   
10: psy\_ls TopicMatchLing(st\_dial.counseling\_topic, ls\_list);   
11: psy\_tt TopicMatchTher( st\_dial.counseling\_topic, tt\_list);   
12: mt\_dial GPT-4.MultiTurnPromptGen(topic\_rc, psy\_ls, emo\_gui, psy\_tt, st\_dial, st\_dial\_cp, res\_gui);   
13: mt\_dial\_list.append(mt\_dial);   
14: end for   
15: return mt\_dial\_list;

“咨询话题”：人际 咨询话题”：人际

“来访者问题主题”：敏感多疑，一听到她们就 会影响自己的情绪，无助痛苦？   
“来访者问题细节”：我之前有一个从初中玩到 高中也是我唯一最好的朋友A，高二时候又有一 个女生B和我们一起玩，她们之间有共同话题， 平时一起走，B很爱站中间，于是一路上我基本 不说话，B说话很毒，可能她是开玩笑的，总之 她的言语伤害到了我，我的别班朋友C就去骂了 她，于是我与B绝交了。但是后来我发现我如果 不去找A，A就不会主动来找我，我们的友谊都是 靠我维持我感觉很累，没绝交前A说生日会叫我 和B，绝交后慢慢到了A生日那天，A叫了B和B的 朋友们，A连电话都没有给我打，我很伤心。在 班级我只和A玩后来我变得没有朋友。高三后换 了新同桌好了一点，可是还是很敏感，一听到A 和B笑的很开心，我就会处于很低落的情绪，我 该怎么办，我知道学业为重，可是我没办法控制 自己一听到她们就会影响自己的情绪，很无助痛 苦。   
“咨询师长文本回复”：你好，我很理解你目前 的感受。你与朋友A之间的友情受到了其他人的 干扰，让你感到无助和痛苦。这种情况下，你可 能会感到被抛弃，以及对于友情的失落和伤感。 对于A的行为，你感到很困惑和伤心，这些情绪 都是非常正常的。在这种情况下，你的自尊和自 信可能会受到挑战，这也会影响到你的学业和生 活。\n\n在这个过程中，你可能也会对A产生怀 疑和不信任的情绪。我建议你可以尝试跟A进行 一次单独的交流，表达你的感受和困惑。也许她 并不是故意要伤害你，而是在处理人际关系时产 生了困扰。通过坦诚的对话，也许可以解开一些 误会，或者找到一种更好的处理方式。\n\n同时， 我明白这段经历会让你感到疲惫和沮丧。在这个 过程中，保持对自己的关爱和呵护是非常重要的。 你可以尝试关注一些自己喜欢的事情，找到一些 能够让自己放松的方式，比如运动、阅读、绘画 等等。通过这些方式，你可以慢慢地调整自己的 情绪状态，让自己更加平静和坚强地面对这段经 历。\n\n最后，我希望你知道，这段经历并不会 一直影响你的生活。每个人都会在人际关系中遇 到挑战，这也是成长的一部分。相信自己，你一 定可以度过这个难关，也会遇到更多真正懂得珍 惜你的朋友。如果你觉得情绪上的困扰影响到了 你的日常生活，也可以考虑寻求专业心理咨询师 的帮助，他们会给予你更专业的支持和指导。 “Counseling Topic”: Relationship   
“Title of Client’s Question”: Interpersonal Sensitivity and Emotional Distress “Detail of Client’s Question”: I previously had a best friend, A, whom I had known since middle school, and she was the only friend I had. In high school, another girl, B, joined our group. B and A shared common interests, and they often walked together, leaving me mostly silent during these moments. B’s remarks, though perhaps intended as jokes, were often hurtful. Eventually, my friend C, from another class, confronted B on my behalf, which led to my falling out with her. Over time, I realized that if I didn’t reach out to A, she wouldn’t initiate contact, and it felt like I was the one keeping our friendship alive. This was draining. Before we parted ways, A had invited both B and me to her birthday, but after the fallout, A invited B and her friends without even calling me. I felt deeply hurt. In class, I only interacted with A, and as a result, I ended up isolated, with no other friends. Later, after switching desks, I became less lonely, but I still felt very sensitive. Hearing A and B laugh together would instantly depress me, and I found myself unable to control my emotions. I feel helpless and distressed. Although I understand the importance of focusing on my studies, I cannot stop how I feel whenever I hear them; it’s overwhelming and painful.   
“Counselor’s Long-Text Response”: Hello, I can deeply empathize with the feelings you’re experiencing. The dynamics in your friendship with A,   
especially the interference caused by B, have left you feeling helpless and distressed. It is completely natural to feel a sense of abandonment and loss when a close friendship begins to deteriorate. Your confusion and hurt   
regarding A’s actions are understandable. These emotions may challenge your sense of self-worth and confidence, which can, in turn, affect both your   
academic and personal life. It seems that this experience has also led to feelings of doubt and mistrust towards A. I would suggest considering the possibility of having a direct conversation with her, where you can openly share your   
feelings and concerns. It is possible that A did not intend to hurt you and may have found herself uncertain about how to navigate the complex dynamics in your friendship. A candid dialogue might help clear up misunderstandings or offer a better way to resolve these issues. At the same time, I recognize how emotionally exhausting and disheartening this situation has been for you. In times like these, it is crucial to show kindness and care towards yourself. I encourage you to engage in activities that you enjoy and that can help alleviate your stress, such as exercising, reading, or engaging in a creative hobby like painting. These activities may help you regain emotional balance, allowing you to approach your situation with a clearer and more resilient mindset. Lastly, please remember that this difficult chapter will not define your life. Everyone encounters challenges in relationships, and these experiences are part of personal growth. Trust in your ability to overcome this, and believe that you will find friends who truly appreciate and value you. If you feel that these emotional struggles are affecting your day-to-day life significantly, seeking guidance from a professional therapist could provide you with more structured support and strategies to cope.

Figure 7: An example of the single-turn long-text dialogues.

来访者：我现在在班级里担任班长一职，在处理班级事务的过程中感觉到同学之间很冷淡，对于我组织的一些活动，同学们的参与度普遍不高，让我很苦恼。

咨询师：您的预期是大家都来参与，但实际结果是同学们参与度不高，是这样吗？您说的同学之间很冷淡，是指的同学们不爱彼此交往，还是只对组织的活动冷淡？

来访者：我的预期倒也不是希望班里的同学都来，因为上了研究生了大家有自己的科研压力，无暇参加班级活动倒也正常，只是我后面听说很多人其实那天并没有科研任务，只是单纯不想参与班级活动而已。

咨询师：我们组织了活动，肯定是希望大家都来参与。辛辛苦苦付出了，没有被看到，心里会有些不舒服。

来访者：就是他们觉得班级里同学之间的交流是没什么必要的，他们只在实验室自己的小团体活跃，跟大班级的交流感觉像无效社交。

咨询师：因为大家没来参与，这件事，对您的影响大吗？

来访者：是的呀，那时候辛辛苦苦组织了一场活动，还拉了个群，我们班总共有60个人，进群有40个人，最后只来了15个人，让我感到很失望。

咨询师：之前有过类似经历吗？比如大学期间，组织活动，来参加的人远远小于预期。您提到“很失望”，看得出来，您是一位责任心很强的人，因为有强烈的责任心，所以对活动的参与人数也有期待

来访者：本科的时候还好，班级的同学还是很团结的，记得当时班级破冰30个人才一两个人没来，当时玩得还是挺开心的，上了研究生之后感觉大家都很冷漠。我作为班长，还是希望能够提升班级的凝聚力，不会做到以后大家都不认识我这个班长，大家彼此相互不认识的现象，如果这样的话会让我感到我很失职。

咨询师：明白了，同学们没来参与活动，会让您感觉到，是您哪些地方没有做到位，自己失职了。有没有这种可能性：您已经把活动准备的很精细到位只是大家因为年龄、兴趣、科研任务等原因，很少人来参与活动？

来访者：可能是吧，因为那次活动是一次桌游活动，首先会有一个自我介绍环节，就是想让大家相互熟悉，然后有后面的娱乐活动。

咨询师：本来您是想通过这场活动，组织大家相互认识。看得出来，您确实动了一番脑筋，精心策划了的，这种情况下，来的人不到预期的一半，确实很失望。

来访者：后面我猜测大家是不是喜欢球类运动多一些，也拉了一个羽毛球群，但是进群的人很多，很多次我在上面喊大家来打羽毛球，但是只有三四个人来打，群里只有我一个人的独角戏。

咨询师：有您做大家的班长，感觉他们挺幸运的，即便错过了上一场活动，大概率还会有下一场。

Client: I currently serve as the class president. However, I've noticed a general sense of indifference among my classmates when it comes to participating in class activities. Their lack of engagement has been particularly frustrating for me.

Counselor: It seems you expected greater participation, but the actual turnout has been disappointing. When you mention "indifference," do you mean that your classmates are generally unengaged with one another, or is it specifically related to the activities you organize?

Client: My expectation wasn't that everyone would participate, especially since we're now in graduate school, where research demands can be overwhelming. It's understandable if some people don't attend class activities. However, I later heard that many classmates didn't have any research commitments that day—they simply chose not to participate. Counselor: Organizing an event naturally comes with the hope that many will attend. It can be disheartening when your efforts aren't recognized or appreciated.

Client: It seems that many of them view interactions within the class as unnecessary. They are more active within their smaller lab groups and see engagement with the larger class as a form of ineffective socializing. Counselor: Did the low attendance have a significant impact on you? Client: Yes, it did. I put a lot of effort into organizing the event, even creating a group chat for it. Out of the 60 classmates, 40 joined the chat, but only 15 attended the event. I was very disappointed.

Counselor: Have you experienced similar situations before? For example, during your undergraduate years, did you organize events where attendance fell short of expectations? You mentioned feeling "disappointed," which suggests you have a strong sense of responsibility. This likely contributes to your high expectations for event participation.

Client: During my undergraduate years, things were different. My classmates were more united. I recall organizing an ice-breaking event where only one or two out of 30 people didn't show up. It was a lot of fun back then. But in graduate school, the atmosphere feels much colder. As class president, I want to strengthen our class cohesion. It would be disheartening if, by the end of our studies, everyone barely knows who I am, or worse, if they don’t know each other at all. That would make me feel like I've failed in my role.

Counselor: I understand. When your classmates don't participate, it makes you question whether you've done enough in your role. Is it possible that you've organized the event meticulously, but due to factors like age, differing interests, and research obligations, many didn't feel compelled to join?

Client: That could be the case. The event I organized was a board game session, beginning with introductions to help everyone get to know each other, followed by the games themselves.

Counselor: It’s clear that you put a lot of thought into organizing the event with the aim of fostering connections among your classmates. Given that less than half of the expected attendees showed up, it's understandable that you'd feel disappointed.

Client: Later on, I thought they might prefer sports, so I created a badminton group chat. Although many joined the group, when I invited them to play, only three or four people ever showed up. It felt like I was just talking to myself.

Counselor: Your classmates are fortunate to have you as their class president.   
Even if they missed last event, there's a good chance you’ll organize another.

![](images/67423283b66c07be015c7a9efadef2e816ccd6325e152532636ff98d7421c17e.jpg)  
Figure 8: An example of real-world counseling cases of psychological counselor with specific counseling style.

![](images/5963d664a37e82e85616bb7700ed0d91912890d9fd92dbf9708e5e33d8bef0ed.jpg)  
Figure 9: The prompt used for summarizing linguistic style of real-world counseling cases.

![](images/6c2e3849a5aedfa6ec912a884b0600d0a22e7d79641f35332334289159adff02.jpg)  
Figure 10: An example of summarized linguistic style of counseling case.

![](images/0096f83203bc9e03d0cfb9f6f6d0dcc562f413682e0720faa75a5eb97ea5a71c.jpg)  
Figure 11: The prompt used for summarizing therapeutic type of real-world counseling cases.

Table 3: Some common therapeutic types
<table><tr><td>School of Thought</td><td>Therapeutic Type</td></tr><tr><td>Psychodynamic</td><td>Psychoanalysis Object-Relations Theory</td></tr><tr><td>Cognitive-Behavioral</td><td>Systematic Desensitization Cognitive Behavioral Therapy Rational Emotive Behavior Therapy Acceptance and Commitment Therapy</td></tr><tr><td>Humanistic</td><td>Person-Centered Therapy Existential Therapy</td></tr><tr><td>Postmodern</td><td>Solution-Focused Therapy Narrative Therapy</td></tr></table>

Table 4: Knowledge base of some common therapy techniques
<table><tr><td>Therapy Technique</td><td>Description</td></tr><tr><td></td><td>REBT involves several stages. Below are the dialogue stages and a brief description of the focus of each stage: (1) Examine Irrational Beliefs and Self-Defeating Thoughts: In REBT, cognitive intervention is viewed as the “lifeblood&quot; of treatment. Thus, almost from the beginning of therapy, during the problem exploration phase, the counselor actively and persuasively helps the client explore the reasons behind their emotional distress. This includes under- standing the client&#x27;s thought process regarding events, the antecedents, and consequences of their emotions to clarify the problem. The counselor firmly encourages the client to reflect on what they “tell&quot;themselves when feeling anxious, depressed, or angry after encountering a stimulus event. (2) Debate Irrational Beliefs: The counselor uses various techniques (primarily cog- nitive techniques) to help the client question and challenge their irrational beliefs and thoughts, demonstrating their unrealistic and unreasonable aspects. The goal is for the</td></tr><tr><td>Rational Emotive Behavior Therapy</td><td>client to recognize the harm of these beliefs and develop a desire and behavior to abandon them. (3) Develop Rational Beliefs and Learn Rational Thinking: After identifying and refuting irrational beliefs, the counselor further guides the client to find appropriate, rational responses to triggering situations and events. The counselor helps the client replace irrational beliefs and self-defeating thoughts with rational beliefs and objective, problem-solving thinking statements. To reinforce rational beliefs, the counselor repeat- edly teaches the client why rational beliefs are reasonable, how they differ from irrational</td></tr><tr><td></td><td>beliefs, why irrational beliefs lead to emotional disturbances, and why rational beliefs lead to more positive, healthy outcomes. (4) Apply Therapeutic Gains in Real Life: The counselor encourages the client to internalize the objective, realistic attitudes and scientifically rational thinking methods learned in therapy and to persistently apply them to solve new problems in their future life.</td></tr></table>

![](images/4796c1b25d19886cf5b5294fceaf99fd29f4cde4d2a670300baa0604d134281e.jpg)  
Figure 12: The prompt used for simulating client personality traits.

## Personality in English

（1）开放性：来访者可能在开放性上得分不高。她似乎在遇到新朋友B的时候，没有表现出强烈的新事物接受度或者对不同类型的友谊的好奇。她更倾向于与熟悉的朋友A保持关系，而不是尝试与B建立新的联系。同时，来访者对B的开玩笑可能没能很好地适应或理解，这表明了她对新奇或具有挑战性的社交互动的适应能力可能较低。

（2）责任心：来访者在责任心上可能表现中等。在描述中，她没有明显显示出组织性或自律性的问题，也没有涉及任何与勤勉或规则遵守直接相关的行为。她确实尽力维持与朋友A的友谊，这显示了一定程度的承诺和努力。然而，由于没有更多信息，对于她的责任心水平难以作出更具体的判断。

（3）外向性：来访者似乎在外向性上得分较低。她在社交场合中表现出来的行为和感受显示，她可能更喜欢独处或仅与很亲近的朋友互动。她描述了在A和B一起走的时候自己不说话的情况，这可能暗示她在社交活动中更保守和被动。她也表示失去了朋友A之后变得没有朋友，可能显示了对于建立新的社交联系的不积极。

（4）宜人性：来访者在宜人性方面可能表现中等或偏低。她对朋友B的行为感到受伤害，并没有选择直接和B沟通，而是通过另一个朋友C来解决冲突，这可能表示她在处理人际关系的冲突时可能不够直接。她也体现出对A的友谊感到失望，因为A在不经她同意的情况下改变了他们的关系动态，这表明她对友好行为的期待可能较高，但在直接的社交互动中可能比较被动。

（5）神经质：来访者在神经质上可能得分偏高。她提到听到A和B开心地笑时感到低落，表明她的情绪受到周围人的影响较大。她表述自己感到“无助痛苦”，这些都是情绪稳定性低的典型迹象。此外，她提到无法控制听到她们时的情绪反应，这表明她可能经历了频繁的情绪波动，并且难以管理自己的情绪状态。

1. \*\*Openness\*\*: The client may score lower in openness. She does not seem to exhibit a strong receptiveness to new experiences or curiosity about different types of friendships when meeting a new acquaintance, B. Instead, she appears to prefer maintaining her relationship with the familiar friend, A, rather than exploring a new connection with B. Furthermore, the client seems to have difficulty adapting to or understanding B's sense of humor, suggesting a lower adaptability to novel or challenging social interactions.

2. \*\*Conscientiousness\*\*: The client’s conscientiousness appears to be moderate. In her narrative, there is no clear indication of issues with organization or self-discipline, nor does she discuss behaviors related to diligence or rule-following. However, her efforts to maintain the friendship with A demonstrate a certain level of commitment and effort. Without further information, it is difficult to make a more definitive assessment of her conscientiousness.

3. \*\*Extraversion\*\*: The client seems to score low in extraversion. Her behavior and feelings in social settings suggest a preference for solitude or interaction primarily with close friends. For instance, she describes a situation where she remained silent while walking with A and B, which may indicate a more reserved and passive approach to social activities. Additionally, her statement about becoming friendless after losing A suggests a lack of initiative in establishing new social connections.

4. \*\*Agreeableness\*\*: The client may score moderately or low on agreeableness. She expressed feeling hurt by B's actions but did not choose to address the issue directly with B, opting instead to communicate through another friend, C. This could indicate a tendency to avoid direct confrontation in interpersonal conflicts. Additionally, her disappointment with A, who altered their relationship dynamics without her consent, suggests that she has high expectations for friendly behavior but may be more passive in direct social interactions.

5. \*\*Neuroticism\*\*: The client may score high in neuroticism. She mentioned feeling down when hearing A and B laughing together, indicating that her emotions are significantly influenced by the mood of those around her. Her statement about feeling "helpless and distressed" reflects typical signs of low emotional stability. Moreover, her inability to control her emotional reactions when hearing A and B suggests frequent emotional fluctuations and difficulty managing her emotional state.

Figure 13: An example of simulated client personality traits.

![](images/cedf6a523e6aa8fcbcbb3b32ee7952692ba4390ac9ed029aa9c0a7e05d2e876a.jpg)  
Figure 14: The prompt used for synthesizing multi-turn mental health dialogues.

<table><tr><td>来访者：我感觉最近自己情绪特别不稳定，一点小事 就能让我烦躁一整天。 咨询师：听起来您最近遇到了一些让您情绪起伏的事 情。要不您先跟我说说，是什么小事会影响到您的情 绪呢?</td><td>Client: Recently, I&#x27;ve been feeling really emotionally unstable. Even the smallest things can make me irritated for the whole day. Counselor: It seems like you&#x27;ve been encountering situations recently that have caused fluctuations in your emotions. Could you tell me more about the kinds of things that are affecting you?</td></tr><tr><td>来访者：就是我和我从小一起长大的朋友A的关系， 现在好像变了，让我很不舒服。 咨询师：您跟朋友A的关系遭遇了一些变化，这件事 显然让您感觉很难受。您能否分享一下发生了什么事 情？</td><td>Client: It&#x27;s about my relationship with my childhood friend, A. It feels like things have changed, and it&#x27;s making me uncomfortable. Counselor: It sounds like there have been some changes in your relationship with A, which have understandably made you feel upset. Could you share more about what has happened?</td></tr><tr><td>来访者：以前我和A是最好的朋友，后来班上来了个 新同学B，A和B很聊得来，我就慢慢地感觉被排除在 外了。 咨询师：您感觉您和A的友情因为B的到来而变得不同 了，这让您感到被边缘化了是吗？这种感觉肯定不好</td><td>Client: We used to be best friends, but then a new student, B, joined our class. A and B got along really well, and I gradually felt excluded. Counselor: You&#x27;re feeling that your friendship with A has changed because of B&#x27;s arrival, and that sense of exclusion is really difficult. Is that how you would describe it?</td></tr><tr><td>受。 来访者：是的，有次我们一起走路，B老是站在中间， 我就被挤到后面去，整路上我都没怎么说话。 咨询师：您感到自己被物理上和情感上都被隔离了，</td><td>Client: Yes. There was this one time we were walking together, and B kept standing in the middle, pushing me to the back. I hardly said anything the entire time.</td></tr><tr><td>很自然地，这让您感觉被孤立。那时候您是怎么应对 这种情况的呢？ 来访者：我没有当场说什么，我把这件事告诉了另一</td><td>Counselor: You felt physically and emotionally isolated during that moment, which understandably made you feel excluded. How did you handle that situation at the time? Client: I didn&#x27;t say anything at that moment. I told another friend, C,</td></tr><tr><td>个朋友C，她就去骂了B。从那以后，我就没再和B说 过话。 咨询师：您选择通过朋友C来表达自己的不满，这导 致了您和B的关系破裂。这件事后来对您有什么影响 呢？</td><td>about it, and she ended up confronting B. Since then, I haven&#x27;t spoken to B at all. Counselor: You chose to express your dissatisfaction through C, which led to a breakdown in your relationship with B. How did that impact you afterwards?</td></tr><tr><td>来访者：我发现如果我不去找A，她就不会来找我。 我们的友谊感觉都是我一个人在维持，我感觉很累。 咨询师：您感觉自己一直在单方面地努力维持着与A 的友谊，这种不平等的努力让您感到筋疲力尽。这对</td><td>Client: I realized that if I don&#x27;t reach out to A, she won&#x27;t contact me. It feels like I’m the only one trying to maintain our friendship, and it&#x27;s exhausting.</td></tr><tr><td>您来说是个重大的心理负担吧。 来访者：对，后来A生日那天，她没叫我，却叫了B和 B的朋友们，我连个电话都没收到，我真的好伤心。</td><td>Counselor: You&#x27;re feeling that you&#x27;re the only one making an effort to sustain the friendship, and this one-sided effort is draining you. This must be a heavy emotional burden for you.</td></tr><tr><td>咨询师：您因为被忽略而感到伤心，那是非常合理的 感受。您是否和A谈过这个问题，表达了您的感受呢？ 来访者：没有，我不知道该怎么开始，怕让事情变得</td><td>Client: Yes. And on A&#x27;s birthday, she didn&#x27;t invite me, but she invited B and B&#x27;s friends. I didn&#x27;t even get a phone call. It really hurt. Counselor: Feeling left out like that is understandably painful. Have you talked to A about how you&#x27;re feeling?</td></tr><tr><td>更糟。 咨询师：我理解您害怕事情变得更加尴尬的担心。不 过有时候，直接沟通可以帮助我们澄清误解，您认为 呢？</td><td>Client: No, I don&#x27;t know how to start the conversation. I&#x27;m afraid it&#x27;ll make things worse. Counselor: I understand your fear of making things more awkward. However, sometimes direct communication can help clarify</td></tr><tr><td>来访者：我觉得这很难，因为自从那以后我们就没怎 么说过话。而且每当我听到A和B在一起笑的很开心， 我就觉得自己好失败。 咨询师：您感受到了被遗弃的情绪，听到他们的笑声 时会提醒您那种不愉快的感觉。这时候的您肯定感觉 Counselor: It seems like hearing them laugh brings back those feelings</td><td>misunderstandings. What do you think? Client: I think it&#x27;s difficult, especially since we haven&#x27;t talked much since then. And every time I hear A and B laughing together, I feel like a failure.</td></tr></table>

## Dialogue in Chinese

## Dialogue in English

![](images/c66c9f4550ef9ffd1f670999078026f41744f29c0af0ace055747a13191170d9.jpg)  
Figure 15: An example of PsyDTCorpus. The red segments represent client personality from Figure 13, the blue segments indicate the linguistic style from Figure 10, and the green segments correspond to the application of therapy techniques.

![](images/f7a5fd8c588ad2bf318b5ac443989546b372b1055a2bf565864eccc350cdbb53.jpg)  
Figure 16: The prompt used for evaluating linguistic style similarity between dialogues.

![](images/51b58b82458a065448daee8702369318598818b102aba949c96e72fe30844738.jpg)  
Figure 17: The prompt used for evaluating therapy technique similarity between dialogues.

Table 5: Evaluation Metrics.
<table><tr><td>Dimension</td><td>Ability</td><td>Description</td></tr><tr><td rowspan="3">Conversation Strategy</td><td>Inquiry and Questioning</td><td>The questions posed by the counselor facilitate deeper reflection and self- exploration by the client, encouraging dialogue and thorough discussion.</td></tr><tr><td>Feedback and Summary</td><td>During sessions, the counselor effec- tively provides feedback and summa- rizes key points when necessary, ensur- ing mutual understanding and alignment in the discussion.</td></tr><tr><td>Problem Solving and Guidance</td><td>The counselor guides clients in problem- solving, encouraging autonomous reflec- tion and self-discovery rather than pro- viding direct solutions.</td></tr><tr><td rowspan="2">State and Attitude</td><td>Openness and Value Neutrality</td><td>The counselor approaches the client&#x27;s opinions, feelings, and experiences with an open and non-judgmental attitude, refraining from immediate value judg- ments or excessive persuasion, allow- ing the client to make independent deci- sions.</td></tr><tr><td>Emotional Control</td><td>The counselor maintains professional emotional regulation throughout the counseling process, neither over- whelmed by the client&#x27;s emotions nor appearing overly detached.</td></tr><tr><td>Relationship Building</td><td>Relationship Building</td><td>The counselor establishes and maintains a positive relationship with the client characterized by trust, warmth, and un- derstanding.</td></tr><tr><td>Application of Therapy Technique</td><td>Application of Therapy Technique</td><td>The counselor effectively applies strate- gies based on one or more theoretical frameworks to guide the client through problem resolution during the dialogue</td></tr></table>

![](images/3c7a3a6d8a688dc711ebe0517f70df231e4c0e8e92042a7da7a6f2f38eafe1e9.jpg)  
Figure 18: The prompt used for evaluating emotional empathy of LLMs.

![](images/3006f0f9c7d8ef1ad37df0d890d77bd14f6872aa2d53638f0a4c72c00d1fff24.jpg)  
Figure 19: The prompt used for evaluating cognitive empathy of LLMs.

![](images/ac37029667772056dc21015f0f0d52f0d3111feda78c2b219eaf58e492f08f0f.jpg)  
Figure 20: The prompt used for evaluating conversation strategy of LLMs.

![](images/86f53a1da247b77915b6f46e22752237a232a7a12ea77e7ff68f450befd82de0.jpg)  
Figure 21: The prompt used for evaluating state and attitude of LLMs.

![](images/c421b5b2489377274d5cb6aba2c0fb7076224cde7d3cb5ad935387f53a71a90c.jpg)  
Figure 22: The prompt used for evaluating safety of LLMs.

![](images/ed047afec319f90c6c4a5b1f4a9c7eb53f3000ffc4e7796f36335464bd8064fd.jpg)

![](images/9dd68155dcfb6b874a23be4ce6f87ea12110acffc80e790e2626ff1f7a6f13d3.jpg)

Figure 23: A case of a client confiding to ChatGPT.  
![](images/18a7424e04620fdb020f3b04c27b3569b03412b7d399c2ae20630b0f4c799cd9.jpg)  
Figure 24: A case of a client confiding to Qwen2-7B-Instruct.

![](images/d09458e5cd26af5a6a39ddbf5fce6a27f6bfb47a4ea6793c4111baa3e1d4fc91.jpg)

![](images/85d14dea94492dfb8053e9731172e8dcc87af381e0e407f1f896152aa23d1d71.jpg)  
Figure 25: A case of a client confiding to SoulChat.

![](images/25ce4e36cb2da51a94aff930dd347328263e196185f0aaaba6b14d77b8eadafe.jpg)  
Figure 26: A case of a client confiding to CPsyCounX.

![](images/93e3aa802a6f5c59c539dac7078e4a032d49f32871b48af2a5b00e8856d65bf9.jpg)

我最近总觉得自己很焦虑，身心俱疲。不知道该怎么办。   
I have been feeling anxious and physically and mentally exhausted lately. I don't know what to do.

听起来你现在承受了很多压力，这一定很难受。能具体告诉我，是什么事情让你感到焦虑吗？ It sounds like you are currently under a lot of pressure, which must be very challenging. Could you specify what exactly is causing your anxiety?

![](images/6ccc01fbf74afa192ccf63e6a7bed9ca6e59e387066780643db1ef89e3d76ea0.jpg)

![](images/1a03165052ff3077742a9db75da57e3b4b266bb0083e6be40df28a34e8d5c067.jpg)

我身边的同学太卷了，感觉自己已经很努力了，但还是比不上别人，做什么事情都比别人差，感觉自己不适合 读研，但又怕毕业的时候找不到满意的工作，所以很焦虑。 My classmates are very competitive. I feel that, despite my efforts, I cannot match their performance. I am worried I am not suited for graduate study, yet I also fear not finding a satisfactory job after graduation, which makes me very anxious.

你觉得自己在同学中处于一个比较劣势的位置，这种感觉一定很让人沮丧。你对“不适合读研”的感觉是怎样 的呢？   
You feel disadvantaged among your peers, and this must be very frustrating. How do you experience the feeling of being "unsuited for graduate studies"?

![](images/97fd9947e801ef6c385bfa96648b71f862e729a2930fbc7d16afb28fd576b2e6.jpg)

![](images/91d1398990a4f287c12710f222956bbd19d83c88c92677841198f72eac77cae6.jpg)

当时读这个研究生，纯粹是因为身边的很多同学都去考研，我也就随大流去考了，后面发现自己并不是做科研的料，有时也挺痛苦的。

I pursued this graduate program mainly because many of my peers were doing so, and I followed suit. Later, I realized I am not cut out for research, which has often been quite distressing.

选择考研时可能更多的是受到周围环境的影响，而不是出于你个人的兴趣或优势。你提到了痛苦，能分享一下具体是什么让你感到痛苦吗？

Your decision to pursue graduate studies seems to have been influenced more by your surroundings than by your personal interests or strengths. You mentioned distress—could you elaborate on what specifically causes you this pain?

![](images/371dd069dd6bd608a80d24dfc6f9b4edd5bc5fe42277bc97f9bbf5f9e13896dc.jpg)

![](images/006e641f707a5b129f5c257f81d769f8420f657b32f1dcdf1e3512b6fa1577aa.jpg)

因为我是跨专业考研，研究生的很多课程我都听不懂。另外导师也会额外布置很多科研任务，有时也会pua我，身边的大佬也特别多，无形之中给了我巨大的压力和痛苦。

I switched fields for my graduate studies, and I find many of the courses incomprehensible. Additionally, my advisor assigns numerous research tasks and sometimes employs manipulative tactics. There are also many outstanding peers around me, which adds to my immense pressure and distress.

听你这么说，感觉你承受了很大的学习和心理压力。这种感觉肯定很沉重。在这样的环境中，你通常是如何处理这些压力的呢？

From what you describe, it seems you are under significant academic and psychological pressure. This must be a heavy burden. How do you typically cope with these pressures in such an environment?

![](images/1c063b560882ae3d7a30545e863954cf44a0c8e680f65d67bfde016e0da0a746.jpg)

![](images/c3810e993d5bb49cc9850eed49908c218196a02fd60d9973bd44b85b6086953c.jpg)  
Figure 27: A case of a client confiding to PsyDTLLM.