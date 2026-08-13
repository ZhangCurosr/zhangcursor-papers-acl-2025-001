# PunchBench: Benchmarking MLLMs in Multimodal Punchline Comprehension

Kun Ouyang†‡, Yuanxin Liu†, Shicheng Li†, Yi Liu†, Hao Zhou‡, Fandong Meng‡, Jie Zhou‡, Xu Sun†

† State Key Laboratory of Multimedia Information Processing, School of Computer Science, Peking University ‡ WeChat AI, Tencent Inc., China

kunouyang10@gmail.com, liuyuanxin@stu.pku.edu.cn, {lisc99, imliuyi}@pku.edu.cn, {tuxzhou, fandongmeng, withtomzhou} @tencent.com, xusun@pku.edu.cn

## Abstract

Multimodal punchlines, which involve humor or sarcasm conveyed in image-caption pairs are a popular way of communication on online multimedia platforms. With the rapid development of multimodal large language models (MLLMs), it is essential to assess their ability to effectively comprehend these punchlines. However, existing benchmarks on punchline comprehension suffer from three major limitations: 1) language shortcuts that allow models to solely rely on text, 2) lack of question diversity, and 3) narrow focus on a specific domain of multimodal content (e.g., cartoon). To address these limitations, we introduce a multimodal Punchline comprehension Benchmark. named PunchBench, which is tailored for accurate and comprehensive evaluation of punchline comprehension. To enhance the evaluation accuracy, we generate synonymous and antonymous captions by modifying original captions, which mitigates the impact of shortcuts in the captions. To provide a comprehensive evaluation, PunchBench incorporates diverse question formats and image-captions from various domains. On this basis, we conduct extensive evaluations and reveal a significant gap between state-of-the-art MLLMs and humans in punchline comprehension. To improve punchline comprehension, we propose Simple-to-Complex Chain-of-Question (SC-CoQ) strategy, enabling the models to incrementally address complicated questions by first mastering simple ones. SC-CoQ effectively enhances the performance of various MLLMs on PunchBench, surpassing in-context learning and chain-of-thought. Datasets, codes are publicly available at https://github.com/ OuyangKun10/PunchBench.

## 1 Introduction

Recent research on Multimodal Large Language Models (MLLMs) (Wang et al., 2024; OpenAI,

![](images/8cc9025e1476c030b7055e73b3d2ef19bff8b48202e470346e949cceb6a3f19a.jpg)  
Figure 1: An example of multimodal punchline comprehension. We illustrate the response of CogVLM2 when provided with different captions and question formats.

2024) has made rapid progress in vision-language tasks such as visual question answering (Antol et al., 2015), dense image captioning (Johnson et al., 2016) and optical character recognition (Islam et al., 2017). Despite the advanced capabilities of modern MLLMs in comprehending factual information from visual content, whether they can effectively grasp punchlines within the multimodal context remains an open question.

As illustrated in Figure 1, multimodal punchlines are typically presented as image-caption pairs (Cai et al., 2019), where humor or sarcasm is elicited through a striking contrast or alignment between visual and textual elements. Understanding these punchlines is important yet challenging for the development of MLLMs. On the one hand, multimodal punchlines are an essential way of communication on online multimedia platforms. Improving comprehension of punchlines is crucial for many real-world applications, including Human-AI interaction (Hempelmann and Petrenko, 2015) and sentiment analysis (Mahdaouy et al., 2021). On the other hand, unlike conventional visual question answering and captioning tasks, multimodal punchline understanding necessitates a nuanced perception of visual content, a strong grasp of language prior knowledge, as well as a deep understanding of the interplay between visual and textual information (Jing et al., 2023).

There are some prior studies on multimodal punchline comprehension, attempting to evaluate sarcasm explanation (Desai et al., 2022) and humor comprehension (Hessel et al., 2023), respectively. However, despite the valuable benchmarks presented by these studies, they suffer from three major limitations that hinder an accurate and comprehensive assessment of multimodal punchline comprehension. First, existing benchmarks overlook the potential shortcuts in the captions. As shown in the Yes/No QA task from Figure 1, CogVLM2 (Hong et al., 2024) can correctly identify that the original caption conveys a punchline regarding the image but fails when some words in the original caption are replaced with antonymous or synonymous ones. Additionally, the model can correctly answer Yes/No QA solely based on an inconsistent caption without visual input. This suggests that the model may exploit biased words (e.g. "enjoy," "plenty of") or text-only inconsistencies (e.g., "enjoy flying" versus "not enough legroom") to arrive at the correct answer rather than genuinely understanding the multimodal punchline. Second, most previous benchmarks are constrained to a single question format (Cai et al., 2019; Desai et al., 2022), limiting their ability to assess the robustness of MLLMs across various user question formats. As depicted in Figure 1, the model can answer the Yes/No OA correctly but struggle with the Matching QA, highlighting performance variations across question formats. Third, prior works (Qiao et al., 2023; Hessel et al., 2023) solely focus on humor or sarcasm within a narrow domain (e.g., cartoon). This limits their applicability to broader real-world scenarios that convey punchlines, and hence causes insufficient evaluations.

In light of the above limitations, we introduce a novel multimodal Punchline comprehension Benchmark, PunchBench for short, designed to provide an accurate and comprehensive evaluation of this task. To enhance evaluation accuracy, we modify captions to mitigate the impact of potential shortcuts. Specifically, we apply context consistency adaptation to eliminate inconsistent captions, and then use word substitution and inversion to generate synonymous and antonymous captions with the help of ChatGPT (OpenAI, 2022). Regarding evaluation comprehensiveness, Punch-Bench features diversity across multiple dimensions. For punchline types, it includes both humor and sarcasm. For task types, it involves two levels of punchline understanding: shallow-level punchline perception and deep-level punchline reasoning. Each task employs diverse question formats: Yes/No QA, Matching QA, Multi-option QA and Generation QA. Furthermore, PunchBench spans a wide range of multimodal content domains, including posts, cartoons, comments, and memes. In total, PunchBench comprises 6, 000 image-caption pairs and 54, 000 question-answer pairs, allowing a comprehensive evaluation.

Leveraging PunchBench, we evaluate a range of state-of-the-art MLLMs. The results reveal a significant gap between MLLMs and humans in punchline comprehension. Additionally, the performance of MLLMs varies across different question formats, and shows notable degradation when faced with synonymous or antonymous captions. These observations emphasize the importance of incorporating diverse question formats, synonymous and antonymous captions in the evaluation process.

To improve the punchline understanding ability of MLLMs, we propose a strategy called Simpleto-Complex Chain-of-Question (SC-CoQ), inspired by the simple-to-complex progression for solving complicated problems. SC-CoQ structures questions from simple to complex within and across tasks, enabling the models to incrementally develop the capability to address complex questions by first mastering simple ones. Compared to in-context learning (Brown et al., 2020) and chainof-thought (Wei et al., 2022) methods, SC-CoQ demonstrates superior performance, further validating its effectiveness in promoting punchline comprehension.

In a nutshell, our contributions can be summarized as follows.

• We introduce PunchBench, which, to the best of our knowledge, is the first benchmark for accurate and comprehensive evaluation of multimodal punchline comprehension.

• Extensive evaluations on PunchBench reveal a significant gap between MLLMs and humans in punchline comprehension, and highlights the performance variations across question formats in each task.

• We propose Simple-to-Complex Chain-of-Question (SC-CoQ), which follows a progression from simple to complex questions to effectively improve punchline comprehension.

## 2 Related Works

## 2.1 Multimodal Large Language Models

Large Language Models (LLMs) for pure text like ChatGPT (OpenAI, 2022), GPT-4 (OpenAI et al., 2024), and LLaMA (Touvron et al., 2023) have proved impressive comprehension capabilities of text. Following this success and to expand it on multimodal tasks, many efforts (Li et al., 2023; Liu et al., 2023a) have been made to integrate visual comprehension capability into LLMs, and lead to a blowout of Multimodal Large Language Models (MLLMs), both closed-source models (e.g., GPT-4V (OpenAI, 2023a) and GPT-4o (OpenAI, 2024)) and open-source models (e.g., LLaVA series (Liu et al., 2023a, 2024a,b), CogVLM series (Wang et al., 2023; Hong et al., 2024), Qwen-VL family (Bai et al., 2023; Wang et al., 2024) and GLM-4V (GLM et al., 2024)). They demonstrate unprecedented and surprising multimodal understanding capabilities in vision-language tasks such as visual question answering (Antol et al., 2015), dense image captioning (Johnson et al., 2016) and optical character recognition (Islam et al., 2017).

## 2.2 Punchline Comprehension

Despite significant progress of MLLMs in understanding factual information from visual content (Long et al., 2023; Jian et al., 2024), the punchline comprehension capabilities (Cai et al., 2019; Ouyang et al., 2024) of MLLMs still lack sufficient evaluations. Prior works (Desai et al., 2022; Kumar et al., 2022; Hessel et al., 2023) related to multimodal punchline comprehension have concentrated on sarcasm or humor. For example, Desai et al. curated the MORE dataset for multimodal sarcasm explanation, which aims to explain the ironic semantics of multimodal post. Furthermore, previous benchmarks overlooked potential shortcuts in captions that MLLMs may exploit to answer questions, undermining true comprehension of punchlines. Noticing these concerns, our benchmark is introduced to provide an accurate and comprehensive evaluation of multimodal punchline comprehension.

## 3 PunchBench

As illustrated in Figure 2, our PunchBench is constructed in four steps: Source Data Collection & Annotation (§ 3.1), Synonymous & Antonymous Caption Generation (§ 3.2), Instruction Construction (§ 3.3), Quality Checking (§ 3.4). In this section, we elaborate on the construction process as well as the data statistics (§ 3.5).

## 3.1 Source Data Collection & Annotation

The image-caption pairs in our dataset are obtained from two sources. 1) Prior datasets. Recognizing the wealth of resources in prior datasets that contribute to punchline comprehension, we select three relevant datasets, i.e., MTSD (Castro et al., 2019), MORE (Kumar et al., 2022) and HUB (Hessel et al., 2023). Then, we meticulously filter the high-quality image-caption pairs using a hybrid approach that combines both manual and automatic filtering, as detailed in Appendix A.1. 2) Multimedia platforms. To ensure up-to-date of our dataset, we gather image-caption pairs from the social media platforms, such as X, Instagram, and YouTube. Additionally, we include image-caption pairs from the cartoon websites like CartoonMovement and CartoonStock. The information about these multimeida platforms is provided in Appendix F.

After obtaining the raw set of image-caption pairs, we implement a crowd voting process, which is outlined in Appendix A.1, to identify a label indicating whether the image-caption pair contains punchline. Ultimately, we compile a collection of 6, 000 image-caption pairs spanning diverse scenarios (e.g., cartoon, post, comment, and meme), half of which are identified as containing punchline. To explain why the particular pair contains punchline, we employ three human annotators to handcraft reasoning sentence for it, which is detailed in Appendix A.1. Finally, we acquire 6, 000 image-caption pairs along with their corresponding labels and reasoning sentences. To emphasize the superiority of PunchBench, we provide a comparison between our PunchBench and prior datasets in Table 4.

## 3.2 Synonymous & Antonymous Caption Generation

As aforementioned, MLLMs may exploit shortcuts in the captions, such as word bias and context inconsistency, to answer the question without truly understanding the image-caption pair. To prevent these shortcuts, we generate synonymous caption and antonymous caption for each image-caption pair through following methods. 1) Word substitution and inversion. Assisted by gpt-3.5-turbo-0125, we substitute the sentiment, action, object and other words with synonymous words to generate synonymous caption, and we invert the semantics by replacing these words with their antonyms to obtain antonymous caption. 2) Context consistency adaption. To adapt the consistency of captions containing semantically conflicting components, $e . g .$ , "I am so glad today! What a disgusting rainy day!", we first leverage gpt-3.5-turbo-0125 to identify and isolate the two conflicting parts, “I am so glad today" contradicts “What a disgusting rainy day". And we then employ word substitution and inversion for the two parts to generate synonymous and antonymous caption. We supplement additional implementation details in Appendix A.2.

![](images/b35c51d33a6eecc44a55327a9e519a2249dac1f535b0c0db859ec24e5ae4e372.jpg)  
Figure 2: Upper: Data collection workflow for PunchBench. We first collect image-caption pairs from prior datasets and multimedia platforms with meticulous filtering, conduct human annotation to obtain the corresponding labels and reasoning sentences for the pairs. And we then utilize gpt-3.5-turbo-0125 to generate synonymous and antonymous captions corresponding to the original captions. Based on these image-caption pairs, we construct corresponding instructions for punchline perception and reasoning. Finally, we perform quality checking to ensure the reliability of our PunchBench. Lower: Data examples for Punchline Perception and Punchline Reasoning.

## 3.3 Instruction Construction

Based on the collected image-caption pairs and corresponding annotations, we now construct instructions for two types of tasks: Punchline Perception, which assesses whether an MLLM can identify the existence of punchline in image-caption pairs, and Punchline Reasoning, which requires the model to understand the reason why a particular imagecaption pair contains punchline. Figure 2 illustrates some examples of the instructions. Before delving into the details, we first clarify some notations.

Notations. Each image-caption pair $P _ { i } ^ { x }$ ${ < } I _ { i } , C _ { i } ^ { x } { > }$ consists of an image $I _ { i }$ and a caption $C _ { i } ^ { x }$ where $x \in \{ o , s , a \}$ denotes the original $( C ^ { o } )$ , synonymous $( C ^ { s } )$ and antonymous $( C ^ { a } )$ caption. And each pair is assigned a label $L _ { i } ^ { x } \in \{ 0 , 1 \}$ , where 1 indicates that the pair contains punchline while 0 is opposite. Notably, $P _ { i } ^ { s }$ shares the same label as $P _ { i } ^ { o }$ while $P _ { i } ^ { a }$ serves as the contrast. We detail instruction construction process as follows, temporally omitting the subscript i that indexes the samples for simplicity.

## 3.3.1 Punchline Perception

Yes/No QA. The model is required to answer whether the given image-caption pair $P ^ { x }$ contains punchline. The instruction is derived based on various instruction templates, with the answer “Yes" or “No" being determined by the label $L ^ { x }$ . To attain a balance, the number of negative answers is equal to that of positive answers.

Matching QA. The model is asked to select between two captions, recognizing which one effectively conveys punchline with the given image. For pair $P ^ { x }$ containing punchline, we utilize $\mathsf { g p t } - 4 0 - 2 \theta 2 4 - \theta 5 - 1 3 ^ { 1 }$ to generate a distractor caption $C ^ { d }$ for the image I. The distractor caption $C ^ { d }$ just describes the content of image I without conveying the punchline. Finally, the image-caption pair $P ^ { x }$ , as well as $C _ { d }$ are subsequently integrated into several templates to obtain the instructions. To prevent bias associated with the position of captions, we randomize the order in which the two captions are displayed for each instruction.

![](images/4075353fbe6a7c1bf09c3aaa61c775fc8adc9c8d2809b3fac50ee1ec942e41ab.jpg)  
(a) Distribution of Question Formats

![](images/2c90cd74d5ec41bc8fcb964639a7c96ae32d949b9b4dc1816beb460ab9346df6.jpg)  
(b) Distribution of Image-Caption Pairs  
Figure 3: The overall data statistics of our PunchBench.

Multi-option QA. The model aims to discern the correct one from four options $i . e . , O _ { 1 } , O _ { 2 } , O _ { 3 } , O _ { 4 }$ describing the image-caption pair $P ^ { x }$ . The four options are generated by gpt-3.5-turbo-0125 based on the caption $C ^ { x }$ and former distractor caption $C ^ { d }$ , with only one being correct. These options, along with $P ^ { x }$ are incorporated into the instruction templates. The sequence of the four options are shuffled to avoid the positional bias.

## 3.3.2 Punchline Reasoning

We utilize the 3, 000 pairs $P ^ { o }$ containing punchline, their synonymous captions $C ^ { s }$ and annotated reasoning sentences $R ^ { a }$ to construct instructions for punchline reasoning.

Yes/No QA. Presented with an image-caption pair and a reasoning sentence, the model is asked to identify whether the reasoning sentence succeeds in explaining why the pair contains punchline. Specifically, we first resort to $\mathsf { g p t } - 3 . 5 \mathsf { - t u r b o - } \theta 1 2 5$ to generate distractor reasoning sentence $R ^ { d }$ based on our annotated reasoning sentence $R ^ { a }$ . And we then randomly assign half of the image-caption pairs to annotated reasoning sentences $R ^ { o }$ , while the other part is linked to the distractor ones $R ^ { d }$ , incorporate them into instruction templates. The answer to instruction using $R ^ { a }$ is “Yes” and using $R ^ { d }$ is $\mathbf { \tilde { \Sigma } } ^ { 6 6 } \mathbf { N } \mathbf { o } ^ { \prime 9 }$ Finally, we have an equal number of positive and

negative instructions.

Matching QA. Given an image-caption pair and two reasoning sentences, $i . e . , R ^ { a }$ (correct) and $R ^ { d }$ (distractor), only one of which appropriately interprets the punchline in the pair, the model is required to select the correct reasoning sentence. Specifically, $R ^ { a }$ and $R ^ { d }$ are paired with $P ^ { o }$ or $P ^ { s }$ in several templates to construct the instructions, with the order of $R ^ { d }$ and $R ^ { a }$ being randomly shuffled.

Generation QA. In this task, the image-caption pair is utilized in various instruction templates to prompt the model to generate a reasoning sentence to explain the punchline, with $R ^ { a }$ serving as the reference answer.

The above instructions undergo a thorough review and refinement process by human annotators. The instruction templates and more details of this construction process are supplied in Appendix A.3.

## 3.4 Quality Checking

To ensure the quality of PunchBench, we randomly sample 100 instructions for each question format, excluding Generation $Q A .$ , for quality checking process. Three human annotators are employed to answer the questions guided by the sampled instructions. Human annotators have an extra option $\mathbf { \vec { \Delta } C B A ^ { \prime } }$ that means “Cannot Be Answered"for each question. Among 500 instructions, only 1 is labeled by $\mathbf { \vec { \tau } } \mathbf { C B A } ^ { \prime \prime }$ , which verifies the high quality of the instructions. Moreover, they answer the questions with high accuracy as results reported in Table 1, which further demonstrates the superior quality of our dataset.

## 3.5 Dataset Statistics

We illustrate Figure 3 to exhibit the dataset statistics of our PunchBench. PunchBench consists of 6, 000 image-caption pairs, spanning cartoon, post, comment and meme. Each image has three types of captions: original, synonymous, and antonymous captions. Our question formats include Yes/No QA, Matching QA, and Multi-option QA for punchline perception, and Yes/No QA, Matching QA, and Generation QA for punchline reasoning. Above all, our PunchBench covers a diverse question formats and domains, which can provide a comprehensive evaluation. We also compare our PunchBench with previous Benchmarks in Appendix A.4.

## 4 Simple-to-Complex Chain-of-Question

In our initial evaluation (the “Zero-shot" results in Table 1), we observe that different question formats present varying levels of difficulty for the MLLMs. The general trend for punchline perception is Yes/No QA < Matching QA < Multi-option QA, and for punchline reasoning, it is Yes/No QA < Matching QA < Generation QA, where < indicates easier than. Inspired by these observations, we propose a Simple-to-Complex Chain-of-Question (SC-CoQ) strategy, which prompts MLLMs to answer the simpler questions before solving the most complex questions. Specifically, we introduce two variations of SC-CoQ, Intra-task and Inter-task:

Intra-task SC-CoQ integrates the various formats of questions within the same task to improve performance on the most challenging question (i.e., Multioption QA and Generation QA). We sequence the questions in a specific order mirroring simple to complex, i.e., <Yes/No QA, Matching QA, Multioption QA or Generation QA>.

Inter-task SC-CoQ incorporates similar question formats (i.e., Yes/No QA and Matching QA) across different tasks to enhance punchline comprehension. For Yes/No QA, we sequentially link the questions from the two tasks, $i . e . ,$ <Yes/No $Q A _ { m }$ , Yes/No $Q A _ { n } > \mathrm { o r } < Y e s / N o ~ Q A _ { n } ,$ Yes/No $Q A _ { m } >$ , where m refers to punchline perception task and n denotes punchline reasoning task. For Matching QA, this chain utilizes both Yes/No QA and Matching QA to reinforce punchline comprehension across tasks, i.e., <Yes/No $Q A _ { m }$ , Yes/No $\mathcal { Q } A _ { n } .$ Matching $\begin{array} { r } { Q A _ { m } , } \end{array}$ Matching $Q A _ { n } >$ or <Yes/No $Q A _ { n }$ Yes/No $\mathcal { Q } A _ { m } ,$ Matching $\mathcal { Q } A _ { n } .$ Matching $Q A _ { m } >$ . More details of SC-CoQ and specific prompting examples can be

found in Appendix B.

## 5 Experiments

## 5.1 Baselines

We include both MLLMs and human baseline for evaluation as follows.

Evaluated MLLMs. We evaluate eight opensource MLLMs (i.e., LLaVA (Liu et al., 2024b), GLM-4V (GLM et al., 2024), Qwen2- VL (Wang et al., 2024), CogVLM2 (Hong et al., 2024)), LLaVA-OneVision (Li et al., 2024a), InternVL2.5 (Chen et al., 2024a), MiniCPM-o 2.6 (Yao et al., 2024), and Aria (Li et al., 2024b)) and two closed-source MLLMs (i.e., GPT-4V (OpenAI, 2023a) and GPT-4o (OpenAI, 2024)). And we adopt zero-shot, 3-shot (in-context learning) and Chain-of-Thought (CoT) as the baselines for prompting MLLMs. A detailed description of these models, their parameter settings, and introduction for in-context learning (Brown et al., 2020) and CoT (Wei et al., 2022) are provided in Appendix C. Human Baseline. To make a comparison with human performance on punchline comprehension, we introduce a human baseline. Specifically, 1) for punchline perception, we first randomly select 100 instructions for each question format except Generation QA, and we then recruit human annotators (three undergraduates outside of the work) to answer the questions guided by the instructions. Notably, the manually annotated reasoning sentences serve as the performance of human baseline for the Generation QA.

## 5.2 Evaluation Metric

For Yes/No QA, Matching QA and Multi-option QA, we utilize accuracy as the metric. A response is deemed correct when the candidate option (e.g., Yes/No, Option A/Option B, or A/B/C/D) mentioned in the response matches the ground truth option. The accuracy is then calculated as the ratio of correct responses to the total number of questions. For Generation QA, where the responses from MLLMs are free-form, we resort to gpt-3.5-turbo-0125² to assess whether the response matches the semantics of the annotated reasoning sentence with a binary judgment “Yes" or “No". Responses marked by "Yes" are considered correct and their ratio serves as the accuracy metric. To ensure the reliability of automatic evaluation, we analyze the correlation between automatic and human assessments. The details provided in the Appendix D.3 demonstrate that the automatic metrics align well with human judgments.

<table><tr><td rowspan="2">Model</td><td rowspan="2">#Params</td><td colspan="4">Yes/No QA</td><td colspan="4">Matching QA</td><td colspan="4">Multi-choice QA</td></tr><tr><td>Zero-shot</td><td>CoT</td><td>3 shot</td><td>SC-CoQ</td><td>Zero-shot</td><td>CoT</td><td>3 shot</td><td>SC-CoQ</td><td>Zero-shot</td><td>CoT</td><td>3 shot</td><td>SC-CoQ</td></tr><tr><td>LLaVA</td><td>7B</td><td>62.7</td><td>61.5</td><td>63.5</td><td>64.8*</td><td>54.2</td><td>54.9</td><td>55.8</td><td>57.1*</td><td>36.4</td><td>37.5</td><td>37.2</td><td>39.1*</td></tr><tr><td>GLM-4V</td><td>9B</td><td>61.4</td><td>61.8</td><td>62.2</td><td>63.7*</td><td>55.3</td><td>53.1</td><td>56.9</td><td>57.7*</td><td>38.2</td><td>38.8</td><td>39.5</td><td>40.6*</td></tr><tr><td>Qwen2-VL-2B-Instruct</td><td>2B</td><td>56.9</td><td>57.2</td><td>57.4</td><td>58.0*</td><td>52.3</td><td>52.0</td><td>51.8</td><td>53.2*</td><td>33.1</td><td>33.5</td><td>33.4</td><td>34.1*</td></tr><tr><td>Qwen2-VL-7B-Instruct</td><td>7B</td><td>70.1</td><td>71.9</td><td>72.4</td><td>73.2*</td><td>58.0</td><td>58.4</td><td>59.2</td><td>61.3*</td><td>41.7</td><td>43.0</td><td>42.4</td><td>44.1*</td></tr><tr><td>Qwen2-VL-72B-Instruct</td><td>72B</td><td>73.7</td><td>74.8</td><td>74.5</td><td>76.1*</td><td>60.2</td><td>61.5</td><td>61.7</td><td>62.9*</td><td>48.8</td><td>49.7</td><td>50.1</td><td>51.7*</td></tr><tr><td>CogVLM2</td><td>19B</td><td>68.2</td><td>67.6</td><td>69.5</td><td>71.3*</td><td>57.3</td><td>58.9</td><td>58.6</td><td>60.8*</td><td>43.4</td><td>44.2</td><td>44.7</td><td>46.3*</td></tr><tr><td>LLaVA-OneVision</td><td>7B</td><td>64.3</td><td>65.8</td><td>66.0</td><td>67.2*</td><td>55.9</td><td>56.4</td><td>56.8</td><td>57.9*</td><td>39.7</td><td>41.1</td><td>40.3</td><td>42.4*</td></tr><tr><td>InternVL2.5</td><td>8B</td><td>69.5</td><td>70.1</td><td>70.7</td><td>71.4*</td><td>58.4</td><td>59.0</td><td>59.2</td><td>60.0*</td><td>42.0</td><td>42.9</td><td>43.1</td><td>44.3*</td></tr><tr><td>MiniCPM-0 2.6</td><td>8B</td><td>70.8</td><td>71.7</td><td>71.4</td><td>72.3*</td><td>59.1</td><td>59.6</td><td>60.1</td><td>61.2*</td><td>43.1</td><td>43.7</td><td>43.5</td><td>45.4*</td></tr><tr><td>Aria</td><td>3.5B×8</td><td>72.1</td><td>72.9</td><td>73.2</td><td>74.5*</td><td>61.8</td><td>62.7</td><td>62.3</td><td>63.6*</td><td>47.9</td><td>49.0</td><td>48.6</td><td>50.8*</td></tr><tr><td>GPT-4V</td><td></td><td>75.0</td><td>74.2</td><td>76.2</td><td>78.1*</td><td>62.1</td><td>63.2</td><td>63.9</td><td>65.0*</td><td>48.1</td><td>50.5</td><td>50.3</td><td>51.9*</td></tr><tr><td>GPT-40</td><td></td><td>77.5 98.3</td><td>78.6</td><td>79.2</td><td>80.7*</td><td>64.2</td><td>66.3</td><td>65.4</td><td>67.9*</td><td>50.8</td><td>51.4</td><td>52.0</td><td>53.1*</td></tr><tr><td>Human</td><td></td><td></td><td></td><td></td><td></td><td>97.7</td><td></td><td></td><td></td><td>90.7</td><td></td><td>=</td><td></td></tr><tr><td colspan="10">(a) Punchline Perception</td><td colspan="4"></td></tr><tr><td>Model</td><td>#Params</td><td></td><td>Yes/No QA</td><td></td><td></td><td></td><td>Matching QA</td><td></td><td></td><td></td><td>Generation QA</td><td></td><td></td></tr><tr><td></td><td></td><td>Zero-shot</td><td>CoT</td><td>3 shot</td><td>SC-CoQ</td><td>Zero-shot</td><td>CoT</td><td>3 shot</td><td>SC-CoQ</td><td>Zero-shot</td><td>CoT</td><td>3 shot</td><td>SC-CoQ</td></tr><tr><td>LLaVA GLM-4V</td><td>7B</td><td>60.1</td><td>61.7</td><td>61.3</td><td>62.6*</td><td>50.7</td><td>51.3</td><td>51.9</td><td>53.0*</td><td>35.2</td><td>37.1</td><td>36.6</td><td>38.7*</td></tr><tr><td>Qwen2-VL-2B-Instruct</td><td>9B 2B</td><td>59.7 54.2</td><td>60.8 55.1</td><td>61.3</td><td>62.9*</td><td>53.1</td><td>52.2</td><td>54.8</td><td>55.9*</td><td>37.1</td><td>38.5</td><td>38.2</td><td>39.8*</td></tr><tr><td>Qwen2-VL-7B-Instruct</td><td>7B</td><td>64.5</td><td>65.3</td><td>54.0</td><td>55.9*</td><td>49.5</td><td>49.0</td><td>50.6</td><td>51.4*</td><td>31.7</td><td>32.1</td><td>31.5</td><td>33.2*</td></tr><tr><td>Qwen2-VL-72B-Instruct</td><td>72B</td><td>72.0</td><td>72.7</td><td>66.0</td><td>67.4*</td><td>55.7 57.5</td><td>56.1</td><td>57.2</td><td>58.4*</td><td>40.6</td><td>41.5</td><td>41.9</td><td>43.7*</td></tr><tr><td>CogVLM2</td><td>19B</td><td>66.3</td><td>67.2</td><td>73.0 68.0</td><td>74.9*</td><td>54.2</td><td>59.1</td><td>59.4</td><td>60.4*</td><td>45.0</td><td>46.1</td><td>46.7</td><td>48.0*</td></tr><tr><td>LLaVA-OneVision</td><td>7B</td><td>61.7</td><td>61.2</td><td>62.8</td><td>69.6* 63.9*</td><td>52.4</td><td>54.9 53.5</td><td>55.4 53.9</td><td>56.3*</td><td>41.8</td><td>42.7</td><td>42.5</td><td>43.4* 40.1*</td></tr><tr><td>InternVL2.5</td><td>8B</td><td>63.8</td><td>64.9</td><td></td><td></td><td>54.6</td><td></td><td></td><td>54.7*</td><td>37.5</td><td>38.2</td><td>38.7</td><td>43.0*</td></tr><tr><td></td><td>8B</td><td>67.2</td><td>68.0</td><td>64.3</td><td>65.8*</td><td></td><td>55.8</td><td>55.5</td><td>56.9*</td><td>40.7</td><td>41.6</td><td>41.8</td><td></td></tr><tr><td>MiniCPM-0 2.6</td><td></td><td>70.9</td><td>72.1</td><td>68.4 72.5</td><td>69.7*</td><td>56.0</td><td>56.9</td><td>57.1</td><td>58.4*</td><td>42.5</td><td>43.9</td><td>43.1</td><td>45.2*</td></tr><tr><td>Aria</td><td>3.5B×8</td><td></td><td>74.7</td><td>75.4</td><td>73.8*</td><td>57.6</td><td>58.0</td><td>58.7</td><td>59.8*</td><td>43.9</td><td>45.0</td><td>44.8</td><td>46.3*</td></tr><tr><td>GPT-4V GPT-40</td><td></td><td>73.9</td><td>75.9</td><td></td><td>76.5*</td><td>57.1</td><td>59.0</td><td>58.2</td><td>60.6*</td><td>44.7</td><td>46.4</td><td>45.9</td><td>47.5*</td></tr><tr><td>Human</td><td></td><td>75.1</td><td></td><td>76.2</td><td>77.4*</td><td>59.2</td><td>61.5</td><td>61.2</td><td>62.8*</td><td>47.2</td><td>47.6</td><td>48.7</td><td>50.1*</td></tr><tr><td></td><td></td><td>96.0</td><td>=</td><td></td><td>(b) Punchline Reasoning</td><td>93.0</td><td>-</td><td></td><td>=</td><td>100.0</td><td>-</td><td></td><td></td></tr></table>

Table 1: Evaluation results on PunchBench. The best results among the MLLMs are in boldface, while the second best are underlined. \* denotes the best results among the prompting methods. The results are the average of four replicates. And the P-value between SC-CoQ performance and other prompting method results is consistently less than 0.01.

## 5.3 Main Results

The evaluation results of punchline perception and reasoning are presented in Table 1, and we conclude the following findings from five aspects.

Overall Performance. The evaluated MLLMs exhibit limited capability of punchline comprehension, with the accuracy across different question formats for both punchline perception and reasoning falling below 80% in zero-shot setting. As can be seen, the closed-source models consistently surpass the open-source models, where GPT-4o achieves the leading performance among the evaluated MLLMs. Regrettably, GPT-4o still lags substantially behind human-level performance, revealing a substantial gap in punchline comprehension between MLLMs and humans.

Cross-task Performance. Comparing performance of MLLMs cross the two tasks, we can see that punchline reasoning poses greater challenges than punchline perception, since MLLMs perform worse in punchline reasoning. This disparity is expected, as punchline reasoning demands a deeper understanding to explain why a particular pair contains punchline, rather than simply identifying its presence. Consequently, punchline reasoning proves to be a more complex task for MLLMs compared to punchline perception.

Cross-question Performance. Comparing the results cross question formats within each task, we can observe that there exists a significant variation in performance. The reasons can be two folds. On the one hand, the complexity of the question formats varies inherently. From simplest to most complex, the question formats can be ranked as follows: Yes/No QA, Matching QA, Multi-option QA/Generation QA. MLLMs show a noticeable decline in performance as the complexity of the questions increases. On the other hand, individual models have varying innate strengths and weaknesses across different question formats. For instance, LLaVA exceeds GLM-4V in Yes/No QA but falls behind GLM-4V in Matching QA for punchline perception task.

Effectiveness of SC-CoQ. Compared to the zeroshot setting, both 3-shot and SC-CoQ methods consistently improve performance across all question formats. While CoT method slightly degrades performance in Yes/No QA for punchline perception, it enhances performance in other question formats. Notably, SC-CoQ outperforms both 3-shot and CoT approaches corss various question formats, highlighting its superiority. The effectiveness of SC-CoQ is further validated in Section 5.4, where its performance improvements in synonymous and antonymous caption settings are analyzed.

![](images/9f3f2eea5b5b800f4723601bcd248ba9e70c01d2a41a1457f2fbd606250ed5a4.jpg)  
Figure 4: Performance comparison cross original, synonymous and antonymous captions in zero-shot, 3-shot, CoT and our SC-CoQ.

## 5.4 Effect of Synonymous and Antonymous Captions

To explore the effect of synonymous and antonymous captions, we compare the performance of CogVLM2 cross the original, synonymous and antonymous captions, as illustrated in Figure 4. And the performance comparison for other models are provided in Appendix D.3. We analyze the results from two perspectives: 1) There is a notable drop in model performance across different question formats when replacing the original caption with synonymous or antonymous captions. It suggests that synonymous and antonymous captions effectively successfully eliminate shortcuts found in the original captions and hence challenge models to achieve a thorough comprehension of image-caption pair, which leads to a more comprehensive assessment for punchline comprehension capabilities. 2) When using 3-shot and CoT methods, model performance with synonymous and antonymous captions lags behind that with the original captions. However, the models show significant improvement across original, synonymous and antonymous captions when applying SC-CoQ. It proves that SC-CoQ can enhance the models’ ability to effectively capture the semantics of imagecaption pairs and hence achieve better punchline comprehension.

![](images/0bdf8f656d01f7d9d8e20f02d6d079191621755399cfd1191aa2c8810bf3753e.jpg)  
Figure 5: Example responses from CogVLM2 and GPT-4o to the Yes/No QA with zero-shot prompts. Responses from CogVLM2 to Multi-option QA with different prompting methods are also presented.

## 5.5 Qualitative Analysis

To provide an intuitive display, we illustrate some testing samples in Figure 5 for qualitative analysis. Part (a) showcases the responses from two representative models CogVLM2 and GPT-4o in the Yes/No QA. Both of them answer correctly when given the original caption, but fail when the original caption is replaced by the synonymous or antonymous caption. This indicates the biases existing in the captions and hence the models may not truly understand the inherent semantics of the image-caption pair to attain the answer. And it underscores the significance of introducing synonymous and antonymous captions in assessing punchline comprehension. Part (b) exhibits the responses of CogVLM2 with zero-shot, 3-shot, CoT and SC-CoQ for Multi-option QA. Notably, with the guidance of SC-CoQ, CogVLM2 successfully answers the question, whereas it fails under the other settings (i.e., zero-shot, 3-shot, and CoT). It highlights the effectiveness of SC-CoQ in enhancing punchline comprehension. More qualitative results for other question formats can be found in Appendix D.4.

## 6 Conclusions

We introduce PunchBench, a benchmark designed to evaluate the ability of MLLMs to comprehend multimodal punchlines. PunchBench distinguish itself from existing benchmarks in two key ways: First, it incorporates synonymous and antonymous captions to mitigate the risk of models relying on shortcuts in the original captions, achieving a more accurate assessment of their capabilities. Second,

PunchBench includes a diverse range of punchline types, evaluation tasks, question formats, and multimodal content domains, ensuring a comprehensive evaluation. Our evaluation results highlight a significant gap between the performance of state-of-the-art MLLMs and human capabilities in understanding multimodal punchlines. To address this, we design the Simple-to-Complex Chain-of-Question (SC-CoQ), which effectively enhances the punchline comprehension ability of MLLMs and outperforms widely-used inference-time techniques such as in-context learning and chain-ofthought.

## Limitations

In this work, we focus on multimodal punchline comprehension for the image-caption pairs, which only consist of static content. According to the evaluation results, MLLMs struggle with the punchline comprehension and fall behind humans. Extending this challenge to videos, where punchlines are often embedded in dynamic flows of information, poses even greater complexity. Unlike static images, videos require models to process temporal dynamics and integrate contextual cues across frames, demanding more advanced comprehension capabilities. Given the added challenges of punchline comprehension in video content, such as comedy, this area presents a meaningful avenue for further exploration. In future work, we aim to evaluate MLLMs'ability to understand punchlines within videos, advancing their capability to process and interpret dynamic multimodal content.

## Acknowledgements

We thank all the anonymous reviewers for their constructive comments. This research was partially supported by the National Natural Science Foundation of China under Grant No. 92470205 and No. 62176002. Xu Sun is the corresponding author of this paper.

## References

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh. 2015. VQA: visual question answering. In ICCV, pages 2425–2433. IEEE Computer Society.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-vl: A versatile

vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

Yitao Cai, Huiyu Cai, and Xiaojun Wan. 2019. Multimodal sarcasm detection in twitter with hierarchical fusion model. In Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pages 2506–2515. Association for Computational Linguistics.

Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, et al. 2024. Internlm2 technical report. arXiv preprint arXiv:2403.17297.

Santiago Castro, Devamanyu Hazarika, Verónica Pérez-Rosas, Roger Zimmermann, Rada Mihalcea, and Soujanya Poria. 2019. Towards multimodal sarcasm detection (an \_Obviously\_ perfect paper). In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4619–4629, Florence, Italy. Association for Computational Linguistics.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. 2024a. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. 2024b. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198.

Poorav Desai, Tanmoy Chakraborty, and Md. Shad Akhtar. 2022. Nice perfume. how long did you marinate in it? multimodal sarcasm explanation. In Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI, pages 10563–10571. AAAI.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,

Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021. An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, Hao Yu, Hongning Wang, Jiadai Sun, Jiajie Zhang, Jiale Cheng, Jiayi Gui, Jie Tang, Jing Zhang, Juanzi Li, Lei Zhao, Lindong Wu, Lucen Zhong, Mingdao Liu, Minlie Huang, Peng Zhang, Qinkai Zheng, Rui Lu, Shuaiqi Duan, Shudan Zhang, Shulin Cao, Shuxun Yang, Weng Lam Tam, Wenyi Zhao, Xiao Liu, Xiao Xia, Xiaohan Zhang, Xiaotao Gu, Xin Lv, Xinghan Liu, Xinyi Liu, Xinyue Yang, Xixuan Song, Xunkai Zhang, Yifan An, Yifan Xu, Yilin Niu, Yuantao Yang, Yueyan Li, Yushi Bai, Yuxiao Dong, Zehan Qi, Zhaoyu Wang, Zhen Yang, Zhengxiao Du, Zhenyu Hou, and Zihan Wang. 2024. Chatglm: A family of large language models from glm-130b to glm-4 all tools.

Kilem L. Gwet. 2014. Handbook of inter-rater reliability: The definitive guide to measuring the extent of agreement among raters. In 4th edition edition, pages 1–38. Advanced Analytics, LLC.

Christian F. Hempelmann and Max Petrenko. 2015. An AI for humorously reframing interaction narratives with human users. In Distributed, Ambient, and Pervasive Interactions - Third International Conference, DAPI 2015, Held as Part of HCI International 2015, Los Angeles, CA, USA, August 2-7, 2015, Proceedings, volume 9189 of Lecture Notes in Computer Science, pages 651–658. Springer.

Jack Hessel, Ana Marasovic, Jena D. Hwang, Lillian Lee, Jeff Da, Rowan Zellers, Robert Mankoff, and Yejin Choi. 2023. Do androids laugh at electric sheep? humor "understanding" benchmarks from the new yorker caption contest. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 688– 714. Association for Computational Linguistics.

Wenyi Hong, Weihan Wang, Ming Ding, Wenmeng Yu, Qingsong Lv, Yan Wang, Yean Cheng, Shiyu Huang, Junhui Ji, Zhao Xue, et al. 2024. Cogvlm2: Visual language models for image and video understanding. arXiv preprint arXiv:2408.16500.

Noman Islam, Zeeshan Islam, and Nazia Noor. 2017. A survey on optical character recognition system. ArXiv, abs/1710.05703.

Pu Jian, Donglei Yu, and Jiajun Zhang. 2024. Large language models know what is key visual entity: An llmassisted multimodal retrieval for VQA. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024, pages 10939– 10956. Association for Computational Linguistics.

Liqiang Jing, Xuemeng Song, Kun Ouyang, Mengzhao Jia, and Liqiang Nie. 2023. Multi-source semantic graph-based multimodal sarcasm explanation generation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 11349–11361. Association for Computational Linguistics.

Justin Johnson, Andrej Karpathy, and Li Fei-Fei. 2016. Densecap: Fully convolutional localization networks for dense captioning. In 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016, pages 4565–4574. IEEE Computer Society.

Shivani Kumar, Atharva Kulkarni, Md. Shad Akhtar, and Tanmoy Chakraborty. 2022. When did you become so smart, oh wise one?! sarcasm explanation in multi-modal multi-party dialogues. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 5956–5968. Association for Computational Linguistics.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. 2024a. Llavaonevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326.

Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao Niu, Fan Zhou, Chengen Huang, Yanpeng Li, et al. 2024b. Aria: An open multimodal native mixture-of-experts model. arXiv preprint arXiv:2410.05993.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. 2023. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 19730–19742. PMLR.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024a. Improved baselines with visual instruction tuning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 26286–26296. IEEE.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. 2024b. Llavanext: Improved reasoning, ocr, and world knowledge.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023a. Visual instruction tuning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023b. Visual instruction tuning.

Yanxin Long, Youpeng Wen, Jianhua Han, Hang Xu, Pengzhen Ren, Wei Zhang, Shen Zhao, and Xiaodan Liang. 2023. Capdet: Unifying dense captioning and open-world detection pretraining. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 15233–15243. IEEE.

Abdelkader El Mahdaouy, Abdellah El Mekki, Kabil Essefar, Nabil El Mamoun, Ismail Berrada, and Ahmed Khoumsi. 2021. Deep multi-task model for sarcasm detection and sentiment analysis in arabic language. In Proceedings of the Sixth Arabic Natural Language Processing Workshop, WANLP 2021, Kyiv, Ukraine (Virtual), April 9, 2021, pages 334–339. Association for Computational Linguistics.

OpenAI. 2022. Introducing chatgpt. CoRR.

OpenAI. 2023a. Gpt-4v(ision) system card.

OpenAI. 2024. Gpt-4o.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal

Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O'Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. 2024. Gpt-4 technical report.

Kun Ouyang, Liqiang Jing, Xuemeng Song, Meng Liu, Yupeng Hu, and Liqiang Nie. 2024. Sentimentenhanced graph-based sarcasm explanation in dia-1ogue. CoRR, abs/2402.03658.

Yang Qiao, Liqiang Jing, Xuemeng Song, Xiaolin Chen, Lei Zhu, and Liqiang Nie. 2023. Mutual-enhanced incongruity learning network for multi-modal sarcasm detection. In Thirty-Seventh AAAI Conference on Artificial Intelligence, AAAI 2023, Thirty-Fifth Conference on Innovative Applications of Artificial Intelligence, IAAI 2023, Thirteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2023, Washington, DC, USA, February 7-14, 2023, pages 9507–9515. AAAI Press.

Noam Shazeer. 2020. GLU variants improve transformer. CoRR, abs/2002.05202.

The Mistral AI Team. 2023. Mistral-7b-instruct-v0.2

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. 2024. Qwen2- vl: Enhancing vision-language model's perception of the world at any resolution. arXiv preprint arXiv:2409.12191.

Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. 2023. Cogvlm: Visual expert for pretrained language models.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. 2024a. Qwen2 technical report. CoRR, abs/2407.10671.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. 2024b. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115.

Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. 2024. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. 2023. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975-11986.

## A More Details for PunchBench

Here we provide more details for the dataset construction for both punchline perception and reasoning.

## A.1 Source Data Collection & Annotation

We detail the data collection process.

Data Collection. 1) Data filtering. To reduce timeconsuming and labor-cost, we introduce MLLMbased filtering method to answer the above questions to help filter the image-caption pairs. To prevent biases from MLLM, we randomly select a model from the set of evaluated MLLMs as the judge. It then is required to assess the quality of image-caption pairs by responding the following questions. Q1: “Whether it contains possible ethics conflict?" If No, go to the next question. Q2: “Whether the content of image is clearly visible?" If Yes, go to the next question. Q3: “Whether the caption is well-written from the aspects of fluency, length and readability?"If Yes, this image-caption pair passes the filtering process. To make sure the filtering quality, we randomly sample 500 imagecaption pairs and then employ three undergraduates outside of this work to answer the above questions. Only 1 pairs of 500 fail to pass the manual filtering process, which verifies the reliability of automatic filtering process. 2) Crowd voting. To determine whether a collected image-caption pair contains a punchline, we conducted a crowd voting process using a questionnaire. Participants were asked, “Does the given image-caption pair make you laugh?" and could choose between “Yes" and “No." Each questionnaire was considered valid if it received more than 10 votes. If one option garnered over 80% of the votes, it was assigned as the label for the corresponding pair. Notably, for the pairs collected from the prior datasets, we adopted the original labels Specifically, if the pair is identified as humorous or sarcastic in previous datasets, we regarded it as containing punchline.

Data Annotation. To acquire reasoning sentences for particular pairs containing punchline, we employ three human annotators to write reasoning sentence based on the content of image and caption. Specifically, we provide the annotated sarcasm or humor explanations for the pairs existing in the previous datasets, which can be referred to write reasoning sentence. Reasoning sentence must cover the key components in image and caption that convey punchline, and the annotators should state how the interplay between visual content and textual information conveys punchline.

## A.2 Synonymous & Antonymous Caption

We illustrate Figure 15 to present the prompts to guide gpt-3.5-turbo-0125 to generate synonymous and antonymous captions. And we provide more implementation details for context consistency adaption as follows. After identifying and isolating the two conflicting parts of inconsistent caption, we adopt word substitution and inversion to derive synonymous and antonymous captions. Specifically, we conduct word substitution for the former part and utilize word inversion for the latter part, if the generated caption maintain the punchline, we regard it as the synonymous caption. And we then conduct word substitution for the latter part and utilize word inversion for the former part, if the generated caption loses the punchline, we regard it as the antonymous caption.

## A.3 Instruction Construction

Instruction Template. We provide various instruction templates for each question format, as follows. For punchline perception, the templates for Yes/NO QA are shown in Figure 20. The prompts for distractor captions generation and instruction templates for Matching QA are exhibited in Figure 21. The prompts for distractor options generation and instruction templates for Multi-option QA are exhibited in Figure 22. For punchline reasoning, the prompts for distractor reasoning sentence generation and instruction templates for Yes/No QA are exhibited in Figure 23. The instruction templates for Matching QA are exhibited in Figure 24. The instruction templates for Generation QA are exhibited in Figure 25.

## A.4 Benchmark Comparison

We compare our PunchBench with the prior benchmarks related to multimodal punchline comprehension in Table 4. PunchBench shows superiority in domain, task, question format, punchline type.

## B More Details for SC-CoQ

For the simplest question format Yes/No QA, we construct Inter-task SC-CoQ, i.e., <Yes/No $\begin{array} { r } { Q A _ { m } , } \end{array}$

Yes/No $Q A _ { n } > , < Y e s / N o$ $Q A _ { n } .$ , Yes/No $Q A _ { m } >$ . m denotes punchline perception and n means punchline reasoning. Specifically, For a specific Yes/No $Q A _ { m }$ in punchline perception task, Yes/No $Q A _ { n } >$ is filled by a randomly sampled Yes/No QA from punchline reasoning task. For a specific Yes/No $Q A _ { n }$ in punchline reasoning task, Yes/No $Q A _ { m } >$ is implemented by the Yes/No QA from punchline reasoning task which shares the same image-caption pair. Notably, we integrate the response to the former question before the final question in the chain, as shown in n Figure 16. Similarly, for Matching QA, we adopt the same process. Then we can obtain SC-CoQ for Yes/No QA and Matching QA. Additionally, we exhibit some prompt examples of Matching QA using SC-CoQ in Figure 17 and Figure 18. For Multi-option QA and Generation QA, we implement <Yes/No QA, Matching QA, Multi-option QA or Generation QA> for a specific image-caption pair. The prompt examples of Multi-option QA are shown in Figure 19.

## C More Details for Evaluation

Introduction for the MLLMs.

• LLaVA (Liu et al., 2024b). We use llavav1.6-mistral-7b in our experiment. It reuses the pre-trained connector of LLaVA-1.5 (Liu et al., 2023b) and adopts Mistral (Team, 2023) as the base LLM.

• GLM-4V (GLM et al., 2024). It consists of GLMTransformer with 40 GLM Blocks and an EVA2CLIP Model with 63 Transformer Layers, along with a GLU mechanism.

• Qwen2-VL (Wang et al., 2024). Qwen2- VL employs a 675M parameter ViT across various-sized LLMs, ensuring that the computational load of the ViT remains constant regardless of the scale of the LLM. In terms of language processing, we have opted for the more powerful Qwen2 (Yang et al., 2024a).

• CogVLM2 (Hong et al., 2024). It is a stronger version of CogVLM, which is an extension of Vicuna, incorporating ViT (Dosovitskiy et al., 2021) as the vision encoder, a two-layer MLP (Shazeer, 2020) as adapter, and introducing Visual expert module.

• LLaVA-OneVision (Li et al., 2024a). It integrates the Qwen2 (Yang et al., 2024b) language backbone with the SigLIP (Zhai et al.,

2023) vision encoder, enhancing performance on tasks that demand fine-grained visual understanding.

• InternVL2.5 (Chen et al., 2024a). This highperforming open-source MLLM integrates InternViT-300M-448px-V2\_5 (Chen et al., 2024b) as the vision encoder and internlm2\_5- 7b-chat (Cai et al., 2024) as the language model backbone.

• MiniCPM-o 2.6 (Yao et al., 2024). The model is built upon SigLIP-400M (Zhai et al., 2023) and Qwen2.5-7B-Instruct (Yang et al., 2024b), comprising a total of 8B parameters.

• Aria (Li et al., 2024b). The model features a fine-grained mixture-of-experts (MoE) decoder that activates 3.5B of its 24.9B total parameters per token, enabling faster and more efficient training and inference through expert specialization.

• GPT-4V (OpenAI, 2023a) and GPT-4o (OpenAI, 2024). They are the leading MLLMs proposed by OpenAI.

<table><tr><td>A</td><td>Strategy</td><td>Parameters</td></tr><tr><td>LLaVA</td><td>Random</td><td>T=0.7</td></tr><tr><td>GLM-4V</td><td>Top-k</td><td>k=3</td></tr><tr><td>Qwen2-VL</td><td>Top-p</td><td>p=0.7</td></tr><tr><td>CogVLM2</td><td>Random</td><td>T=0.7</td></tr><tr><td>LLaVA-OneVision</td><td>Greedy</td><td></td></tr><tr><td>InternVL2.5</td><td>Greedy</td><td></td></tr><tr><td>MiniCPM-o 2.6</td><td>Greedy</td><td></td></tr><tr><td>Aria</td><td>Greedy</td><td></td></tr><tr><td>GPT-4V</td><td>Greedy</td><td></td></tr><tr><td>GPT-40</td><td>Greedy</td><td></td></tr></table>

Table 2: Decoding strategy and parameters for the evaluated MLLMs.

Inference settings of the MLLMs. We present the inference settings, including decoding strategy and parameters of MLLMs in Table 2.

Introduction for in-conext learning and chainof-thought. 1) In-context learning (ICL) (Brown et al., 2020). ICL enables models to perform tasks without explicit parameter updates by conditioning on a sequence of input-output examples, often referred to as a prompt. The model implicitly learns the task by observing these examples within the context, leveraging its pre-trained knowledge to generate predictions for new inputs. In this work, we adopt 3-shot prompt as one of the baselines. 2) Chain-of-Thought (CoT) (Wei et al., 2022). CoT prompting encourages models to generate intermediate reasoning steps in natural language, leading to more accurate and interpretable outputs for complex problems. By including step-by-step explanations in the prompt, CoT facilitates the decomposition of multi-step tasks, such as arithmetic, logical reasoning, or commonsense inference, into manageable sub-tasks. This approach significantly improves performance on reasoning-heavy benchmarks and highlights the potential of leveraging language models for tasks requiring structured thought processes.

## D Evaluation and Analysis

## D.1 Performance Variations

We compare the results cross the original, synonymous, and antonymous captions for all the evaluated MLLMs. The results for LLaVA, GLM-4V, Qwen2-VL, GPT-4V and GPT-4o cross different captions are exhibited in Figure 7, Figure 8, Figure 9, Figure 10, and Figure 11. As can be seen, synonymous and antonymous captions effectively eliminate shortcuts in the original captions, challenging models to fully comprehend the imagecaption pairs. This leads to a more comprehensive evaluation of punchline comprehension capabilities. When using 3-shot and CoT methods, model performance with synonymous and antonymous captions lags behind that with original captions. However, when applying SC-CoQ, models show significant improvement across all caption types. This demonstrates that SC-CoQ enhances the models’ ability to grasp the semantics of image-caption pairs, leading to better punchline comprehension.

## D.2 Human Evaluation

To validate the reliability of automatic evaluation for Generation QA, we conduct human evaluation through pairwise test. Specifically, we first randomly sample 100 pairs of reasoning sentences from two candidate models. And we then involve three independent annotators (undergraduate students uninvolved in this work) to compare reasoning sentences generated by two models (A and B) for the same image-caption pair. The annotators are supposed to choose one of three options: i.e., “A Wins", “A Draws B” and "B Wins". Finally, the winner is determined by the “Win" votes. If both models receive an equal number of “Win" votes, the final result is recorded as “A Draws B". In addition, we calculate Gwet's γ (Gwet, 2014) to represent inter-annotator agreement. The results for human evaluation of the generated reasoning sentences from evaluated models are shown in Table 3.

<table><tr><td>A</td><td>B</td><td>A Wins (%)</td><td>A Draws B (%)</td><td>B Wins (%)</td><td>G-γ (%)</td></tr><tr><td>GLM-4V</td><td>Llava</td><td>57.0</td><td>18.0</td><td>25.0</td><td>82.6</td></tr><tr><td>Qwen2-VL</td><td>GLM-4V</td><td>67.0</td><td>23.0</td><td>10.0</td><td>77.4</td></tr><tr><td>CogVLM2</td><td>Qwen2-VL</td><td>41.0</td><td>37.0</td><td>22.0</td><td>80.4</td></tr><tr><td>GPT-4V</td><td>CogVLM2</td><td>59.0</td><td>20.0</td><td>21.0</td><td>78.1</td></tr><tr><td>GPT-40</td><td>GPT-4V</td><td>47.0</td><td>30.0</td><td>23.0</td><td>74.6</td></tr><tr><td>GPT-40 (CoT)</td><td>GPT-40 (Zero-shot)</td><td>31.0</td><td>48.0</td><td>21.0</td><td>71.2</td></tr><tr><td>GPT-40 (3-shot)</td><td>GPT-40 (CoT)</td><td>39.0</td><td>38.0</td><td>23.0</td><td>76.3</td></tr><tr><td>GPT-40 (SC-CoQ)</td><td>GPT-40 (3-shot)</td><td>46.0</td><td>32.0</td><td>22.0</td><td>82.7</td></tr></table>

Table 3: Human estimation for Generation QA. Inter-annotator agreement is emphasized by Gwet's γ (Gwet, 2014), which is consistently larger than 70.0%, indicating substantial agreement.

![](images/a8a69ae1c4b330a1228343ee659e7ee7a1c915079fb551e53d8828adb5f8ddb2.jpg)

![](images/1235d4728fd5adaa0260794c8b991909b41728401dd88ecf36fe492adb57ca04.jpg)  
Figure 6: We show the relation between accuracy of automatic evaluation and ranking of human evaluation for evaluated MLLMs and different prompting methods.

## D.3 Correlation between Automatic and Human Evaluation

Human evaluation results, which are presented in Appendix D.2, show substantial agreement among annotators since Gwet's γ (Gwet, 2014) is consistently larger than 70%. And we exhibit the correlation between Automatic and Human evaluation in Figure 6 to emphasize the reliability of automatic evaluation for Generation QA. As observed, the models or methods that rank higher in human evaluation also show better accuracy in automatic evaluation. And our SC-CoQ achieves the best performance in both automatic and human evaluation. It not only verifies the credibility of the automatic evaluation results, but also further demonstrates the advantages of our SC-CoQ.

![](images/364545a76ef4ceef9e653a669d4d5a0c1f335cb3df2e457d9461a270fb1002d8.jpg)  
Figure 7: Performance comparison for LLaVA across original, synonymous and antonymous captions in zeroshot, 3-shot, CoT and our SC-CoQ.

![](images/ed8b6c4fb331545ec6f70c791b32427e5de76530461a77d3f050e64c3b9b697e.jpg)  
Figure 8: Performance comparison for GLM-4V across original, synonymous and antonymous captions in zeroshot, 3-shot, CoT and our SC-CoQ.

![](images/0f988145febe5c44a7e3b6c7c639517cef536fd151ecf3e8a1ac83a8eb17bbc9.jpg)  
Figure 9: Performance comparison for Qwen2-VL across original, synonymous and antonymous captions in zero-shot, 3-shot, CoT and our SC-CoQ.

![](images/8b54f72ca7da7c25d761a3427420ef83f31b12c5f0a6407709f74fe8266dfc69.jpg)  
Figure 10: Performance comparison for GPT-4V across original, synonymous and antonymous captions in zeroshot, 3-shot, CoT and our SC-CoQ.

![](images/00ab5bf22568916ef217217cbd9c83eb59ee4b5416a147a1babec26327cadd3b.jpg)  
Figure 11: Performance comparison for GPT-4o across original, synonymous and antonymous captions in zeroshot, 3-shot, CoT and our SC-CoQ.

![](images/47c914c50b43eb2c13a9296bc4bd70955de352ad75662baf4f2f972c32f985a3.jpg)  
Figure 12: An example for qualitative analysis, where we show the responses from CogVLM2 to the Matching QA with different settings (i.e., zero-shot, 3-shot, CoT and SC-CoQ).

![](images/b9b44eab55361295019e905ea63bdc8a8ebe4ece47a082fe7101b2aecc0f8f3d.jpg)  
Figure 13: An example for qualitative analysis, where we show the responses from CogVLM2 to the Yes/No QA and Matching QA of punchline reasoning with different settings (i.e., zero-shot, 3-shot, CoT and SC-CoQ).

## D.4 More Qualitative Results

We provide result examples for Matching QA of punchline perception in Figure 12. As we can see, when using SC-CoQ, the model correctly answers the question, while failing when utilizing other prompting methods. For punchline reasoning task, we supply result examples for Yes/No QA and Matching QA in Figure 13. In addition, we present result examples for Generation QA in Figure 14.

## E Documentation, Licensing, Potential risk and Intended Use of PunchBench

PunchBench encompasses 6, 000 image-caption pairs and 54, 000 question-answer pairs for multimodal punchline comprehension. We evaluate punchline comprehension in two levels: shallowlevel punchline perception and deep-level punchline reasoning. We introduce three question formats for each task. We release the dataset without ground truth answers, along with a validation set that includes ground truth annotations, under the CC BY-NC 4.0 license³. Notably, there may be some offensive information in the images, despite we have made efforts to exclude the potential offensive information in the collection and filtering process. Furthermore, PunchBench should only be used for research purpose only.

![](images/ef44addf3121d1a89f3bd2e718309b7488ea896f7b5d9631c6f9aee9e3ae6a65.jpg)  
Figure 14: Two random samples of explanations generated by CogVLM2, GPT-4o, and human-written reasoning sentences. Notably, we present the generated reasoning sentences by CogVLM2 and GPT-4o prompted by 3-shot, CoT and SC-CoQ.

<table><tr><td>Benchmarks</td><td>Domain</td><td>Task</td><td>Question Format</td><td>Punchline Type</td><td>#Num of Image-caption Pairs</td><td>#Num of Question-answer Pairs</td></tr><tr><td>MTSD (Cai et al., 2019)</td><td>Post</td><td>Sarcasm Classification</td><td>Single</td><td>Sarcasm</td><td>19,816</td><td>19,816</td></tr><tr><td>MORE (Desai et al., 2022)</td><td>Post</td><td>Sarcasm Explanation Matching,</td><td>Single</td><td>Sarcasm</td><td>3,510</td><td>3,510</td></tr><tr><td>HUB (Hessel et al., 2023)</td><td>Cartoon</td><td>Ranking and Explanation</td><td>Single</td><td>Humor</td><td>704</td><td>5,973</td></tr><tr><td>PunchBech</td><td>Cartoon, Post, Comment, Meme.</td><td>Punchline Perception, Punchline Reasoning</td><td>Yes/No QA, Matching QA, Multi-option QA, Generation QA.</td><td>Humor, Sarcasm</td><td>6,000</td><td>54,000</td></tr></table>

Table 4: Comparison between our PunchBench and previous benchmarks.

## F Annotators Recruitment and Multimedia Platforms

For human baseline, we employed three undergraduates outside of the work as the annotators. For human evaluation, we asked another three undergraduate students to evaluate the quality of generated reasoning sentences. The information about the multimedia platforms we used is listed as follows. The social media platforms $\mathrm { X ^ { 4 } }$ , Instagram⁵5, and YouTube6. Additionally, we include image-caption pairs from the cartoon websites like CartoonMovement7 and CartoonStock8.

![](images/01fa1be91db9b7c636ce9173ff9aa5714fd46831ec619d1272453c572a4136ab.jpg)  
Figure 15: Prompts used to guide gpt-3.5-turbo-0125, where Prompt1 guides the model to generate synonymous caption, Prompt2 guides it to derive antonymous caption, and Prompt3 guides it to identify the context inconsistency.

![](images/3aca57bbf87e892743a5a78ad6d3f5d086c60c397066e388e3872017d3d160a6.jpg)  
Figure 16: Prompt examples for Yes/No QA of punchline perception and reasoning using SC-CoQ.

![](images/7701401677fc8e2263f292db0b30b69d641f3c9a8b14265f4e50a7a25787a30d.jpg)  
Figure 17: Prompt examples for Matching QA of punchline perception using SC-CoQ

![](images/37eb221f5ffa8e6439d26b02cd4e90ae1c2ff206d11405ed51e15702bc89a18c.jpg)  
Figure 18: Prompt examples for Matching QA of punchline reasoning using SC-CoQ

![](images/17c9497c87946e69495635000c489f1502ed8b88b12e2cf4d24c908ea17af50b.jpg)  
Figure 19: Prompt examples for Multi-option QA and Generation QA using SC-CoQ

![](images/6070273275f08f5255e04a676058e23bb5745d0379fccd811041993cfb36f8d4.jpg)  
Figure 20: Instruction templates for Yes/No QA of punchline perception.

![](images/fe350b096849aaf7568e42cd69730dac4dec21331d65e4f4f52be4ee13236cf2.jpg)  
Figure 21: Prompts used to guide GPT-4o to generate distractor caption and instruction templates for Matching QA of punchline perception.

![](images/33bb1155d792944faf6d10a0547ead79cec579574109481bae61244573530dd5.jpg)  
Figure 22: Prompts used to guide GPT-4o to generate distractor options and instruction templates for Multi-option QA of punchline perception.

![](images/54b0fba524a87e59f035009d7ed99887a598539180127a8538ac5a7fbdde7ad0.jpg)  
Figure 23: Prompts used to guide ChatGPT to generate distractor reasoning sentence and instruction templates for Yes/No QA of punchline reasoning.

![](images/d86de538f4ea664a03a45b1fa8bcfd6c9903aaab607e3e0c444594278f349da3.jpg)  
Figure 24: Instruction templates for Matching QA of punchline reasoning.

![](images/91fe1fbeee2e0868d97af75f96296356054da1d37c6029816021199f753b6937.jpg)  
Figure 25: Instruction templates for Generation QA of punchline reasoning.