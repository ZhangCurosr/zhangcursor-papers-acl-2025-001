# TUNA: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos

Fanheng Kong<sup>1\*</sup>, Jingyuan Zhang<sup>2</sup>, Hongzhi Zhang<sup>2</sup>, Shi Feng<sup>1†</sup>, Daling Wang<sup>1</sup>, Linhao Yu<sup>2</sup>, Xingguang Ji<sup>2</sup>, Yu Tian<sup>2</sup>, Victoria W., Fuzheng Zhang<sup>2</sup> <sup>1</sup>Northeastern University <sup>2</sup>Kuaishou Technology kongfanheng426@gmail.com, fengshi@cse.neu.edu.cn

## Abstract

Videos are unique in their integration of temporal elements, including camera, scene, action, and attribute, along with their dynamic relationships over time. However, existing benchmarks for video understanding often treat these properties separately or narrowly focus on specific aspects, overlooking the holistic nature of video content. To address this, we introduce TUNA, a temporal-oriented benchmark for fine-grained understanding on dense dynamic videos, with two complementary tasks: captioning and QA. Our TUNA features diverse video scenarios and dynamics, assisted by interpretable and robust evaluation criteria. We evaluate several leading models on our benchmark, providing finegrained performance assessments across various dimensions. This evaluation reveals key challenges in video temporal understanding, such as limited action description, inadequate multi-subject understanding, and insensitivity to camera motion, offering valuable insights for improving video understanding models. The data and code are available at https:// friedrichor.github.io/projects/TUNA.

## 1 Introduction

Vision enables us to perceive the world, and video, as a key form of visual media, offers rich spatial and temporal information (Tang et al., 2023; Madan et al., 2024). With the rapid growth of video content, video understanding has become a crucial area of research, enabling applications that address the increasing volume of video data (Nguyen et al., 2024) and facilitate video generation as generalpurpose simulators of the physical world (Brooks et al., 2024). Despite these advancements, the lack of robust evaluation methods remains a pressing challenge for the community. Accurate and comprehensive benchmarks are essential to assess the performance of video understanding models and improve their ability to interpret and analyze diverse video data effectively.

![](images/357171bb7b9277554b18e33afe6e8d03a5d481c1ff8d30244ca93b219a54f5f3.jpg)  
Figure 1: Performance of several advanced models on our TUNA. TUNA offers robust and interpretable evaluations on video captioning and QA tasks, providing clear guidance for advancements in video understanding.

Recent works (Fu et al., 2024; Zhou et al., 2024) have evaluated video understanding across various tasks such as temporal perception and reasoning, video captioning, and long-video comprehension, providing metrics to guide the development of video LMMs. However, these evaluations often focus on specific aspects, such as subject actions, while neglecting other crucial video elements like camera states and background scenes along with the relationships between these elements (Chai et al., 2024; Xiong et al., 2024; Polyak et al., 2024). Additionally, the bias toward long-form videos (Fu et al., 2024; Li et al., 2024e; Mangalam et al., 2023) entangles video understanding with long-context modeling, making it difficult to attribute performance to specific capabilities. Furthermore, existing benchmarks lack an analysis of the model’s sensitivity towards key factors affecting video understanding, such as diversity of video dynamics and visual characteristics. These limitations hinder comprehensive evaluation and effective error analysis to advance video understanding models.

To address the need for comprehensive video understanding, we introduce TUNA, a challenging multimodal benchmark for Temporal Understanding of dense dyNAmic videos. Unlike previous evaluations that focus on isolated video elements, TUNA emphasizes holistic video comprehension. We carefully curated 1,000 representative videos from diverse sources, spanning 12 domains such as Film and Driving, categorized across four visual characteristics: High-Dynamic, Low-Dynamic, Multi-Scene, and Multi-Subject. Each video in our dataset, TUNA-1K, is meticulously segmented into fine-grained events and annotated with detailed temporal captions, capturing camera states, background scenes, subject actions, object attributes. Table 1 shows the comparison with various vidoe understanding benchmarks.

<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">#Videos</td><td rowspan="2">#Samp.</td><td rowspan="2">Anno.</td><td rowspan="2">Domain</td><td rowspan="2">Temporal Oriented</td><td rowspan="2">Scene Trans.</td><td colspan="5">Captioning</td><td colspan="2">VQA</td></tr><tr><td>Camera</td><td>Scene</td><td>Key.</td><td>Sem.</td><td>M.D.</td><td>Global</td><td>Fine.</td></tr><tr><td>VQA Benchmark</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>NExT-QA (Xiao et al., 2021)</td><td>1,000</td><td>8,564</td><td>M</td><td>daily life</td><td>x</td><td>x</td><td></td><td></td><td></td><td></td><td></td><td>X</td><td></td></tr><tr><td>EgoSchema (Mangalam et al., 2023)</td><td>5,063</td><td>5,063</td><td>M&amp;A</td><td>egocentric</td><td></td><td>x</td><td></td><td></td><td></td><td></td><td></td><td></td><td>x</td></tr><tr><td>PerceptionTest (Patraucean et al., 2024)</td><td>11,620</td><td>44,000</td><td>M</td><td>indoor</td><td></td><td>x</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MVBench (Li et al., 2024d)</td><td>3,641</td><td>4,000</td><td>A</td><td>open</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Video-MME (Fu et al., 2024)</td><td>900</td><td>2,700</td><td>M</td><td>open</td><td>x</td><td></td><td></td><td></td><td></td><td></td><td></td><td>X</td><td></td></tr><tr><td>MMBench-Video (Fang et al., 2024)</td><td>609</td><td>1,998</td><td>M</td><td>open</td><td>x</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VideoVista (Li et al., 2024e)</td><td>894</td><td>24,906</td><td>A</td><td>open</td><td>x</td><td></td><td></td><td></td><td></td><td></td><td></td><td>X</td><td></td></tr><tr><td>TOMATO (Shangguan et al., 2024)</td><td>1,417</td><td>1,484</td><td>M</td><td>open</td><td>」</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>x</td></tr><tr><td>Captioning Benchmark</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DREAM-1K (Wang et al., 2024a)</td><td>1,000</td><td>1,000</td><td>M</td><td>open</td><td>√</td><td></td><td>x</td><td>x</td><td>x</td><td></td><td>x</td><td></td><td></td></tr><tr><td>VDC (Chai et al., 2024)</td><td>1,027</td><td>1,027</td><td>A</td><td>open</td><td>√</td><td></td><td>√</td><td>√</td><td>x</td><td></td><td>√</td><td></td><td></td></tr><tr><td>Multi-task Benchmark</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MLVU (Zhou et al., 2024)</td><td>1,334</td><td>2,593</td><td>M</td><td>open</td><td>x</td><td></td><td>x</td><td>1</td><td>x</td><td></td><td>x</td><td>x</td><td></td></tr><tr><td>TempCompass (Liu et al., 2024f)</td><td>410</td><td>7,540</td><td>M&amp;A</td><td>open</td><td>V</td><td></td><td>x</td><td>x</td><td>X</td><td></td><td>V</td><td></td><td></td></tr><tr><td>E.T.Bench (Liu et al., 2024e)</td><td>7,002</td><td>7,289</td><td>M</td><td>open</td><td></td><td></td><td>x</td><td>x</td><td>x</td><td></td><td>x</td><td></td><td></td></tr><tr><td>TemporalBench (Cai et al., 2024)</td><td>2,179</td><td>2,179</td><td>M</td><td>open</td><td></td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td></td><td></td></tr><tr><td>TUNA</td><td>1,000</td><td>2,432</td><td>M&amp;A</td><td>open</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 1: Comparison with various video understanding benchmarks across several aspects: number of videos (#Videos); number of samples (#Samp.); annotation method (Anno., with M/A denoting manual/automatic); domain (Domain); temporal orientation (Temporal Orientated); presence of scene transitions (Scene Trans.); consideration of camera (Camera) and scene (Scene); use of keypoints (Key.) for controllability and interpretability; Judgement of semantically identical yet diverse representations (Sem.); availability of multi-dimensional scores (M.D.); if global (Global) and fine-grained (Fine.) understanding are concerned.

Building on TUNA-1K, we propose TUNA, a multi-task benchmark towards temporal dynamics through two complementary tasks: TUNA-CAP for captioning and TUNA-MCQ for VQA. TUNA-CAP features an automated evaluation pipeline that performs event splitting, matching, and relationship classification, closely aligning with human judgment to assess dense captioning capabilities. TUNA-MCQ comprises 1,432 carefully crafted multiple-choice questions that specifically require full video context for accurate answers, ensuring that answers cannot be derived from a single frame or limited frames, providing a rigorous test of temporal understanding. Together, these tasks provide comprehensive evaluation metrics and valuable insights for advancing video understanding research.

We benchmark 21 popular LMMs on TUNA, revealing key challenges in video understanding.

Figure 1 shows the performance of selected models. Dense video captioning remains a difficult task, with GPT-4o (OpenAI, 2024) achieving the best performance but only reaching an F1 score of 58.5%, yet open-source models lag notably behind commercial models. Additionally, LMMs struggle with complex scenarios involving multi-scene, multi-subject, and high-dynamic video content. Interestingly, in the VQA task, open-source models demonstrate competitive performance. However, all models show consistent weaknesses in comprehending camera motion and action sequence. The notable performance disparity between captioning and VQA tasks underscores the current limitations in holistic video understanding capabilities. These findings provide crucial insights for advancing video LMMs, particularly in temporal and visual comprehension capabilities.

## In summary, our contributions are:

• We introduce TUNA-1K, a meticulously annotated video-caption dataset that captures finegrained temporal dynamics across camera, scene, action, and attribute on dense dynamic videos.

• We develop TUNA, a novel benchmark for comprehensive temporal video understanding, measuring the performance across various novel dimensions, such as different visual characteristics, temporal elements and video complexities.

• We conduct a comprehensive evaluation of several popular models, uncovering their strengths and weaknesses across various dimensions.

![](images/37c851467751d1e8f4284d76eb712e8f20b6350208b2c6bc8bca3ce5ccdd86cd.jpg)  
Figure 2: Overview of TUNA-1K construction. We collect and filter high-quality, short videos featuring dynamic temporal content from various sources. Each video is then categorized based on its visual characteristics and domain. Trained annotators provide temporally dense descriptions, followed by cross-validation. Video experts continuously review annotations, guiding annotators to refine their works, thus ensuring quality of the annotations.

Hopefully, this provides solid guidance for advancing video understanding.

## 2 Related Work

Video Captioning. Recent works (Zhang et al., 2024d,f; Chen et al., 2024a; Liu et al., 2024d) have revealed the importance of detailed captions for video understanding. Compared to image captioning, video captioning presents a greater challenge as it requires advanced techniques to handle the di versity of human and object appearances in various scenes, as well as their evolving relationships over time (de Souza Inácio and Lopes, 2023). While video captioning data is served as training data for video LMMs, it is challenging to robustly and interpretably evaluate video captioning. Traditional ngram overlaps based metrics (Papineni et al., 2002; Lin, 2004; Vedantam et al., 2015) fail to measure genuine semantic similarity, with weakly consistency with human judgement. LLM-based scoring methods (Chan et al., 2023; Maaz et al., 2023) can deal with captions with the same semantics yet distinct expressions, but directly asking LLM to generate digital scores is not dependable due to their ambiguous meaning of each rating. Recently, Dream-1K (Wang et al., 2024a) evaluates captions from events, providing a robust results. However, these efforts don’t centre on temporal dynamics, overlooking the essential features of video and pay minimal attention to changes in camera and scene. Video QA. Recent works has provided benchmarks for comprehensively evaluating video LMM’s ability to understand video, e.g., Video-MME (Fu et al., 2024), MLVU (Zhou et al., 2024). Temporal dynamics are crucial as a unique feature of video. Existing temporal understanding benchmarks (Patraucean et al., 2024; Li et al., 2024d) focus on restricted scenes (e.g., indoor, egocentric), or just on subject’s actions and attributes, without attention to changes in camera and scene, which are incomplete for temporal understanding evaluation. Our TUNA aims to comprehensively evaluate temporal perception skills towards open-domain videos.

## 3 TUNA

In this section, we present TUNA-1K, a temporally dense video-caption dataset, and TUNA, a multitask temporal understanding benchmark.

## 3.1 TUNA-1K

The construction workflow of TUNA-1K is shown in Figure 2, consisting of four major phases: video collection, filter, cluster, and annotation.

Video Collection. Temporally dense videos should have diverse contents and include changes in the camera states and scenes besides subject actions and object attributes (Polyak et al., 2024; Xiong et al., 2024). To capture these complexities, we carefully collect 1,000 open-domain videos from 10 sources: (1) Academic Video Understanding Data: DREAM-1K (Wang et al., 2024a), Perception Test (Patraucean et al., 2024), VELOCITI (Saravanan et al., 2024), YouCook2 (Zhou et al., 2018); (2) Academic Video Generation Data: MiraData (Ju et al., 2024), VIDGEN-1M (Tan et al., 2024)); (3) Other Academic Video Data: CoVLA (Arai et al., 2024); and (4) Web Data: Pexels (Pexels, 2023), Pixabay (pixabay, 2023), MixKit (mixkit, 2023). Unlike concurrent works (Cai et al., 2024), we maintain the original videos containing multiple scenes or complex actions without segmenting them into clips, as these are essential for our tasks. Video Filter. We remove blurry, low-resolution and long duration videos to ensure that our videos are high-quality and short, with an average resolution of 1579 892, and an average duration of 14.5s. We select short videos to ensure that sampling frame strategies can extract all keyframes from videos, to purely examine the video understanding ability. To ensure the videos are temporal-dynamic, one criterion is rich in either camera motion, scene transitions, or subject activities. Coarse filtering (e.g., resolution) is achieved by rules, and humans make complex filters (e.g., dynamic degree).

![](images/c084f508efc8cf36e1e3ead84be2704c7fac4084c308f3dfa5c7f6c2c3319585.jpg)  
Figure 3: An instance in TUNA-1K consists of three levels of description: (a) an overall caption (Narrative-level), (b) a chronological sequence of events ( Event-level ), and (c) fine-grained visual elements (Atomic-level) along with their types and weights. A complete sample can be found in Figure 15.

Video Cluster. We use GPT-4o (OpenAI, 2024) to generate description for each video, and cluster videos based on their descriptions, including four visual characteristics and 12 domains. Annotators then correct and complete the classification results. Annotation. Existing models often miss critical events (Wang et al., 2024a), and lack sensitivity to camera states, struggling to accurately describe camera changes (Chai et al., 2024). Consequently, generating temporally dense video captions through automatic methods is challenging. Instead, our data is manually annotated. Trained human annotators are tasked to provide detailed video descriptions, focusing on camera states, background scenes, subject actions, and object attributes. The target captions features several chronologically evolving events, without summaries and subjective feelings. Additionally, annotators split each event into multiple visual elements, assigning types and weights to these elements. The types include camera, scene, action, and attribute, while weights indicates the element’s importance for the video on a scale of 1-3.

Formally, a typical instance in TUNA-1K involves a collection of temporally evolving events $E _ { r e f } = [ r _ { 1 } , r _ { 2 } , . . . , r _ { T } ]$ forming an overall caption $C _ { r e f }$ , where $T$ denotes the count of events in the sequence. Each event $r _ { i }$ further contains various visual elements $V _ { i } = \{ v _ { i 1 } , \ldots , v _ { i , n _ { i } } \}$ , where n<sub>i</sub> represents the number of visual elements in event $r _ { i }$

Moreover, each visual element $v _ { i j }$ is labelled with a type $t \in \{ \mathsf { c a m e r a } .$ scene, action, attribute and their weight $w _ { i j } \in \{ 1 , 2 , 3 \}$ . An example of TUNA-1K is shown in Figure 3.

Quality Review. All annotated video-caption pairs undergo cross-inspection by annotators. In parallel, video experts (non-authors) review the annotations, providing feedback and prompting annotators to refine results to ensure high-quality annotation.

## 3.2 TUNA

## 3.2.1 Task Definition

Temporal dynamics distinguish videos from static images. While several benchmarks consider temporal sequences, they sole focus on actions and attributes, neglecting changes in camera state and scene. Additionally, some evaluation tasks fail to capture the perception ability of relationships and evolution of various elements in the video. For example, many questions are just about a single frame cue in the video. To fill these gaps, we emphasize the in-context understanding throughout the entire video, and measure temporal understanding across 4 key dynamic elements: camera state, background scene, subject action, and object attribute. Specifically, we introduce two complementary tasks: TUNA-CAP for captioning and TUNA-MCQ for VQA.

## 3.2.2 TUNA-CAP

An effective way to evaluate the temporal understanding ability of LMMs is reflected by their captioning skills (de Souza Inácio and Lopes, 2023; Chen et al., 2024a). However, it remains a challenge to reliably and interpretably assess the correctness and completeness of video captions. Event-level methods (Wang et al., 2024a, 2022) have proven effective but solely focus on subject actions, overlooking camera states and scenes. To this end, we propose a strategy to assess the temporally dense captions that incorporate dynamic elements evolutions over time.

![](images/ed7543231f1c78a909fbc0699afd999085ef6f7c2387149e87c049768b0b803b.jpg)  
Figure 4: Overview of the evaluation workflow for TUNA-CAP. We first split candidate caption into multiple events and match them to reference events in TUNA-1K. Then we discard the mismatched events (useless content or inconsistent chronology), and connect the matched candidate events with the same reference event, considering the temporal sequence of the captions. Finally, we classify the relationship of visual elements to the candidate event.

As shown in Figure 4, our evaluation proceeds in three stages: (1) Event Splitting, (2) Event Matching, and (3) Relationship Classification.

Event Splitting & Matching. To examine temporal perception skills through model-generated captions, we consider that an effective solution is to verify whether the models accurately describe several events in the correct temporal sequence. To achieve this, the candidate caption $C _ { \mathrm { g e n } }$ is first split into an event sequence $G \ =$ $[ g _ { 1 } , g _ { 2 } , \ldots , g _ { k } ]$ Then, each candidate event $g _ { i }$ is matched to a reference event $r _ { j } .$ . Formally, the target is to obtain $\{ ( i , i d _ { i } ) \} _ { i = 1 } ^ { k }$ pairs, where $i d _ { i } \in \{ 1 , \ldots , T$ , None denotes the index of the reference event $r _ { i d _ { i } }$ matched with the candidate event $g _ { i }$ and $i d _ { 1 } \leq i d _ { 2 } \leq \cdots \leq i d _ { k }$ . These ensures that events which are effective and described in a correct temporal order are extracted.

Relationship Classification. For the captioning task, the classification-based approach is more interpretable and robust than the direct scoring methods (Wang et al., 2024a). Each reference event $r _ { j }$ corresponds to a set of visual elements $V _ { j }$ . Thus, we can transition from a tuple of concatenated candidate events with reference events $( g _ { i } ^ { \prime } , r _ { j } )$ to a tuple of candidate events with visual elements $( g _ { i } ^ { \prime } , V _ { j } )$ . Subsequently, the relationship $\phi ( v _ { i j } , g _ { i } ^ { \prime } ) \in$ entailment, lack, contradiction between visual element $v _ { i j }$ and candidate event $g _ { i } ^ { \prime }$ is classified. This element-based approach improves the interpretability of the evaluation. The workflow is implemented by GPT-4o (OpenAI, 2024), an LLM with powerful instruction-following capabilities.

Metrics. We employ precision (P) and recall (R)

to measure the correctness and completeness of the captions, introducing a novel metric calculation:

$$
\begin{array} { r l } & { \mathbf { P } = \frac { \sum _ { i = 1 } ^ { T } \sum _ { j = 1 } ^ { n _ { i } } \mathbb { 1 } \left( \phi \left( v _ { i j } , g _ { i } ^ { \prime } \right) = \mathrm { e n t . } \right) \cdot w _ { i j } } { \sum _ { i = 1 } ^ { T } \sum _ { j = 1 } ^ { n _ { i } } \mathbb { 1 } \left( \phi \left( v _ { i j } , g _ { i } ^ { \prime } \right) \in \left\{ \mathrm { e n t . } , \mathrm { c o n . } \right\} \right) \cdot w _ { i j } } } \\ & { \mathbf { R } = \frac { \sum _ { i = 1 } ^ { T } \sum _ { j = 1 } ^ { n _ { i } } \mathbb { 1 } \left( \phi \left( v _ { i j } , g _ { i } ^ { \prime } \right) = \mathrm { e n t . } \right) \cdot w _ { i j } } { \sum _ { i = 1 } ^ { T } \sum _ { j = 1 } ^ { n _ { i } } w _ { i j } } } \\ & { \mathbf { F } 1 = \frac { 2 \times \mathbf { P } \times \mathbf { R } } { \mathbf { P } + \mathbf { R } } } \end{array}\tag{1}
$$

(2)

(3)

where $\mathbb { 1 } ( \cdot )$ denotes the indicator function. Recognizing that each visual element $v _ { i j }$ has a distinct importance within the video, each element is weighted by its corresponding factor $w _ { i j }$

## 3.2.3 TUNA-MCQ

Based on fine-grained TUNA-1K, we design a pipeline, integrating automatic construction and manual refinement, to create instructions for multichoice questions. The pipeline involves two main flows: error-prone points extraction and multichoice QA generation. We consider 10 task types: 1) camera motion, e.g, zooming, panning, and rotating. 2) camera transition. 3) scene description. 4) scene transition. 5) action recognition. 6) action sequence. 7) action-subject matching. 8) object recognition. 9) object appearance, e.g., age, dress, color, shape, number. 10) object location. Beyond previous works (Li et al., 2024d; Liu et al., 2024f) that focus on subject actions and object attributes, we additionally emphasize camera states and scene transitions, to provide a more comprehensive assessment of temporal understanding.

Error-prone Points Extraction. To generate challenging questions, we develop an automatic approach to identify error-prone points in videos. The process involves feeding video frames and their ground-truth descriptions to video LMMs, which then identify visual elements that appear inconsistent with the textual descriptions. Leveraging LMMs’ inherent limitations in visual interpretation, we utilize their misidentified elements as naturally occurring error-prone points for question generation.

<table><tr><td rowspan="2">Model</td><td colspan="4">Dynamic Element Type</td><td colspan="4">Visual Characteristic</td><td rowspan="2">Overall</td></tr><tr><td>Camera</td><td>Scene</td><td>Action</td><td>Attribute</td><td>Low-Dynamic</td><td>High-Dynamic</td><td>Multi-Scene</td><td>Multi-Subject</td></tr><tr><td colspan="9">Open-Source LMMs</td></tr><tr><td>PLLaVA-7B</td><td>49.4/22.6/28.9</td><td>52.2/30.9/36.6</td><td>30.5/12.6/16.5</td><td>44.5/19.5/25.3</td><td>66.5/23.0/32.7</td><td>56.6/17.1/24.7</td><td>55.7/15.5/22.8</td><td>56.2/15.3/22.5</td><td>60.0/19.1/27.4</td></tr><tr><td>LongVA-7B</td><td>52.3/26.0/32.5</td><td>56.5/34.4/40.6</td><td>38.9/17.2/22.0</td><td>50.6/22.0/28.4</td><td>75.9/26.5/37.3</td><td>69.4/20.1/29.0</td><td>68.3/19.0/27.6</td><td>67.3/15.7/23.7</td><td>71.6/22.3/31.8</td></tr><tr><td>Tarsier-7B</td><td>56.9/27.3/34.8</td><td>45.3/28.2/33.1</td><td>56.7/28.9/36.2</td><td>56.4/26.0/33.3</td><td>81.2/34.3/46.5</td><td>68.7/24.5/34.5</td><td>71.7/25.3/35.8</td><td>67.8/23.2/33.2</td><td>73.0/27.9/38.6</td></tr><tr><td>Kangaroo</td><td>65.2/36.5/44.1</td><td>67.8/45.4/51.9</td><td>49.3/26.0/31.9</td><td>59.8/32.2/39.5</td><td>73.2/34.7/45.6</td><td>67.6/31.3/41.1</td><td>66.2/29.7/39.3</td><td>63.5/26.3/35.7</td><td>69.5/32.5/42.7</td></tr><tr><td>LLaVA-OV-7B</td><td>75.2/42.0/51.0</td><td>71.8/51.2/57.6</td><td>54.1/30.4/36.8</td><td>66.2/42.0/49.3</td><td>78.6/38.4/50.0</td><td>71.0/38.8/48.9</td><td>71.7/38.3/48.4</td><td>67.1/33.8/43.8</td><td>73.6/38.6/49.3</td></tr><tr><td>LLaVA-Video-7B</td><td>74.0/41.5/50.4</td><td>73.6/52.3/58.9</td><td>57.0/30.8/37.8</td><td>72.1/44.8/53.1</td><td>80.7/40.0/52.2</td><td>75.1/39.5/50.3</td><td>77.1/38.6/50.0</td><td>73.5/34.6/45.8</td><td>77.0/39.7/51.0</td></tr><tr><td>Qwen2-VL-7B</td><td>72.3/40.7/49.0</td><td>71.9/50.0/56.7</td><td>55.9/30.1/37.0</td><td>68.2/38.4/46.7</td><td>81.2/42.0/53.8</td><td>76.0/35.3/46.4</td><td>76.8/33.2/44.4</td><td>73.6/28.9/39.9</td><td>77.8/37.6/48.9</td></tr><tr><td>InternVL2-8B</td><td>64.8/33.7/41.7</td><td>59.4/38.7/44.7</td><td>45.2/24.7/30.0</td><td>59.8/35.5/42.3</td><td>71.6/34.0/44.5</td><td>64.9/29.7/38.9</td><td>65.6/29.1/38.4</td><td>61.5/26.6/35.2</td><td>67.2/31.1/40.8</td></tr><tr><td>MiniCPM-V-2.6</td><td>76.5/47.8/56.0</td><td>75.0/54.1/60.6</td><td>57.2/31.8/38.8</td><td>68.7/42.3/50.2</td><td>79.3/41.4/53.0</td><td>74.3/40.4/51.0</td><td>76.5/40.8/51.7</td><td>73.5/38.3/49.0</td><td>76.0/40.7/51.7</td></tr><tr><td>PLLaVA-34B</td><td>60.8/29.6/37.4</td><td>56.2/33.7/39.9</td><td>38.7/17.3/22.3</td><td>55.1/26.1/33.2</td><td>74.5/28.1/38.9</td><td>64.3/22.6/31.8</td><td>63.9/21.3/30.2</td><td>60.7/19.2/27.6</td><td>67.8/24.5/34.2</td></tr><tr><td>Tarsier-34B</td><td>63.6/34.3/42.3</td><td>59.0/38.4/44.4</td><td>65.6/39.9/47.6</td><td>63.6/34.3/42.2</td><td>79.6/37.2/49.1</td><td>75.8/36.5/47.8</td><td>77.6/38.1/49.6</td><td>74.4/36.0/47.3</td><td>77.1/36.7/48.2</td></tr><tr><td>LLaVA-OV-72B</td><td>73.5/43.7/51.9</td><td>71.5/51.1/57.5</td><td>51.2/30.2/36.0</td><td>65.7/41.4/48.8</td><td>75.4/37.3/48.6</td><td>71.3/36.7/45.9</td><td>71.4/40.1/50.1</td><td>72.3/39.1/49.4</td><td>72.7/39.2/49.6</td></tr><tr><td>LLaVA-Video-72B</td><td>72.7/41.7/50.3</td><td>71.1/49.9/56.4</td><td>55.7/32.7/39.3</td><td>68.1/43.2/50.8</td><td>77.3/39.2/50.6</td><td>71.9/39.8/50.0</td><td>73.9/38.6/49.3</td><td>70.5/35.1/45.7</td><td>73.7/39.6/50.2</td></tr><tr><td>Qwen2-VL-72B</td><td>73.6/45.9/54.0</td><td>67.6/46.3/52.8</td><td>59.1/35.7/42.6</td><td>66.6/40.7/48.5</td><td>79.2/44.6/55.7</td><td>72.4/39.3/49.7</td><td>73.6/37.2/48.0</td><td>69.1/32.8/43.3</td><td>74.7/41.1/51.7</td></tr><tr><td>InternVL2-76B</td><td>75.1/45.4/53.9</td><td>73.3/55.8/61.4</td><td>55.7/34.9/41.2</td><td>64.3/44.5/50.9</td><td>72.0/43.1/52.8</td><td>70.1/41.9/51.5</td><td>71.4/41.1/51.1</td><td>68.6/39.7/49.3</td><td>70.7/42.3/51.9</td></tr><tr><td colspan="8">Closed-Source LMMs</td><td></td><td></td></tr><tr><td>Gemini 1.5 Flash</td><td>74.6/52.8/59.6</td><td>77.2/59.3/65.1</td><td>58.7/36.4/42.9</td><td>69.0/48.4/55.2</td><td>74.0/46.5/56.0</td><td>72.0/46.4/55.5</td><td>73.4/46.2/55.9</td><td>73.4/46.2/55.9</td><td>72.7/46.4/55.7</td></tr><tr><td>Gemini 1.5 Pro</td><td>78.7/53.0/60.7</td><td>75.7/57.4/63.3</td><td>59.0/40.3/46.3</td><td>69.0/49.4/56.0 73.8/50.1/57.8</td><td>76.7/48.7/58.7 79.1/47.3/58.2</td><td>72.1/47.8/56.7</td><td>73.4/47.7/57.0</td><td>69.9/44.1/53.3 76.8/44.4/55.5</td><td>73.7/48.1/57.4</td></tr><tr><td>GPT-40</td><td>80.1/53.3/61.3</td><td>79.5/60.2/66.4</td><td>64.0/41.1/48.0</td><td></td><td></td><td>77.0/48.6/58.7</td><td>78.7/47.2/58.1</td><td></td><td>77.7/48.2/58.5</td></tr></table>

Table 2: TUNA-CAP performance of representative video LMMs. We provide detailed scores for selected tested models in various perception skills and visual characteristic categories. Each cell contains "Precision / Recall / F1 Score". The best and second-best results are marked with bold and underline, respectively.

Multi-Choice QA Generation. Based on predefined task types, error-prone points and textual descriptions, LLM generates several multi-choice questions for each video. To ensure these questions effectively capture temporal dynamics, we employ a temporal-indispensability filtering mechanism similar to MMBench-Video (Fang et al., 2024). Specifically, a question is considered temporalindispensable only if it cannot be correctly answered using a single frame but requires n frames (default n = 16) for accurate comprehension. This rigorous filtering process helps maintain a high temporal-indispensability ratio in TUNA-MCQ.

Quality Review. To ensure that data is highquality and time-sensitive, we employ crowdsourcing to further filter and refine the automatically constructed data. In addition, human annotators perform cross-inspections to ensure annotation quality.

## 4 Experiments

## 4.1 Settings

We evaluate 21 closed-source models and opensource models with various sizes, including: Gemini 1.5 Pro (Reid et al., 2024), Gemini 1.5 Flash (Reid et al., 2024), GPT-4o (OpenAI, 2024), PLLaVA (Xu et al., 2024), LongVA (Zhang et al., 2024c), Tarsier (Wang et al., 2024a), InternVL2 (Chen et al., 2024b), Kangaroo (Liu et al., 2024c), LLaVA-OneVision (Li et al., 2024a), MiniCPM-V-2.6 (Yao et al., 2024), LLaVA-Video (Zhang et al., 2024f), and Qwen2-VL (Wang et al., 2024b).

By default, we uniformly sample 32 frames from each video, which is sufficient to capture the entire content of videos in our TUNA. Some models have varying constraints on input length or specific recommended settings. To accommodate these variations, we employ tailored sampling strategies for these models. More details are available in Appendix B.1 and Appendix C.3.

## 4.2 Video Captioning

We evaluate the temporal understanding skills of the models and their abilities to perceive videos towards different dynamic elements and visual characteristics. Precision reflects the correctness of the content mentioned in the descriptions, while recall reflects the completeness of the descriptions. As shown in Table 2, majority of video LMMs achieve a precision over 70%, but recall is below 50%, indicating that many visual elements in videos are often overlooked or misdescribed. The state-of-the-art model GPT-4o only achieve an F1 score of 58.5%, with a recall of 48.2%, highlighting that LMMs still have a great potential for improvement in the task of temporally dense captioning.

Temporal Dynamic Elements. Recent researches in video understanding and video generation have increasingly emphasized the dynamics of camera states and scenes (Chai et al., 2024; Xiong et al., 2024; Polyak et al., 2024). In this work, we comprehensively thoroughly analyze four key dynamic element types: camera, scene, action, and attribute, aiming to explore the challenges that existing models face in captioning dynamic videos. As shown in Table 2, LMMs demonstrate superior performance in scene perception compared to the other dimensions. Existing LMMs often extract multiple frames from videos and treat them as a series of static images, facilitating a better grasp of static visual scenes. However, camera and attribute elements, which assess overall dynamics and fine-grained perception respectively, remain challenging, with highest scores reaching only 61.3% (56.0% for open-source models) for camera and 57.8% (53.1% for open-source models) for attribute. Notably, action perception shows consistently weaker performance across almost all models compared to the other dimensions, indicating substantial shortcomings in accurately describing the dynamic actions. An interesting exception is Tarsier-34B, which performs exceptionally well in the action dimension, falling only 0.4% behind GPT-4o. This aligns with its strong performance on DREAM-1K (Wang et al., 2024a), a video captioning benchmark focused on action events.

![](images/fb59de28ab0fc966c99cb342669a611d5210ad8a056b4c8e6832662a2c66ebd3.jpg)

![](images/2b4c6130755cc4d39a7d0c2fbe6ac702ec356c292f64cd13e2a9e1f34b022151.jpg)  
(a) Number of Events

![](images/e289c991c64fd0bb7f36cbf9a9f945fff80f02bddad705b2b4f13fd2ac9a9b0f.jpg)

![](images/f3602ffcca70d18a74be252e92bcd9d907cd94c8a184c65633ed4dc9035a54e7.jpg)  
(b) Number of Visual Elements  
Figure 5: Performance comparison of different input frames with different video complexity for models trained in long contexts (over 8K tokens). The horizontal coordinate is the number of input frames.

Diverse Visual Characteristics. As shown in Table 2, large performance disparities emerge when models process videos with different visual char-<sub>70.0</sub> acteristics. All tested models perform better with60.0 low-dynamic content, but they struggle with high-<sub>o</sub>re dynamic and multi-scene videos, and show theF weakest performance when handling videos con-20.0 taining multiple subjects.Number of V

![](images/a2a8b3e4a806e7f3d3c08496910f1f48fde3ec886996050647310d73aaf168f2.jpg)

![](images/1e82777a8358f6592c5842bb05925374357339ba53378e461e78ca6b2aaa98a5.jpg)  
Figure 6: Performance comparison across different video complexities.

Video Complexity. We partition TUNA-1K based on the number of events and visual elements to investigate how increasing video complexity affects model performance. As shown in Figure 6, the F1 scores demonstrate a consistent downward trend as video complexity increases, indicating that the comprehension of complex videos remains a formidable challenge for current models. More details are available in Appendix B.2.1.

Enrichment of Visual Inputs. To explore the challenges posed by complex videos, we further investigate the impact of increasing frame number on videos with varying complexity. As shown in Figure 5, we analyze LLaVA-Video and Qwen2- VL, both trained with longer context lengths. Our findings reveal that F1 scores decrease with increasing video complexity at any given number of input frames. Generally, increasing the number of frames results in greater improvements for more complex samples, suggesting that complex videos require more frames for a complete and precise description. Counterintuitively, an unexpected pattern emerges: for the most complex videos, increasing frames from 32 to 64 actually reduces performance, indicating that highly complex videos remain a prominent challenge for LMMs. Further details be found in Appendix B.2.2.

<table><tr><td>Measure</td><td>Kendall&#x27;s τ</td><td>Spearman&#x27;s ρ</td><td>Pearson r</td></tr><tr><td>METEOR (Banerjee and Lavie, 2005)</td><td>30.8</td><td>44.8</td><td>54.7</td></tr><tr><td>BERT-Score (Zhang et al., 2019)</td><td>27.4</td><td>34.8</td><td>49.2</td></tr><tr><td>CLAIR (Chan et al., 2023)</td><td>45.6</td><td>56.6</td><td>41.0</td></tr><tr><td>DREAM-1K (Wang et al., 2024a)</td><td>22.2</td><td>31.3</td><td>24.7</td></tr><tr><td>TUNA-CAP</td><td>57.2</td><td>76.7</td><td>69.9</td></tr></table>

Table 3: Human judgment correlation scores for our automatic evaluation. All p-values < 0.05.

Correlation with Human Judgments. To validate the effectiveness and robustness of our automatic evaluation method, we calculate Kendall’s τ, Spearman’s $\rho ,$ and Pearson r correlation scores between several methods and human evaluation. As shown in Table 3, these results demonstrate strong correlation, confirming that our method provides a robust and accurate solution for captioning evaluation. More details are available in Appendix B.2.4.

<table><tr><td rowspan="2">Model</td><td colspan="2">Camera State</td><td colspan="2">Background Scene</td><td colspan="3">Subject Action</td><td colspan="3">Object Attribute</td><td rowspan="2">Overall</td></tr><tr><td>Motion</td><td>Transition</td><td>Description</td><td>Transition</td><td>Recognition</td><td>Sequence</td><td>Matching</td><td>Recognition</td><td>Appearance</td><td>Location</td></tr><tr><td colspan="10">Open-Source LMMs</td><td></td></tr><tr><td>PLLaVA-7B</td><td>29.7</td><td>31.9</td><td>48.1</td><td>22.4</td><td>43.6</td><td>34.6</td><td>30.4</td><td>32.3</td><td>38.1</td><td>45.2</td><td>33.7</td></tr><tr><td>LongVA-7B</td><td>37.5</td><td>41.5</td><td>63.0</td><td>30.8</td><td>44.6</td><td>44.7</td><td>43.5</td><td>41.7</td><td>47.6</td><td>40.5</td><td>42.4</td></tr><tr><td>Tarsier-7B</td><td>23.0</td><td>24.6</td><td>40.7</td><td>20.6</td><td>38.6</td><td>26.9</td><td>45.7</td><td>20.9</td><td>25.9</td><td>23.8</td><td>26.5</td></tr><tr><td>Kangaroo</td><td>33.2</td><td>47.3</td><td>53.7</td><td>38.3</td><td>49.5</td><td>38.8</td><td>54.3</td><td>47.2</td><td>43.5</td><td>59.5</td><td>42.9</td></tr><tr><td>LLaVA-OV-7B</td><td>42.2</td><td>54.6</td><td>57.4</td><td>48.6</td><td>42.6</td><td>41.4</td><td>60.9</td><td>47.9</td><td>50.0</td><td>59.5</td><td>47.4</td></tr><tr><td>LLaVA-Video-7B</td><td>39.1</td><td>50.7</td><td>59.3</td><td>46.7</td><td>52.5</td><td>52.4</td><td>56.5</td><td>53.6</td><td>61.9</td><td>47.6</td><td>50.6</td></tr><tr><td>Qwen2-VL-7B</td><td>41.0</td><td>51.7</td><td>66.7</td><td>45.8</td><td>54.5</td><td>52.8</td><td>65.2</td><td>49.0</td><td>60.2</td><td>57.1</td><td>51.3</td></tr><tr><td>InternVL2-8B</td><td>41.0</td><td>53.1</td><td>66.7</td><td>40.2</td><td>45.5</td><td>50.5</td><td>50.0</td><td>45.8</td><td>56.8</td><td>45.2</td><td>48.4</td></tr><tr><td>MiniCPM-V-2.6</td><td>39.8</td><td>45.9</td><td>59.3</td><td>34.6</td><td>49.5</td><td>51.1</td><td>52.2</td><td>42.2</td><td>46.6</td><td>50.0</td><td>45.7</td></tr><tr><td>PLLaVA-34B</td><td>42.6</td><td>41.5</td><td>63.0</td><td>43.9</td><td>45.5</td><td>48.5</td><td>56.5</td><td>43.2</td><td>56.8</td><td>57.1</td><td>46.9</td></tr><tr><td>Tarsier-34B</td><td>43.0</td><td>48.3</td><td>72.2</td><td>45.8</td><td>51.5</td><td>50.2</td><td>56.5</td><td>49.7</td><td>53.7</td><td>61.9</td><td>50.1</td></tr><tr><td>LLaVA-OV-72B</td><td>46.5</td><td>67.6</td><td>75.9</td><td>57.0</td><td>59.4</td><td>56.6</td><td>73.9</td><td>63.5</td><td>69.5</td><td>59.5</td><td>60.0</td></tr><tr><td>LLaVA-Video-72B</td><td>47.7</td><td>67.6</td><td>77.8</td><td>61.7</td><td>61.4</td><td>57.0</td><td>65.2</td><td>62.5</td><td>73.7</td><td>57.1</td><td>60.7</td></tr><tr><td>Qwen2-VL-72B</td><td>52.7</td><td>64.7</td><td>74.1</td><td>55.1</td><td>62.4</td><td>54.4</td><td>67.4</td><td>63.0</td><td>76.3</td><td>66.7</td><td>60.7</td></tr><tr><td>InternVL2-76B</td><td>43.8</td><td>61.8</td><td>74.1</td><td>43.0</td><td>50.5</td><td>50.5</td><td>54.3</td><td>52.1</td><td>66.1</td><td>57.1</td><td>53.1</td></tr><tr><td colspan="10">Closed-Source LMMs</td><td></td></tr><tr><td>Gemini 1.5 Flash</td><td>40.8</td><td>58.3</td><td>70.4</td><td>52.3</td><td>48.0</td><td>54.2</td><td>63.0</td><td>49.0</td><td>66.7</td><td>64.3</td><td>53.3</td></tr><tr><td>Gemini 1.5 Pro</td><td>49.4</td><td>68.4</td><td>64.8</td><td>59.8</td><td>55.0</td><td>60.4</td><td>69.6</td><td>64.6</td><td>65.0</td><td>66.7</td><td>60.8</td></tr><tr><td>GPT-40</td><td>53.9</td><td>56.0</td><td>81.5</td><td>56.1</td><td>59.4</td><td>67.6</td><td>58.7</td><td>56.8</td><td>63.6</td><td>59.5</td><td>60.3</td></tr></table>

Table 4: TUNA-MCQ performance of representative video LMMs. We provide detailed scores for selected tested models on 10 temporal tasks. The best and second-best results are marked with bold and underline, respectively.

## 4.3 Video QA

TUNA-MCQ specializes in temporal understanding in videos, emphasizing the necessity of the entire video observation rather than single-frame analysis. We assess the temporal understanding skills across 4 dynamic elements and 10 task types.

Overall Performance. Table 4 showcases the performance of selected models on TUNA-MCQ. All tested models demonstrate limited capabilities, with even the best-performing model barely achieving a passing score. However, a promising trend emerges as open-source models illustrate performance on par with commercial counterparts. Specifically, LLaVA-Video-72B and Qwen2-VL-72B achieve an identical score of 60.7%, matching the performance of GPT-4o (60.3%) and Gemini 1.5 Pro (60.8%). This competitive performance of open-source models aligns with findings from recent studies, such as Video-MME (Short) (Fu et al., 2024) and TempCompass (Liu et al., 2024f), suggesting a promising direction for open-source development in video understanding.

Camera State. Recent works (Chai et al., 2024; Tan et al., 2024) emphasize the crucial role of camera state in video understanding and generation. However, open-source video understanding datasets minimally involve this aspect. Our evaluation reveals a considerable weakness in models camera understanding skill, with average scores notably lower than overall scores. While models show some promise in detecting camera transitions, they struggle particularly with camera motion analysis, achieving a maximum score of only 53.9%.

Subject Action. Action understanding is another challenge, as it requires tracking and interpreting character state evolutions across multiple frames. The action sequence task is notoriously difficult due to its complexity, demanding models to simultaneously recognize individual actions while understanding their temporal order and causal relationships. While GPT-4o leads performance with 67.6% accuracy, all other models struggle to the passing threshold. Additionally, temporal action recognition remains challenging, with even the best-performing model achieving only 62.4%.

Background Scene & Object Attribute. Advanced video LMMs show promising capabilities in scene and attribute understanding. For background scene tasks, models achieve impressive results with GPT-4o reaching 81.5% on scene description, while LLaVA-Video-72B attains 61.7% on scene transition understanding. In object attribute tasks, models also perform well, with top scores of 64.6% in recognition, 76.3% in appearance, and 66.7% in location tasks. These strong performance can be attributed to the transfer of knowledge from well-established image-text understanding techniques, as these tasks share similar characteristics with multi-image analysis scenarios.

These comprehensive results underscore the complex challenges in understanding temporal dynamics in videos, while offering clear directions for future improvements in video LMMs.

## 4.4 Synthesizing Analysis

Through comprehensive analysis of TUNA-CAP and TUNA-MCQ results, commercial models demonstrate superior performance across both tasks. While open-source models (Qwen2-VL-72B and LLaVA-Video-72B) achieve comparable results on TUNA-MCQ, they notably underperform in TUNA-CAP. This performance gap reveals a critical limitation of open-source LMMs in captioning and even open-ended QA tasks, indicating areas demanding further research efforts.

## 5 Conclusion

In this paper, we present TUNA-1K, a temporally dense video-caption dataset, and its derivative benchmark TUNA. Our work focuses on temporal dynamics, the distinctive feature between videos and static images, by examining four critical temporal aspects: camera, scene, action, and attribute. TUNA-1K features comprehensive coverage across diverse visual domains with detailed, fine-grained captions. TUNA evaluates LMMs temporal understanding skills through two complementary tasks: captioning and MCQ. This comprehensive evaluation provides precise insights into models’ strengths and weaknesses, offering interpretable metrics for advancing video understanding technology. We envision TUNA serving as a catalyst for future research in video understanding. Moreover, the meticulously annotated TUNA-1K, with its high accuracy and completeness, offers versatile applications beyond our current scope. We anticipate its broad utility in diverse research directions and look forward to seeing its impact on future studies in the field.

## Limitations

Our dataset is highly fine-grained, but the data annotation is extremely labor-intensive. making it costly to apply this construction method to other video datasets. For TUNA-CAP, we conduct a comprehensive evaluation of the video captioning capabilities of video LMMs using an interpretable and robust approach. However, our method has certain limitations. Our scoring system focuses on the alignment with annotated visual elements. If the model outputs visual elements that fail to match the annotated events or elements, our method cannot assess their precision. Specifically, when a generated caption includes excessive irrelevant content, even if this content contains substantial hallucinatory information, our method would be unable to provide a valid assessment in such cases.

## Ethics Policy

To increase the diversity of our dataset, we collected videos from several sources. These include a number of movies spanning many years and several types. While we made an effort to remove some videos that were poorly observed or NSFW, there may be unintentional data that involve potential social biases and stereotypes, including stereotypical items related to gender, race, ethnicity, age, and socioeconomic status. This requires careful judgment and utilization of the data.

## Acknowledgments

This work is supported by the National Natural Science Foundation of China (No. 62272092, No. 62172086).

## References

Elmira Amirloo, Jean-Philippe Fauconnier, Christoph Roesmann, Christian Kerl, Rinu Boney, Yusu Qian, Zirui Wang, Afshin Dehghan, Yinfei Yang, Zhe Gan, et al. 2024. Understanding alignment in multimodal llms: A comprehensive study. arXiv preprint arXiv:2407.02477.

Hidehisa Arai, Keita Miwa, Kento Sasaki, Yu Yamaguchi, Kohei Watanabe, Shunsuke Aoki, and Issei Yamamoto. 2024. Covla: Comprehensive vision-language-action dataset for autonomous driving. arXiv preprint arXiv:2408.10845.

Satanjeev Banerjee and Alon Lavie. 2005. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. pages 65–72, Ann Arbor, Michigan.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. 2024. Video generation models as world simulators.

Davide Caffagni, Federico Cocchi, Luca Barsellotti, Nicholas Moratelli, Sara Sarto, Lorenzo Baraldi, Lorenzo Baraldi, Marcella Cornia, and Rita Cucchiara. 2024. The revolution of multimodal large language models: A survey. pages 13590–13618, Bangkok, Thailand and virtual meeting.

Mu Cai, Reuben Tan, Jianrui Zhang, Bocheng Zou, Kai Zhang, Feng Yao, Fangrui Zhu, Jing Gu, Yiwu Zhong, Yuzhang Shang, et al. 2024. Temporalbench: Benchmarking fine-grained temporal understanding for multimodal video models. arXiv preprint arXiv:2410.10818.

Wenhao Chai, Enxin Song, Yilun Du, Chenlin Meng, Vashisht Madhavan, Omer Bar-Tal, Jeng-Neng Hwang, Saining Xie, and Christopher D Manning. 2024. Auroracap: Efficient, performant video detailed captioning and a new benchmark. arXiv preprint arXiv:2410.03051.

David Chan, Suzanne Petryk, Joseph Gonzalez, Trevor Darrell, and John Canny. 2023. CLAIR: Evaluating image captions with large language models. pages 13638–13646, Singapore.

Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. 2024a. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325.

Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. 2024b. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821.

Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. 2024. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476.

Andrei de Souza Inácio and Heitor Silvério Lopes. 2023. Evaluation metrics for video captioning: A survey. Machine Learning with Applications, 13:100488.

Xinyu Fang, Kangrui Mao, Haodong Duan, Xiangyu Zhao, Yining Li, Dahua Lin, and Kai Chen. 2024. Mmbench-video: A long-form multi-shot benchmark for holistic video understanding. Advances in Neural Information Processing Systems, 37:89098–89124.

Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. 2024. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075.

Xuan Ju, Yiming Gao, Zhaoyang Zhang, Ziyang Yuan, Xintao Wang, Ailing Zeng, Yu Xiong, Qiang Xu, and Ying Shan. 2024. Miradata: A large-scale video dataset with long durations and structured captions. arXiv preprint arXiv:2407.06358.

Fanheng Kong, Jingyuan Zhang, Yahui Liu, Hongzhi Zhang, Shi Feng, Xiaocui Yang, Daling Wang, Yu Tian, Victoria W, Fuzheng Zhang, and Guorui Zhou. 2025. Modality curation: Building universal embeddings for advanced multimodal information retrieval. arXiv preprint arXiv:2505.19650.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2024a. Llavaonevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326.

Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, Jianfeng Gao, et al. 2024b. Multimodal foundation models: From specialists to general-purpose assistants. Foundations and Trends® in Computer Graphics and Vision, 16(1- 2):1–214.

Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. 2024c. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR.

Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. 2024d. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206.

Yunxin Li, Xinyu Chen, Baotian Hu, Longyue Wang, Haoyuan Shi, and Min Zhang. 2024e. Videovista: A versatile benchmark for video understanding and reasoning. arXiv preprint arXiv:2406.11303.

Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. 2023. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. pages 74–81, Barcelona, Spain.

Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. 2024. Vila: On pretraining for visual language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26689–26699.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024a. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2024b. Visual instruction tuning. Advances in neural information processing systems, 36.

Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, Xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu. 2024c. Kangaroo: A powerful videolanguage model supporting long-context video input. arXiv preprint arXiv:2408.15542.

Shilong Liu, Hao Cheng, Haotian Liu, Hao Zhang, Feng Li, Tianhe Ren, Xueyan Zou, Jianwei Yang, Hang Su, Jun Zhu, et al. 2025. Llava-plus: Learning to

use tools for creating multimodal agents. In European Conference on Computer Vision, pages 126– 142. Springer.

Tingkai Liu, Yunzhe Tao, Haogeng Liu, Qihang Fang, Ding Zhou, Huaibo Huang, Ran He, and Hongxia Yang. 2024d. DeVAn: Dense video annotation for video-language models. pages 14305–14321, Bangkok, Thailand.

Ye Liu, Zongyang Ma, Zhongang Qi, Yang Wu, Chang Wen Chen, and Ying Shan. 2024e. E.t. bench: Towards open-ended event-level video-language understanding. In Neural Information Processing Systems (NeurIPS).

Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. 2024f. Tempcompass: Do video llms really understand videos? arXiv preprint arXiv:2403.00476.

Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. 2023. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424.

Neelu Madan, Andreas Møgelmose, Rajat Modi, Yogesh S Rawat, and Thomas B Moeslund. 2024. Foundation models for video understanding: A survey. arXiv preprint arXiv:2405.03770.

Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. 2023. Egoschema: A diagnostic benchmark for very long-form video language understanding. Advances in Neural Information Processing Systems, 36:46212–46244.

mixkit. 2023. mixkit. https://mixkit.com/ videos/.

Thong Nguyen, Yi Bin, Junbin Xiao, Leigang Qu, Yicong Li, Jay Zhangjie Wu, Cong-Duy Nguyen, See-Kiong Ng, and Luu Anh Tuan. 2024. Video-language understanding: A survey from model architecture, model training, and data perspectives. arXiv preprint arXiv:2406.05615.

OpenAI. 2024. Hello gpt-4o.

Xichen Pan, Li Dong, Shaohan Huang, Zhiliang Peng, Wenhu Chen, and Furu Wei. 2023. Kosmos-g: Generating images in context with multimodal large language models. arXiv preprint arXiv:2310.02992.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. pages 311–318, Philadelphia, Pennsylvania, USA.

Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. 2024. Perception test: A diagnostic benchmark for multimodal video models. Advances in Neural Information Processing Systems, 36.

Pexels. 2023. Pexels. https://www.pexels.com/ videos/.

pixabay. 2023. pixabay. https://pixabay.com/ videos/.

Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. 2024. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720.

Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530.

Darshana Saravanan, Darshan Singh, Varun Gupta, Zeeshan Khan, Vineet Gandhi, and Makarand Tapaswi. 2024. Velociti: Can video-language models bind semantic concepts through time? arXiv preprint arXiv:2406.10889.

Ziyao Shangguan, Chuhan Li, Yuxuan Ding, Yanan Zheng, Yilun Zhao, Tesca Fitzgerald, and Arman Cohan. 2024. Tomato: Assessing visual temporal reasoning capabilities in multimodal foundation models. arXiv preprint arXiv:2410.23266.

Zhiyu Tan, Xiaomeng Yang, Luozheng Qin, and Hao Li. 2024. Vidgen-1m: A large-scale dataset for text-tovideo generation. arXiv preprint arXiv:2408.02629.

Yunlong Tang, Jing Bi, Siting Xu, Luchuan Song, Susan Liang, Teng Wang, Daoan Zhang, Jie An, Jingyang Lin, Rongyi Zhu, et al. 2023. Video understanding with large language models: A survey. arXiv preprint arXiv:2312.17432.

Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. 2015. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4566–4575.

Jiawei Wang, Liping Yuan, and Yuchen Zhang. 2024a. Tarsier: Recipes for training and evaluating large video description models. arXiv preprint arXiv:2407.00634.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. 2024b. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191.

Yuxuan Wang, Difei Gao, Licheng Yu, Weixian Lei, Matt Feiszli, and Mike Zheng Shou. 2022. Geb+: A benchmark for generic event boundary captioning, grounding and retrieval. In European Conference on Computer Vision, pages 709–725. Springer.

Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. 2021. Next-qa: Next phase of questionanswering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777–9786.

Tianwei Xiong, Yuqing Wang, Daquan Zhou, Zhijie Lin, Jiashi Feng, and Xihui Liu. 2024. Lvd-2m: A longtake video dataset with temporally dense captions. arXiv preprint arXiv:2410.10816.

Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. 2024. Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994.

Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. 2024. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800.

Duzhen Zhang, Yahan Yu, Chenxing Li, Jiahua Dong, Dan Su, Chenhui Chu, and Dong Yu. 2024a. Mmllms: Recent advances in multimodal large language models. arXiv preprint arXiv:2401.13601.

Jingyuan Zhang, Hongzhi Zhang, Zhou Haonan, Chenxi Sun, Jiakang Wang, Fanheng Kong, Yahui Liu, Qi Wang, Fuzheng Zhang, et al. 2025. Data metabolism: An efficient data design schema for vision language model. arXiv preprint arXiv:2504.12316.

Pan Zhang, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Rui Qian, Lin Chen, Qipeng Guo, Haodong Duan, Bin Wang, Linke Ouyang, et al. 2024b. Internlmxcomposer-2.5: A versatile large vision language model supporting long-contextual input and output. arXiv preprint arXiv:2407.03320.

Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. 2024c. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852.

Ruohong Zhang, Liangke Gui, Zhiqing Sun, Yihao Feng, Keyang Xu, Yuanhan Zhang, Di Fu, Chunyuan Li, Alexander Hauptmann, Yonatan Bisk, et al. 2024d. Direct preference optimization of video large multimodal models from language model reward. arXiv preprint arXiv:2404.01258.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. 2019. Bertscore: Evaluating text generation with bert. arXiv preprint arXiv:1904.09675.

Yiqun Zhang, Fanheng Kong, Peidong Wang, Shuang Sun, SWangLing SWangLing, Shi Feng, Daling Wang, Yifei Zhang, and Kaisong Song. 2024e. STICKERCONV: Generating multimodal empathetic responses from scratch. pages 7707–7733, Bangkok, Thailand.

Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. 2024f. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713.

Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. 2023. Multimodal chain-of-thought reasoning in language models. arXiv preprint arXiv:2302.00923.

Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. 2024. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264.

Luowei Zhou, Chenliang Xu, and Jason Corso. 2018. Towards automatic learning of procedures from web instructional videos. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. 2023. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592.

## A TUNA-1K

## A.1 Statistics

![](images/8def413ebcfa53192cb85b7e2985bdabb75c33a42c2843c9d2682001ce4712b3.jpg)  
Figure 7: The sample distribution of TUNA-1K, videos covering 4 visual characteristics and 12 domains.

As shown in Table 5, we illustrate the detailed statistics of TUNA-1K. Each video must belong to one of Low-Dynamic or High-Dynamic categories, while Multi-Scene and Multi-Subject are optional.

![](images/054757fec8461ace93171f4610ef7b7395d44dea30a981b751afca9bb52f5736.jpg)  
Figure 8: Sample distribution of domains in the TUNA-1K, covering 12 domains.

There are 12 domains contained in TUNA-1K, including: (1) Film, (2) Daily Life, (3) Cooking, (4) Sports Activity, (5) Driving, (6) Animals & Pets, (7) Natural Landscape, (8) Cityscape, (9) Urban Activity, (10) Foods, (11) Plants, and (12) Autos & Vehicles. As shown in Figure 8, we illustrate the domain statistics of the videos in TUNA-1K.

We visualize the sample distribution of video complexity in TUNA-1K in Figure 9, in terms of (a) the number of events, (b) the number of visual elements in each video, and (c) the number of visual elements in each event.

## A.2 More Details of TUNA-1K Construction

## A.2.1 Video Collection

Table 6 shows the video sources that make up the TUNA-1K, along with their descriptions.

## A.2.2 Annotators

We employ crowdsourcing for data annotation. All annotators have TEM-4 or TEM-8 English proficiency, and have experience in video captioning annotation (e.g., several annotators have previously annotated video-caption pairs for Kling<sup>1</sup> project). Prior to formal annotation, they undergo our specialized training to guarantee the quality of their annotation.

## A.2.3 Annotator Training

We prepare a detailed note document for instructing human annotators on annotation. The documentation encompasses 5 key components: (1) Visual Characteristic Classification, (2) Video Element Guidelines, (3) Video Captioning Protocol, (4) Event Splitting and Element Extraction Criterion, and (5) Annotation Examples.

Visual Characteristic Classification. Detailed criteria for categorizing videos based on their visual characteristics.

• Low/High-Dynamic: Based on the number and frequency of dynamic elements in the video.

• Multi-Scene: Presence of at least one camera transition or scene transition. Excludes those that just have camera zooming, panning, or rotating.

• Multi-Subject: Presence of at least two subjects. Non-major objects are not counted.

Video Element Guidelines. Comprehensive definitions and key considerations for essential video elements:

• Camera: Camera states, including panning, rotating, zooming, following, shaking, transition, etc. It is necessary to indicate a specific direction.

• Scene: Describe the background scene, including environment, weather, time, etc.

• Action: Recognize actions and their temporal evolving sequences.

• Attribute: Identify objects and describe their appearance (e.g., characters’ gender, age, and dress, objects’ color, shape, and number) and spatial orientation (location and relative positional relationships).

Video Captioning Protocol. We emphasizing:

• Strict chronological ordering of events.

• Objective descriptions without summarization and subjective feelings.

• If multiple similar characters/objects appear, distinguish them in expression by unique attributes (e.g., age, dress, etc.).

<table><tr><td></td><td>Low-Dynamic</td><td>High-Dynamic</td><td>Multi-Scene</td><td>Multi-Subject</td><td>Total</td></tr><tr><td>#Videos</td><td>340</td><td>660</td><td>493</td><td>385</td><td>1,000</td></tr><tr><td>Duration</td><td>18.0s</td><td>12.8s</td><td>12.5s</td><td>9.5s</td><td>14.53s</td></tr><tr><td>#Events</td><td>2.8</td><td>3.4</td><td>3.8</td><td>3.8</td><td>3.2</td></tr><tr><td>#Elements (Narrative-level)</td><td>15.8</td><td>18.3</td><td>19.9</td><td>20.2</td><td>17.5</td></tr><tr><td>#Elements (Event-level)</td><td>5.7</td><td>5.4</td><td>5.3</td><td>5.3</td><td>5.48</td></tr><tr><td>#Tokens</td><td>198.8</td><td>247.1</td><td>255.9</td><td>267.6</td><td>230.7</td></tr></table>

Table 5: Detailed statistics for TUNA-1K, including: number of videos (#Videos), video duration (Duration), number of events (#Events), number of visual elements in captions (#Elements (Narrative-level)), number of visual elements in events (#Elements (Narrative-level)), number of tokens of caption (#Tokens).

![](images/09a6d714e480400b1269914ba209de4002e0f9d8c9e0e91c4acb7be6787ee751.jpg)  
(a) Number of Events

![](images/44c1fff8c4acbd623855b1a68f322b3ac4144d843c9e8349b94b1eb82ca59df6.jpg)  
(b) Number of Visual Elements (Narrative-level)

![](images/edb6267b461adf3104b7dce2ac5a5117fcfe2f90386c6d6b9b81715eb9bd6031.jpg)  
(c) Number of Visual Elements (Event-level)  
Figure 9: Visual statistics of the number of events and the number of visual elements in TUNA-1K.

Event Splitting and Element Extraction Criterion. To ensure systematic and standardized annotation, we establish the following comprehensive guidelines:

• Divide captions into chronologically ordered events, where each event represents distinct temporal activities. Further decompose each event into its constituent visual elements.

• Ensure explicit subject identification in all visual elements. Replace missing subjects and pronouns with their corresponding specific noun references to maintain clarity and precision.

• Element weighting criteria for scoring: (1) Weight 3: primary and conspicuous contents in the video. (2) Weight 2: primary and inconspicuous contents, or secondary but conspicuous contents. (3) Weight 1: secondary and inconspicuous contents.

Annotation Examples. Some complete annotation examples provide human annotators with a further guidance for annotation.

To ensure annotation quality and consistency, we implemented a rigorous annotator selection and training process. Initially, all potential annotators underwent a trial annotation phase using a shared subset of videos. This phase served both as a training exercise and a qualification assessment. Through careful evaluation of their trial annotations, we selected only those annotators who demonstrated high consistency, accuracy, and thorough understanding of the annotation guidelines. These qualified annotators then proceeded to participate in the main annotation task. This systematic approach helped maintain annotation quality while minimizing potential inconsistencies across different annotators.

## A.2.4 Annotation

Video Filter. We first filter out undesired videos based on specific rules, e.g., videos with low resolution (not satisfying 480p) and long duration (>40s). Then, human annotators filter out near-static or NSFW videos to ensure the high quality and temporal dynamics of the selected videos.

Video Cluster. Firstly, we assign a caption for each video. If the original source provides a caption, it is utilized; otherwise, a caption is generated using gpt-4o-2024-05-13. Then, we utilize GPT-4o to classify visual characteristic category and domain for each video. Thus far, we have obtained raw videos with initial model-generated visual characteristics and domains. Initially, we obtain raw videos with model-generated visual characteristics and domains. Annotators then observe the videos, correcting and supplementing the visual characteristic categories and domains as needed. The prompt instruction used in this step is shown in Figure 23.

Temporally Dense Caption Annotation. Annotators are tasked with providing a detailed chronological description of each video. They must divide the caption into multiple events based on criteria such as camera transitions, scene transitions, or story advancements. Each event is further split into multiple atomic visual elements, categorized by type and weighted by importance on a scale of 1-3. The types include camera, scene, action, and attribute.

![](images/2e45dc61ce7e6eaef69c6f8c617b090afc1d475dea4b4438b3c1014c4b88e456.jpg)  
(a) Open-source models (< 34B)

![](images/d2ee7b36ca8bf7e4b9f7d3b44e644dcf715964d0de2fc6408fdacf9fbb658c02.jpg)

![](images/6a5d75d741273daf58c3ad619224350ccb6704f001f223a1a12b4e7756e3b9c8.jpg)  
(c) Open-source models (< 34B)

(b) Open-source models (≥ 34B) and commercial models  
![](images/4486f6cbcf12dd84836f8d7afd9e3edcfe595985130ab82f6c01719d3576b489.jpg)  
(d) Open-source models (≥ 34B) and commercial models  
Figure 10: The whole performance comparison on TUNA-CAP across different video complexities.

Quality Review. For quality assurance, crossinspections are performed between annotators. Furthermore, trained video experts (non-authors) continuously review the annotations, offering feedback and prompting annotators to refine their work to ensure the high-quality annotations. During crossinspections and expert reviews, the checking covers all annotation results including video caption, event splitting and visual element extraction as well as the type and weight of the elements.

## A.2.5 Visualized Examples

A detailed example in TUNA-1K is shown in Figure 15.

## B TUNA-CAP

## B.1 Experimental Settings

The configuration and experimental settings for all test models are shown in Table 7.

The specific version of the closed-source models we tested are gemini-1.5-flash-002, gemini-1.5-pro-002, and gpt-4o-2024-08-06. Incidentally, a few samples (less than 5) in our TUNA-CAP and TUNA-MCQ do not receive any responses from the Gemini (Reid et al., 2024) series, possibly due to security mechanisms. Therefore, we calculated the scores using only the samples with responses, rather than assigning a score of 0 to those without responses.

Input Frames. By default, we uniformly sample 32 frames from each video, which is sufficient to capture the entire content of the video in our TUNA. For Qwen2-VL (Wang et al., 2024b) and PLLaVA (Xu et al., 2024), the official strategy is followed to sample frames at 2 FPS and uniformly sample 16 frames, respectively. For closed-source models, we sampled frames dynamically with 1/2 FPS, meaning that when the video event is less than 16s, it is sampled at 2 FPS, otherwise at 1 FPS.

Detailed Prompts. The default prompt template for captioning is shown in Figure 19. Figures 20, 21, and 22 illustrate the prompt templates used to evaluate TUNA-CAP.

## B.2 More Experimental Analysis

TUNA-CAP results of all tested models are shown in Table 8 and Table 9, as a complement result to Table 2.

## B.2.1 Video Complexity

We partition the video complexity according to the number of events and the number of visual elements in the video, to observe the impact of the model on increasing video complexity. The visualization results of selected models are shown in Figure 6. The detailed results of all tested models are in Table 10, and its visualization results are shown in Figure 10.

As demonstrated in Table 10 and Figure 10, model performance consistently declines with increasing video complexity. Larger models $\scriptstyle ( \geq 3 4 \mathrm { B }$ parameters) exhibit better robustness to complex videos, showing smaller performance drops (2.8% for event count, 2.5% for element count) compared to their smaller counterparts (<34B parameters), which experience steeper declines (4.7% and 3.5% respectively). Moreover, the performance gap between large and small models becomes more pronounced in highly complex videos. When event count exceeds 9 (from 7\~8 events), small models suffer a substantial 6.2% performance drop, while large models remain stable with only a 0.7% variation. Similarly, for videos with more than 31 elements (increased from 26\~30), small models show a 3.0% fluctuation compared to just 0.7% for large models. This evidence strongly suggests that larger models possess superior adaptability to complex video content.

## B.2.2 Enrichment of Visual Inputs

The number of input frames is crucial for video understanding, as it directly impacts whether the model receives sufficient visual content. This is particularly important in long-video scenarios, where the model’s ability to answer a question depends on whether the sampled frames contain the necessary visual information. Limited by the number of input frames in existing LMMs, our TUNA-1K ensures that 32 frames are sufficient to cover the content of each video, considering that TUNA-1K has an average duration of 15s and a maximum duration50.0<sub>(</sub>% of 38s. To explore the effect of frame number on Sc performance, we compare the TUNA-CAP performance with different input frame numbers across40.0 several classical models.Frames

![](images/835647e7bae1bee0de9c614c8d684885051a566e6d4e81764e04f0a34f164fcc.jpg)

![](images/1f6f315d9c0e15dadf049256671703ca2ea8648f3a53c1778807ec8dcd3a0d3c.jpg)  
Figure 11: Performance comparison across different number of input frames.

As shown in Figure 11, increasing the number of frames generally improves the F1 score, with an average increase of 1.86% from 8 to 16 frames and

1.62% from 16 to 32 frames. This underscores the importance of providing sufficient visual information, especially when the frame count is low. Similar pattern is shown in action perception, with average improvements of 3.48% from 8 to 16 frames and 2.48% from 16 to 32 frames, indicating that dynamic actions are more sensitive to frame numbers. However, we observe a performance drop in some earlier models, including LLaVA-OV-7B (Li et al., 2024a), InternVL2-26B (Chen et al., 2024b), MiniCPM-V-2.6 (Yao et al., 2024), when the frame number is increased to 64. We attribute this decline to the fact that these earlier models rarely involve 64 frames of input (context length over 8K) during training, leading to poorer performance at 64 frames. In contrast, LLaVA-Video-7B (Zhang et al., 2024f) and Qwen2-VL-7B (Wang et al., 2024b), which are trained on longer contexts, achieve better results when the number of frames reaches 64. This indicates that providing more frames can indeed enhance performance when the context length is not constrained. More frames can improve the ability to capture intricate temporal dynamics and rich contextual information in videos. Consequently, exploring how to efficiently utilize more frames for training will emerge as a pivotal topic in the field of multimodal video understanding.

To further explore the effect of frame numbers on video understanding, we select LLaVA-Video and Qwen2-VL, which are trained with longer contexts, to illustrate the performance disparity across different video complexities with varying input frame numbers. Figure 5 presents the visualized results, while Table 12 provides the corresponding specific scores. These results demonstrate that increasing the number of frames is more beneficial for understanding more complex videos. However, excessive complexity can lead to performance anomalies, indicating that understanding highly complex videos remains a prominent challenge.

## B.2.3 Scaling Law

As shown in Table 2, there is a general law that the performance of models increases as the model scale increases. Therefore, the scaling law is equally valid for video captioning task. Larger models typically have more parameters, enabling them to capture more complex patterns and nuances in the data, leading to improved performance. However, we notice that the LLaVA-Video series shows inconsistent performance scaling with model size. This anomaly may be attributed to the Slow-Fast approach used in LLaVA-Video-72B, which results in $2 / 3$ of the visual tokens being compressed to $1 / 4$ of the others. This compression leads to a extensive loss of fine-grained information, which is crucial for detailed video understanding and accurate captioning. This observation suggests that the efficient usage of visual information is essential and may even outweigh the impact yielded by the language model scale. The quality and richness of the visual tokens play a critical role in the overall performance of video captioning models.

Discussion. This observation has sparked an intriguing discussion in the field: video LMMs demonstrate superior performance when processing a higher number of input frames. While increased frame coverage provides a more comprehensive representation of video content, capturing nuanced details and temporal dynamics, this advantage is constrained by context length limitations. Specifically, accommodating more frames typically involves the compression of visual tokens, a process that remains a key technical challenge. Future research should focus on the development of more efficient visual token compression techniques and the innovation in architectural designs that can handle extended context lengths, to unlock the full potential of large-scale models in video understanding tasks.

## B.2.4 Correlation with Human Judgments

Given a video-caption pair, this task is to check whether the metric is consistent with human scoring. Specifically, we randomly sample 40 videos containing 687 visual elements. We provided human scorers with reference meta-information and model-generated captions. The scorers were asked to sequentially determine whether each reference visual element appeared accurately and completely in the candidate captions in the correct temporal order, ultimately resulting in human-assigned scores. Finally, we calculate Kendall’s $\tau ,$ Spearman’s $\rho ,$ and Pearson $r$ to test the consistency of TUNA-$\mathrm { C A P ^ { \prime } s }$ automatic evaluation method with human scoring. The calculated Kendall’s τ , Spearman’s $\rho ,$ and Pearson $r$ are 57.2%, 76.7%, and 69.9%, respectively, with all p-values < 0.05, demonstrating the validity of our automatic evaluation method. CLAIR (Chan et al., 2023), an evaluation method for image captioning, is an LLM-based strategy for scoring based on reference captions. We migrate this approach seamlessly to assess video captioning as a comparative object. DREAM-1K (Wang et al.,

2024a) is a recently proposed method for video captioning evaluation with interpretability. However, it only focuses on subject actions, leading to its weak performance on our comprehensive video captioning data that focuses on camera, scene, action, and attributes.

## C TUNA-MCQ

## C.1 Statistics

![](images/6bdf3ed988d618b043de2b85fe2361dbbd33c7671bfbd0d68d39ca8c64f2e2c7.jpg)  
Figure 12: Sample distribution of task types in the TUNA-MCQ, covering 10 task types.

Figure 12 illustrates the sample distribution of the TUNA-MCQ, across 10 tasks: (1) camera motion, e.g, zooming, panning, and rotating. (2) camera transition. (3) scene description. (4) scene transition. (5) action recognition. (6) action sequence. (7) action-subject matching. (8) object recognition. (9) object appearance, e.g., gender, age, dress, color, shape, number. and (10) object location.

![](images/8d76ced74e53e120ed1fbbfe04002305507b414730f8e1df09d85f8646932649.jpg)  
Figure 13: Sample distribution of correct option in the TUNA-MCQ.

To eliminate the bias and varied sensitivity of the models towards order and token, we ensure that the distribution of correct options is uniform, as shown in Figure 13.

## C.2 More Details of TUNA-MCQ Construction

Error-prone Points Extraction. To obtain challenging questions, we obtain some error-prone points through an automated approach. Specifically, we provide the video LMM with 8 frames from the video and its ground-truth textual description, and ask it to generate what it thinks it sees that the video is inconsistent with the textual description. The prompt instruction used in this step is shown in Figure 25.

Multi-Choice QA Generation. Based on a predefined set of task types, error-prone points and textual descriptions, LLM generates several multichoice QAs for each video. The prompt instruction used in this step is shown in Figure 26.

Quality Review. To ensure that data is high-quality and time-sensitive, we employ crowdsourcing to optimize the automatically generated data. In addition, human annotators perform cross-inspections to ensure quality. To guarantee that the questions are relevant to capture temporal dynamics, we employ LLaVA-Video-7B to filter them. A question is deemed temporal-indispensable if it can be accurately answered using both a single frame and multiple frames. Specifically, we deem the question to be temporal-indispensable if it can be answered correctly by both 1-frame and 16-frame inputs.

## C.2.1 Visualized Examples

Several examples in TUNA-MCQ are shown in Figure 16, 17, and 18.

## C.3 Experimental Settings

The number of input frames in TUNA-MCQ is consistent with TUNA-CAP, which is shown in Table 7. The default prompt template for multi-choice QA is shown in Figure 24.

Incidentally, a few samples (less than 10) in our TUNA-MCQ do not receive any responses from the Gemini series, possibly due to security mechanisms. Therefore, we calculated the scores using only the samples with responses, rather than assigning a score of 0 to those without responses.

## C.4 More Experimental Analysis

TUNA-MCQ results of all tested models are shown in Table 13, as a complement result to Table 4.

## C.4.1 Scaling Law

On TUNA-MCQ, while most models demonstrate predictable scaling patterns, InternVL2 (Chen et al., 2024b) exhibits an unexpected trend where its 76B variant underperforms the 40B version and its 26B variant underperforms the 8B version. This anomaly is consistently observed across multiple video comprehension benchmarks: Video-MME (76B: 64.7% vs. 40B: 66.1%), MVBench (76B:

69.6% vs. 40B: 72.0%), MMBench-Video (76B: 1.71% vs. 40B: 1.78%), MLVU (76B: 69.9% vs. 40B: 71.0%). Notably, this counter-intuitive scaling behavior can be attributed to architectural differences: each InternVL2 variant employs distinct LLM backbone families and vision encoders, making direct performance comparisons less meaningful for establishing scaling laws.

## D Future Work

Considering that different models have diverse capabilities in following complex instructions, we deliberately adopted simple prompting templates to ensure fair comparison and clear assessment. While this approach helps isolate models’ inherent temporal understanding abilities, advanced prompting strategies like Multimodal-CoT (Zhang et al., 2023) reasoning show promising potential for performance enhancement. Although such sophisticated prompting techniques may improve performance on TUNA-MCQ, their applicability to captioning tasks like TUNA-CAP remains challenging. We encourage future research to explore advanced prompting strategies that can effectively enhance temporal understanding across different tasks while maintaining a balance between performance optimization and the assessment of fundamental temporal comprehension abilities.

## E More Related Work

Video LMMs. Large Mulitmodal Models (LMMs) have mushroomed, showcasing impressive visual understanding capabilities (Li et al., 2024b; Zhang et al., 2024a; Caffagni et al., 2024; Amirloo et al., 2024; Zhang et al., 2025). These advances have catalyzed the development of diverse and innovative applications across multiple domains. (Pan et al., 2023; Zhang et al., 2024e; Liu et al., 2025; Kong et al., 2025). Existing works bridge visual encoders and Large Language Models (LLMs) using a small intermediate architecture, as seen in models like LLaVA (Liu et al., 2024b,a), BLIP-2 (Li et al., 2023), and MiniGPT-4 (Zhu et al., 2023), which facilitate the evolution of visual-language LMMs. On this basis, recent researches (Li et al., 2024c; Zhang et al., 2024b; Lin et al., 2024; Cheng et al., 2024; Lin et al., 2023; Maaz et al., 2023) have extended these techniques from static images to dynamic videos, demonstrating promising results in video understanding by processing videos as multiple image frames.

<table><tr><td rowspan=1 colspan=1>Type </td><td rowspan=1 colspan=1>Source</td><td rowspan=1 colspan=1>Domain</td><td rowspan=1 colspan=1>Visual Characteristic</td><td rowspan=1 colspan=1>Description</td></tr><tr><td rowspan=3 colspan=1>Webb Dta</td><td rowspan=1 colspan=1>Pexels (Pexels,2023)</td><td rowspan=1 colspan=1>Animals &amp; PetsAutos &amp; VehiclesCityscapeFoodsNatural LandscapeUrban Activity</td><td rowspan=1 colspan=1>Low-Dynamic</td><td rowspan=1 colspan=1>A website offer stock videos and motion graphics free from copyright issues, which are usuallyexceptionally high-quality videos uploaded by skilled photographers. We sample 46 videos, as a source ofLow-Dynamic scenarios, covering diverse domains.</td></tr><tr><td rowspan=1 colspan=1>Pixabay (pixabay,2023)</td><td rowspan=1 colspan=1>Animals &amp; PetsCityscapeFoodsNatural LandscapeUrban Activity</td><td rowspan=1 colspan=1>Low-Dynamic</td><td rowspan=1 colspan=1>A website offer stock videos and motion graphics free from copyright issues, which are usuallyexceptionally high-quality videos uploaded by skilled photographers. We sample 13 videos, as a source ofLow-Dynamic scenarios, covering diverse domains.</td></tr><tr><td rowspan=1 colspan=1>MixKit (mixkit,2023)</td><td rowspan=1 colspan=1>Natural Landscape</td><td rowspan=1 colspan=1>Low-Dynamic</td><td rowspan=1 colspan=1>A website offer stock videos and motion graphics free from copyright issues, which are usuallyexceptionally high-quality videos uploaded by skilled photographers. We sample 7 videos, as a source ofLow-Dynamic scenarios.</td></tr><tr><td rowspan=4 colspan=1>Acai A  Vata</td><td rowspan=1 colspan=1>DREAM-1K(Wang et al.,2024a)</td><td rowspan=1 colspan=1>Film</td><td rowspan=1 colspan=1>Low-DynamicHigh-DynamicMulti-SceneMulti-Subject</td><td rowspan=1 colspan=1>DREAM-1K consists of 1,000 video clips from five categories: live-action movies, animated movies,stock videos, YouTube videos, and TikTok-style short videos. These videos typically feature multipleevents and subjects across various shots. We sample 148 videos from live-action movies that meet ourselection principles, mostly as a source of High-Dynamic, Multi-Scene, and Multi-Subject scenarios.</td></tr><tr><td rowspan=1 colspan=1>VELOCITI(Saravanan et al.,2024)</td><td rowspan=1 colspan=1>Film</td><td rowspan=1 colspan=1>Low-DynamicHigh-DynamicMulti-SceneMulti-Subject</td><td rowspan=1 colspan=1>A benchmark using complex movie clips and dense semantic role label annotations to test perception andbinding in video LMMs. The videos feature challenging scenarios with frequent shot changes, fast actionsequences, multi-event situations, role switching, and entity co-referencing over time. We sample 266videos, mostly as a source of High-Dynamic, Multi-Scene, and Multi-Subject scenarios.</td></tr><tr><td rowspan=1 colspan=1>PerceptionTest(Patraucean et al.,2024)</td><td rowspan=1 colspan=1>Daily Life(Indoor)</td><td rowspan=1 colspan=1>Low-DynamicHigh-DynamicMulti-Scene</td><td rowspan=1 colspan=1>A dataset evaluates performance across skill areas (memory, abstraction, physics, semantics) andreasoning types (descriptive, explanatory, predictive, counterfactual). We sample 114 videos, mostly as asource of High-Dynamic scenarios.</td></tr><tr><td rowspan=1 colspan=1>YouCook2 (Zhouet al., 2018)</td><td rowspan=1 colspan=1>Cooking</td><td rowspan=1 colspan=1>High-DynamicMulti-SceneMulti-Subject</td><td rowspan=1 colspan=1>A dataset of YouTube videos covering 89 recipes from four major cuisines (Africa, Americas, Asia,Europe), featuring diverse cooking styles and challenges like fast camera motion, camera zooms, videodefocus, and scene-type changes. We sample 100 videos, mostly as a source of High-Dynamic scenarios.</td></tr><tr><td rowspan=2 colspan=1>Aci   ata</td><td rowspan=1 colspan=1>VIDGEN-1M (Tanet al., 2024)</td><td rowspan=1 colspan=1>Animals &amp; PetsAutos &amp; VehiclesCityscapeFoodsNatural LandscapePlantsUrban ActivitySports Activity</td><td rowspan=1 colspan=1>Low-DynamicHigh-DynamicMulti-Scene</td><td rowspan=1 colspan=1>Open-domain Text-to-Video dataset with high video quality, high temporal consistency, and balancedcategories. We sample 154 videos, as a source of High-Dynamic, Multi-Scene (Sports Activity) scenarios,and Low-Dynamic (other domains) scenarios.</td></tr><tr><td rowspan=1 colspan=1>MiraData (Ju et al.,2024)</td><td rowspan=1 colspan=1>Animals &amp; PetsAutos &amp; VehiclesCityscapeFoodsNatural LandscapePlantsUrban Activity</td><td rowspan=1 colspan=1>Low-DynamicMulti-Scene</td><td rowspan=1 colspan=1>A large-scale, high-quality video dataset designed to meet the key expectations of video generation tasks:diverse content, high visual quality, long duration, and significant motion strength. Unlike existingtext-to-video datasets that primarily source videos from YouTube, MiraData includes videos fromYouTube, Videvo, Pixabay, and Pexels, ensuring a more comprehensive and suitable data source. Wesample 102 videos, mostly as a source of Low-Dynamic scenarios, covering diverse domains.</td></tr><tr><td rowspan=1 colspan=1>Odrs</td><td rowspan=1 colspan=1>CoVLA (Arai et al.,2024)</td><td rowspan=1 colspan=1>Driving</td><td rowspan=1 colspan=1>Low-DynamicMulti-SceneMulti-Subject</td><td rowspan=1 colspan=1>The CoVLA (Comprehensive Vision-Language-Action) dataset is a novel large-scale resource designed toadvance autonomous driving research. The dataset includes synchronized multi-modal data streams fromfront-facing cameras, in-vehicle signals, and other sensors, providing a comprehensive view of diversedriving scenarios. We choose it due to its complex scene variations. We sample 50 videos as a source ofMulti-Scene scenarios.</td></tr></table>

Table 6: Rich video sources within TUNA-1K. Domain denote the domains represented in the sampled data. Visual Characteristic indicates the visual characteristics present in the sampled data, with bold representing major features and grey representing minor features.. We also provide a brief description of each dataset, along with our our selection criteria and counts.

![](images/ad4ed81196555f29920ae09945ede91c7ec7210b7ea1f9897329efd9a2591ab9.jpg)  
Figure 14: Several video understanding benchmark examples and analysis.

<table><tr><td>Model</td><td>LLM</td><td>Vision Model</td><td>#Frames</td></tr><tr><td colspan="4">Open-Source LMMs</td></tr><tr><td>Qwen2-VL-72B</td><td>Qwen2-72B</td><td>ViT-600M</td><td>2FPS</td></tr><tr><td>Qwen2-VL-7B</td><td>Qwen2-7B</td><td>ViT-600M</td><td>2FPS</td></tr><tr><td>LLaVA-Video-72B</td><td>Qwen2-72B</td><td>SigLIP-400M</td><td>32</td></tr><tr><td>LLaVA-Video-7B</td><td>Qwen2-7B</td><td>SigLIP-400M</td><td>32</td></tr><tr><td>LLaVA-OneVision-72B</td><td>Qwen2-72B</td><td>SigLIP-400M</td><td>32</td></tr><tr><td>LLaVA-OneVision-7B</td><td>Qwen2-7B</td><td>SigLIP-400M</td><td>32</td></tr><tr><td>InternVL2-76B</td><td>Llama-3-70B-Instruct</td><td>InternViT-6B</td><td>32</td></tr><tr><td>InternVL2-40B</td><td>Nous-Hermes-2-Yi-34B</td><td>InternViT-6B</td><td>32</td></tr><tr><td>InternVL2-26B</td><td>InternLM2-20B</td><td>InternViT-6B</td><td>32</td></tr><tr><td>InternVL2-8B</td><td>InternLM2.5-7B</td><td>InternViT-300M</td><td>32</td></tr><tr><td>Tarsier-34B</td><td>Nous-Hermes-2-Yi-34B</td><td>CLIP ViT-L/14</td><td>32</td></tr><tr><td>Tarsier-7B</td><td>Vicuna-v1.5-7B</td><td>CLIP ViT-L/14</td><td>32</td></tr><tr><td>PLLaVA-34B</td><td>Nous-Hermes-2-Yi-34B</td><td>CLIP ViT-L/14</td><td>16</td></tr><tr><td>PLLaVA-13B</td><td>Vicuna-v1.5-13B</td><td>CLIP ViT-L/14</td><td>16</td></tr><tr><td>PLLaVA-7B</td><td>Vicuna-v1.5-7B</td><td>CLIP ViT-L/14</td><td>16</td></tr><tr><td>MiniCPM-V-2.6</td><td>Qwen2-7B</td><td>SigLIP-400M</td><td>32</td></tr><tr><td>Kangaroo</td><td>Llama3-8B-Instruct</td><td>EVA-CLIP-L</td><td>32</td></tr><tr><td>LongVA-7B</td><td>Qwen2-7B-Instruct-224K</td><td>CLIP ViT-L/14</td><td>32</td></tr><tr><td colspan="4">Closed-Source LMMs</td></tr><tr><td>GPT-40</td><td>Unknown</td><td>Unknown</td><td>1/2 FPS*</td></tr><tr><td>Gemini 1.5 Pro</td><td>Unknown</td><td>Unknown</td><td>1/2 FPS*</td></tr><tr><td>Gemini 1.5 Flash</td><td>Unknown</td><td>Unknown</td><td>1/2 FPS*</td></tr></table>

Table 7: The number of frames used in the TUNA evaluation in Section 4.2, 4.3. By default, 32 frames are sampled uniformly, which is enough to cover the content of each video in TUNA-CAP. Some models take a different number of frames because they are limited by the input length or according to their sampling recommendations. ∗ indicates that 2 FPS is employed when the video duration < 16s, otherwise 1 FPS is employed. The versions of the closed-source models are gpt-4o-2024-08-06, gemini-1.5-pro-002, gemini-1.5-flash-002.

<table><tr><td rowspan="2">Model</td><td colspan="3">Camera</td><td colspan="3">Scene</td><td colspan="3">Action</td><td colspan="3">Attribute</td><td colspan="3">Overall</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td colspan="10">Open-Source LMMs</td><td colspan="3"></td><td colspan="3"></td></tr><tr><td>PLLaVA-7B</td><td>49.4</td><td>22.6</td><td>28.9</td><td>52.2</td><td>30.9</td><td>36.6</td><td>30.5</td><td>12.6</td><td>16.5</td><td>44.5</td><td>19.5</td><td>25.3</td><td>60.0</td><td>19.1</td><td>27.4</td></tr><tr><td>LongVA-7B</td><td>52.3</td><td>26.0</td><td>32.5</td><td>56.5</td><td>34.4</td><td>40.6</td><td>38.9</td><td>17.2</td><td>22.0</td><td>50.6</td><td>22.0</td><td>28.4</td><td>71.6</td><td>22.3</td><td>31.8</td></tr><tr><td>Tarsier-7B</td><td>56.9</td><td>27.3</td><td>34.8</td><td>45.3</td><td>28.2</td><td>33.1</td><td>56.7</td><td>28.9</td><td>36.2</td><td>56.4</td><td>26.0</td><td>33.3</td><td>73.0</td><td>27.9</td><td>38.6</td></tr><tr><td>Kangaroo</td><td>65.2</td><td>36.5</td><td>44.1</td><td>67.8</td><td>45.4</td><td>51.9</td><td>49.3</td><td>26.0</td><td>31.9</td><td>59.8</td><td>32.2</td><td>39.5</td><td>69.5</td><td>32.5</td><td>42.7</td></tr><tr><td>LLaVA-OV-7B</td><td>75.2</td><td>42.0</td><td>51.0</td><td>71.8</td><td>51.2</td><td>57.6</td><td>54.1</td><td>30.4</td><td>36.8</td><td>66.2</td><td>42.0</td><td>49.3</td><td>73.6</td><td>38.6</td><td>49.3</td></tr><tr><td>LLaVA-Video-7B</td><td>74.0</td><td>41.5</td><td>50.4</td><td>73.6</td><td>52.3</td><td>58.9</td><td>57.0</td><td>30.8</td><td>37.8</td><td>72.1</td><td>44.8</td><td>53.1</td><td>77.0</td><td>39.7</td><td>51.0</td></tr><tr><td>Qwen2-VL-7B</td><td>72.3</td><td>40.7</td><td>49.0</td><td>71.9</td><td>50.0</td><td>56.7</td><td>55.9</td><td>30.1</td><td>37.0</td><td>68.2</td><td>38.4</td><td>46.7</td><td>77.8</td><td>37.6</td><td>48.9</td></tr><tr><td>InternVL2-8B</td><td>64.8</td><td>33.7</td><td>41.7</td><td>59.4</td><td>38.7</td><td>44.7</td><td>45.2</td><td>24.7</td><td>30.0</td><td>59.8</td><td>35.5</td><td>42.3</td><td>67.2</td><td>31.1</td><td>40.8</td></tr><tr><td>MiniCPM-V-2.6</td><td>76.5</td><td>47.8</td><td>56.0</td><td>75.0</td><td>54.1</td><td>60.6</td><td>57.2</td><td>31.8</td><td>38.8</td><td>68.7</td><td>42.3</td><td>50.2</td><td>76.0</td><td>40.7</td><td>51.7</td></tr><tr><td>PLLaVA-13B</td><td>57.0</td><td>25.8</td><td>33.0</td><td>57.3</td><td>34.0</td><td>40.3</td><td>36.2</td><td>13.8</td><td>18.5</td><td>50.0</td><td>23.3</td><td>29.8</td><td>65.0</td><td>21.4</td><td>30.6</td></tr><tr><td>InternVL2-26B</td><td>73.2</td><td>43.2</td><td>51.6</td><td>72.5</td><td>52.6</td><td>58.7</td><td>51.7</td><td>30.9</td><td>37.0</td><td>63.9</td><td>42.3</td><td>49.1</td><td>70.0</td><td>39.2</td><td>49.0</td></tr><tr><td>PLLaVA-34B</td><td>60.8</td><td>29.6</td><td>37.4</td><td>56.2</td><td>33.7</td><td>39.9</td><td>38.7</td><td>17.3</td><td>22.3</td><td>55.1</td><td>26.1</td><td>33.2</td><td>67.8</td><td>24.5</td><td>34.2</td></tr><tr><td>Tarsier-34B</td><td>63.6</td><td>34.3</td><td>42.3</td><td>59.0</td><td>38.4</td><td>44.4</td><td>65.6</td><td>39.9</td><td>47.6</td><td>63.6</td><td>34.3</td><td>42.2</td><td>77.1</td><td>36.7</td><td>48.2</td></tr><tr><td>InternVL2-40B</td><td>77.8</td><td>46.3</td><td>55.1</td><td>71.9</td><td>53.1</td><td>59.0</td><td>53.4</td><td>33.1</td><td>39.3</td><td>65.9</td><td>45.7</td><td>52.3</td><td>71.3</td><td>42.1</td><td>51.7</td></tr><tr><td>LLaVA-OV-72B</td><td>73.5</td><td>43.7</td><td>51.9</td><td>71.5</td><td>51.1</td><td>57.5</td><td>51.2</td><td>30.2</td><td>36.0</td><td>65.7</td><td>41.4</td><td>48.8</td><td>72.7</td><td>39.2</td><td>49.6</td></tr><tr><td>LLaVA-Video-72B</td><td>72.7</td><td>41.7</td><td>50.3</td><td>71.1</td><td>49.9</td><td>56.4</td><td>55.7</td><td>32.7</td><td>39.3</td><td>68.1</td><td>43.2</td><td>50.8</td><td>73.7</td><td>39.6</td><td>50.2</td></tr><tr><td>Qwen2-VL-72B</td><td>73.6</td><td>45.9</td><td>54.0</td><td>67.6</td><td>46.3</td><td>52.8</td><td>59.1</td><td>35.7</td><td>42.6</td><td>66.6</td><td>40.7</td><td>48.5</td><td>74.7</td><td>41.1</td><td>51.7</td></tr><tr><td>InternVL2-76B</td><td>75.1</td><td>45.4</td><td>53.9</td><td>73.3</td><td>55.8</td><td>61.4</td><td>55.7</td><td>34.9</td><td>41.2</td><td>64.3</td><td>44.5</td><td>50.9</td><td>70.7</td><td>42.3</td><td>51.9</td></tr><tr><td colspan="10">Closed-Source LMMs</td><td colspan="7"></td></tr><tr><td>Gemini 1.5 Flash</td><td>74.6</td><td>52.8</td><td>59.6</td><td>77.2</td><td>59.3</td><td>65.1</td><td>58.7</td><td>36.4</td><td>42.9</td><td>69.0</td><td>48.4</td><td>55.2</td><td>72.7</td><td>46.4</td><td>55.7</td></tr><tr><td>Gemini 1.5 Pro</td><td>78.7</td><td>53.0</td><td>60.7</td><td>75.7</td><td>57.4</td><td>63.3</td><td>59.0</td><td>40.3</td><td>46.3</td><td>69.0</td><td>49.4</td><td>56.0</td><td>73.7</td><td>48.1</td><td>57.4</td></tr><tr><td>GPT-40</td><td>80.1</td><td>53.3</td><td>61.3</td><td>79.5</td><td>60.2</td><td>66.4</td><td>64.0</td><td>41.1</td><td>48.0</td><td>73.8</td><td>50.1</td><td>57.8</td><td>77.7</td><td>48.2</td><td>58.5</td></tr><tr><td rowspan="2">Model</td><td colspan="3">Low-Dynamic</td><td colspan="3">High-Dynamic</td><td colspan="3">Multi-Scene</td><td colspan="3">Multi-Subject</td><td colspan="3">Overall</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td colspan="10">Open-Source LMMs</td><td colspan="3"></td><td colspan="3"></td></tr><tr><td>PLLaVA-7B</td><td>66.5</td><td>23.0</td><td>32.7</td><td>56.6</td><td>17.1</td><td>24.7</td><td>55.7</td><td>15.5</td><td>22.8</td><td>56.2</td><td>15.3</td><td>22.5</td><td>60.0</td><td>19.1</td><td>27.4</td></tr><tr><td>LongVA-7B</td><td>75.9</td><td>26.5</td><td>37.3</td><td>69.4</td><td>20.1</td><td>29.0</td><td>68.3</td><td>19.0</td><td>27.6</td><td>67.3</td><td>15.7</td><td>23.7</td><td>71.6</td><td>22.3</td><td>31.8</td></tr><tr><td>Tarsier-7B</td><td>81.2</td><td>34.3</td><td>46.5</td><td>68.7</td><td>24.5</td><td>34.5</td><td>71.7</td><td>25.3</td><td>35.8</td><td>67.8</td><td>23.2</td><td>33.2</td><td>73.0</td><td>27.9</td><td>38.6</td></tr><tr><td>Kangaroo</td><td>73.2</td><td>34.7</td><td>45.6</td><td>67.6</td><td>31.3</td><td>41.1</td><td>66.2</td><td>29.7</td><td>39.3</td><td>63.5</td><td>26.3</td><td>35.7</td><td>69.5</td><td>32.5</td><td>42.7</td></tr><tr><td>LLaVA-OV-7B</td><td>78.6</td><td>38.4</td><td>50.0</td><td>71.0</td><td>38.8</td><td>48.9</td><td>71.7</td><td>38.3</td><td>48.4</td><td>67.1</td><td>33.8</td><td>43.8</td><td>73.6</td><td>38.6</td><td>49.3</td></tr><tr><td>LLaVA-Video-7B</td><td>80.7</td><td>40.0</td><td>52.2</td><td>75.1</td><td>39.5</td><td>50.3</td><td>77.1</td><td>38.6</td><td>50.0</td><td>73.5</td><td>34.6</td><td>45.8</td><td>77.0</td><td>39.7</td><td>51.0</td></tr><tr><td>Qwen2-VL-7B</td><td>81.2</td><td>42.0</td><td>53.8</td><td>76.0</td><td>35.3</td><td>46.4</td><td>76.8</td><td>33.2</td><td>44.4</td><td>73.6</td><td>28.9</td><td>39.9</td><td>77.8</td><td>37.6</td><td>48.9</td></tr><tr><td>InternVL2-8B</td><td>71.6</td><td>34.0</td><td>44.5</td><td>64.9</td><td>29.7</td><td>38.9</td><td>65.6</td><td>29.1</td><td>38.4</td><td>61.5</td><td>26.6</td><td>35.2</td><td>67.2</td><td>31.1</td><td>40.8</td></tr><tr><td>MiniCPM-V-2.6</td><td>79.3</td><td>41.4</td><td>53.0</td><td>74.3</td><td>40.4</td><td>51.0</td><td>76.5</td><td>40.8</td><td>51.7</td><td>73.5</td><td>38.3</td><td>49.0</td><td>76.0</td><td>40.7</td><td>51.7</td></tr><tr><td>PLLaVA-13B</td><td>69.8</td><td>25.7</td><td>36.0</td><td>62.5</td><td>19.1</td><td>27.8</td><td>62.3</td><td>17.6</td><td>26.0</td><td>60.3</td><td>16.3</td><td>24.3</td><td>65.0</td><td>21.4</td><td>30.6</td></tr><tr><td>InternVL2-26B</td><td>71.9</td><td>39.1</td><td>49.4</td><td>69.0</td><td>39.2</td><td>48.9</td><td>70.3</td><td>38.6</td><td>48.4</td><td>67.2</td><td>36.3</td><td>45.8</td><td>70.0</td><td>39.2</td><td>49.0</td></tr><tr><td>PLLaVA-34B</td><td>74.5</td><td>28.1</td><td>38.9</td><td>64.3</td><td>22.6</td><td>31.8</td><td>63.9</td><td>21.3</td><td>30.2</td><td>60.7</td><td>19.2</td><td>27.6</td><td>67.8</td><td>24.5</td><td>34.2</td></tr><tr><td>Tarsier-34B</td><td>79.6</td><td>37.2</td><td>49.1</td><td>75.8</td><td>36.5</td><td>47.8</td><td>77.6</td><td>38.1</td><td>49.6</td><td>74.4</td><td>36.0</td><td>47.3</td><td>77.1</td><td>36.7</td><td>48.2</td></tr><tr><td>InternVL2-40B</td><td>75.0</td><td>43.8</td><td>53.9</td><td>69.5</td><td>41.2</td><td>50.5</td><td>70.7</td><td>40.8</td><td>50.5</td><td>67.9</td><td>38.7</td><td>48.0</td><td>71.3</td><td>42.1</td><td>51.7</td></tr><tr><td>LLaVA-OV-72B</td><td>75.4</td><td>37.3</td><td>48.6</td><td>71.3</td><td>36.7</td><td>45.9</td><td>71.4</td><td>40.1</td><td>50.1</td><td>72.3</td><td>39.1</td><td>49.4</td><td>72.7</td><td>39.2</td><td>49.6</td></tr><tr><td>LLaVA-Video-72B</td><td>77.3</td><td>39.2</td><td>50.6</td><td>71.9</td><td>39.8</td><td>50.0</td><td>73.9</td><td>38.6</td><td>49.3</td><td>70.5</td><td>35.1</td><td>45.7</td><td>73.7</td><td>39.6</td><td>50.2</td></tr><tr><td>Qwen2-VL-72B</td><td>79.2</td><td>44.6</td><td>55.7</td><td>72.4</td><td>39.3</td><td>49.7</td><td>73.6</td><td>37.2</td><td>48.0</td><td>69.1</td><td>32.8</td><td>43.3</td><td>74.7</td><td>41.1</td><td>51.7</td></tr><tr><td>InternVL2-76B</td><td>72.0</td><td>43.1</td><td>52.8</td><td>70.1</td><td>41.9</td><td>51.5</td><td>71.4</td><td>41.1</td><td>51.1</td><td>68.6</td><td>39.7</td><td>49.3</td><td>70.7</td><td>42.3</td><td>51.9</td></tr><tr><td colspan="10">Closed-Source LMMs</td><td colspan="7"></td></tr><tr><td>Gemini 1.5 Flash</td><td>74.0</td><td>46.5</td><td>56.0</td><td>72.0</td><td>46.4</td><td>55.5</td><td>73.4</td><td>46.2</td><td>55.9</td><td>73.4</td><td>46.2</td><td>55.9</td><td>72.7</td><td>46.4</td><td>55.7</td></tr><tr><td>Gemini 1.5 Pro</td><td>76.7</td><td>48.7</td><td>58.7</td><td>72.1</td><td>47.8</td><td>56.7</td><td>73.4</td><td>47.7</td><td>57.0</td><td>69.9</td><td>44.1</td><td>53.3</td><td>73.7</td><td>48.1</td><td>57.4</td></tr><tr><td>GPT-40</td><td>79.1</td><td>47.3</td><td>58.2</td><td>77.0</td><td>48.6</td><td>58.7</td><td>78.7</td><td>47.2</td><td>58.1</td><td>76.8</td><td>44.4</td><td>55.5</td><td>77.7</td><td>48.2</td><td>58.5</td></tr></table>

Table 8: Evaluation results in terms of dynamic element categoryies on TUNA-CAP. The best and second-best results are marked with orange and blue , respectively.

Table 9: Evaluation results in terms of visual characteristic categoryies on TUNA-CAP. The best and second-best results are marked with orange and blue , respectively.

<table><tr><td rowspan="3">Model</td><td colspan="5">#Events</td><td colspan="5">#Elements</td><td rowspan="3">Overall</td></tr><tr><td>≤2</td><td>3~4</td><td>5~6</td><td>7~8</td><td>≥9</td><td>≤ 15</td><td>16~20</td><td>21~25</td><td>26~30</td><td>≥31</td></tr><tr><td colspan="10">Open-Source LMMs</td></tr><tr><td>PLLaVA-7B</td><td>32.1</td><td>25.8</td><td>21.9</td><td>16.9</td><td>14.6</td><td>32.1</td><td>26.9</td><td>21.9</td><td>20.6</td><td>17.6</td><td>27.4</td></tr><tr><td>LongVA-7B</td><td>35.5</td><td>31.1</td><td>25.1</td><td>24.1</td><td>19.9</td><td>37.4</td><td>30.3</td><td>26.3</td><td>24.6</td><td>22.9</td><td>31.8</td></tr><tr><td>Tarsier-7B</td><td>42.5</td><td>37.7</td><td>33.5</td><td>29.2</td><td>18.5</td><td>43.7</td><td>36.3</td><td>34.6</td><td>33.0</td><td>31.5</td><td>38.6</td></tr><tr><td>Kangaroo</td><td>45.9</td><td>42.6</td><td>35.0</td><td>35.0</td><td>19.0</td><td>46.6</td><td>42.4</td><td>40.0</td><td>34.5</td><td>28.7</td><td>42.7</td></tr><tr><td>LLaVA-OV-7B</td><td>52.1</td><td>48.8</td><td>45.2</td><td>38.7</td><td>35.5</td><td>54.0</td><td>47.6</td><td>46.4</td><td>42.0</td><td>38.8</td><td>49.3</td></tr><tr><td>LLaVA-Video-7B</td><td>53.5</td><td>50.7</td><td>45.1</td><td>44.2</td><td>39.6</td><td>55.1</td><td>50.0</td><td>47.4</td><td>44.1</td><td>42.9</td><td>51.0</td></tr><tr><td>Qwen2-VL-7B</td><td>53.3</td><td>48.9</td><td>39.3</td><td>30.3</td><td>22.0</td><td>55.0</td><td>46.9</td><td>43.9</td><td>42.8</td><td>34.6</td><td>48.9</td></tr><tr><td>InternVL2-8B</td><td>44.2</td><td>40.5</td><td>34.4</td><td>33.2</td><td>13.5</td><td>45.9</td><td>39.5</td><td>36.2</td><td>35.4</td><td>25.9</td><td>40.8</td></tr><tr><td>MiniCPM-V-2.6</td><td>52.8</td><td>51.2</td><td>52.3</td><td>47.3</td><td>47.0</td><td>54.9</td><td>49.4</td><td>49.4</td><td>48.4</td><td>52.6</td><td>51.7</td></tr><tr><td>PLLaVA-13B</td><td>35.0</td><td>30.0</td><td>22.2</td><td>12.2</td><td>14.0</td><td>35.9</td><td>29.9</td><td>24.7</td><td>22.9</td><td>17.8</td><td>30.6</td></tr><tr><td>InternVL2-26B</td><td>50.4</td><td>48.8</td><td>46.2</td><td>45.4</td><td>44.8</td><td>52.4</td><td>47.4</td><td>46.1</td><td>45.3</td><td>47.8</td><td>49.0</td></tr><tr><td>Avg (&lt;34B)</td><td>45.2</td><td>41.5 (-3.7)</td><td>36.4 (-5.1)</td><td>32.4 (-4.0)</td><td>26.2 (-6.2)</td><td>46.6</td><td>40.6 (-6.0)</td><td>37.9 (-2.7)</td><td>35.8 (-2.1)</td><td>32.8 (-3.0)</td><td>42.0</td></tr><tr><td>PLLaVA-34B</td><td>39.6</td><td>33.0</td><td>24.5</td><td>24.6</td><td>15.9</td><td>40.7</td><td>32.4</td><td>27.3</td><td>27.4</td><td>22.7</td><td>34.2</td></tr><tr><td>Tarsier-34B</td><td>48.7</td><td>48.3</td><td>47.0</td><td>46.6</td><td>41.1</td><td>50.9</td><td>46.6</td><td>45.9</td><td>47.7</td><td>43.8</td><td>48.2</td></tr><tr><td>InternVL2-40B</td><td>54.2</td><td>51.2</td><td>45.0</td><td>45.4</td><td>53.9</td><td>55.9</td><td>50.5</td><td>47.3</td><td>46.4</td><td>46.2</td><td>51.7</td></tr><tr><td>LLaVA-OV-72B</td><td>50.3</td><td>49.6</td><td>49.9</td><td>42.6</td><td>40.1</td><td>52.8</td><td>48.2</td><td>46.4</td><td>44.7</td><td>49.1</td><td>49.6</td></tr><tr><td>LLaVA-Video-72B</td><td>51.5</td><td>50.4</td><td>44.2</td><td>48.0</td><td>48.9</td><td>54.1</td><td>49.8</td><td>44.9</td><td>43.7</td><td>47.7</td><td>50.2</td></tr><tr><td>Qwen2-VL-72B</td><td>55.1</td><td>51.9</td><td>44.2</td><td>32.3</td><td>33.0</td><td>56.9</td><td>51.0</td><td>46.4</td><td>43.8</td><td>39.5</td><td>51.7</td></tr><tr><td>InternVL2-76B</td><td>54.8</td><td>51.2</td><td>48.8</td><td>41.2</td><td>43.1</td><td>56.0</td><td>49.7</td><td>49.1</td><td>47.5</td><td>47.0</td><td>51.9</td></tr><tr><td>Avg (≥34B)</td><td>50.6</td><td>47.9 (-2.7)</td><td>43.4 (-4.6)</td><td>40.1 (-3.3)</td><td>39.4 (-0.7)</td><td>52.5</td><td>46.9 (-5.6)</td><td>43.9 (-3.0)</td><td>43.0 (-0.9)</td><td>42.3 (-0.7)</td><td>48.2</td></tr><tr><td colspan="10">Closed-Source LMMs</td><td></td><td></td></tr><tr><td>Gemini 1.5 Flash</td><td></td><td>54.8</td><td>55.8</td><td></td><td></td><td></td><td>53.6</td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini 1.5 Pro</td><td>57.6 59.4</td><td>57.0</td><td>54.7</td><td>48.9 44.7</td><td>48.3 54.7</td><td>59.1 60.9</td><td>55.2</td><td>53.0 54.6</td><td>53.8 55.2</td><td>52.2 54.8</td><td>55.7 57.4</td></tr><tr><td>GPT-40</td><td>60.9</td><td>58.2</td><td>55.6</td><td>50.2</td><td>41.3</td><td>61.7</td><td>57.9</td><td>56.0</td><td>53.7</td><td>50.4</td><td>58.5</td></tr><tr><td>Avg (close-source)</td><td>59.3</td><td>56.7 (-2.6)</td><td>55.4 (-1.3)</td><td>47.9 (-7.4)</td><td>48.1 (+0.2)</td><td>60.6</td><td>55.6 (-5.0)</td><td>54.5 (-1.0)</td><td></td><td>52.5 (-1.8)</td><td>57.2</td></tr><tr><td>Avg (Total)</td><td>49.0</td><td>45.8 (-3.2)</td><td>41.4 (-4.4)</td><td>37.2 (-4.2)</td><td>33.7 (-3.4)</td><td>50.6</td><td>44.8 (-5.7)</td><td>42.3 (-2.6)</td><td>54.2 (-0.3) 40.8 (-1.4)</td><td>38.8 (-2.0)</td><td>46.2</td></tr></table>

Table 10: Detailed performance comparison with varying video complexities. Video complexity is measured by the number of events and the number of visual elements in the video. The inference setup is consistent with Table 7.

<table><tr><td>Model</td><td>Frames</td><td>Camera</td><td>Scene</td><td>Action</td><td>Attribute</td><td>Low-Dynamic</td><td>High-Dynamic</td><td>Multi-Scene</td><td>Multi-Subject</td><td>Overall</td></tr><tr><td rowspan="5"> $\mathrm { L L a V A – O V  – 7 B }$ </td><td>8</td><td>50.2</td><td>56.6</td><td>33.0</td><td>47.8</td><td>50.4</td><td>46.1</td><td>46.2</td><td>42.6</td><td>47.5</td></tr><tr><td>16</td><td> $4 9 . 3 \ ( . 0 . 9 )$ </td><td> $5 7 . 2 \ ( + 0 . 6 ) $ </td><td> $3 5 . 6 \ \AA ( + 2 . 6 )$ </td><td> $4 8 . 9 \ _ { ( + 1 . 1 ) }$ </td><td> $5 0 . 2 \ ( . 0 . 2 )$ </td><td> $4 7 . 1 \ ( + 1 . 0 )$ </td><td> $4 7 . 0 \ _ { ( + 0 . 8 ) }$ </td><td> $4 2 . 5 \ _ { ( + 0 . 1 ) }$ </td><td> $4 8 . 2 \ ( + 0 . 7 ) $ </td></tr><tr><td>32</td><td> $5 1 . 0 \ ( + 1 . 7 )$ </td><td> $5 7 . 6 \ ( + 0 . 4 )$ </td><td>36.8 (+1.2)</td><td> $4 9 . 3 \ _ { ( + 0 . 4 ) }$ </td><td> $5 0 . 0 \ ( . 0 . 2 ) $ </td><td> $4 8 . 9 \ _ { ( + 1 . 8 ) }$ </td><td> $4 8 . 4 \ ( + 1 . 4 ) $ </td><td> $4 3 . 8 \ _ { ( + 1 . 3 ) }$ </td><td> $4 9 . 3 \ _ { ( + 1 . 1 ) }$ </td></tr><tr><td>64</td><td> $4 7 . 4 \ ( . 3 . 6 )$ </td><td> $5 4 . 6 \ ( . 3 )$ </td><td> $3 3 . 5 \ ( . 3 . 3 )$ </td><td> $4 5 . 9 \ ( . 3 . 4 )$ </td><td> $4 8 . 8 \ ( . 1 . 2 )$ </td><td> $4 4 . 8 \ ( - 4 . 1 )$ </td><td> $4 4 . 6 \ ( - 3 . 8 )$ </td><td> $3 9 . 9 \ _ { ( - 3 . 9 ) }$ </td><td> $4 6 . 2 \ ( - 3 . 1 )$ </td></tr><tr><td>8</td><td>56.3</td><td>59.8</td><td>33.0</td><td>47.3</td><td>52.9</td><td>47.1</td><td>48.3</td><td>44.8</td><td>49.1</td></tr><tr><td rowspan="4">MiniCPM-V-2.6</td><td>16</td><td> $5 5 . 5 \ ( . 0 . 8 )$ </td><td> $6 0 . 5 \ _ { ( + 0 . 7 ) }$ </td><td> $3 6 . 7 \ _ { ( + 3 . 7 ) }$ </td><td> $4 7 . 9 \ _ { ( + 0 . 6 ) }$ </td><td> $5 2 . 6 \ \left( . 0 . 3 \right)$ </td><td> $4 9 . 7 \ _ { ( + 2 . 6 ) }$ </td><td> $5 0 . 8 \ _ { ( + 2 . 5 ) }$ </td><td> $4 8 . 1 \ _ { ( + 3 . 3 ) }$ </td><td> $5 0 . 7 \ ( + 1 . 6 )$ </td></tr><tr><td>32</td><td> $5 6 . 0 \ _ { ( + 0 . 5 ) }$ </td><td> $6 0 . 6 \ ( + 0 . 1 ) $ </td><td> $3 8 . 8 \ ( + 2 . 1 ) $ </td><td> $5 0 . 2 \ ( + 2 . 3 )$ </td><td> $5 3 . 0 \ \mathrm { _ { ( + 0 . 4 ) } }$ </td><td> $5 1 . 0 \ ( + 1 . 3 )$ </td><td> $5 1 . 7 \ ( + 0 . 9 )$ </td><td> $4 9 . 0 \ _ { ( + 0 . 9 ) }$ </td><td> $5 1 . 7 \ ( + 1 . 0 )$ </td></tr><tr><td>64</td><td> $5 2 . 6 \ ( . 3 . 4 )$ </td><td> $5 8 . 2 \ ( - 2 . 4 ) $ </td><td> $3 9 . 1 \ _ { ( + 0 . 3 ) }$ </td><td> $4 8 . 6 \ ( . 1 . 6 )$ </td><td> $5 0 . 5 \ ( . 2 . 5 )$ </td><td> $5 0 . 3 \ ( . 0 . 7 ) $ </td><td> $5 0 . 0 \ ( . 1 . 7 ) $ </td><td> $4 6 . 9 \ ( . 2 . 1 )$ </td><td> $5 0 . 3 \ ( . 1 . 4 )$ </td></tr><tr><td>8</td><td>50.1</td><td>58.3</td><td>35.0</td><td>48.8</td><td>49.6</td><td>47.1</td><td></td><td></td><td></td></tr><tr><td rowspan="4">InternVL2-26B</td><td>16</td><td> $5 0 . 0 _ { \ : ( - 0 . 1 ) }$ </td><td> $5 9 . 1 \ ( + 0 . 8 )$ </td><td> $3 6 . 3 \ _ { ( + 1 . 3 ) }$ </td><td> $4 9 . 9 \ _ { ( + 1 . 1 ) }$ </td><td> $4 9 . 4 \ ( . 0 . 2 )$ </td><td> $4 8 . 4 \ ( + 1 . 3 )$ </td><td>47.0  $4 8 . 3 \ ( + 1 . 3 )$ </td><td>43.9  $4 5 . 4 \ _ { ( + 1 . 5 ) }$ </td><td>47.9  $4 8 . 7 \ _ { ( + 0 . 8 ) }$ </td></tr><tr><td>32</td><td> $5 1 . 6 \ ( + 1 . 6 )$ </td><td> $5 8 . 7 ~ ( - 0 . 4 ) $ </td><td> $3 7 . 0 _ { ( + 0 . 7 ) }$ </td><td> $4 9 . 1 \ ( - 0 . 8 )$ </td><td> $4 9 . 4 ~ ( - )$ </td><td> $4 8 . 9 \ _ { ( + 0 . 5 ) }$ </td><td> $4 8 . 4 \ ( + 0 . 1 ) $ </td><td> $4 5 . 8 \ _ { ( + 0 . 4 ) }$ </td><td> $4 9 . 0 \ _ { ( + 0 . 3 ) }$ </td></tr><tr><td>64</td><td> $4 9 . 6 ( - 2 )$ </td><td> $5 5 . 1 \ ( - 3 . 6 )$ </td><td> $3 3 . 3 \ ( . 3 . 7 )$ </td><td> $4 6 . 4 \ ( . 2 . 7 )$ </td><td> $4 7 . 5 \ ( . 1 . 9 )$ </td><td> $4 5 . 7 \ ( . 3 . 2 )$ </td><td> $4 4 . 3 \ ( - 4 . 1 )$ </td><td> $4 2 . 2 \ ( - 3 . 6 )$ </td><td> $4 6 . 3 \ ( - 2 . 7 )$ </td></tr><tr><td>8</td><td>49.3</td><td>55.1</td><td>31.8</td><td>46.8</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">LLaVA-Video-7B</td><td>16</td><td> $5 0 . 7 \ ( + 1 . 4 )$ </td><td> $5 7 . 0 \ _ { ( + 1 . 9 ) }$ </td><td> $3 6 . 3 \ _ { ( + 4 . 5 ) }$ </td><td></td><td>49.8</td><td>44.6  $4 7 . 9 \ _ { ( + 3 . 3 ) }$ </td><td>44.3</td><td>41.0</td><td>46.3</td></tr><tr><td>32</td><td> $5 0 . 4 \ ( + 0 . 3 )$ </td><td> $5 8 . 9 \ _ { ( + 1 . 9 ) }$ </td><td> $3 7 . 8 \ _ { ( + 1 . 5 ) }$ </td><td> $4 9 . 0 \ _ { ( + 2 . 2 ) }$   $5 3 . 1 \ ( + 4 . 1 )$ </td><td> $5 1 . 7 \ ( + 1 . 9 )$   $5 2 . 2 \ \mathrm { _ { ( + 0 . 5 ) } }$ </td><td> $5 0 . 3 \ ( + 2 . 4 )$ </td><td> $4 7 . 0 \ ( + 2 . 7 )$ </td><td> $4 3 . 0 \ \mathrm { ( + 2 . 0 ) }$   $4 5 . 8 \ _ { ( + 2 . 8 ) }$ </td><td> $4 9 . 2 \ ( + 2 . 9 )$ </td></tr><tr><td>64</td><td> $5 1 . 0 \ ( + 0 . 6 )$ </td><td> $5 8 . 7 ~ ( - 0 . 2 ) $ </td><td> $3 9 . 0 \ _ { ( + 1 . 2 ) }$ </td><td> $5 2 . 4 \ ( . 0 . 7 ) $ </td><td> $5 1 . 3 ~ ( . 0 . 9 )$ </td><td> $5 1 . 4 \ ( + 1 . 1 ) $ </td><td> $5 0 . 0 \ ( + 3 . 0 )$ </td><td></td><td> $5 1 . 0 \ \mathrm { { ( + 1 . 8 ) } }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $5 0 . 1 \ ( + 0 . 1 )$ </td><td> $4 6 . 9 \ _ { ( + 1 . 1 ) }$ </td><td> $5 1 . 4 \ ( + 0 . 4 )$ </td></tr><tr><td rowspan="4">Qwen2-VL-7B</td><td>8</td><td>44.2</td><td>55.5</td><td>27.8</td><td>41.7</td><td>49.6</td><td>39.0</td><td>37.1</td><td>33.2</td><td>42.6</td></tr><tr><td>16</td><td> $4 7 . 7 \ ( + 3 . 5 )$ </td><td> $5 5 . 9 \ _ { ( + 0 . 4 ) }$ </td><td> $3 3 . 1 \ ( + 5 . 3 )$ </td><td> $4 3 . 6 \ _ { ( + 1 . 9 ) }$ </td><td> $5 1 . 5 \ _ { ( + 1 . 9 ) }$ </td><td> $4 3 . 0 \ _ { ( + 4 . 0 ) }$ </td><td> $4 2 . 0 \ _ { ( + 4 . 9 ) }$ </td><td> $3 6 . 7 \ _ { ( + 3 . 5 ) }$ </td><td> $4 5 . 9 \ _ { ( + 3 . 3 ) }$ </td></tr><tr><td>32 64</td><td> $4 8 . 8 \ \substack { ( + 1 . 1 ) }$ </td><td> $5 7 . 0 \ \mathrm { { ( + 1 . 1 ) } }$ </td><td> $4 0 . 0 \ _ { ( + 6 . 9 ) }$ </td><td> $4 7 . 1 \ ( + 3 . 5 )$ </td><td> $5 2 . 6 \ \AA _ { ( + 1 . 1 ) }$ </td><td> $4 8 . 4 \ ( + 5 . 4 )$ </td><td> $4 6 . 5 \ _ { ( + 4 . 5 ) }$ </td><td> $4 3 . 0 \ _ { ( + 6 . 3 ) }$ </td><td> $4 9 . 8 \ _ { ( + 3 . 9 ) }$ </td></tr><tr><td></td><td> $5 0 . 1 \ ( + 1 . 3 )$ </td><td> $5 3 . 1 \ ( . 3 . 9 )$ </td><td>40.0 (-)</td><td> $4 9 . 4 \ ( + 2 . 3 )$ </td><td> $5 3 . 2 \ \mathrm { _ { ( + 0 . 6 ) } }$ </td><td> $4 8 . 7 \ _ { ( + 0 . 3 ) }$ </td><td> $4 7 . 1 \ \mathrm { { \Omega } } _ { ( + 0 . 6 ) }$ </td><td> $4 3 . 4 \ ( + 0 . 4 )$ </td><td> $5 0 . 2 \ ( + 0 . 4 ) $ </td></tr></table>

Table 11: Detailed performance comparison with different number of input frames. Consistent visual results in Figure 11.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Frames</td><td colspan="5">#Events</td><td colspan="5">#Elements</td><td rowspan="2">Overall</td></tr><tr><td> $\leq 2$ </td><td>3~4</td><td>5~6</td><td>7~8</td><td> $\geq 9$ </td><td> $\leq 1 5$ </td><td>16~20</td><td>21~25</td><td> $2 6 { \sim } 3 0 $ </td><td> $\geq 3 1$ </td></tr><tr><td rowspan="4">LLaVA-Video-7B</td><td>8</td><td>49.4</td><td>46.3</td><td>39.6</td><td>35.2</td><td>24.9</td><td>51.7</td><td>45.3</td><td>40.9</td><td>38.6</td><td>35.0</td><td>46.3</td></tr><tr><td>16</td><td> $5 2 . 9 \ _ { ( + 3 . 5 ) }$ </td><td> $4 8 . 8 \ ( + 2 . 5 )$ </td><td> $4 1 . 2 \ ( + 1 . 6 )$ </td><td> $3 6 . 7 \ _ { ( + 1 . 5 ) }$ </td><td> $3 5 . 7 \ _ { ( + 1 0 . 8 ) }$ </td><td> $5 4 . 6 ~ ( + 2 . 9 )$ </td><td> $4 7 . 7 \ ( + 2 . 4 )$ </td><td> $4 4 . 0 \ _ { ( + 3 . 1 ) }$ </td><td> $4 2 . 4 \ ( + 3 . 8 )$ </td><td> $3 8 . 2 \ ( + 3 . 2 ) $ </td><td> $4 9 . 2 \ ( + 2 . 9 )$ </td></tr><tr><td>32</td><td> $5 3 . 5 \ \mathrm { _ { ( + 0 . 6 ) } }$ </td><td> $5 0 . 7 \ ( + 1 . 9 )$ </td><td> $4 5 . 1 \ ( + 3 . 9 )$ </td><td> $4 4 . 2 \ ( + 7 . 5 )$ </td><td> $3 9 . 6 \ _ { ( + 3 . 9 ) }$ </td><td> $5 5 . 1 \ ( + 0 . 5 )$ </td><td> $5 0 . 0 \ ( + 2 . 3 )$ </td><td> $4 7 . 4 \ ( + 3 . 4 )$ </td><td> $4 4 . 1 \ ( + 1 . 7 )$ </td><td> $4 2 . 9 \ _ { ( + 4 . 7 ) }$ </td><td> $5 1 . 0 \ \mathrm { { ( + 1 . 8 ) } }$ </td></tr><tr><td>64</td><td> $5 3 . 7 \ ( + 0 . 2 )$ </td><td> $5 1 . 1 \ ( + 0 . 4 )$ </td><td> $4 6 . 6 \ _ { ( + 1 . 5 ) }$ </td><td>44.2 (-)</td><td> $3 9 . 4 \ ( . 0 . 2 )$ </td><td> $5 5 . 2 \ ( + 0 . 1 ) $ </td><td> $5 0 . 9 \ ( + 0 . 9 ) $ </td><td> $4 7 . 9 \ _ { ( + 0 . 5 ) }$ </td><td> $4 4 . 7 \ ( + 0 . 6 )$ </td><td> $4 1 . 4 \ ( . 1 . 5 )$ </td><td> $5 1 . 4 \ ( + 0 . 4 )$ </td></tr><tr><td rowspan="4">Qwen2-VL-7B</td><td>8</td><td>46.8</td><td>43.2</td><td>29.7</td><td>25.5</td><td>16.0</td><td>48.6</td><td>42.1</td><td>36.9</td><td>30.6</td><td>29.1</td><td>42.6</td></tr><tr><td>16</td><td> $5 0 . 4 \ ( + 3 . 6 )$ </td><td> $4 6 . 1 \ _ { ( + 2 . 9 ) }$ </td><td> $3 2 . 9 \ _ { ( + 3 . 2 ) }$ </td><td> $3 2 . 7 \ ( + 7 . 2 )$ </td><td> $1 4 . 6 ~ ( - 1 . 4 )$ </td><td> $5 2 . 3 \ _ { ( + 3 . 7 ) }$ </td><td> $4 4 . 9 \ _ { ( + 2 . 8 ) }$ </td><td> $3 9 . 1 \ ( + 2 . 2 )$ </td><td> $3 6 . 3 \ _ { ( + 5 . 7 ) }$ </td><td> $3 2 . 6 \ _ { ( + 3 . 5 ) }$ </td><td> $4 5 . 9 \ _ { ( + 3 . 3 ) }$ </td></tr><tr><td>32</td><td> $5 3 . 5 \ \mathrm { ( + 3 . 1 ) }$ </td><td> $4 9 . 8 \ ( + 3 . 7 )$ </td><td> $4 0 . 0 \ ( + 7 . 1 )$ </td><td> $3 7 . 7 \ ( + 5 )$ </td><td> $3 0 . 6 \ \AA _ { ( + 1 6 ) }$ </td><td> $5 5 . 0 \ ( + 2 . 7 )$ </td><td> $4 8 . 5 \ _ { ( + 3 . 6 ) }$ </td><td> $4 5 . 3 \ ( + 6 . 2 )$ </td><td> $4 2 . 3 \ ( + 6 )$ </td><td> $3 8 . 3 \ ( + 5 . 7 ) $ </td><td> $4 9 . 8 \ _ { ( + 3 . 9 ) }$ </td></tr><tr><td>64</td><td> $5 2 . 7 \ ( . 0 . 8 )$ </td><td> $5 0 . 5 \ ( + 0 . 7 ) $ </td><td> $4 2 . 7 \ ( + 2 . 7 )$ </td><td> $4 5 . 1 \ ( + 7 . 4 )$ </td><td> $2 7 . 6 \ ( . 3 )$ </td><td> $5 5 . 1 \ ( + 0 . 1 )$ </td><td> $4 9 . 8 \ ( + 1 . 3 )$ </td><td> $4 5 . 2 \ ( . 0 . 1 )$ </td><td> $4 2 . 9 \ _ { ( + 0 . 6 ) }$ </td><td> $3 6 . 8 \ ( . 1 . 5 )$ </td><td> $5 0 . 2 \ ( + 0 . 4 )$ </td></tr></table>

Table 12: Performance comparison across different video complexities with varying input frame numbers. Consistent visualization results in Figure 5.

<table><tr><td rowspan="2">Model</td><td colspan="2">Camera State</td><td colspan="2">Background Scene</td><td colspan="3">Subject Action</td><td colspan="3">Object Attribute</td><td rowspan="2">Overall</td></tr><tr><td>Motion</td><td>Transition</td><td>Description</td><td>Transition</td><td>Recognition</td><td>Sequence</td><td>Matching</td><td>Recognition</td><td>Appearance</td><td>Location</td></tr><tr><td colspan="10">Open-Source LMMs</td><td></td></tr><tr><td>PLLaVA-7B</td><td>29.7</td><td>31.9</td><td>48.1</td><td>22.4</td><td>43.6</td><td>34.6</td><td>30.4</td><td>32.3</td><td>38.1</td><td>45.2</td><td>33.7</td></tr><tr><td>LongVA-7B</td><td>37.5</td><td>41.5</td><td>63.0</td><td>30.8</td><td>44.6</td><td>44.7</td><td>43.5</td><td>41.7</td><td>47.6</td><td>40.5</td><td>42.4</td></tr><tr><td>Tarsier-7B</td><td>23.0</td><td>24.6</td><td>40.7</td><td>20.6</td><td>38.6</td><td>26.9</td><td>45.7</td><td>20.9</td><td>25.9</td><td>23.8</td><td>26.5</td></tr><tr><td>Kangaroo</td><td>33.2</td><td>47.3</td><td>53.7</td><td>38.3</td><td>49.5</td><td>38.8</td><td>54.3</td><td>47.2</td><td>43.5</td><td>59.5</td><td>42.9</td></tr><tr><td>LLaVA-OV-7B</td><td>42.2</td><td>54.6</td><td>57.4</td><td>48.6</td><td>42.6</td><td>41.4</td><td>60.9</td><td>47.9</td><td>50.0</td><td>59.5</td><td>47.4</td></tr><tr><td>LLaVA-Video-7B</td><td>39.1</td><td>50.7</td><td>59.3</td><td>46.7</td><td>52.5</td><td>52.4</td><td>56.5</td><td>53.6</td><td>61.9</td><td>47.6</td><td>50.6</td></tr><tr><td>Qwen2-VL-7B</td><td>41.0</td><td>51.7</td><td>66.7</td><td>45.8</td><td>54.5</td><td>52.8</td><td>65.2</td><td>49.0</td><td>60.2</td><td>57.1</td><td>51.3</td></tr><tr><td>InternVL2-8B</td><td>41.0</td><td>53.1</td><td>66.7</td><td>40.2</td><td>45.5</td><td>50.5</td><td>50.0</td><td>45.8</td><td>56.8</td><td>45.2</td><td>48.4</td></tr><tr><td>MiniCPM-V-2.6</td><td>39.8</td><td>45.9</td><td>59.3</td><td>34.6</td><td>49.5</td><td>51.1</td><td>52.2</td><td>42.2</td><td>46.6</td><td>50.0</td><td>45.7</td></tr><tr><td>PLLaVA-13B</td><td>31.2</td><td>31.9</td><td>46.3</td><td>23.4</td><td>48.5</td><td>41.1</td><td>45.7</td><td>37.0</td><td>41.5</td><td>45.2</td><td>37.2</td></tr><tr><td>InternVL2-26B</td><td>38.7</td><td>45.4</td><td>63.0</td><td>42.1</td><td>48.5</td><td>46.0</td><td>58.7</td><td>42.7</td><td>55.1</td><td>50.0</td><td>45.9</td></tr><tr><td>PLLaVA-34B</td><td>42.6</td><td>41.5</td><td>63.0</td><td>43.9</td><td>45.5</td><td>48.5</td><td>56.5</td><td>43.2</td><td>56.8</td><td>57.1</td><td>46.9</td></tr><tr><td>Tarsier-34B</td><td>43.0</td><td>48.3</td><td>72.2</td><td>45.8</td><td>51.5</td><td>50.2</td><td>56.5</td><td>49.7</td><td>53.7</td><td>61.9</td><td>50.1</td></tr><tr><td>InternVL2-40B</td><td>40.2</td><td>58.0</td><td>74.1</td><td>51.4</td><td>56.4</td><td>53.4</td><td>63.0</td><td>57.3</td><td>66.9</td><td>61.9</td><td>54.7</td></tr><tr><td>LLaVA-OV-72B</td><td>46.5</td><td>67.6</td><td>75.9</td><td>57.0</td><td>59.4</td><td>56.6</td><td>73.9</td><td>63.5</td><td>69.5</td><td>59.5</td><td>60.0</td></tr><tr><td>LLaVA-Video-72B</td><td>47.7</td><td>67.6</td><td>77.8</td><td>61.7</td><td>61.4</td><td>57.0</td><td>65.2</td><td>62.5</td><td>73.7</td><td>57.1</td><td>60.7</td></tr><tr><td>Qwen2-VL-72B</td><td>52.7</td><td>64.7</td><td>74.1</td><td>55.1</td><td>62.4</td><td>54.4</td><td>67.4</td><td>63.0</td><td>76.3</td><td>66.7</td><td>60.7</td></tr><tr><td>InternVL2-76B</td><td>43.8</td><td>61.8</td><td>74.1</td><td>43.0</td><td>50.5</td><td>50.5</td><td>54.3</td><td>52.1</td><td>66.1</td><td>57.1</td><td>53.1</td></tr><tr><td colspan="10">Closed-Source LMMs</td><td></td></tr><tr><td>Gemini 1.5 Flash Gemini 1.5 Pro</td><td>40.8</td><td>58.3</td><td>70.4</td><td>52.3</td><td>48.0</td><td>54.2</td><td>63.0</td><td>49.0</td><td>66.7</td><td>64.3</td><td>53.3</td></tr><tr><td></td><td>49.4</td><td>68.4</td><td>64.8</td><td>59.8</td><td>55.0</td><td>60.4</td><td>69.6</td><td>64.6</td><td>65.0</td><td>66.7</td><td>60.8</td></tr><tr><td>GPT-40</td><td>53.9</td><td>56.0</td><td>81.5</td><td>56.1</td><td>59.4</td><td>67.6</td><td>58.7</td><td>56.8</td><td>63.6</td><td>59.5</td><td>60.3</td></tr></table>

Table 13: TUNA-MCQ performance of all tested video LMMs. We provide detailed scores on 10 temporal-dynamic tasks. The best and second-best results are marked with orange and blue , respectively.

![](images/76541bddce88e96bfa78643443e25796f22d0e13c72730e0a458c5a4fd76e725.jpg)

## Video Caption

The video begins with the camera focused on a wooden decorative piece, behind which a man is watching through it.

Then, the camera cuts to an outdoor scene with blurred edges and a clear center. A white news van with the logo “KXBD 6 News at 6” is visible by the roadside. Next to the van is a set-up camera, and a military green vehicle passes in front of the lens. In the background, greenery and a pedestrian path are visible. A woman with a bag on her right shoulder and a bag in her left hand walks along the sidewalk. The camera moves to the right, where a person is standing by the front passenger door of the news van, making a phone call.

Next, the camera cuts back indoors, where a man in a black suit suddenly turns to look inside. Behind him is an ornately decorated wall. The man in the suit turns again to look outside, then steps back while closing the door in front of him. He then turns and walks further into the room.

Events & Visual Elements
<table><tr><td rowspan=1 colspan=1>Event</td><td rowspan=1 colspan=3>The video begins with the camera focused on a wooden decorative piece, behind which a man is watching through it.</td></tr><tr><td rowspan=7 colspan=1>VisualElementsEvent</td><td rowspan=1 colspan=1>The video begins with the camera focused on a wooden decorative piece.</td><td rowspan=1 colspan=1>camera</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>Behind the wooden decoration, there is a man.</td><td rowspan=1 colspan=1>attribute</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>The man looks outside through the wooden decoration.</td><td rowspan=1 colspan=1>action</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>Then, the camera cuts to an outdoor scene with blurred edges and a clear center. A white news van with the logo “KXBD 6</td><td rowspan=1 colspan=2>News at 6&quot; is</td></tr><tr><td rowspan=3 colspan=1>visible by the roadside. Next to the van is a set-up camera, and a military green vehicle passes in front of the lens. Igreenery and a pedestrian path are visible. A woman with a bag on her right shoulder and a bag in her left hand walks aThe camera moves to the right, where a person is standing by the front passenger door of the news van, making a phone call.</td><td rowspan=1 colspan=2>n the background,</td></tr><tr><td rowspan=1 colspan=1>long the sidewalk.</td><td rowspan=1 colspan=1>alk.</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=17 colspan=1>VisualElements1Event Visual</td><td rowspan=1 colspan=1>Then, the camera cuts to an outdoor scene.</td><td rowspan=1 colspan=1>camera</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>The edges of the frame are blurred, with a clear center.</td><td rowspan=1 colspan=1>attribute</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>A white news van is visible by the roadside.</td><td rowspan=1 colspan=1>attribute</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>The van has the logo “KXBD 6 News at 6” on its side.</td><td rowspan=1 colspan=1>attribute</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>A set-up camera stands beside the van.</td><td rowspan=1 colspan=1>scene</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>A military green vehicle passes in front of the lens.</td><td rowspan=1 colspan=1>action</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>Greenery and a pedestrian walkway are visible in the background.</td><td rowspan=1 colspan=1>scene</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>A woman with a bag on her right shoulder and another in her left hand walks along the sidewalk.</td><td rowspan=1 colspan=1>action</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>The camera moves to the right.</td><td rowspan=1 colspan=1>camera</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>A person is standing outside the front passenger door of the news van.</td><td rowspan=1 colspan=1>attribute</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=2 colspan=1>The person is making a phone call.Next, the camera cuts back indoors, where a man in a black suit suddenly turns to look inside. Behind him is an ornatel</td><td rowspan=1 colspan=1>action</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>y decorated wall.</td><td rowspan=1 colspan=1>vall.</td></tr><tr><td rowspan=1 colspan=1>The man in the suit turns again to look outside, then steps back while closing the door in front of him. He then turns and w</td><td rowspan=1 colspan=1>alks further into</td><td rowspan=1 colspan=1>into</td></tr><tr><td rowspan=1 colspan=1>the room.</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Next, the camera cuts back to the interior scene.</td><td rowspan=1 colspan=1>camera</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>A man in a black suit suddenly turns around, looking inside.</td><td rowspan=1 colspan=1>attribute</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>Behind the man in the suit is an ornately decorated wall.</td><td rowspan=1 colspan=1>attribute</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=3 colspan=1> Elements</td><td rowspan=1 colspan=1>The man in the suit turns again to look outside.</td><td rowspan=1 colspan=1>action</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>The man then steps back while closing the door in front of him.</td><td rowspan=1 colspan=1>action</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>The man in the suit turns and walks further into the room.</td><td rowspan=1 colspan=1>action</td><td rowspan=1 colspan=1>3</td></tr></table>

Figure 15: A detailed example in TUNA-1K.

![](images/3a8bdbba5d99c248e533a217d1e8e46e8991a5e123bc797bb3a5178426660a33.jpg)  
Figure 16: Several examples in TUNA-MCQ, involving Camera Motion, Camera Transition, Scene Description and Scene Transition tasks

![](images/59e6d6ef3e5b191ce101624fbdb81328fd8f7e3e31141fea18f6f0a5c0894337.jpg)

What happens after the man in the gray sweater waves his arms widely? A. The man in the white shirt turns and enters the kitchen.   
B. The man in the white shirt walks out of the kitchen.   
C. The man in the gray sweater lowers his hands and glances back.   
D. The man in the gray sweater raises both hands and waves them. What is the order of the woman's actions involving the paper? A. (1) places bottle caps. (2) holds the paper up. (3) draws shapes. (4) picks up a pen.   
B. (1) picks up a pen. (2) draws shapes. (3) holds the paper up. (4) places bottle caps.   
C. (1) draws shapes. (2) picks up a pen. (3) places bottle caps. (4) holds the paper up.   
D. (1) holds the paper up. (2) picks up a pen. (3) draws shapes. (4) places bottle caps.

![](images/456ac97430718a41e0eb5cb721ea56c935948d96ab96d7274071312329945b81.jpg)

![](images/f77dc59e9caa5e28ec442550f6369a707bfaa4e1d7d7ebc0d2443f8712fc74c4.jpg)

What is the order of the woman's actions involving the paper? A. Player in white jersey No. 7 B. Player in blue jersey No. 14 C. Player in blue jersey No. 7 D. Player in white jersey No. 11

Figure 17: Several examples in TUNA-MCQ, involving Action Recognition, Action Sequence, and Action-Subject Matching tasks.

![](images/5f5c64f6af1930f1e5cd822056a374a1e715549300039879831a4f36492a0c5b.jpg)

![](images/41e4116ce80aca4cda3dc2b89b6b46643d98463c778d8df5246ba74d814ff3a2.jpg)

Task Type: Object Recognition   
What is the sequence of movements of the vehicles in the video?   
A. (1) Black car. (2) White truck. (3) Blue truck. B. (1) White truck. (2) Black car. (3) Blue truck. C. (1) Blue truck. (2) White truck. (3) Black car. D. (1) Blue truck. (2) Black car. (3) White truck.

![](images/4f72f95c2c5a81ff1e6da4bf6df472723f312b342308a3c8e2b69d5be2517290.jpg)  
Figure 18: Several examples in TUNA-MCQ, involving Object Recognition, Object Appearance, and Object Location tasks.

![](images/cd9fbd25fb82426e4ca4878fe78f9868e8afd2c47efe03b7c55c99c5f9e7eefc.jpg)  
Figure 19: The default prompt used for the TUNA-CAP experiments in Section 4.2.

![](images/248304bf34fd2372b0c121d8f3403e9dda187313f9960d5a9013d2cc067f9fca.jpg)  
Figure 20: The prompt used to split events for the TUNA-CAP experiments in Section 3.2.2.

![](images/411ad4efcf7fb6acfcb3ca53ad415325a37ee9626b8231ae7e30a85c025c627c.jpg)  
Figure 21: The prompt used to match events for the TUNA-CAP experiments in Section 3.2.2.

![](images/bb6ae07821ca025e65d2fb734f75ddb0ad248a8866432de69b71d9ae6a492e8e.jpg)  
Figure 22: The prompt used to classify relationships for the TUNA-CAP experiments in Section 3.2.2.

![](images/210fe8f75d1b85be8df55e08d7bbe8e7873a11d109c4b99206841b49a4c5424d.jpg)  
Figure 23: The prompt used to classify videos for the TUNA-CAP construction in Section 3.2.2.

![](images/cbcbe5eb077c5469f3b2b6bdb8aa11a87937fdf32517984a8dc9914a5a80dff9.jpg)  
Figure 24: The default prompt used for the TUNA-MCQ experiments in Section 4.3.

![](images/6b91a95cfd3a9bf4e50b36ea7d9d8a9fabe8c1ffd34e23d3252640778a256259.jpg)  
Figure 25: The prompt used to generate error-prone points for the TUNA-MCQ construction in Section 3.2.3.

```jsonl
Default Prompt for Multi-Choice Q&As Generation
You are an AI visual assistant specialized in analyzing videos.
Given a human-annotated video description, and several error-prone points, your task is to propose five challenging
multi-choice QAs about complex temporal-dynamic fine-grained understanding and reasoning. These questions
should be require integrating information across multiple events and frames.
# Requirements
- Questions can include temporal understanding and reasoning, counterfactual reasoning, causal reasoning, etc.
- You only need to focus on key events and objects. Ensure that the question can be answered from the given video
description. Ensure that a single event or frame cue cannot answer these questions.
- The four options should be confusing and of similar length. Ensure that it is unable to judge the correct option just
by the textual question and four textual options. For example, if there are multiple elements in an answer, then each
correct element should not be the one that appears the most times among the four options.
- If the options are an disordered list of some elements (at least 3), normalise the elements to "(a) element_i\n(b)
element_j\n..." as a part of the question, and answer should be formatted as something like "(b)(c)(a)(d)".
## Task Type
1. camera motion; 2. camera transition; 3. scene description; 4. scene transition; 5. action recognition; 6. action
sequence; 7. action-subject matching; 8. object recognition; 9. object appearance; 10. object location.
Output a JSON formed as:
[
{"question": "", "task_type": "", "answer": "", "options": {"A": "", "B": "", "C": "", "D": ""}, "correct_option", ""}
]
## Reference Examples (for reference only, not limited)
{"question": "What is the order of camera state changes throughout the video?", "task_type": "camera motion",
"answer": "rotate left.", "options": {"A": "stationary.", "B": "zoom out.", "C": "rotate left.", "D": "pan left."},
"correct_option", "C"}
{"question": "How many times does the camera switch in the video? How many of these camera shots are close-ups
of the woman?", "task_type": "camera transition", "answer": "4, 2.", "options": {"A": "3, 2.", "B": "4, 2.", "C": "4, 0.",
"D": "4, 1."}, "correct_option", "B"}
{"question": "Reasoning which team will score based on the video?","task_type": "action recognition", "answer":
"The team in red uniforms.", "options": {"A": "Not Sure.", "B": "The team in yellow uniforms.", "C": "The team in
blue uniforms.", "D": "The team in red uniforms."}, "correct_option", "D"}
{"question": "What is the order in which this person picks up the objects?\n(a) a book\n(b) a pen\n(c) an apple",
"task_type": "action sequence", "answer": "11", "options": {"A": "(c) (b) (a)", "B": "(a) (b) (c)", "C": "(b) (c) (a)", "D":
"(b) (a) (c)"}, "correct_option", "D"}
{"question": "Which number player scored the goal?", "task_type": "action-subject matching", "answer": "11",
"options": {"A": "11", "B": "17", "C": "7", "D": "1"}, "correct_option", "A"}
{"question": "What is the temporal order of occurrence of the following objects?\n(a) an apple\n(b) a guava\n(c) a
banana\n(d) a loaf of bread", "task_type": "object recognition", "answer": "(c) (d) (b) (a)", "options": {"A": "(c) (b) (d)
(a)", "B": "(d) (c) (a) (b)", "C": "(c) (d) (b) (a)", "D": "(d) (b) (c) (a)"}, "correct_option", "C"}
# Input
## Video Description
{video_caption}
## Error-prone Points
{error_prone_points}
# Output
DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the List. Output:
```  
Figure 26: The prompt used to generate multi-choice QAs for the TUNA-MCQ construction in Section 3.2.3.