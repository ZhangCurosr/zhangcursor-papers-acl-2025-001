# ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use

Junjie Ye<sup>1,2</sup>, Zhengyin Du<sup>2</sup>, Xuesong Yao<sup>2</sup>, Weijian Lin<sup>2</sup>,

Yufei Xu<sup>2</sup>, Zehui Chen<sup>2</sup>, Zaiyuan Wang<sup>2</sup>, Sining Zhu<sup>2</sup>, Zhiheng Xi<sup>1</sup>, Siyu Yuan<sup>1</sup>, Tao Gui<sup>1,3</sup>\*, Qi Zhang<sup>1,3</sup>∗, Xuanjing Huang<sup>1,3</sup>∗, Jiecao Chen<sup>2</sup> <sup>1</sup> Fudan University <sup>2</sup> ByteDance

<sup>3</sup> Shanghai Collaborative Innovation Center of Intelligent Visual Computing jjye23@m.fudan.edu.cn, {qz, tgui}@fudan.edu.cn

## Abstract

Effective evaluation of multi-hop tool use is critical for analyzing the understanding, reasoning, and function-calling capabilities of large language models (LLMs). However, progress has been hindered by a lack of reliable evaluation datasets. To address this, we present ToolHop, a dataset comprising 995 user queries and 3,912 associated tools, specifically designed for rigorous evaluation of multi-hop tool use. ToolHop ensures diverse queries, meaningful interdependencies, locally executable tools, detailed feedback, and verifiable answers through a novel querydriven data construction approach that includes tool creation, document refinement, and code generation. We evaluate 14 LLMs across five model families (i.e., LLaMA3.1, Qwen2.5, Gemini1.5, Claude3.5, and GPT), uncovering significant challenges in handling multi-hop tool-use scenarios. The leading model, GPT-4o, achieves an accuracy of 49.04%, underscoring substantial room for improvement. Further analysis reveals variations in tooluse strategies for various families, offering actionable insights to guide the development of more effective approaches. Code and data can be found in https://huggingface.co/ datasets/bytedance-research/ToolHop.

## 1 Introduction

The task of multi-hop tool use presents a significant challenge for large language models (LLMs) (OpenAI, 2023; Touvron et al., 2023; Bai et al., 2023). As illustrated in Figure 1, it requires LLMs to incrementally decompose a complex multi-hop query into atomic subqueries, invoke the appropriate tools, and iteratively retrieve results from the tool feedback until the final answer is reached. This process demands advanced capabilities such as comprehension, reasoning, and function-calling (Qin et al., 2023; Qu et al.,

![](images/0348a51f43bfd0481193683e479eec32d7e1814b1d73b136fb7f4c6025068491.jpg)  
Figure 1: An illustration of multi-hop tool use. The process entails decomposing a complex multi-hop query into multiple atomic sub-queries, sequentially invoking the appropriate tools, retrieving results from the tool feedback, and iterating until the final answer is derived. This demonstrates the integration of comprehension, reasoning, and function-calling capabilities.

2024), making the evaluation of multi-hop tool use essential for assessing these skills. Furthermore, such evaluations are pivotal for advancing LLMs toward generalized intelligence (Xi et al., 2023).

Existing studies have made progress in evaluating tool use of LLMs. Some focus on evaluating single-step tool use in simulation environments, requiring manual calibration of correct tool-call results (Chen et al., 2024; Ye et al., 2024a,b). Others examine the process of tool use, leveraging advanced models like GPT-4 to go beyond singlestep evaluations and providing some valuable insights (Qin et al., 2024; Huang et al., 2024b; Ye et al., 2025).

However, these works still fall short of offering a reliable evaluation of multi-hop tool use. Specifically, a key limitation of prior work lies in their reliance on tool-driven data construction methods, where a collection of tools is gathered and queries are simulated for them (Tang et al., 2023; Wu et al., 2024; Liu et al., 2024). This approach fails to ensure that the collected tools are interdependent or that the queries involve genuine multi-hop reasoning. Furthermore, the absence of verifiable answers forces these studies to depend on process analysis using models, introducing model bias and evaluation errors (Guo et al., 2023; Eloundou et al., 2024).

![](images/3a4974b1edcf09bc10e59ceb77262863c34caf7e1fcbf021505c8103c6b129d9.jpg)  
Figure 2: An illustration of our proposed query-driven data construction scheme, comprising three key processes: tool creation, document refinement, and code generation. This approach incrementally develops detailed tool document and code implementation for each atomic subquery within a multi-hop query.

To address these challenges, we introduce Tool-Hop, a novel dataset specifically designed to evaluate LLMs’ multi-hop tool use capabilities. ToolHop comprises 995 multi-hop queries and 3,912 locally executable tools, constructed using a query-driven data construction scheme. This methodology involves tool creation, document refinement, and code generation, which can expand a single multi-hop query into a comprehensive multihop tool use test case. An analysis of ToolHop demonstrates its effectiveness in accommodating diverse queries, ensuring meaningful interdependencies, supporting locally executable tools, and delivering detailed feedback alongside verifiable answers. This design rigorously evaluates LLMs multi-hop tool use capabilities.

We evaluate ToolHop on 14 LLMs from five different families (i.e., LLaMA3.1 (Team, 2024a), Qwen2.5 (Team, 2024b), Gemini1.5 (Reid et al., 2024), Claude3.5 (Bai et al., 2022), and GPT (OpenAI, 2023)). Our results reveal that while tools significantly improve model performance, even the top-performing model, GPT-4, achieves only 49.04% accuracy in multi-hop tool use, highlighting considerable room for improvement. Further studies reveal that different model families exhibit distinct patterns in tool use, leading to varied outcomes. For instance, the Qwen2.5 (Team, 2024b) family of models tends to emphasize parallel calls, which results in hallucinations, while the GPT family leverages tool feedback to improve their performance in tool usage. These insights provide valuable guidance for developing more effective methods.

## Our contributions are as follows:

• We introduce ToolHop, a test set of 995 multihop queries with 3,912 locally executable tools, designed to assess LLMs’ ability to use tools in multi-hop scenarios. It ensures diverse queries, meaningful interdependencies, locally executable tools, detailed feedback, and verifiable answers.

• We propose a novel query-driven data construction process that can expand queries into multi-hop tool use data via tool creation, document refinement, and code generation.

• We provide a comprehensive evaluation of 14 LLMs, identifying significant limitations in current tool-use capabilities and offering insights for future improvements.

## 2 ToolHop

In this section, we introduce ToolHop in detail. Specifically, we first provide a formal definition of multi-hop tool use (Section 2.1), then explain our proposed query-driven data construction scheme (Section 2.2), and finally analyze the quality of the ToolHop dataset (Section 2.3).

## 2.1 Task Formulation

Given a multi-hop query $q ,$ which can be decomposed into subqueries $q _ { 1 } , q _ { 2 } , . . . , q _ { l } ;$ and a collection of tools $\mathbb { T } ~ = ~ ( t _ { 1 } , t _ { 2 } , \ldots , t _ { l } )$ , where each tool $t _ { i }$ is defined by a document doc and a code implementation fun<sub>i</sub>, the description document doc includes the tool name $n _ { i } .$ , a function description $d _ { i }$ , and the corresponding parameters $p _ { i } = ( p _ { i } ^ { 1 } , p _ { i } ^ { 2 } , . . . , p _ { i } ^ { k } )$ .Each parameter $p _ { i } ^ { j }$ is characterized by its name $n p _ { i } ^ { j }$ , a description $d p _ { i } ^ { j }$ , its type $t p _ { i } ^ { j }$ , and whether it is required $r p _ { i } ^ { j }$ The goal of multi-hop tool use is for the model to utilize the information in $\mathbb { T }$ to sequentially invoke the appropriate tools $t _ { 1 } , t _ { 2 } , . . . , t _ { l }$ , where each tool $t _ { i }$ is used to solve subquery $q _ { i }$ and produce an intermediate answer $a _ { i }$ . The output $a _ { i }$ then serves as one of the inputs to the next tool $t _ { i + 1 }$ , enforcing the need for the model to correctly understand complex dependencies between tools and accurately decompose the original query. The final answer a is obtained after executing the full sequence of tool calls. A visual illustration of this process is provided in Figure 1.

## 2.2 Query-Driven Data Construction

As illustrated in Figure 2, we propose a novel query-driven data construction scheme that departs from traditional tool-driven approaches. This scheme comprises three key stages that involves tool creation, document refinement, and code generation. Given a multi-hop user query $q ,$ the scheme extends $q$ to produce a sequence of corresponding tool documents $\mathrm { d o c } _ { i . . l }$ and their associated code implementations fun<sub>i..l</sub>.

Tool Creation The query-driven data construction begins with the multi-hop user query $q ,$ which serves as the foundation for building dynamic tools. The tool creation process accepts $q$ and generates a preliminary set of tool documents $\mathrm { d o c } _ { 1 \ldots l } ^ { \prime }$ . These documents are designed to be both relevant to q and interdependent.

![](images/875cc1d5d51d638db65f106e2e820650768017c0fa862ce716d44a8952fc9945.jpg)  
Figure 3: Distribution of user queries across 47 domains in the ToolHop dataset.

<table><tr><td># Tools</td><td>Three</td><td>Four</td><td>Five</td><td>Six</td><td>Seven</td></tr><tr><td># Data</td><td>428</td><td>353</td><td>136</td><td>10</td><td>68</td></tr></table>

Table 1: Distribution of the number of tools required to solve each query in ToolHop.

To achieve this, q is decomposed into a sequence of atomic subqueries $q _ { 1 } , q _ { 2 } , \ldots , q _ { l }$ , where each subquery $q _ { i }$ depends on resolving the preceding ones $( \mathrm { i . e . , ~ } q _ { i - 1 } )$ . For each $q _ { i } ^ { . }$ , a preliminary document $\operatorname* { d o c } _ { i } ^ { \prime }$ is created. . These documents not only capture the input-output logic of $q _ { i }$ , but are also structured to generalize to similar queries. By maintaining backward and forward dependencies between documents, this approach ensures both modularity and cohesion, simplifying the tool creation process.

Document Refinement The initial tool documents $\operatorname* { d o c } _ { i } ^ { \prime } ,$ derived directly from atomic queries, are typically rudimentary due to the limited information in $q _ { i }$ . The document refinement process transforms $\operatorname* { d o c } _ { i } ^ { \prime }$ into a more comprehensive document doc , designed to better support the evaluation of models in complex multi-hop scenarios.

This process involves two key aspects. On the one hand, the tool’s functionality is expanded by introducing features such as result filtering and customizable formats, all while maintaining compatibility with the original functionality. On the other hand, the number of parameters is increased, and their types are optimized. For instance, parameters initially represented as simple strings are replaced with structured types such as arrays or objects, enabling the tools to handle more complex inputs. These refinements ensure that the resulting tool documents are robust, versatile, and capable of addressing intricate cases.

![](images/6fe138a81cfce859eff8b98d6d869ff8c5502f34a6ab6c0bdbb63f671777ae9c.jpg)  
Figure 4: Distribution of the number of tool parameters before and after document refinement.

Code Generation Once refined tool documents doc<sub>i</sub> are complete, the code generation process produces corresponding locally executable functions fun<sub>i</sub>. These functions allow external invocation of tools, enabling seamless multi-turn interactions between the model and tools.

Code generation systematically maps document information to code. For instance, the tool name in doc is converted into the function name, while parameter specifications are used to define the function signature. To ensure the correctness of fun<sub>i</sub>, the atomic query $q _ { i }$ and its answer $a _ { i }$ are included as inputs, requiring the function to return $a _ { i }$ when executed with $q _ { i } ^ { . }$ . Additionally, a robust exception-handling mechanism is implemented, enabling tools to provide informative error messages for invalid inputs while maintaining normal operation. Moreover, the generated code is verified to ensure it functions as intended.

Dataset Construction To effectively implement our approach, we draw on queries from the MoreHopQA dataset (Schnitzler et al., 2024), which consists of multi-hop questions that can be decomposed into at least three atomic queries with answers. Using this foundation, we generate 995 user queries and 3,912 corresponding locally executable tools, which collectively form the ToolHop dataset.<sup>1</sup>

## 2.3 Dataset Analysis

To ensure that the ToolHop dataset rigorously evaluates the multi-hop tool-use capabilities of LLMs, we conduct a comprehensive analysis across five critical dimensions. This analysis validates ToolHop’s ability to represent diverse and challenging multi-hop tool-use scenarios.<sup>2</sup>

![](images/dcbd5e12cf4e6f759152233e898f9132e95eb6c66b8073b46d8df26458de9d9a.jpg)  
Figure 5: Distribution of tool parameter types before and after document refinement.

Diverse Queries Real-world user needs vary widely, requiring an effective LLM to flexibly utilize tools to address queries spanning multiple domains. To evaluate such capabilities, a suitable dataset must encompass queries from a broad range of topics. ToolHop is explicitly designed to prioritize diversity in its multi-hop queries, reflecting real-world scenarios.

To verify this diversity, we use GPT-4o to categorize all queries in ToolHop into distinct domains. Similar categories are merged to ensure clarity and independence. As shown in Figure 3, the categorization reveals that ToolHop spans 47 unique domains, including topics such as movies and television, academic subjects, and family relationships. This broad coverage ensures that ToolHop effectively evaluates LLM performance across diverse query types, enhancing its representativeness and practical applicability for real-world tool-use scenarios.

Meaningful Interdependencies Previous evaluation for tool use (Song et al., 2023; Yang et al., 2023; Ye et al., 2024b; Han et al., 2024) typically assemble tools from disparate sources and then generate user queries for them. However, these approaches fail to account for interdependencies between tools, often producing queries that inadequately represent multi-hop reasoning. To address this limitation, ToolHop employs a novel query-driven framework. It begins by formulating multi-hop queries and subsequently constructs the required tools based on each atomic query. This approach inherently preserves the multihop structure of queries and enforces meaningful interdependencies between tools.

<table><tr><td>Refinement</td><td>Zero</td><td>One</td><td>Two</td><td>Three</td><td>Four</td></tr><tr><td>Before</td><td>2</td><td>2433</td><td>1250</td><td>202</td><td>25</td></tr><tr><td>After</td><td>2</td><td>2490</td><td>1198</td><td>200</td><td>22</td></tr></table>

Table 2: Distribution of the number of required parameters before and after document refinement.
<table><tr><td>Refinement</td><td>string</td><td>boolean</td><td>array</td><td>integer</td><td>object</td><td>number</td></tr><tr><td>Before</td><td>4758</td><td>2</td><td>404</td><td>333</td><td>24</td><td>114</td></tr><tr><td>After</td><td>4473</td><td>2</td><td>755</td><td>241</td><td>44</td><td>102</td></tr></table>

Table 3: Distribution of required tool parameter types before and after refinement.

To validate the effectiveness of this approach, we analyze the distribution of tools associated with each query in ToolHop. As shown in Table 1, the number of tools required per query ranges from three to seven, which corresponds to the minimum number of reasoning hops required to resolve the queries, emphasizing the importance of multi-hop reasoning. This distribution underscores the complexity of queries handled by ToolHop and its capability to support scalable multi-hop tool use.

Locally Executable Tools Tools are a core component of the tool use task. ToolHop includes 3,912 locally deployable and directly executable tools, enabling zero-cost invocation and seamless interaction by LLMs. To better align the constructed tools with the diverse requirements of realworld applications, we enhance their complexity through a document refinement process.

Figure 4 shows that the average number of parameters per tool increased from 3.49 to 5.91 after refinement. This reflects an intentional shift toward more expressive tools, which better capture the complexity of real-world tasks. Concurrently, Figure 5 illustrates a 12% reduction in simple string parameters, replaced by more structured types such as arrays, booleans, and objects, which enable richer and more precise tool interactions. Table 2 and Table 3 further demonstrate that the refinement process preserves the number and types of required parameters while increasing the diversity of optional parameters.

Detailed Feedback Effective multi-turn interaction between LLMs and tools requires not only correct outputs for valid inputs but also meaningful error messages for invalid ones. Our approach incorporates two key strategies to address this need.

On the one hand, we include atomic queries and their corresponding answers as part of the input during code generation, ensuring tools reliably produce correct outputs for solvable problems. On the other hand, we integrate robust exceptionhandling mechanisms into the generated code. Since the tools are locally executable, we can validate LLM-generated call instances using a compiler, providing detailed error reports and feedback to guide subsequent interactions.

![](images/24b7dcaa588c95e9380941a39937217b144e399c2d67cc4986d67df0a31ba08a.jpg)  
Figure 6: Distribution of answer types for the second atomic subquery and final answers in ToolHop.

Verifiable Answers A key limitation of earlier tool-driven datasets is the absence of predetermined answers, which makes validation difficult. ToolHop overcomes this issue by predefining both queries and answers, enabling straightforward comparison with model outputs.

To ensure verifiability, we analyze the answer types for the second atomic subquery and the final query, which is presented in Figure 6. The result demonstrates that ToolHop supports diverse and flexible answer types while standardizing final answers into objective entities. This design simplifies validation, enhances robustness, and enables consistent performance evaluation.

## 3 Experimental Setup

We use ToolHop to evaluate the representative families of LLMs for multi-hop tool use. In this section, we introduce the families of LLMs evaluated (Section 3.1) and describe the implementation details of our experiments (Section 3.2).

## 3.1 Models

We use ToolHop to evaluate 14 LLMs from five families, including LLaMA3.1-Instruct-8B and LLaMA3.1-Instruct-70B from the LLaMA3.1 family, Qwen2.5-Instruct-7B, Qwen2.5- Instruct-14B, Qwen2.5-Instruct-32B, and Qwen2.5-Instruct-72B from the Qwen2.5 fmaily,

<table><tr><td rowspan="2">Source</td><td rowspan="2">Family</td><td rowspan="2">Version</td><td colspan="3">Answer Correctness (↑)</td><td colspan="2">Invocation Error (↓)</td></tr><tr><td>Direct</td><td>Mandatory</td><td>Free</td><td>Query</td><td>Instance</td></tr><tr><td></td><td></td><td>Avg.</td><td>19.83</td><td>32.12</td><td>32.84</td><td>18.72</td><td>8.68</td></tr><tr><td rowspan="6">Open-Source</td><td rowspan="2">LLaMA3.1</td><td>Instruct-8B</td><td>13.17</td><td>12.76</td><td>13.47</td><td>41.61</td><td>21.10</td></tr><tr><td>Instruct-70B</td><td>18.79</td><td>19.10</td><td>12.76</td><td>35.08</td><td>14.24</td></tr><tr><td rowspan="4">Qwen2.5</td><td>Instruct-7B</td><td>11.46</td><td>9.85</td><td>16.18</td><td>28.84</td><td>7.09</td></tr><tr><td>Instruct-14B</td><td>17.39</td><td>26.38</td><td>26.13</td><td>15.78</td><td>6.82</td></tr><tr><td>Instruct-32B</td><td>20.00</td><td>25.03</td><td>22.61</td><td>12.46</td><td>3.46</td></tr><tr><td>Instruct-72B</td><td>17.89</td><td>45.43</td><td>38.29</td><td>13.27</td><td>4.93</td></tr><tr><td rowspan="8">Closed-Source</td><td rowspan="2">Gemini1.5</td><td>flash-002</td><td>18.59</td><td>29.35</td><td>32.76</td><td>13.59</td><td>6.69</td></tr><tr><td>pro-002</td><td>18.89</td><td>31.16</td><td>33.07</td><td>14.57</td><td>6.61</td></tr><tr><td rowspan="2">Claude3.5</td><td>Haiku</td><td>36.08</td><td>38.09</td><td>44.72</td><td>23.48</td><td>15.81</td></tr><tr><td>Sonnet</td><td>27.14</td><td>39.90</td><td>45.23</td><td>19.60</td><td>15.83</td></tr><tr><td rowspan="4">GPT</td><td>3.5-Turbo</td><td>17.09</td><td>35.38</td><td>36.58</td><td>11.76</td><td>6.03</td></tr><tr><td>40-mini</td><td>19.40</td><td>40.20</td><td>43.42</td><td>11.66</td><td>3.58</td></tr><tr><td>4-Turbo</td><td>18.59</td><td>47.94</td><td>46.83</td><td>10.95</td><td>4.97</td></tr><tr><td>40</td><td>23.12</td><td>49.04</td><td>47.74</td><td>9.45</td><td>4.31</td></tr></table>

Table 4: Performance of various LLMs on ToolHop, including answer correctness and invocation error. ‘Direct, ‘Mandatory,’ and ‘Free’ denote the direct answer, mandatory tool use, and free choice scenarios, respectively. ‘Query’ and ‘Instance’ refer to the percentage of queries and tool invocation instances with errors, respectively. ‘Avg.’ represents the average across all LLMs. Values above the average are highlighted in teal , and those below are highlighted in maroon , with darker shades indicating larger deviations.

Gemini1.5-flash-002 and Gemini1.5-pro-002 from the Gemini1.5 family, textbfClaude3.5-Haiku and Claude3.5-Sonnet from the Claude3.5 family, and GPT-3.5-Turbo, GPT-4o-mini, GPT-4-Turbo, and GPT-4o from the GPT family.<sup>3</sup>

## 3.2 Implementation Details

In the data construction stage, we use GPT-4o to assist with processing.<sup>4</sup> For evaluation, opensource LLMs are tested using their chat templates with greedy decoding, while closed-source LLMs are evaluated via their APIs with a temperature setting of 0. To ensure consistency across evaluations, all tools are implemented through the models’ function call interfaces.

## 4 Main Results

In this section, we present the key evaluation dimensions (Section 4.1) and observations (Section 4.2).

## 4.1 Evaluation Dimensions

Evaluating the capabilities of LLMs requires a comprehensive approach that assesses both their ability to provide correct answers and their effectiveness in invoking external tools. We analyze these dimensions through answer correctness and invocation error.

Answer Correctness For the accuracy of LLM responses, our query-driven data construction scheme enables direct comparison with predefined standard answers. We consider three evaluation scenarios: the direct answer scenario, where LLMs solve queries independently without external tools; the mandatory tool use scenario, where models are required to use provided tools extensively to maximize their tool-use capabilities; and the free choice scenario, where external tools are available but optional, allowing LLMs to balance independent problem-solving with tool use.

Invocation Error In the mandatory tool use scenario, we assess errors made when invoking tools, leveraging detailed feedback for each tool to identify errors. We focus on three types: tool hallucination, where models invoke tools not included in the provided toolset; parameter hallucination, where unprovided parameters are used for a given tool; and parameter missing, where required parameters for a tool are omitted. Errors are quantified from the percentage of queries containing incorrect calls relative to total queries, and the percentage of incorrect tool invocations relative to all tool use instances.

## 4.2 Evaluation Observations

From the results presented in Table 4, we can make several notable observations.<sup>5</sup>

While LLMs have significantly enhanced their ability to solve complex multi-hop queries with the use of tools, their multi-hop tool use capabilities still leave considerable room for improvement. Comparing the direct answer scenario (i.e., Direct) versus the mandatory tool use scenario (i.e., Mandatory), we observe that the use of tools increases LLMs’ answer correctness by an average of 12.29%. Notably, the GPT family of models improves its accuracy by an average of 23.59% through tool use, underscoring how effective tool-use capabilities enhance their performance in solving complex multi-hop problems. Despite these improvements, the overall accuracy in the mandatory tool use scenario remains limited. Even the best-performing model, GPT-4o, achieves only 49.04% answer correctness in this scenario. Furthermore, 9.45% of queries exhibit hallucinations. The performance of LLaMA3.1-Instruct-8B reveals further challenges, with over 40% of queries containing invocation errors, underscoring the need for better documentation understanding.

The performance of different LLM families indicates that most are optimized for tool use, but they exhibit distinct characteristics when solving multi-hop queries. In both the mandatory tool use scenario and the free choice scenario (i.e., Free), LLMs generally opt to use tools, with answer correctness in these two conditions differing by only 0.62%. This indicates that most LLMs are specifically optimized for tool use. However, different LLM families show varying strengths in their tool use. For instance, Qwen2.5-Instruct-72B improves its answer correctness by 27.54% through tool use, while the Claude3.5 family excels in the direct answer scenario without tool reliance. The underlying reasons for these differences are explored in depth in Section 5.

Examining the performance of different versions within each LLM family, larger models generally demonstrate better tool use to meet user needs, aligning with the scaling law (Kaplan et al., 2020; Chung et al., 2022). Both opensource and closed-source LLMs show an increase in answer correctness and a decrease in invocation error in the mandatory tool use scenario as model size grows. Notably, the correlation between invocation errors and answer correctness is stronger at the query level than at the instance level, suggesting that invocation errors in specific queries significantly impair problem-solving. Interestingly, this pattern enables the inference of relative model sizes within families. For instance, based on performance patterns, GPT-4o is likely a larger and more advanced version compared to other models in the GPT family.

## 5 Further Studies

From the results in Section 4.2, we observe significant variation in the performance across different families of LLMs. To further investigate these differences, we analyze each family in detail and present the following key observations.<sup>6</sup>

The LLaMA3.1 and Gemini1.5 families perform poorly in multi-hop tool use scenarios compared to other LLMs from the same source, primarily due to their incomplete support for tool use capabilities. In the case of LLaMA3.1, the inability to output both natural language text and tool call instances simultaneously restricts its capacity to perform chain-of-thought (CoT) (Wei et al., 2022) reasoning during tool use, hampering its understanding and analysis of user intent. On the other hand, the Gemini1.5 family of models lack support for union-type parameters, which prevents them from handling tool lists that include complex parameter structures. This limitation significantly reduces their effectiveness in such scenarios.

The enhancement of the Qwen2.5 family with parallel tool calls introduces a trade-off between efficiency and accuracy. Compared to the LLaMA3.1 family, the Qwen2.5 family has improved its ability to utilize tools, particularly with the addition of parallel invocation, which is intended to increase the problem-solving efficiency. However, in multi-hop tool use scenarios, forcing parallel invocation without first processing the results of previous tool calls leads to hallucinations in parameter value assignments, resulting in incorrect answers. For instance, in the mandatory tool use scenario, the percentage of queries involving parallel tool calls is 70.1% for Qwen2.5-Instruct-14B and even higher at 75.08% for Qwen2.5- Instruct-32B, contributing to their relatively poor performance. In contrast, Qwen2.5-Instruct-72B reduces the percentage of parallel calls to just 3.82%, significantly improving its performance.

The optimization of CoT reasoning in the Claude family of models gives them a distinct advantage in the direct answer scenario. Even without explicit CoT prompts, the Claude3.5 family of models independently adopt a step-by-step CoT approach to decompose user queries and generate answers. This method significantly improves their accuracy compared to other LLMs in such scenarios. For instance, in the direct answer scenario, Claude3.5-Haiku applies CoT reasoning to 64.92% of queries, while Claude3.5-Sonnet does so for 8.5%. Additionally, the Claude3.5 family of models do not fully rely on the answers returned by tools. This allows them to produce correct responses using their own internal knowledge when tool invocations lead to errors. Despite a relatively high tool invocation error rate, this ability explains why overall answer correctness remains high.

The GPT family of models demonstrates some ability to correct tool call behavior after an error occurs, but this heavily depends on the level of detail in the feedback provided. Leveraging our query-driven data construction process, we offer detailed feedback when a tool call fails. We calculate the percentage of queries with call errors in the mandatory tool use scenario where the GPT family of models ultimately provide the correct answer. We compare this to the percentage of correct answers when only minimal feedback is given, such as a simple hint indicating the call failed (e.g., ‘Failed!’). As shown in Table 5, the GPT family of models exhibit a significant improvement in performance when detailed feedback is provided, successfully correcting their behavior to arrive at the correct answer. However, when only basic error hints are provided, the correctness of their final answers drops by 20.66%. This highlights not only the importance of detailed feedback but also the challenges in further enhancing the models correction capabilities.

Based on these observations, we propose the following recommendations to enhance the model’s tool use capabilities in the future: 1) Develop a robust and adaptable tool-use model that can support a wide range of complex tools; 2) Optimize the model’s parallelism and other capabilities while prioritizing improvements in its understanding of user intent to avoid potential negative effects; and 3) Investigate effective strategies for leveraging rich tool feedback to enhance the model’s error correction abilities.

<table><tr><td>Version</td><td>w/ Feedback</td><td>w/o Feedback</td><td> $\pmb { \Delta } _ { \mathbf { C }  \mathbf { I } }$ </td><td> $\Delta _ { \mathrm { I }  \mathrm { C } }$ </td></tr><tr><td>3.5-Turbo</td><td>36.75</td><td>21.37</td><td>20.51</td><td>5.13</td></tr><tr><td>4o-mini</td><td>38.53</td><td>11.93</td><td>29.36</td><td>2.75</td></tr><tr><td>4-Turbo</td><td>29.31</td><td>12.07</td><td>17.24</td><td>0.00</td></tr><tr><td>40</td><td>47.87</td><td>24.47</td><td>25.53</td><td>2.13</td></tr></table>

Table 5: Answer correctness of the GPT family of models in queries containing invocation error. ‘w/ Feedback’ and ‘w/o Feedback’ represent cases where detailed feedback or only simple error reporting is provided, respectively. $\mathbf { \partial } \cdot \mathbf { \partial } \mathbf { \Delta C } \to \mathbf { I } ^ { ' }$ denotes the proportion of correct answers that become incorrect, while $\mathbf { \Gamma } ^ { \bullet } \Delta _ { \mathbf { I }  \mathbf { C } } \mathbf { \Psi } ^ { \bullet }$ represents the proportion of incorrect answers that become correct, when transitioning from detailed feedback to simple error reporting.

## 6 Related Works

LLMs in Tool Use The use of tools is a prominent hallmark of biological intelligence (Shumaker et al., 2011). Equipping LLMs with the ability to use tools is therefore a key milestone in advancing their capabilities toward artificial general intelligence (Ye et al., 2023; Xi et al., 2023). Tools broadly encompass APIs, online services, application software, and other models that can be represented in formats accessible to LLMs (Qin et al., 2023). A critical factor in enhancing tool-use performance is constructing extensive datasets that detail tool use (Tang et al., 2023; Liu et al., 2024). This involves generating diverse user queries and their corresponding tool sets. Existing approaches often employ a tool-driven methodology, collecting tools from various sources and using models to simulate user queries (Zhuang et al., 2023; Yu et al., 2024). However, these methods lack diversity, fail to ensure dependency consistency, and cannot reliably verify data correctness. In this paper, we propose a query-driven data construction approach. This method extends the range of locally executable tools through multi-hop queries, improving dataset quality and better supporting the development of LLM tool-use capabilities.

Evaluation of Tool Use Effectively evaluating the tool-use capabilities of LLMs is crucial for identifying their strengths and weaknesses. Existing methods, such as manual verification (Tang et al., 2023) or checking for the presence of a final answer (Qin et al., 2024), fall short in providing objective and reliable measures of performance. Multi-dimensional approaches (Ye et al., 2025, 2024a) attempt to evaluate the process and outcomes of tool use but risk introducing model bias and inconsistencies. In this paper, we focus on evaluating LLMs in multi-hop tool use scenarios. Our query-driven data construction scheme predefines verifiable answers, ensuring accurate assessments and providing a robust framework for evaluation.

## 7 Conclusion

In this paper, we introduce ToolHop, a novel dataset designed to evaluate LLMs in multihop tool use. ToolHop employs a query-driven data construction framework, encompassing tool creation, document refinement, and code generation. This approach overcomes the limitations of previous methods, ensuring diverse queries, meaningful interdependencies, locally executable tools, detailed feedback, and verifiable answers. Using ToolHop, we benchmark 14 LLMs across five families, providing a comprehensive evaluation of their tool-use capabilities. Further studies illuminate the distinct characteristics of different LLM families, offering actionable insights to enhance their performance. By setting a robust standard for multi-hop tool use evaluation, ToolHop lays the groundwork for advancing LLMs’ ability to perform complex tool-based reasoning tasks.

## Limitations

While our dataset effectively evaluates the performance of LLMs in multi-hop tool use, one limitation of this work is the lack of an immediate strategy for enhancing these capabilities. Nonetheless, the scalability of our data construction scheme represents a significant advantage, as it can be readily adapted to create training datasets aimed at addressing this challenge. We hypothesize that targeted training using such datasets could markedly improve the ability of LLMs to perform multi-hop tool use tasks. Additionally, we provide a detailed analysis of current tool-use characteristics in LLMs, offering valuable insights that can serve as a foundation for future research and advancements in this area.

## Acknowledgments

The authors wish to thank the anonymous reviewers for their helpful comments. This work was partially funded by National Natural Science Foundation of China (No. 62476061,62206057,62076069), Shanghai Rising-Star Program (23QA1400200), Natural Science Foundation of Shanghai (23ZR1403500), Program of Shanghai Academic Research Leader under grant 22XD1401100.

## References

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. 2023. Qwen technical report. CoRR, abs/2309.16609.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosiute, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemí Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. 2022. Constitutional AI: harmlessness from AI feedback. CoRR, abs/2212.08073.

Zehui Chen, Weihua Du, Wenwei Zhang, Kuikun Liu, Jiangning Liu, Miao Zheng, Jingming Zhuo, Songyang Zhang, Dahua Lin, Kai Chen, and Feng Zhao. 2024. T-eval: Evaluating the tool utilization capability of large language models step by step. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 9510–9529. Association for Computational Linguistics.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun

Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Y. Zhao, Yanping Huang, Andrew M. Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. 2022. Scaling instructionfinetuned language models. CoRR, abs/2210.11416.

Yu Du, Fangyun Wei, and Hongyang Zhang. 2024. Anytool: Self-reflective, hierarchical agents for largescale API calls. CoRR, abs/2402.04253.

Tyna Eloundou, Alex Beutel, David G. Robinson, Keren Gu-Lemberg, Anna-Luisa Brakman, Pamela Mishkin, Meghan Shah, Johannes Heidecke, Lilian Weng, and Adam Tauman Kalai. 2024. First-person fairness in chatbots. CoRR, abs/2410.19803.

Zishan Guo, Renren Jin, Chuang Liu, Yufei Huang, Dan Shi, Supryadi, Linhao Yu, Yan Liu, Jiaxuan Li, Bojian Xiong, and Deyi Xiong. 2023. Evaluating large language models: A comprehensive survey. CoRR, abs/2310.19736.

Han Han, Tong Zhu, Xiang Zhang, Mengsong Wu, Hao Xiong, and Wenliang Chen. 2024. Nestools: A dataset for evaluating nested tool learning abilities of large language models. CoRR, abs/2410.11805.

Shijue Huang, Wanjun Zhong, Jianqiao Lu, Qi Zhu, Jiahui Gao, Weiwen Liu, Yutai Hou, Xingshan Zeng, Yasheng Wang, Lifeng Shang, Xin Jiang, Ruifeng Xu, and Qun Liu. 2024a. Planning, creation, usage: Benchmarking llms for comprehensive tool utilization in real-world complex scenarios. In Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, pages 4363– 4400. Association for Computational Linguistics.

Yue Huang, Jiawen Shi, Yuan Li, Chenrui Fan, Siyuan Wu, Qihui Zhang, Yixin Liu, Pan Zhou, Yao Wan, Neil Zhenqiang Gong, and Lichao Sun. 2024b. Metatool benchmark for large language models: Deciding whether to use tools and which to use. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, and Geoffrey E. Hinton. 1991. Adaptive mixtures of local experts. Neural Comput., 3(1):79–87.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. CoRR, abs/2001.08361.

Minghao Li, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, Haiyang Yu, Zhoujun Li, Fei Huang, and Yongbin Li. 2023. Api-bank: A comprehensive benchmark for tool-augmented llms. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 3102–3116. Association for Computational Linguistics.

Weiwen Liu, Xu Huang, Xingshan Zeng, Xinlong Hao, Shuai Yu, Dexun Li, Shuai Wang, Weinan Gan, Zhengying Liu, Yuanqing Yu, Zezhong Wang, Yuxian Wang, Wu Ning, Yutai Hou, Bin Wang, Chuhan Wu, Xinzhi Wang, Yong Liu, Yasheng Wang, Duyu Tang, Dandan Tu, Lifeng Shang, Xin Jiang, Ruiming Tang, Defu Lian, Qun Liu, and Enhong Chen. 2024. Toolace: Winning the points of LLM function calling. CoRR, abs/2409.00920.

Weijie Lv, Xuan Xia, and Sheng-Jun Huang. 2024. Codeact: Code adaptive compute-efficient tuning framework for code llms. CoRR, abs/2408.02193.

OpenAI. 2023. GPT-4 technical report. CoRR, abs/2303.08774.

Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. 2023. Gorilla: Large language model connected with massive apis. arXiv preprint arXiv:2305.15334.

Yujia Qin, Shengding Hu, Yankai Lin, Weize Chen, Ning Ding, Ganqu Cui, Zheni Zeng, Yufei Huang, Chaojun Xiao, Chi Han, Yi Ren Fung, Yusheng Su, Huadong Wang, Cheng Qian, Runchu Tian, Kunlun Zhu, Shihao Liang, Xingyu Shen, Bokai Xu, Zhen Zhang, Yining Ye, Bowen Li, Ziwei Tang, Jing Yi, Yuzhang Zhu, Zhenning Dai, Lan Yan, Xin Cong, Yaxi Lu, Weilin Zhao, Yuxiang Huang, Junxi Yan, Xu Han, Xian Sun, Dahai Li, Jason Phang, Cheng Yang, Tongshuang Wu, Heng Ji, Zhiyuan Liu, and Maosong Sun. 2023. Tool learning with foundation models. CoRR, abs/2304.08354.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. 2024. Toolllm: Facilitating large language models to master 16000+ real-world apis. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Changle Qu, Sunhao Dai, Xiaochi Wei, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Jun Xu, and Ji-Rong Wen. 2024. Tool learning with large language models: A survey. CoRR, abs/2405.17935.

Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy P. Lillicrap, Jean-Baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, Ioannis Antonoglou, Rohan Anil, Sebastian Borgeaud, Andrew M. Dai, Katie Millican, Ethan Dyer, Mia Glaese, Thibault Sottiaux, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, James Molloy, Jilin Chen, Michael Isard, Paul Barham, Tom Hennigan, Ross McIlroy, Melvin Johnson, Johan Schalkwyk, Eli Collins, Eliza Rutherford, Erica Moreira, Kareem Ayoub, Megha Goel, Clemens Meyer, Gregory Thornton, Zhen Yang, Henryk Michalewski, Zaheer Abbas, Nathan Schucher, Ankesh Anand, Richard Ives, James Keeling, Karel Lenc, Salem

Haykal, Siamak Shakeri, Pranav Shyam, Aakanksha Chowdhery, Roman Ring, Stephen Spencer, Eren Sezener, and et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. CoRR, abs/2403.05530.

Julian Schnitzler, Xanh Ho, Jiahao Huang, Florian Boudin, Saku Sugawara, and Akiko Aizawa. 2024. Morehopqa: More than multi-hop reasoning. CoRR, abs/2406.13397.

Yongliang Shen, Kaitao Song, Xu Tan, Wenqi Zhang, Kan Ren, Siyu Yuan, Weiming Lu, Dongsheng Li, and Yueting Zhuang. 2023. Taskbench: Benchmarking large language models for task automation. CoRR, abs/2311.18760.

Robert W Shumaker, Kristina R Walkup, and Benjamin B Beck. 2011. Animal tool behavior: the use and manufacture of tools by animals. JHU Press.

Yifan Song, Weimin Xiong, Dawei Zhu, Cheng Li, Ke Wang, Ye Tian, and Sujian Li. 2023. Restgpt: Connecting large language models with real-world applications via restful apis. CoRR, abs/2306.06624.

Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, and Le Sun. 2023. Toolalpaca: Generalized tool learning for language models with 3000 simulated cases. CoRR, abs/2306.05301.

Meta Team. 2024a. Introducing llama 3.1: Our most capable models to date.

Qwen Team. 2024b. Qwen2.5: A party of foundation models!

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open foundation and fine-tuned chat models. CoRR, abs/2307.09288.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS.

Mengsong Wu, Tong Zhu, Han Han, Chuanyuan Tan, Xiang Zhang, and Wenliang Chen. 2024. Seal-tools: Self-instruct tool learning dataset for agent tuning and detailed benchmark. CoRR, abs/2405.08355.

Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, Zhangyue Yin, Shihan Dou, Rongxiang Weng, Wensen Cheng, Qi Zhang, Wenjuan Qin, Yongyan Zheng, Xipeng Qiu, Xuanjing Huang, and Tao Gui. 2023. The rise and potential of large language model based agents: A survey. CoRR, abs/2309.07864.

Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu Li, and Ying Shan. 2023. Gpt4tools: Teaching large language model to use tools via selfinstruction. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Junjie Ye, Xuanting Chen, Nuo Xu, Can Zu, Zekai Shao, Shichun Liu, Yuhan Cui, Zeyang Zhou, Chao Gong, Yang Shen, Jie Zhou, Siming Chen, Tao Gui, Qi Zhang, and Xuanjing Huang. 2023. A comprehensive capability analysis of GPT-3 and GPT-3.5 series models. CoRR, abs/2303.10420.

Junjie Ye, Guanyu Li, Songyang Gao, Caishuang Huang, Yilong Wu, Sixian Li, Xiaoran Fan, Shihan Dou, Tao Ji, Qi Zhang, Tao Gui, and Xuanjing Huang. 2025. Tooleyes: Fine-grained evaluation for tool learning capabilities of large language models in real-world scenarios. In Proceedings of the 31st International Conference on Computational Linguistics, COLING 2025, Abu Dhabi, UAE, January 19-24, 2025, pages 156–187. Association for Computational Linguistics.

Junjie Ye, Sixian Li, Guanyu Li, Caishuang Huang, Songyang Gao, Yilong Wu, Qi Zhang, Tao Gui, and Xuanjing Huang. 2024a. Toolsword: Unveiling safety issues of large language models in tool learning across three stages. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 2181–2211. Association for Computational Linguistics.

Junjie Ye, Yilong Wu, Songyang Gao, Caishuang Huang, Sixian Li, Guanyu Li, Xiaoran Fan, Qi Zhang, Tao Gui, and Xuanjing Huang. 2024b. Rotbench: A multi-level benchmark for evaluating the robustness of large language models in tool learning. In Proceedings of the 2024 Conference on Empirical

Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024, pages 313–333. Association for Computational Linguistics.

Yuanqing Yu, Zhefan Wang, Weizhi Ma, Zhicheng Guo, Jingtao Zhan, Shuai Wang, Chuhan Wu, Zhiqiang Guo, and Min Zhang. 2024. Steptool: A step-grained reinforcement learning framework for tool learning in llms. CoRR, abs/2410.07745.

Yuchen Zhuang, Yue Yu, Kuan Wang, Haotian Sun, and Chao Zhang. 2023. Toolqa: A dataset for LLM question answering with external tools. CoRR, abs/2306.13304.

## A Comparison of Various Benchmarks

<table><tr><td rowspan="2">Benchmarks</td><td rowspan="2"># Tools</td><td rowspan="2"># Instances</td><td rowspan="2">Multi-Tool?</td><td rowspan="2">Meaningful Interdependencies?</td><td rowspan="2">Locally Executable Tools?</td><td rowspan="2">Detailed Feedback?</td><td rowspan="2">Verifiable Answers?</td></tr><tr><td></td></tr><tr><td>API-Bank (Li et al., 2023)</td><td>73</td><td>314</td><td>√</td><td>X</td><td>√</td><td>√</td><td>√</td></tr><tr><td>ToolAlpaca (Tang et al., 2023)</td><td>426</td><td>3938</td><td>×</td><td>X</td><td>×</td><td>×</td><td>x</td></tr><tr><td>RestBench (Song et al., 2023)</td><td>94</td><td>157</td><td>V</td><td>X</td><td>×</td><td>X</td><td>V</td></tr><tr><td>ToolBench (Qin et al., 2024)</td><td>16464</td><td>126484</td><td>√</td><td>X</td><td>√</td><td>√</td><td>x</td></tr><tr><td>MetaTool (Huang et al., 2024b)</td><td>199</td><td>21127</td><td>√</td><td>X</td><td>X</td><td>×</td><td></td></tr><tr><td>TaskBench (Shen et al., 2023)</td><td>103</td><td>23271</td><td>√</td><td>X</td><td>√</td><td>y</td><td>x</td></tr><tr><td>T-Eval (Chen et al., 2024)</td><td>15</td><td>533</td><td></td><td>X</td><td>√</td><td></td><td>V</td></tr><tr><td>ToolEyes (Ye et al., 2025)</td><td>568</td><td>382</td><td></td><td>X</td><td>√</td><td>√</td><td>×</td></tr><tr><td>UltraTool (Huang et al., 2024a)</td><td>2032</td><td>5824</td><td></td><td>X</td><td>×</td><td>×</td><td>×</td></tr><tr><td>Seal-Tools (Wu et al., 2024)</td><td>4076</td><td>14076</td><td>1</td><td>X</td><td>X</td><td>X</td><td></td></tr><tr><td>AnyToolBench (Du et al., 2024)</td><td></td><td>400</td><td></td><td>×</td><td>√</td><td>V</td><td>X</td></tr><tr><td>BFCL v3 (multi-turn) (Patil et al., 2023)</td><td></td><td>1000</td><td></td><td>√</td><td>×</td><td>×</td><td></td></tr><tr><td>ToolHop</td><td>3912</td><td>995</td><td></td><td>√</td><td>5</td><td>√</td><td></td></tr></table>

Table 6: A comparison between ToolHop and existing benchmarks related to tool use.

To illustrate the necessity and advantages of ToolHop, Table 6 presents a comparison between ToolHop and existing benchmarks related to tool use. Most existing benchmarks (e.g., API-Bank) lack meaningful interdependencies between tools. This limitation prevents them from evaluating a model’s ability to perform multi-hop tool use, where the output of one tool serves as the input to another. The only benchmark that includes meaningful interdependencies, BFCL v3 (multi-turn), lacks locally executable tools and detailed feedback. Moreover, it only supports models that generate all tool calls at once, rather than allowing multi-turn interactions with the environment. This makes it unsuitable for evaluating dynamic tool use, such as reasoning updates based on feedback.

In contrast, ToolHop is the only benchmark that incorporates all five key features: diverse queries, meaningful interdependencies, locally executable tools, detailed feedback, and verifiable answers. These features make it a comprehensive and realistic dataset for evaluating model performance in multi-hop tool use scenarios.

Additionally, most existing datasets become obsolete within 1–2 months as models quickly surpass their challenges. However, even the most advanced models continue to struggle with ToolHop. To maintain its effectiveness over time, we actively expand the dataset with more difficult tasks, establishing it as a long-term and instructive benchmark for evaluating tool-use capabilities.

## B Prompt for Data Construction

Our proposed query-driven data construction scheme involves tool creation, document refinement, and code generation. The prompts used for each process are provided in Table 7, Table 8, and Table 9, respectively.

Identify the appropriate tool to solve the given problem and provide an analysis of the tool design. The output should be in JSON format, following the specified structure.

## # Steps

1. \*\*Analyze the Problem\*\*: Understand the question and determine the type of information required to answer it.

2. \*\*Tool Design\*\*: Design a tool that can solve the problem, considering the complexity and additional functionalities it might need.

3. \*\*Parameter Specification\*\*: Define the parameters for the tool, ensuring they are comprehensive and flexible for various use cases.

4. \*\*Output Construction\*\*: Format the output in JSON, including both the analysis and the tool schema.

\# Notes

\- Ensure the tool is versatile enough to handle similar queries for different sports figures.

\- Consider edge cases.

\# Output Format

The output should be a JSON object with the following structure \*\*without any other contents\*\*:

\- "analysis": A detailed analysis of the ideas behind the tool design.

\- "tool": A JSON schema characterizing the tool, including its name, description, and parameters.

\# Example

{Example}

\*\*Question\*\*: {Question}

\*\*Output\*\*:

Table 7: The prompt for tool creation, where ‘{Example}’ and ‘{Question}’ represent the example and subquery, respectively.

![](images/b501f911f147918e73dcf4927c0526a31c5dd328cc7b988eba875a48ca6594bd.jpg)

Table 8: The prompt for document refinement, where ‘{Tool}’ represents the preliminary document.

Create a function implementation based on a provided tool document, question, and answer. The function should strictly adhere to the tool’s specifications, including the function name, parameter names, and types. Ensure the function is fully realized and capable of returning different feedback based on the input parameters.

## # Steps

1. \*\*Understand the Tool Document\*\*: Review the tool document to identify the function name, parameter names, and types.

2. \*\*Analyze the Question and Answer\*\*: Determine how the function should be used to answer the question.

3. \*\*Implement the Function\*\*:

\- Use the tool name as the function name.

\- Define parameters exactly as specified in the tool document.

\- Implement the function logic to produce the correct answer for the given question.

\- Simulate additional return values as specified in the tool document.

4. \*\*Error Handling\*\*: Develop a robust error handling mechanism to return valid error messages for incorrect inputs or other issues.

## # Notes

\- Ensure parameter types and names match exactly with the tool document.

\- Simulate additional return values as needed based on the tool’s documentation.

\- Implement comprehensive error handling to cover potential issues.

## # Output format

Output the result in JSON format with the following structure \*\*without any other contents\*\*:

{ "analysis": "Detailed analysis of how the function was designed, including reasoning for parameter choices and exception handling.",

"function": "The specific function design, including code and comments explaining each part." }

\*\*Tool Document\*\*:

{document}

\*\*Question\*\*: {question}

\*\*Answer\*\*: {answer}

Table 9: The prompt for code generation, where ‘{document}’, ‘{question}’ and ‘{answer}’ represent the refined document, the subquery and the corresponding answer, respectively.

## C Prompt for Domain Classification

![](images/f7ca9f9946e8bc4369604e9d8dd07ccde2a2de853da21e22c5b63315ca8561bc.jpg)  
Table 10: The prompt for domain classification, where ‘{sentence}’ represents the multi-hop query.

## D Details for Models

We evaluate 14 LLMs from five families, spanning both open- and closed-source models, to provide a comprehensive analysis of their performance in multi-hop tool use.

• LLaMA3.1 Family. The LLaMA3.1 family, developed by Meta, includes open-source LLMs with model sizes of 8B, 70B, and 405B, and context lengths up to 128K. These models are optimized for tasks such as long text summarization, multilingual dialogue, and code generation. Due to computational constraints, this study evaluates LLaMA3.1-Instruct-8B and LLaMA3.1-Instruct-70B.

• Qwen2.5 Family. The Qwen2.5 family, developed by Alibaba, consists of open-source LLMs pre-trained on 18 trillion tokens. These models are designed to excel in mathematics, programming, and knowledge representation, with versions ranging from 0.5B to 72B. Our evaluation focuses on Qwen2.5-Instruct-7B, Qwen2.5-Instruct-14B, Qwen2.5-Instruct-32B, and Qwen2.5-Instruct-72B.

• Gemini1.5 Family. The Gemini1.5 family, developed by DeepMind, utilizes a mixture-ofexperts (Jacobs et al., 1991) architecture for advanced reasoning across large datasets. This family includes flash and pro versions. For this paper, we analyze Gemini1.5-flash-002 and Gemini1.5- pro-002.

• Claude3.5 Family. The Claude3.5 family, developed by Anthropic, includes closed-source Haiku and Sonnet versions, which are known for advancements in instruction-following and nuanced reasoning. This evaluation considers Claude3.5-Haiku and Claude3.5-Sonnet.

• GPT Family. The GPT family, developed by OpenAI, comprises closed-source LLMs designed for text generation, multimodal understanding, and tool use. In this paper, we evaluate GPT-3.5-Turbo, GPT-4o-mini, GPT-4-Turbo, and GPT-4o.

## E Performance under Various Inference Frameworks

<table><tr><td>Source</td><td>Family</td><td>Version</td><td>FC</td><td>ReAct</td><td>CodeAct</td></tr><tr><td rowspan="2">Tool-Use</td><td>ToolLLaMA-2</td><td>7B-v2</td><td></td><td>16.08</td><td>-</td></tr><tr><td>TL-CodeLLaMA-2</td><td>7B</td><td>1</td><td>-</td><td>17.59</td></tr><tr><td rowspan="4">Open-Source</td><td>LLaMA3.1</td><td>Instruct-8B Instruct-70B</td><td>13.47 12.76</td><td>38.59 57.79</td><td>25.63 44.19</td></tr><tr><td></td><td>Instruct-7B</td><td>16.18</td><td>40.50</td><td>22.91</td></tr><tr><td rowspan="2">Qwen2.5</td><td>Instruct-14B</td><td>26.13</td><td>46.13</td><td>36.78</td></tr><tr><td>Instruct-32B</td><td>22.61</td><td>48.04</td><td>45.73</td></tr><tr><td rowspan="6">Closed-Source</td><td rowspan="2">Gemini1.5</td><td>Instruct-72B</td><td>38.29</td><td>48.44</td><td>46.63</td></tr><tr><td>flash-002</td><td>32.76</td><td>4.82</td><td>5.13</td></tr><tr><td rowspan="2">Claude3.5</td><td>pro-002</td><td>33.07</td><td>22.61</td><td>7.24</td></tr><tr><td>Haiku</td><td>44.72</td><td>34.07</td><td>48.34</td></tr><tr><td rowspan="2"></td><td>Sonnet</td><td>45.23</td><td>44.52</td><td>49.15</td></tr><tr><td>3.5-Turbo</td><td>36.58</td><td>28.14</td><td>25.63</td></tr><tr><td rowspan="4"></td><td>GPT</td><td>4o-mini</td><td>43.42</td><td>47.14</td><td>40.80</td></tr><tr><td></td><td>4-Turbo</td><td>46.83</td><td>53.67</td><td>42.61</td></tr><tr><td></td><td>40</td><td>47.74</td><td>54.17</td><td>44.92</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 11: Performance of various LLMs on ToolHop across different inference frameworks. ‘FC’ refers to answer correctness in the Free scenario using built-in function-calling templates.

In addition to using the built-in chat template, we also compare the performance of the models under other commonly used tool-use frameworks. Table 11 shows how different models perform with the ReAct (Yao et al., 2023) and CodeAct (Lv et al., 2024) inference frameworks, and compares these results to performance in the Free scenario. The results show that while model performance varies across frameworks, none of them effectively solves the challenges presented by ToolHop in any setting, which further supports our conclusions.

We also observe the following:

• Open-source models benefit more from the ReAct framework, as it enforces a step-by-step reasoning process before tool invocation, which improves planning ability. In contrast, models using FC template often invoke tools directly without explicit reasoning, suggesting that open-source models still have room to improve their built-in function-calling capabilities.

• Closed-source models are generally well-optimized for FC, and their performance remains more stable across different frameworks compared to open-source models. However, different models still show preferences for specific frameworks. For instance, the GPT series performs best with ReAct, while the Claude 3.5 series prefers CodeAct, likely due to differences in training data and methodology.

## F Case Study

In Section 4.2 and Section 5, we analyze the performance of different LLMs across various scenarios. In this section, we present relevant examples from Figure 7 to Figure 10.

![](images/f0b6c8e9b7ad73b402094ff789e5418504c70f2b4e72219f58c5961027de3ef5.jpg)  
Figure 7: The Qwen2.5 family of LLMs emphasizes parallel tool calls in the mandatory tool use scenario, which can lead to hallucinations and incorrect answers.

![](images/7153662552993569995db16417cfb517902da868c954cd4d2132b5675d06124f.jpg)  
Figure 8: The Claude 3.5 family of LLMs optimizes CoT reasoning in the direct answer scenario, enhancing their analytical and problem-solving capabilities."

![](images/1bac1770f6cd6d5046fc610ed91975a6effd8ec5c2d181d99860b5a5a1be46b2.jpg)  
Figure 9: The GPT family of LLMs improves performance by refining calling behavior through the use of detailed tool feedback.

![](images/abb90943863b325e0f054a8b47100f7f2cdd6201628c014c5f2745d0b59288eb.jpg)  
Figure 10: The GPT fmaily of LLMs struggles to correct their calling behavior when provided with minimal feedback.

## G Examples of Tool Documents

Our query-driven data construction scheme generates preliminary document prior to refinement. Below, we provide examples of documents before and after refinement. As shown, the refinement process enhances the tool’s functionality, increases the number of parameters and introduces more diverse parameter types.

"name": "album\_release\_date\_finder",   
"description": "A tool designed to find the release date of music   
albums. It queries a database or API to retrieve accurate   
information about album release dates, accommodating variations in   
album titles and artist names.",   
"parameters": {   
"type": "object",   
"properties": {   
"album\_name": {   
"type": "string",   
"description": "The name of the album for which the   
release date is being queried."   
},   
"artist\_name": {   
"type": "string",   
"description": "The name of the artist or band   
associated with the album, to ensure accuracy in case   
of albums with similar names."   
},   
"output\_format": {   
"type": "string",   
"enum": [   
"date",   
"text"   
],   
"description": "The format of the output. Defaults to   
date (the release date in YYYY-MM-DD format)."   
}   
},   
"required": [   
"album\_name"   
]   
}   
}

"description": "An advanced tool designed to find the release date   
of music albums. It queries a comprehensive database or API to   
retrieve accurate information about album release dates,   
accommodating variations in album titles, artist names, album   
versions, release regions, and languages. This tool ensures   
precision and flexibility in retrieving album release information.",   
"parameters": {

```csv
"album_name": {
"type": "string",
"description": "The name of the album for which the
release date is being queried."
},
"artist_name": {
"type": "string",
"description": "The name of the artist or band
associated with the album, to ensure accuracy in case
of albums with similar names."
},
"album_version": {
"type": "string",
"description": "The specific version of the album
(e.g., deluxe, remastered) to refine the search."
},
"release_region": {
"type": "string",
"description": "The geographical region where the albu
was released, which can affect the release date."
},
"language": {
"type": "string",
"description": "The language of the album, useful for
albums released in multiple languages."
},
"output_format": {
"type": "string",
"enum": [
"date",
"text"
],
"description": "The format of the output. Defaults to
date (the release date in YYYY-MM-DD format)."
}
},
"required": [
"album_name"
]
}
}
```

- artist\_name (str): The name of the artist or band associated with   
the album.

## H Examples of Code Implementations

Our query-driven data construction scheme translates the refined tool document into code, enabling it to function as a locally executable tool. Below, we provide the code implementation of the refined document in Appendix G. The implementation fully realizes the defined functionality, provides valid feedback for correct parameter inputs, and incorporates a robust exception handling mechanism.

```python
def album_release_date_finder(album_name: str, artist_name: str =
'', album_version: str = '', release_region: str = '', language:
str = '', output_format: str = 'date') -> str:
11 1 11
```

```prolog
- album_version (str): The specific version of the album (e.g.,
deluxe, remastered).
```

- release\_region (str): The geographical region where the album was   
released.

- language (str): The language of the album.

- output\_format (str): The format of the output, either 'date' or   
'text'.

## Returns:

- str: The release date of the album in the specified format.   
11 1 11

```python
# Simulated database/API response
album_database = {
'Boy': {
'artist': 'U2',
'release_date': '1980-10-20',
'versions': {
'standard': '1980-10-20',
'deluxe': '2008-07-21'
},
'regions': {
'US': '1980-10-20',
'UK': '1980-10-20'
},
'languages': {
'English': '1980-10-20'
}
}
}
# Error handling for required parameter
if not album_name:
return 'Error: The album_name parameter is required.'
```

# Retrieve album information

```python
album_info = album_database.get(album_name)
if not album_info:
return 'Error: Album not found in the database.'
# Check artist name if provided
if artist_name and album_info['artist'] != artist_name:
return 'Error: Artist name does not match the album record.
# Determine release date based on version, region, and language
release_date = album_info['release_date']
if album_version:
release_date = album_info['versions'].get(album_version,
release_date)
if release_region:
release_date = album_info['regions'].get(release_region,
release_date)
if language:
release_date = album_info['languages'].get(language,
release_date)
# Format the output
if output_format == 'text':
return f'The album "{album_name}" by {album_info["artist"]} was
released on {release_date}.'
return release_date
# Example usage
print(album_release_date_finder(album_name='Boy', artist_name='U2',
output_format='date')) # Output: '1980-10-20'
print(album_release_date_finder(album_name='Boy', artist_name='U2',
output_format='text')) # Output: 'The album "Boy" by U2 was released
on 1980-10-20.'
```