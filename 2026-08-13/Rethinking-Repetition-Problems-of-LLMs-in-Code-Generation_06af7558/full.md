# Rethinking Repetition Problems of LLMs in Code Generation

Yihong Dong, Yuchen Liu, Xue Jiang, Bin Gu†, Zhi Jin, and Ge Li\*

Key Laboratory of High Confidence Software Technologies (Peking University), Ministry of Education; School of Computer Science, Peking University, Beijing, China

†Beijing Institute of Control Engineering

{dongyh, liuyuchen1, jiangxue}@stu.pku.edu.cn,{zhijin, lige}@pku.edu.cn

## Abstract

With the advent of neural language models, the performance of code generation has been significantly boosted. However, the problem of repetitions during the generation process continues to linger. Previous work has primarily focused on content repetition, which is merely a fraction of the broader repetition problem in code generation. A more prevalent and challenging problem is structural repetition. In structural repetition, the repeated code appears in various patterns but possesses a fixed structure, which can be inherently reflected in grammar. In this paper, we formally define structural repetition and propose an efficient decoding approach called RPG, which stands for Repetition Penalization based on Grammar, to alleviate the repetition problems in code generation for LLMs. Specifically, RPG first leverages grammar rules to identify repetition problems during code generation, and then strategically decays the likelihood of critical tokens that contribute to repetitions, thereby mitigating them in code generation. To facilitate this study, we construct a new dataset CodeRepetEval to comprehensively evaluate approaches for mitigating the repetition problems in code generation. Extensive experimental results demonstrate that RPG substantially outperforms the best-performing baselines on CodeRepetEval dataset as well as HumanEval and MBPP benchmarks, effectively reducing repetitions and enhancing the quality of generated code.

## 1 Introduction

Code generation seeks to automatically produce code that aligns with user intents, which is a research hotspot in the fields of artificial intelligence, natural language processing, and software engineering (Le et al., 2022; Chen et al., 2023; Liu et al., 2023). In recent years, the emergence of neural language models has shown remarkable advancements in code generation (Chen et al., 2021; OpenAI, 2023). However, even well-trained large language models (LLMs) may suffer from repetition problems, which hurts the code generation quality of LLMs substantially (Liu et al., 2024).

Recent studies about repetition problems of LLMs are primarily focused on content repetition (Xu et al., 2022; Li et al., 2023a), which refers to the results of the generation system always containing duplicate fragments (Fu et al., 2021a). However, our preliminary investigation of LLM's repetition problem in code generation reveals that content repetition constitutes only a minor portion of them, as shown in Figure 1. In contrast, a predominant form of repetition in the generated results involves the repeated occurrence of similar codes with fixed structural patterns, which we term ‘structural repetition'2. Distinct from content repetitions, the pattern of different structural repetitions varies markedly (e.g., structural repetitions I-IV), making structural repetitions hard to detect and handle. Given the diversity and complexity of structural repetitions, previous approaches tailored for content repetitions are insufficient to address them effectively. Therefore, it is necessary and significant to explore the structural repetitions in code generation.

In this paper, we propose an effective decoding approach RPG: Repetition Penalization based on Grammar, to alleviate repetition problems of LLMs in code generation. Considering different code fragments with the same structural patterns can be represented by identical grammar rules, RPG employs the pushdown automaton built on grammar rules to detect repetition problems during the generation process, and then strategically decreases the likelihood of key tokens that contribute to repetitions. RPG offers two main benefits: 1) it curtails the endless generation of meaningless, repetitive code, thereby saving tokens and time-consuming; 2) it realigns the LLMs’ generation back to the correct generation path, enhancing the quality of code generation. Moreover, we construct a new dataset, named CodeRepetEval, for evaluating approaches to mitigate the repetition problems in code generation. Extensive experimental results and analyses verify the effectiveness and generality of RPG.

![](images/c751f6efa4ff3745c9675d6d806e194c4a47591797ccbc07ab37b85038fa4f44.jpg)  
Figure 1: Examples of repetition problems in code generation, collected from the well-trained LLMs, e.g., CodeLlama (Rozière et al., 2023) and ChatGPT (OpenAI, 2022) (Left). The statistical percentage of two repetition forms occurs in the generated code of LLMs (Right).

Our main contribution can be summarized as fourfold. 1) We first formally define structural repetitions, which are more prevalent than content repetitions in code generation. 2) We present RPG, a novel decoding approach that leverages pushdown automaton to identify and mitigate repetitive problems in code generation from grammar perspective. 3) We construct CodeRepetEval dataset covering three scenarios, with data derived from artificial synthesis, code generation benchmarks, and realworld repositories, to facilitate subsequent research for repetitive problems in code generation. 4) RPG substantially outperforms the best-performing baselines in various scenarios, which alleviates both structural and content repetitions, and achieves better code generation quality.

## 2 Motivation Example

The repetition problem in code generation remains an underexplored challenge, usually resulting in redundancy and errors. Figure 2 showcases an example where LLMs generate the code containing structural repetitions. This generated code is plagued by repetitions with the fixed structural pattern starting with elif’. In each repetition, LLMs generate different conditions following the start token elif and varying statements under these conditions. Despite the content of each repetition differing, both the probability of the start token and the average probabilities of all tokens in each repetition exhibit an upward trend as the number of repetitions increases, showing a self-reinforcement effect as the right side of Figure 2. This phenomenon means that the start token in each repetition will serve as an anchor point. As the model continues to generate code, it relies on this anchor, reinforcing its choice and making it increasingly difficult to diverge from the structural repetition. This ultimately leads to the subsequent generation getting stuck in endless and meaningless (or even erroneous) repetitions.

According to principles of compilation (Alfred et al., 2007), we discover that massive patterns of structural repetition are inherently reflected in grammar, i.e., the potential positions where code can be repeated are determined by explicit grammar rules. For example, the structural repetitions of the code in Figure 2 adhere to (elif'test :' suite)\* within if\_stmt of grammar, where \* denotes that its preceding expression can be repeated zero or more times. Although the grammar rules impose no limit on the number of repetitions, human-written code does not repeat endlessly, with the higher the number of repetitions, the lower the likelihood of their occurrence. Therefore, the prediction confidence of further repetition ought to decrease with a growing number of repetitions, rather than exhibiting the self-reinforcement usually observed in LLMs.

In this paper, we first formally define structural repetitions, and propose RPG to effectively detect and alleviate them in code generation, thereby enhancing the quality of generated code for LLMs.

![](images/c05d40089e36696c0d44f994198ce87752b15c005c64c6ffeab85ed4449883dc.jpg)

![](images/e8a6c4459316bf7ab9033fb7cbe7f0dfe2595dd6a47a34e884232323a5e25774.jpg)  
Figure 2: A case of structural repetition generated by CodeLlama with temperature = 0, where the dashed-underline text is the prompt (Left). The corresponding grammar rules of structural patterns (Top). LLM's probabilities of generated tokens in each repetition (Right)

## 3 Definition of Structural Repetition

Given a token sequence $X = [ x _ { 1 } , x _ { 2 } , \cdot \cdot \cdot , x _ { | X | } ] ,$ where $x _ { i }$ denotes the i-th token and $| X |$ represents the length of X. We denote $X _ { p : q } =$ $[ x _ { p } , x _ { p + 1 } , \cdot \cdot \cdot , x _ { q } ]$ , where $( 1 \leq p < q \leq | X | )$ , as a continuous subsequence of X. Given a mapping function $G$ to represent the underlying structure of X, the mapped sequence ${ \hat { R } } = G ( X )$ is obtained by applying G to X, where $\hat { R } = [ \hat { r } _ { 1 } , \hat { r } _ { 2 } , \cdots , \hat { r } _ { | \hat { R } | } ]$ . In this paper, G indicates the context-free grammar³, which can be defined as a quad tuple $( N , \Sigma , P , S )$ where N is a set of non-terminal symbols, Σ is a set of terminal symbols, representing the basic symbols of the language, P is a set of production rules, with each rule in the form $A  \beta ,$ where $A \in N$ and $\beta$ is a sequence of elements from $N \cup \Sigma$ , and $S \in N$ is the start symbol, used to begin derivations of strings.

Specifically, the generated sequence X will be reduced in accordance with G, ensuring that the reduction of X does not exceed the statement levels of grammar rules, which includes twelve simple statements and nine compound statements⁴. Thus, the patterns of structural repetitions within X can

be defined as:

$$
\begin{array} { r } { \mathrm { S R } ( X ) = \{ \hat { R } _ { p : q } | \exists 1 \leq p < q \leq | \hat { R } | - q + p , } \\ { \forall i \in [ p , q ] , \hat { r } _ { i + q - p } = \hat { r } _ { i } \} , \quad } \end{array}\tag{1}
$$

where structural repetitions exist in X, ${ \mathrm { i f } } \operatorname { S R } ( X ) \neq$ Ø, and the elements in SR(X) represent the patterns of structural repetitions. For example, if X denotes the generated code in Figure 2, then Ř would be $\begin{array} { r l } { ^ { 6 6 } . \ . . } & { { } \ ^ { \cdot } e l i f ^ { , } } \end{array}$ test $\because$ suite $' e l i f '$ test $\because$ suite $\cdots$ Thus, $\hat { R } _ { p : q }$ can be the subsequence of ${ \hat { R } } , \operatorname { i . e . , } \cdots e l i f ^ { \prime }$ test :' suite", where $\forall i \in [ p , q ] , \hat { r } _ { i + q - p } = \hat { r } _ { i }$ . Note that for the same input, G has a unique output, i.e., if content repetitions exist in X, the repetitions will be preserved in $G ( X )$ as well, which implies content repetitions can also be detected by Eq. (1).

Structural repetition negatively impacts code generation in the following two ways: 1) it fails to terminate properly, rendering the code uncompilable. As repetition persists, the generation of LLMs becomes meaningless or incorrect gradually; 2) it disrupts the generation of code, leading to the absence of a portion of functionality, thereby severely hurting the quality of generated code.

## 4 Methodology

In this section, we will introduce RPG in detail, including three parts, i.e., Reduction to Grammar Rules (§ 4.1), Detection of Repetition (§ 4.2), and Penalization of Repetition (§ 4.3).

## 4.1 Reduction to Grammar Rules

In problems of structural repetition, while the forms of repeated statements vary, their underlying grammar rules demonstrate similarities. Following the previous work (Dong et al., 2023b), we employ the pushdown automaton (PDA) for the reduction of generated codes into their underlying grammar rules during code generation⁵. A PDA can be defined as a seven tuple $( Q , \Sigma , \Gamma , \delta , q _ { 0 } , Z , F )$ , where $Q , \Sigma , \Gamma$ are finite sets representing the states, input symbols, and stack symbols, respectively. qo is the initial state, $z _ { \mathrm { 0 } }$ is the initial stack symbol, and A is the set of accepting states, with $q _ { 0 } \in Q , z _ { 0 } \in \Gamma$ 2 and $A \subseteq Q . \delta : Q \times ( \Sigma \cup \{ \epsilon \} ) \times \Gamma \to \mathcal { P } ( Q \times \Gamma ^ { * } )$ is the transition function, where € denotes the empty string, and $\Gamma ^ { * }$ represents all sequences of stack symbols. Based on δ of PDA, we have

$$
q _ { t } , z _ { t } = \delta ( q _ { t - 1 } , x _ { t } , z _ { t - 1 } ) ,\tag{2}
$$

where $q _ { t }$ is the state of t-th time step, $z _ { t }$ is the stack symbol of t-th time step, and $x _ { t }$ is the generated token of t-th time step. According to $q _ { t }$ and $z _ { t } , x _ { t }$ can be reduced to its corresponding grammar rule uniquely, which is expressed as:

$$
\hat { x } _ { t } = g ( x _ { t } ) = [ q _ { t } , z _ { t } ] ,\tag{3}
$$

We merge the same adjacent parts in $\hat { X } _ { 1 : t } ~ =$ $[ \hat { x } _ { 1 } , \hat { x } _ { 2 } , \cdots , \hat { x } _ { t } ]$ to get the final reduction sequence of grammar rules for $x _ { t }$

$$
\hat { R } _ { 1 : t } = \mathrm { m e r g e } ( \hat { X } _ { 1 : t } ) ,\tag{4}
$$

For example, multiple adjacent generated tokens in Figure 2 belong to the same terms such as ‘test’ and suite', the adjacent ones will be merged into a single entity for each term.

## 4.2 Detection of Repetition

Considering that once repetition problems arise in code generation, they tend to persist until the end of the generation process, we need to detect these repetition problems as they emerge during code generation. To detect the repetitions in $\hat { R } _ { 1 : t }$ , we employ suffix arrays and longest common prefix (LCP) arrays, which are efficient with the time complexity of O(n log n) and the space complexity of $O ( n )$ . The pseudo-code for suffix arrays and the LCP array is provided in Appendix D.

Suffix Array: The suffix array is an array of integers representing the starting indices of the suffixes of $\hat { R } _ { 1 : t }$ , sorted in lexicographical order. Thus, Suf[i] points to the starting index of the i-th smallest suffix in $\hat { R } _ { 1 : t }$

Longest Common Prefix Array: The LCP array is defined such that $\mathrm { L C P } [ i ]$ is the length of the longest common prefix between suffixes starting at Suf[i — 1] and Suf[i] for all $1 \leq i < n$ (with LCP[0] typically set to 0 for convenience).

Using the LCP array, we identify all positions of $\hat { R } _ { 1 : t }$ where $\mathrm { L C P } [ i ] > 0$ These positions indicate the presence of repetitions of length at least $\mathrm { L C P } [ i ]$ Therefore, the repetition patterns within $X _ { 1 : t }$ can be expressed as:

$$
\begin{array} { r l } & { { \tt R e p } ( X _ { 1 : t } ) = \{ \hat { R } _ { { \tt S u f } [ i ] : { \tt S u f } [ i ] + { \tt L C P } [ i ] } \mid \forall i \in [ 1 , t - 1 ] , } \\ & { { \tt L C P } [ i ] > 0 \land { \tt S u f } [ i - 1 ] = { \tt S u f } [ i ] + { \tt L C P } [ i ] \} , } \end{array}\tag{5}
$$

## 4.3 Penalization of Repetition

Given the identified repetitions, RPG applies a penalization mechanism to discourage the model from generating them in future outputs. The penalization mechanism integrates into the code generation model's scoring function, modifying how tokens are weighted during the generation process.

Dynamic Weight Adjustment: We define a dynamic weight function, Pn(·), which applies a decreasing factor to the score of a token based on the frequency and recency of its associated grammar rule in the sequence $\hat { R } _ { 1 : t }$ . The weight for each token is adjusted as follows:

$$
\operatorname P \mathbf { n } ( x _ { t } | x _ { < t } ) = \lambda ^ { \operatorname { C o u n t } ( \operatorname { R e p } ( X _ { 1 : t } ) ) } ,\tag{6}
$$

where $x _ { < t }$ represents $X _ { 1 : t - 1 } , \lambda$ is a decay factor between 0 and 1, and Count $\left( \mathrm { R e p } ( X _ { 1 : t } ) \right)$ is the count of times the repetition patterns in $\mathrm { R e p } ( X _ { 1 : t } )$ has appeared. This exponential decay effectively reduces the likelihood of selecting tokens associated with repetitive grammar rules.

Token Scoring Adjustment: During the code generation, each token's score is recalculated by incorporating the dynamic weight:

$$
s ^ { \prime } ( x _ { t } | \boldsymbol { x } _ { < t } ) = s ( \boldsymbol { x } _ { t } | \boldsymbol { x } _ { < t } ) \cdot \mathbf { P n } ( x _ { t } | \boldsymbol { x } _ { < t } ) ,\tag{7}
$$

where $s ( x _ { t } )$ is the original score of the token $x _ { t }$ provided by the model, and $g ( x _ { t } )$ is the grammar rule associated with $x _ { t }$ . The adjusted score $s ^ { \prime } ( x _ { t } )$

influences the token selection process, guiding the model toward less repetitive and more diverse code generation.

Finally, our RPG approach can be defined as:

$$
\mathrm { R P G } ( x _ { t } | \boldsymbol { x } _ { < t } ) = \arg \operatorname* { m a x } _ { \boldsymbol { x } _ { t } } s ^ { \prime } ( x _ { t } | \boldsymbol { x } _ { < t } )\tag{8}
$$

## 5 Experiment Setup

In this section, we will provide the setups of our experiments below. The detailed description of experiment setups can be found in Appendix C.

## 5.1 Datasets

Considering the absence of datasets for repetition problems in code generation, we dedicate more than 400 hours to constructing and examining CodeRepetEval dataset. We simulate repetition problems in code generation covering three scenarios, including artificial synthesis, code generation benchmarks, and real-world repositories. Specifically, Artificial Synthesis scenario involves 512 test samples. Each sample consists of a correct code concatenated with its last repetition patterns 5 to 10 times. Code Generation Benchmarks scenario comprising 512 test samples, which are selected from the generated repetitive codes of three LLMs (i.e., CodeLlama (Rozière et al., 2023), DeepSeek Coder (Guo et al., 2024), CodeGen (Nijkamp et al., 2023), and ChatGPT (OpenAI, 2022)) on HumanEval and MBPP benchmarks. Realworld Repositories scenario includes 1024 test samples, picked from the partial code in real-world repositories, which is identified to induce repetition problems in the generated outputs of the aforementioned LLMs. We employ CodeRepetEval to assess the effectiveness of RPG for addressing repetition problems in code generation scenarios.

Moreover, We also involve four public benchmarks to evaluate the RPG's performance in code generation, including HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021), as well as their extended version HumanEval-ET and MBPP-ET (Dong et al., 2024a).

## 5.2 Baselines

As our approach is based on decoding that does not require modification and training of the model, we compare it to the four most commonly used decoding approaches, including Greedy Sampling, Topk Sampling (Fan et al., 2018), Temperature Sampling (Caccia et al., 2019), Topp Sampling (Holtzman et al., 2020). Furthermore, we also compare two representative baselines for addressing content repetition in text generation, including Repetition Penalty (Keskar et al., 2019) for the decoding phase and Repetition Dropout (Li et al., 2023a) applied on the training phase. These baselines follow the settings in their original paper.

## 5.3 Metrics

We mainly use six metrics to evaluate approaches for addressing the repetition problems in code generation. End-of-sentence Generation Percentage (EGP) quantifies the frequency with which a model successfully interrupts repetitive sequences to conclude generation, which is determined by calculating the proportion of end-ofsentence (EOS) tokens across all samples generated by the model. TR-N is calculated to measure structural repetitions within a generated sequence at phrase-level, which is defined as 1.0 — $\scriptstyle { \big | } \displaystyle { \big \{ } G ( x ) ^ { \prime } | \exists p \in { \dot { [ 1 , | G ( x ) | - n + 1 ] } } , G ( x ) ^ { \prime } = G ( x ) _ { p : p + n - 1 } { \big \} } { \big | }$ , where $| G ( x ) |$ means the number of elements in $G ( x )$ It effectively quantifies the proportion of duplicate n-grams present in its underlying grammar rules, and $n = 4$ in this paper. In contrast, TR-S measures structural repetitions within a generated sequence at statement-level, which is defined as 1.0 – Lunique statements in G(x)l. Compiler Correct- [statements in G(x)| ness Percentage (CCP) evaluates whether the generated code is compilable, which is measured by the proportion of code samples that successfully compile. We use Time and GenLen to verify the approaches' efficiency, which means the average time of model generation and the average length of model-generated outputs, respectively

For HumanEval(-ET) and MBPP(-ET) benchmarks which contain the test cases, we employ Pass@k (Li et al., 2022) metric to measure the functional correctness of the generated code by executing test cases.

## 5.4 Implementation Details

In this paper, all experiments are conducted on an A6000 GPU (48GB). We employ CodeLlama-7B as our base model. The decay factor λ for the penalization of repetition in RPG is set at 0.9 by default. The maximum token length of each approach is set to 1024 in all datasets and scenarios, except for CodeRepetEval (real-world repository) setting to 4096. Following the previous work (Chen et al., 2021; Rozière et al., 2023), the default temperature for the baselines is set at 0.8. To mitigate the instability of the model sampling, we report the average results of five trials in the experiments.

Table 1: Comparison of RPG with commonly used decoding approaches and representative content repetition baselines on CodeRepetEval dataset in three scenarios.
<table><tr><td rowspan="2">Approach</td><td colspan="6">CodeRepetEval</td></tr><tr><td>EGP↑</td><td>TR-N↓</td><td>TR-S ↓</td><td>CCP↑</td><td>Time ↓</td><td>GenLen</td></tr><tr><td colspan="7">Code Generation Benchmarks</td></tr><tr><td>Rep_Penalty</td><td>0.721</td><td>0.374</td><td>0.425</td><td>0.413</td><td>16.88</td><td>689</td></tr><tr><td>Rep_Dropout</td><td>0</td><td>0.569</td><td>0.536</td><td>0.218</td><td>35.65</td><td>1024</td></tr><tr><td>Greedy</td><td>0</td><td>0.598</td><td>0.637</td><td>0.455</td><td>33.87</td><td>1024</td></tr><tr><td>Temp (t=0.1)</td><td>0.007</td><td>0.599</td><td>0.622</td><td>0.39</td><td>37.72</td><td>1019</td></tr><tr><td>Temp (t=0.2)</td><td>0.03</td><td>0.603</td><td>0.63</td><td>0.413</td><td>36.13</td><td>950</td></tr><tr><td>Temp (t=0.8)</td><td>0.578</td><td>0.423</td><td>0.441</td><td>0.433</td><td>27.13</td><td>755</td></tr><tr><td>Topk (k=5)</td><td>0.536</td><td>0.465</td><td>0.484</td><td>0.394</td><td>20.61</td><td>763</td></tr><tr><td>Topk (k=10)</td><td>0.547</td><td>0.441</td><td>0.464</td><td>0.421</td><td>29.11</td><td>752</td></tr><tr><td>Topk (k=30)</td><td>0.628</td><td>0.415</td><td>0.443</td><td>0.442</td><td>33.99</td><td>737</td></tr><tr><td>Topp (p=0.8)</td><td>0.046</td><td>0.559</td><td>0.549</td><td>0.379</td><td>35.17</td><td>995</td></tr><tr><td>Topp (p=0.9)</td><td>0.102</td><td>0.53</td><td>0.512</td><td>0.391</td><td>42.03</td><td>966</td></tr><tr><td>Topp (p=0.95)</td><td>0.114</td><td>0.508</td><td>0.518</td><td>0.399</td><td>30.99</td><td>959</td></tr><tr><td>RPG (Ours)</td><td>0.912</td><td>0.352</td><td>0.391</td><td>0.805</td><td>13.68</td><td>565</td></tr><tr><td colspan="7">Artificial Synthesis</td></tr><tr><td>Rep_Penalty</td><td>0.679</td><td>0.624</td><td>0.521</td><td>0.467</td><td>20.91</td><td>615</td></tr><tr><td>Rep_Dropout</td><td>0</td><td>0.713</td><td>0.597</td><td>0.188</td><td>32.89</td><td>1024</td></tr><tr><td>Greedy</td><td>0</td><td>0.807</td><td>0.789</td><td>0.459</td><td>40.45</td><td>1024</td></tr><tr><td>Temp (t=0.1)</td><td>0.016</td><td>0.798</td><td>0.785</td><td>0.452</td><td>40.16</td><td>1010</td></tr><tr><td>Temp (t=0.2)</td><td>0.018</td><td>0.798</td><td>0.768</td><td>0.438</td><td>39.95</td><td>982</td></tr><tr><td>Temp (t=0.8)</td><td>0.522</td><td>0.613</td><td>0.519</td><td>0.433</td><td>23.30</td><td>675</td></tr><tr><td>Topk (k=5)</td><td>0.503</td><td>0.659</td><td>0.558</td><td>0.45</td><td>24.35</td><td>723</td></tr><tr><td>Topk (k=10)</td><td>0.552</td><td>0.636</td><td>0.524</td><td>0.48</td><td>22.42</td><td>661</td></tr><tr><td>Topk (k=30)</td><td>0.561</td><td>0.628</td><td>0.521</td><td>0.475</td><td>22.00</td><td>653</td></tr><tr><td>Topp (p=0.8)</td><td>0.097</td><td>0.752</td><td>0.691</td><td>0.431</td><td>37.27</td><td>946</td></tr><tr><td>Topp (p=0.9)</td><td>0.146</td><td>0.724</td><td>0.658</td><td>0.501</td><td>35.82</td><td>915</td></tr><tr><td>Topp (p=0.95)</td><td>0.201</td><td>0.681</td><td>0.637</td><td>0.454</td><td>33.19</td><td>868</td></tr><tr><td></td><td>0.871</td><td>0.618</td><td>0.556</td><td>0.731</td><td>17.37</td><td></td></tr><tr><td>RPG (Ours)</td><td></td><td></td><td></td><td></td><td></td><td>489</td></tr></table>

<table><tr><td colspan="7">Real-world Repositories</td></tr><tr><td>Rep_Penalty</td><td>0.828</td><td>0.461</td><td>0.358</td><td>0.395</td><td>62.01</td><td>1753</td></tr><tr><td>Rep_Dropout</td><td>0</td><td>0.705</td><td>0.519</td><td>0.074</td><td>208.51</td><td>4096</td></tr><tr><td>Greedy</td><td>0</td><td>0.738</td><td>0.631</td><td>0.297</td><td>203.96</td><td>4096</td></tr><tr><td>Temp (t=0.1)</td><td>0.031</td><td>0.726</td><td>0.62</td><td>0.292</td><td>189.35</td><td>3997</td></tr><tr><td>Temp (t=0.2)</td><td>0.052</td><td>0.728</td><td>0.627</td><td>0.309</td><td>185.38</td><td>3944</td></tr><tr><td>Temp (t=0.8)</td><td>0.769</td><td>0.496</td><td>0.384</td><td>0.365</td><td>64.72</td><td>1926</td></tr><tr><td>Topk (k=5)</td><td>0.732</td><td>0.534</td><td>0.427</td><td>0.411</td><td>64.58</td><td>1937</td></tr><tr><td>Topk (k=10)</td><td>0.783</td><td>0.51</td><td>0.399</td><td>0.381</td><td>63.29</td><td>1911</td></tr><tr><td>Topk (k=30)</td><td>0.783</td><td>0.495</td><td>0.386</td><td>0.372</td><td>64.72</td><td>1940</td></tr><tr><td>Topp (p=0.8)</td><td>0.128</td><td>0.686</td><td>0.585</td><td>0.344</td><td>172.11</td><td>3728</td></tr><tr><td>Topp (p=0.9)</td><td>0.221</td><td>0.658</td><td>0.55</td><td>0.349</td><td>155.82</td><td>3443</td></tr><tr><td>Topp (p=0.95)</td><td>0.291</td><td>0.642</td><td>0.521</td><td>0.373</td><td>144.96</td><td>3270</td></tr><tr><td>RPG (Ours)</td><td>0.889</td><td>0.416</td><td>0.335</td><td>0.638</td><td>60.36</td><td>1415</td></tr></table>

## 6 Experimenal Results

We systematically evaluate our approach from two main aspects. First, regarding the effectiveness of mitigating repetition problem in code generation, we conduct multi-angle evaluations on CodeRepetEval dataset: 1) We compare the performance of our RPG approach with baselines in three scenarios: Code Generation Benchmarks, Artificial Synthesis, and Real-world Repositories; 2) We valid the effect of RPG on the base LLMs across different series and sizes; 3) We explore the generalizability of RPG across different programming languages (PLs). 4) We evaluate the impact of different values of hyperparameter λ on the performance of RPG. 5) We conduct a case study to qualitatively analyze the RPG's performance (Appendix A). Second, concerning the effectiveness of code generation, we evaluate the RPG’s performance compared to baselines on HumanEval(-ET) and MBPP(-ET) benchmarks.

## 6.1 Repetition Mitigation

Effectiveness of RPG. As illustrated in Tables 1, we assess our RPG approach along with various baselines on CodeRepetEval dataset. Our analysis of experimental results yields several insights: 1) Enhancing the model's confidence in its output tends to improve the likelihood of generating repetitive sequences. Specifically, when the hyperparameters for temperature, Topk, and Top-p sampling are set to lower values, the TR-N and TR-S metrics across the three scenarios of CodeRepetEval dataset show poorer performance, and the generation of EOS tokens becomes more challenging, thereby increasing the generation time and length of models. 2) The approaches for addressing content repetition in text generation are not applicable to the structural repetition problems in code generation. Although Repetition Penalty can reduce TR-N and TR-S to a certain extent, they achieve this at the expense of generated code quality. Its generated code usually terminates prematurely at a position where it should not end. Repetition Dropout masks the content repetitions for self-attention during training, but it has little effect on the structural repetitions. 3) RPG substantially outperforms other baselines on CodeRepetEval dataset, which effectively reduces repetitions and enhances the quality of the generated code. RPG achieves the best performance in terms of EGP and CCP metrics on CodeRepetEval dataset across three scenarios. For the TR-N and TR-S metrics, RPG also achieves the optimal performance except for Artificial Synthesis scenario. This may be attributed to the fact that although artificially synthesized prompts tend to induce repetitions, since these prompts are not inherently natural to the model, the model more readily escapes these repetitions when sampling from a smoother distribution.

Performance on different LLMs. We also conduct experiments for evaluating the performance of RPG based on different LLMs across three scenarios of CodeRepetEval dataset, as presented in Table 2. The experimental results indicate that RPG exhibits consistent and significant enhancements on different base models, highlighting the robustness of RPG for base model selections. Moreover, we find that training LLMs on code is likely to increase their susceptibility to structural repetitions during greedy sampling. Given that CodeLlama, although fine-tuned on Llama2 for coding tasks, demonstrates markedly worse performance in terms of structural repetitions in code generation.

Table 2: The impact of RPG using different base models on CodeRepetEval dataset across three scenarios, where AS, CGB, and RR donate Artificial Synthesis, Code Generation Benchmarks, and Real-world Repositories.
<table><tr><td rowspan="2">Approach</td><td colspan="2">AS</td><td colspan="2">CGB</td><td colspan="2">RR</td></tr><tr><td>TR-S↓</td><td>CCP↑</td><td>TR-S↓</td><td>CCP↑</td><td>TR-S↓</td><td>CCP↑</td></tr><tr><td colspan="7">CodeLlama</td></tr><tr><td>Greedy</td><td>0.789</td><td>0.459</td><td>0.637</td><td>0.455</td><td>0.631</td><td>0.297</td></tr><tr><td>RPG</td><td>0.613</td><td>0.731</td><td>0.435</td><td>0.805</td><td>0.379</td><td>0.638</td></tr><tr><td colspan="7">CodeGen</td></tr><tr><td>Greedy</td><td>0.684</td><td>0.541</td><td>0.657</td><td>0.509</td><td>0.657</td><td>0.369</td></tr><tr><td>RPG</td><td>0.437</td><td>0.841</td><td>0.417</td><td>0.867</td><td>0.375</td><td>0.720</td></tr><tr><td colspan="7">DeepSeek-Coder</td></tr><tr><td>Greedy</td><td>0.453</td><td>0.821</td><td>0.664</td><td>0.479</td><td>0.467</td><td>0.332</td></tr><tr><td>RPG</td><td>0.369</td><td>0.951</td><td>0.515</td><td>0.732</td><td>0.341</td><td>0.585</td></tr><tr><td colspan="7">Llama2</td></tr><tr><td>Greedy</td><td>0.528</td><td>0.758</td><td>0.508</td><td>0.597</td><td>0.542</td><td>0.477</td></tr><tr><td>RPG</td><td>0.422</td><td>0.917</td><td>0.427</td><td>0.912</td><td>0.383</td><td>0.764</td></tr></table>

![](images/3cdc8a6c560eab3c9b246455642629c143c7195cc3b50db6e58dc0b427b21b38.jpg)  
Figure 3: The performance of RPG applied to LLMs of different sizes. This result is the average value across three scenarios on CodeRepetEval dataset.

As shown in Figure 3, we observe that the repetition also occurs on LLMs larger than 7B, i,e., CodeLlama-34B, and RPG is effective for it as well, which demonstrates the same trends as other LLMs evaluated in Table 1 and Table 2, indicating the generalizability across LLMs of varying sizes.

Performance on different PLs. RPG can be applied to other PLs, requiring only their grammatical rules, which are readily obtainable from the web. To demonstrate the convenience of RPG, we have extended it to the PL of Go, and the experimental results are shown in Table 3. We can find that RPG still achieved substantial improvements in all five metrics for the PL of Go.

Table 3: The performance of RPG on CodeRepetEval-Go dataset.
<table><tr><td>Approach</td><td>EGP↑</td><td>TR-N↓</td><td>TR-S↓</td><td>CCP↑</td><td>Time ↓</td></tr><tr><td>Greedy</td><td>0.133</td><td>0.750</td><td>0.353</td><td>0.403</td><td>38.28</td></tr><tr><td>Temp (t=0.8)</td><td>0.601</td><td>0.554</td><td>0.231</td><td>0.396</td><td>26.78</td></tr><tr><td>RPG (Ours)</td><td>0.875</td><td>0.518</td><td>0.215</td><td>0.725</td><td>21.07</td></tr></table>

Influence of hyperparamter λ. In our experiments, we fix the hyperparameter λ intuitively for RPG. As shown in Figure 5 of Appendix B, we investigate the influence of varying λ empirically on all scenarios of CodeRepetEval dataset and HumanEval and MBPP benchmarks by changing itself. The results indicate that the reduction of the hyperparameter λ for RPG leads to a more pronounced suppression of repetition problems in code generation and decreases the sampling time. Furthermore, there is still room for further improvements with the better hyper-parameter setup of λ.

Case Study. In Figure 4 of Appendix A, we showcase two examples from Code Generation Benchmarks and Real-World Repository to conduct qualitative analysis. In these cases, the original model falls into a repetition trap, continuing until it exhausts its token budget. Case (a) features a repetition pattern: num = [i for i in num if i != NUM], where LLMs repeatedly generate statements that increment NUM. Case (b)'s repetition pattern involves a sequence of imports, where modules attention' and conv' are endlessly added. The code generated under these repetition patterns is utterly nonsensical and fails completely. However, our approach can break out of these loops, returning to a normal code generation trajectory, and ultimately succeeding in generating correct code.

## 6.2 Code Generation

In addition to CodeRepetEval dataset, we further validate the effectiveness of RPG on widely used code generation benchmarks, i.e., HumanEval(- ET) and MBPP(-ET), as presented in Table 4. The results demonstrate that RPG outperforms both standard decoding approaches and specialized approaches aimed at reducing content repetition. Although Repetition Penalty and Repetition Dropout forcibly reduce content repetition in generated code, they also significantly impair the performance of code generation. In contrast, RPG not only effectively eliminates both content and structural repetition, but also enhances the accuracy of code generation for LLMs, achieving relative improvements of up to 11.3% in Pass @ 1.

Table 4: The performance of RPG on code generation benchmarks HumanEval(-ET) and MBPP(-ET). Improvement represents the relative improvement of RPG compared to Greedy sampling.
<table><tr><td>Approach</td><td>HumanEval</td><td>HumanEval-ET</td><td>MBPP</td><td>MBPP-ET</td></tr><tr><td>Rep_Penalty</td><td>0.092</td><td>0.067</td><td>0.143</td><td>0.115</td></tr><tr><td>Rep_Dropout</td><td>0.139</td><td>0.116</td><td>0.167</td><td>0.141</td></tr><tr><td>Greedy</td><td>0.301</td><td>0.232</td><td>0.396</td><td>0.303</td></tr><tr><td>Temp (t=0.8)</td><td>0.226</td><td>0.183</td><td>0.305</td><td>0.218</td></tr><tr><td>RPG (Ours)</td><td>0.325</td><td>0.258</td><td>0.421</td><td>0.334</td></tr><tr><td>Improvement</td><td>↑8.0%</td><td>↑11.3%</td><td>↑6.4%</td><td>↑10.3%</td></tr></table>

## 7 Related Work

## 7.1 Code Generation

Since the advent of artificial intelligence in the 1950s, code generation has been considered the Holy Grail of computer science research (Gulwani et al., 2017). With the rapid expansion of codebases and the increasing capacity of deep learning models, using deep learning for program generation has shown great potential and practicality (Raychev et al., 2014; Ling et al., 2016; Wei et al., 2019; Sun et al., 2020; Mukherjee et al., 2021; Jiang et al., 2023; Dong et al., 2023a; Li et al., 2024b; Jiang et al., 2024; Li et al., 2024a; Zhang et al., 2024). In recent years, the rise of pre-training techniques has brought new momentum to the field of code generation. For example, studies like CodeT5 (Wang et al., 2021) and UniXcoder (Guo et al., 2022) pre-train models for code generation tasks. With the continual increase in model parameters, researchers have discovered emergent phenomena in LLMs, leading to new breakthroughs . Against this backdrop, LLMs such as AlphaCode (Li et al., 2022), Codex (Chen et al., 2021), CodeGeeX (Zheng et al. 2023), Starcoder (Li et al., 2023b), CodeLlama (Rozière et al., 2023), and DeepSeek Coder (Guo et al., 2024) have emerged.

Some work focuses on grammar-based code generation approaches (Yin et al., 2018; Sun et al., 2019; Jiang et al., 2021; Dong et al., 2023b), which primarily utilize learning or decoding based on grammar rulers to enhance the grammatical correctness of generated code. However, given that all structural repetitions adhere to the grammar, i.e., they are grammatically correct, merely using grammar rules during decoding or learning grammar rules during training is not applicable to the structural repetition problem. Therefore, these approaches fail to address this problem.

## 7.2 Repetition in Neural Text Generation.

Repetition problems in neural language models have drawn increasing attention, with various interpretations and proposed solutions emerging from recent research, especially in the field of text generation (Holtzman et al., 2020). Repetition Penalty (Holtzman et al., 2020) is a commonly used approach to reduce content repetition, which prevents words or phrases that have already appeared during the generation process from being generated again. However, there are lots of key tokens that appear frequently in code generation, such as ‘=', (', [' (Eghbali et al., 2022). Uniformly preventing these tokens in subsequent generations would be extremely detrimental to code generation.

Previous work (Fu et al., 2021b) points out that repetition is caused by the phenomenon of selfreinforcement. Some works address this problem during the training phase (Fu et al., 2021b; Xu et al., 2022; Su et al., 2022). Repetition dropout (Li et al., 2023a) finds a link between training data degradation and repetition, mitigating it by lowering attention to repeated words. However, compared to our RPG approach, these approaches have three primary disadvantages: 1) They require extensive training and necessitate the construction of a large amount of data for fine-tuning LLMs, which incurs substantial costs. 2) They usually hurt the code generation performance of models obviously. 3) They only focus on addressing content repetitions in text generation, without involving the prevalent issue of structural repetitions in code generation.

## 8 Conclusion

In this paper, we have formally defined structural repetition, which is the major repetition problem in code generation. We have proposed a novel decoding approach called RPG to alleviate repetition problems in code generation from grammar perspective for LLMs. By leveraging the grammar rules, RPG can recognize repetitions and strategically decay the output probability of critical tokens that contribute to repetitions. We also construct a new dataset CodeRepetEval, designed to provide a comprehensive evaluation for addressing repetition problems in code generation. Extensive experiments demonstrate the effectiveness and generalization of RPG in repetition mitigation of code generation. Through our work, we hope to shed light on this direction and call more attention to repetition problems in code generation.

## 9 Limitations

Our work has the following two main limitations.

First, RPG demands slightly more computational resources than sampling to detect the repetitions. However, compared to the enormous computational overhead of LLMs, it is marginal and acceptable.

Second, the potential reasons why LLMs induce structural repetitions in code generation remain unclear. Our current analysis has not touched on this aspect, which we leave for future work.

## 10 Acknowledgments

This research is supported by the National Key R&D Program under Grant No. 2023YFB4503801, the National Natural Science Foundation of China under Grant No. 62192733, 62192730, and the Major Program (JD) of Hubei Province (No.2023BAA024).

## References

V Aho Alfred, S Lam Monica, and D Ullman Jeffrey. 2007. Compilers Principles, Techniques & Tools. pearson Education.

Jacob Austin, Augustus Odena, Maxwell I. Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie J. Cai, Michael Terry, Quoc V. Le, and Charles Sutton. 2021. Program synthesis with large language models. CoRR, abs/2108.07732.

Massimo Caccia, Lucas Caccia, William Fedus, Hugo Larochelle, Joelle Pineau, and Laurent Charlin. 2019. Language gans falling short. In (ICLR).

Bei Chen, Fengji Zhang, Anh Nguyen, Daoguang Zan, Zeqi Lin, Jian-Guang Lou, and Weizhu Chen. 2023. CodeT: Code generation with generated tests. In ICLR.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé de Oliveira Pinto, Jared Kaplan, Harrison Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter,

Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating large language models trained on code. CoRR.

Yihong Dong, Jiazheng Ding, Xue Jiang, Ge Li, Zhuo Li, and Zhi Jin. 2024a. Codescore: Evaluating code generation by learning code execution. ACM TOSEM.

Yihong Dong, Xue Jiang, Zhi Jin, and Ge Li. 2023a. Self-collaboration code generation via chatgpt. CoRR, abs/2304.07590.

Yihong Dong, Xue Jiang, Huanyu Liu, Zhi Jin, Bin Gu, Mengfei Yang, and Ge Li. 2024b. Generalization or memorization: Data contamination and trustworthy evaluation for large language models. In Findings of the Association for Computational Linguistics: ACL 2024, pages 12039–12050, Bangkok, Thailand. Association for Computational Linguistics.

Yihong Dong, Ge Li, and Zhi Jin. 2023b. CODEP: grammatical seq2seq model for general-purpose code generation. In ISSTA, pages 188–198. ACM.

Aryaz Eghbali and Michael Pradel. 2022. Crystalbleu: Precisely and efficiently measuring the similarity of code. In ICSE-Companion, pages 341–342. ACM/IEEE.

Angela Fan, Mike Lewis, and Yann N. Dauphin. 2018. Hierarchical neural story generation. In ACL (1), pages 889–898. Association for Computational Linguistics.

Zihao Fu, Wai Lam, Anthony Man-Cho So, and Bei Shi. 2021a. A theoretical analysis of the repetition problem in text generation. In AAAI, pages 12848– 12856. AAAI Press.

Zihao Fu, Wai Lam, Anthony Man-Cho So, and Bei Shi. 2021b. A theoretical analysis of the repetition problem in text generation. In AAAI, pages 12848– 12856. AAAI Press.

Sumit Gulwani, Oleksandr Polozov, Rishabh Singh et al. 2017. Program synthesis. Foundations and Trends® in Programming Languages, 4(1-2):1–119.

Daya Guo, Shuai Lu, Nan Duan, Yanlin Wang, Ming Zhou, and Jian Yin. 2022. Unixcoder: Unified crossmodal pre-training for code representation. In ACL (1), pages 7212–7225. Association for Computational Linguistics.

Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y. Wu, Y. K. Li, Fuli Luo, Yingfei Xiong, and Wenfeng Liang. 2024. Deepseek-coder: When the large language model meets programming - the rise of code intelligence. CoRR, abs/2401.14196.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2020. The curious case of neural text degeneration. In ICLR. OpenReview.net.

Xue Jiang, Yihong Dong, Yongding Tao, Huanyu Liu, Zhi Jin, Wenpin Jiao, and Ge Li. 2024. ROCODE: integrating backtracking mechanism and program analysis in large language models for code generation. CoRR, abs/2411.07112.

Xue Jiang, Yihong Dong, Lecheng Wang, Qiwei Shang, and Ge Li. 2023. Self-planning code generation with large language model. CoRR, abs/2303.06689.

Xue Jiang, Zhuoran Zheng, Chen Lyu, Liang Li, and Lei Lyu. 2021. Treebert: A tree-based pre-trained model for programming language. In UAI, volume 161 of Proceedings of Machine Learning Research, pages 54–63. AUAI Press.

Nitish Shirish Keskar, Bryan McCann, Lav R. Varshney, Caiming Xiong, and Richard Socher. 2019. CTRL: A conditional transformer language model for controllable generation. CoRR, abs/1909.05858.

Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven Chu-Hong Hoi. 2022. Coderl: Mastering code generation through pretrained models and deep reinforcement learning. In NeurIPS.

Huayang Li, Tian Lan, Zihao Fu, Deng Cai, Lemao Liu, Nigel Collier, Taro Watanabe, and Yixuan Su. 2023a. Repetition in repetition out: Towards understanding neural text degeneration from the data perspective. In NeurIPS.

Jia Li, Ge Li, Xuanming Zhang, Yunfei Zhao, Yihong Dong, Zhi Jin, Binhua Li, Fei Huang, and Yongbin Li. 2024a. Evocodebench: An evolving code generation benchmark with domain-specific evaluations. In NeurIPS.

Jia Li, Ge Li, Yunfei Zhao, Yongmin Li, Huanyu Liu, Hao Zhu, Lecheng Wang, Kaibo Liu, Zheng Fang, Lanshen Wang, Jiazheng Ding, Xuanming Zhang, Yuqi Zhu, Yihong Dong, Zhi Jin, Binhua Li, Fei Huang, Yongbin Li, Bin Gu, and Mengfei Yang. 2024b. Deveval: A manually-annotated code generation benchmark aligned with real-world code repositories. In ACL (Findings), pages 3603–3614. Association for Computational Linguistics.

Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Mishig Davaadorj, Joel Lamy-Poirier, João Monteiro, Oleh Shliazhko, Nicolas Gontier, Nicholas Meade, Armel

Zebaze, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Benjamin Lipkin, Muhtasham Oblokulov, Zhiruo Wang, Rudra Murthy V, Jason Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca, Manan Dey, Zhihan Zhang, Nour Moustafa-Fahmy, Urvashi Bhattacharyya, Wenhao Yu, Swayam Singh, Sasha Luccioni, Paulo Villegas, Maxim Kunakov, Fedor Zhdanov, Manuel Romero, Tony Lee, Nadav Timor, Jennifer Ding, Claire Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra, Alex Gu, Jennifer Robinson, Carolyn Jane Anderson, Brendan Dolan-Gavitt, Danish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Muñoz Ferrandis, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries. 2023b. Starcoder: may the source be with you! CoRR, abs/2305.06161.

Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. 2022. Competition-level code generation with alphacode. Science, 378(6624):1092–1097.

Wang Ling, Phil Blunsom, Edward Grefenstette, Karl Moritz Hermann, Tomás Kociský, Fumin Wang, and Andrew W. Senior. 2016. Latent predictor networks for code generation. In ACL (1). The Association for Computer Linguistics.

Fang Liu, Yang Liu, Lin Shi, Houkun Huang, Ruifeng Wang, Zhen Yang, and Li Zhang. 2024. Exploring and evaluating hallucinations in llm-powered code generation. CoRR, abs/2404.00971.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. 2023. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. In NeurIPS

Rohan Mukherjee, Yeming Wen, Dipak Chaudhari, Thomas W. Reps, Swarat Chaudhuri, and Christopher M. Jermaine. 2021. Neural program generation modulo static analysis. In NeurIPS, pages 18984 18996.

Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. 2023. Codegen: An open large language model for code with multi-turn program synthesis. In ICLR. OpenReview.net.

OpenAI. 2022. ChatGPT.

OpenAI. 2023. GPT-4 technical report. CoRR, abs/2303.08774.

Veselin Raychev, Martin T. Vechev, and Eran Yahav. 2014. Code completion with statistical language models. In PLDI, pages 419–428. ACM.

Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton-Ferrer, Aaron Grattafiori,

Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Synnaeve. 2023. Code llama: Open foundation models for code. CoRR, abs/2308.12950.

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural machine translation of rare words with subword units. In ACL (1). The Association for Computer Linguistics.

Yixuan Su, Tian Lan, Yan Wang, Dani Yogatama, Lingpeng Kong, and Nigel Collier. 2022. A contrastive framework for neural text generation. In NeurIPS.

Zeyu Sun, Qihao Zhu, Lili Mou, Yingfei Xiong, Ge Li, and Lu Zhang. 2019. A grammar-based structural CNN decoder for code generation. In AAAI, pages 7055–7062. AAAI Press.

Zeyu Sun, Qihao Zhu, Yingfei Xiong, Yican Sun, Lili Mou, and Lu Zhang. 2020. Treegen: A tree-based transformer architecture for code generation. In AAAI, pages 8984–8991. AAAI Press.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. Self-instruct: Aligning language models with self-generated instructions. In ACL (1), pages 13484–13508. Association for Computational Linguistics.

Yue Wang, Weishi Wang, Shafiq R. Joty, and teven C. H. Hoi. 2021. Codet5: Identifier-aware unified pre-trained encoder-decoder models for code understanding and generation. In EMNLP (1), pages 8696– 8708.

Bolin Wei, Ge Li, Xin Xia, Zhiyi Fu, and Zhi Jin. 2019. Code generation as a dual task of code summarization. In NeurIPS, pages 6559–6569.

Jin Xu, Xiaojiang Liu, Jianhao Yan, Deng Cai, Huayang Li, and Jian Li. 2022. Learning to break the loop: Analyzing and mitigating repetitions for neural text generation. In NeurIPS.

Pengcheng Yin and Graham Neubig. 2018. TRANX: A transition-based neural abstract syntax parser for semantic parsing and code generation. In EMNLP (Demonstration), pages 7–12. Association for Computational Linguistics.

Kechi Zhang, Ge Li, Yihong Dong, Jingjing Xu, Jun Zhang, Jing Su, Yongfei Liu, and Zhi Jin. 2024. Codedpo: Aligning code models with self generated and verified source code. CoRR, abs/2410.05605.

Qinkai Zheng, Xiao Xia, Xu Zou, Yuxiao Dong, Shan Wang, Yufei Xue, Zihan Wang, Lei Shen, Andi Wang, Yang Li, Teng Su, Zhilin Yang, and Jie Tang. 2023. Codegeex: A pre-trained model for code generation with multilingual evaluations on humaneval-x. CoRR, abs/2303.17568.

## A Case Study

![](images/8bab2a9275c4cd8c34a05018711b8a1fa25f316b9dbd4278fedf248b150b5aa5.jpg)  
(a) A case from HumanEval in Code Generation Benchmark.

![](images/dbc5e346e063193c38b39d638e331141313418869268c849e7246a3731490ba2.jpg)  
(b) A case from Real-world Repository  
Figure 4: Two cases of generating structural repetition and the effect of our approach on them. LLMs succumb to endless loops of repetition. Our proposed approach can effectively break out of these loops, steering back to a normal code generation trajectory, and ultimately succeeding in producing correct code.

## B The Influence of Hyper-parameters λ

We shown the influence of hyper-parameters λ in Figure 5.

(a)  
![](images/2c80937006e6503d6e5d1343d48ad9d992fd15dd32448f378e02ab8f37b391d0.jpg)

![](images/5a555798b11b3c7fb0f325c82334221ae0953adbee4587667d930feded3c1a1f.jpg)  
(b)

![](images/293dc5514ac8f85c703315eed52922411b89b9ad4a3c45904e85fb1603ae9975.jpg)  
(c)

![](images/6299f2be8aaaa6746f289208af29462d6641dd3f605631be9fc8ef63c8761579.jpg)  
(d)

![](images/4ce9c373106e107609966a365f60a95cf6aea155500ce0ebfdb53d7000ad5953.jpg)  
(e)

![](images/863f26765725bbc03e9e8248bec9ddb974a3104fe1f8e5266a63826983b7a58a.jpg)  
(f)  
Figure 5: The influence of hyper-parameters λ on Artificial Synthesis, Code Generation Benchmarks, and Realworld Repositories scenarios of CodeRepetEval dataset, as well as HumanEval and MBPP benchmarks. We use the gray dashed line to represent the employed hyper-parameters.

## C Details of Experiment Setup

## C.1 Dataset Construction and Evaluation Methodology of CodeRepetEval

The purpose of CodeRepetEval datasets is to evaluate the performance of different approaches for avoiding repetitions in code generation under various scenarios. We provide a detailed description of the dataset construction process and the evaluation methodology as follows

The construction processes for three scenarios, i.e., Artificial Synthesis, Code Generation Benchmarks, and Real-world Repositories, are similar, with mere differences in the source of code samples. For Artificial Synthesis, we employ Self-instruct (Wang et al., 2023) to construct instruction sets that induce CodeLlama-7B to generate code samples containing repetitions. For Code Generation Benchmarks, we use CodeLlama-7B to sample codes on HumanEval and MBPP benchmarks. For Real-world Repositories, we select code from 100 high-quality open-source projects on GitHub, following the selection criteria of Starcoder (Li et al., 2023b).

After obtaining the code samples, we employ the TR-S metric to sort them in descending order based on the degree of repetition. We then select the repetitive code segments from the top 2\*N samples, where N equals 512, 512, and 1024 for the respective scenarios. Subsequently, we randomly truncate the code samples at positions containing 2 to 4 repeated statements, retain the first half, and combine it with the original input to generate new prompts. Finally, the researchers conduct a manual evaluation to filter out N prompts, which are then incorporated into our datasets.

For the evaluation, we employ the constructed prompts as input and generate code using different approaches. We apply three metrics—TR-S, TR-N, and EGP—to determine the extent to which the code generated by these approaches continues to manifest repetition issues. Moreover, we apply the CCP metric to assess the compilability and quality of generated codes.

## C.2 Code Generation Benchmarks

We evaluate our approach using four public code generation benchmarks.

HumanEval (Chen et al., 2021) consists of 164 handwritten programming tasks, proposed by OpenAI. Each task includes a function signature, NL description, use cases, function body, and several unit tests (average 7.7 per task).

MBPP (Austin et al., 2021) contains 974 manually verified Python programming tasks, covering programming fundamentals, standard library functionality, and more. Each task consists of an NL description, a code solution, and 3 automated test cases.

HumanEval-ET and MBPP-ET (Dong et al., 2024a) are expanded versions of MBPP and HumanEval respectively, where each includes over 100 additional test cases per task. This updated version enhances the soundness of code evaluation compared to the original benchmarks.

## C.3 Baselines

We detail the baseline approaches compared in this work:

• Greedy Sampling chooses the highest probability token from the model's output distribution at each time step, $P _ { \mathrm { g r e e d y } } ^ { \prime } ( w ) = \mathbb { 1 } ( w = \arg \operatorname* { m a x } _ { w } P ( w | w _ { < t } ) )$

• Topk Sampling (Fan et al., 2018) limits the next-word selection to the top k most likely candidates as determined by the model, $P _ { \mathrm { t o p k } } ^ { \prime } ( w ) = P ( w | w _ { < t } )$ if $w \in$ Topk, otherwise 0.

• Topp Sampling (Holtzman et al., 2020) involves choosing from a smaller set of plausible candidates by dynamically selecting a variable-sized subset of tokens (the "nucleus") that cumulatively make up a certain probability mass (e.g., top 90%), $\begin{array} { r } { P _ { \mathrm { t o p p } } ^ { \prime } ( w ) = P ( w | w _ { < t } ) \mathrm { i f } \sum _ { w ^ { \prime } \in S } P ( w ^ { \prime } | w _ { < t } ) \leq } \end{array}$ p, otherwise 0.

• Temperature Sampling (Caccia et al., 2019) controls the randomness of the token selection process—higher temperatures lead to more uniform distributions, while lower temperatures make high-probability tokens even more likely, $\begin{array} { r } { P _ { \mathrm { t e m p } } ^ { \prime } ( w ) = \frac { \exp \left( \log \left( P ( w | w < t ) \right) / T \right) } { \sum _ { w ^ { \prime } } \exp \left( \log \left( P ( w ^ { \prime } | w < t ) \right) / T \right) } } \end{array}$

• Repetition Penalty (Keskar et al., 2019) penalizes sampling works by discounting the scores of previously generated tokens, $\begin{array} { r l r } { P ^ { \prime } ( w ) } & { = } & { \frac { \exp \left( \log \left( P \left( w | w < t \right) \right) / \left( T \cdot I ( i \in g ) \right) \right. } { \sum _ { w ^ { \prime } } \exp \left( \log \left( P \left( w ^ { \prime } | w < t \right) \right) / \left( T \cdot I ( i \in g ) \right) \right. } } \end{array}$ , where $I ( c )$ 二 θ if c is True else 1. Unless otherwise specified, "the settings of baselines follow their original paper.

• Repetition Dropout (Li et al., 2023a) applies masking vectors to sentences, randomly dropping out repetitive n-grams based on a pre-specified dropout rate, thereby preventing the model from over-relying on repetitive patterns during training.

## C.4 Metrics

Compiler Correctness Percentage (CCP). CCP is defined as the ratio of the number of code samples that pass compilation to the total number of samples in the dataset.

Pass@k. We use the unbiased version (Chen et al., 2021) of Pass@k, where $n > = k$ samples are generated for each problem, count the number of correct samples $c < = n$ which pass test cases and calculate the following estimator, i.e.,

$$
\mathrm { P a s s @ k } = \mathop { \mathbb { E } } _ { \mathrm { P r o b l e m s } } \left[ 1 - \frac { \binom { n - c } { k } } { \binom { n } { k } } \right] .\tag{9}
$$

## D Pseudo-code of Suffix Array and LCP Array

```python
def constructSuffixArray(S):
n = length of S
Create an array suffixes[n] where each element is a tuple (index, suffix)
for i from 0 to n-1:
suffixes[i] = (i, S[i:n]) // Store index and suffix starting at index
i
Sort suffixes based on the suffix part of each tuple
Initialize SA[n]
for i from 0 to n-1:
SA[i] = suffixes[i].index
return SA
def constructLCPArray(S, SA):
n = length of S
Initialize LCP[n] with zeros
Initialize rank[n] to store the rank of each suffix in SA
for i from 0 to n-1:
rank[SA[i]] = i
h = 0 // Length of the longest common prefix
for i from 0 to n-1:
if rank[i] > 0:
j = SA[rank[i] - 1] // Index of the previous suffix in the sorted
list
while i + h < n and j + h < n and S[i + h] == S[j + h]:
h += 1
LCP[rank[i]] = h
if h > 0:
h -= 1 // Decrease h for the next calculation
return LCP
def findConsecutiveRepetitions(S):
SA = constructSuffixArray(S)
LCP = constructLCPArray(S, SA)
repetitions = set()
for i from 1 to length of S - 1:
if LCP[i] > 0:
duplicate_substring = S[SA[i]:SA[i] + LCP[i]]
# Check for consecutive occurrence
previous_suffix_length = SA[i-1]
current_suffix_length = SA[i]
if previous_start == current_start + LCP[i]:
repetitions.add(duplicate_substring)
return repetitions
```

## E PDA for LLMs

LLMs usually employ Byte-Pair Encoding (BPE) (Sennrich et al., 2016) for tokenization, which causes the tokens in the vocabulary of LLMs to deviate from the terminal symbols in grammar. Specifically, this discrepancy manifests in three primary scenarios, i.e., one token corresponds to one terminal symbol, one

token corresponds to multiple terminal symbols, and multiple tokens correspond to one terminal symbol. Therefore, to effectively utilize PDA with LLMs, it is essential to develop an approach that adapts PDA operations to accommodate these tokenization scenarios.

## E.1 One Token to One Terminal Symbol

In scenarios where one token directly corresponds to one terminal symbol, the adaptation of PDA is relatively straightforward. The PDA can process each token as a single unit that matches exactly one terminal symbol in the grammar of the language being parsed. Here, the transition functions of PDA can be directly applied without modification. For instance, if a token from the LLM's output matches a terminal symbol in a programming language's grammar, the PDA can push, pop, or transition based on this token following standard PDA rules. This case represents the simplest form of interaction between LLM outputs and grammar-based parsing.

## E.2 One Token to Multiple Terminal Symbols

This scenario arises when a single token encapsulates multiple grammatical elements, due to a compact or compressed representation of the language. For example, “[]", “),", “)\n", and so on. To handle this, we should first decompose the token into its constituent terminal symbols. Then, we sequentially check whether each terminal symbol is in the PDA candidate set of its prefixes. Finally, the tokens that all constitute terminal symbols satisfying the condition are retained. This approach ensures that even complex tokens can be seamlessly integrated into the grammar-based processing framework of the PDA.

## E.3 Multiple Tokens to One Terminal Symbol

In contrast, when multiple tokens collectively represent a single terminal symbol, we should aggregate these tokens before using the transition function of PDA. This scenario typically occurs with the tokentypes in terminal symbols, such as NAME, NUMBER, and STRING. In this case, we constructed a lexical grammar PDA to accumulate tokens until a complete terminal symbol is formed. The PDA operations then proceed based on these aggregated terminal symbols.

## F Full Grammar specification

For example, the full Python grammar is shown as follows:

```makefile
# Grammar for Python
# NOTE WELL: You should also follow all the steps listed at
# https://devguide.python.org/grammar/
# Start symbols for the grammar:
# single_input is a single interactive statement;
# file_input is a module or sequence of commands read from an input file
# eval_input is the input for the eval() functions.
# func_type_input is a PEP 484 Python 2 function type comment
# NB: compound_stmt in single_input is followed by extra NEWLINE!
# NB: due to the way TYPE_COMMENT is tokenized it will always be followed by
a NEWLINE
single_input: NEWLINE | simple_stmt I compound_stmt NEWLINE
file_input: (NEWLINE | stmt)* ENDMARKER
eval_input: testlist NEWLINE* ENDMARKER
decorator: '@' dotted_name [ '('[arglist]')'] NEWLINE
```

```python
decorators: decorator+
decorated: decorators (classdef | funcdef | async_funcdef)
async_funcdef: ASYNC funcdef
funcdef: 'def'NAME parameters ['->' test]':'[TYPE_COMMENT] func_body_suite
parameters:'('[typedargslist] ')'
# The following definition for typedarglist is equivalent to this set of
rules:
#
# arguments = argument (','[TYPE_COMMENT] argument)*
# argument = tfpdef ['=' test]
# kwargs = '**' tfpdef [','] [TYPE_COMMENT]
# args = '*'[tfpdef]
# kwonly_kwargs = (','[TYPE_COMMENT] argument)* (TYPE_COMMENT | [','
TYPE_COMMENT] [kwargs]])
# args_kwonly_kwargs = args kwonly_kwargs | kwargs
# poskeyword_args_kwonly_kwargs = arguments ( TYPE_COMMENT | [','[
TYPE_COMMENT] [args_kwonly_kwargs]])
# typedargslist_no_posonly = poskeyword_args_kwonly_kwargs |
args_kwonly_kwargs
# typedarglist = (arguments ','[TYPE_COMMENT] '/'[','[[TYPE_COMMENT]
typedargslist_no_posonly]])l(typedargslist_no_posonly)"
#
# It needs to be fully expanded to allow our LL(1) parser to work on it.
typedargslist:(
(tfpdef ['='test] (','[TYPE_COMMENT] tfpdef ['=' test])* ','[
TYPE_COMMENT] '/'[','[ [TYPE_COMMENT] tfpdef ['=' test] (
','[TYPE_COMMENT] tfpdef ['=' test])* (TYPE_COMMENT I [','[
TYPE_COMMENT] [
'*'[tfpdef] (','[TYPE_COMMENT] tfpdef ['='test])* (TYPE_COMMENT I
[','[TYPE_COMMENT] ['**'tfpdef [','] [TYPE_COMMENT]]])
'**'tfpdef [','] [TYPE_COMMENT]]])
I '*'[tfpdef] (',' [TYPE_COMMENT] tfpdef ['=' test])* (TYPE_COMMENT I [','
[TYPE_COMMENT] ['**'tfpdef [','] [TYPE_COMMENT]]])
I '**' tfpdef [','] [TYPE_COMMENT]]] )
| (tfpdef ['='test] (','[TYPE_COMMENT] tfpdef ['='test])* (TYPE_COMMENT |
[','[TYPE_COMMENT] [
'*'[tfpdef] (','[TYPE_COMMENT] tfpdef ['='test])* (TYPE_COMMENT | [',
[TYPE_COMMENT] ['**'tfpdef [','] [TYPE_COMMENT]]])
'**' tfpdef [','] [TYPE_COMMENT]]])
I '*'[tfpdef] (',' [TYPE_COMMENT] tfpdef ['=' test])* (TYPE_COMMENT I [','
[TYPE_COMMENT] ['**'tfpdef [','] [TYPE_COMMENT]]])
'**'tfpdef [','] [TYPE_COMMENT])
tfpdef: NAME [':' test]
```

# The following definition for varargslist is equivalent to this set of rules:   
#   
# arguments = argument (',' argument )\*   
# argument = vfpdef ['=' test]   
# kwargs = '\*\*' vfpdef [',']   
# args = '\*'[vfpdef]   
# kwonly\_kwargs = (',' argument )\* [','[kwargs]]   
# args\_kwonly\_kwargs = args kwonly\_kwargs | kwargs   
# poskeyword\_args\_kwonly\_kwargs = arguments [','[args\_kwonly\_kwargs]]   
# vararglist\_no\_posonly = poskeyword\_args\_kwonly\_kwargs I   
args\_kwonly\_kwargs   
# varargslist = arguments ',' '/'[','[(vararglist\_no\_posonly)]] I(   
vararglist\_no\_posonly)   
#   
# It needs to be fully expanded to allow our LL(1) parser to work on it.   
varargslist: vfpdef ['=' test ](',' vfpdef ['=' test])\* ',' '/' [','[ (   
vfpdef ['=' test] (',' vfpdef ['='test])\* [','[   
'\*'[vfpdef] (',' vfpdef ['=' test])\* [',' ['\*\*' vfpdef [',']]]   
'\*\*' vfpdef [',']]]   
| '\*'[vfpdef] (',' vfpdef ['='test])\* [','['\*\*' vfpdef [',']]]   
| '\*\*' vfpdef [',']) ]] | (vfpdef ['=' test] (',' vfpdef ['='test])\* [',   
'\*'[vfpdef] (','vfpdef ['=' test])\* [','['\*\*' vfpdef [',']]]   
'\*\*' vfpdef [',']]]   
| '\*'[vfpdef] (',' vfpdef ['='test])\* [','['\*\*' vfpdef [',']]]   
'\*\*' vfpdef [',']   
)   
vfpdef: NAME   
stmt: simple\_stmt | compound\_stmt   
simple\_stmt: small\_stmt (';' small\_stmt)\* [';'] NEWLINE   
small\_stmt: (expr\_stmt | del\_stmt | pass\_stmt | flow\_stmt |   
import\_stmt | global\_stmt | nonlocal\_stmt | assert\_stmt)   
expr\_stmt: testlist\_star\_expr (annassign | augassign (yield\_expr|testlist) |   
[('=' (yield\_expr|testlist\_star\_expr))+ [TYPE\_COMMENT]] )   
annassign: ':'test ['='(yield\_expr|testlist\_star\_expr)]   
testlist\_star\_expr: (test|star\_expr) (','(test|star\_expr))\* [',']   
augassign: ('+='|'-='|'\*='| '@='|'/='|'%='| '&='| '|='|'^='|   
'<<=′|'>>='|'\*\*='|//=')   
# For normal and annotated assignments, additional restrictions enforced by   
the interpreter   
del\_stmt: 'del' exprlist   
pass\_stmt: 'pass'   
flow\_stmt: break\_stmt | continue\_stmt | return\_stmt | raise\_stmt | yield\_stmt   
break\_stmt: 'break'   
continue\_stmt: 'continue'   
return\_stmt: 'return'[testlist\_star\_expr]   
yield\_stmt: yield\_expr

```python
raise_stmt: 'raise'[test ['from'test]]
import_stmt: import_name | import_from
import_name:'import'dotted_as_names
# note below: the ('.' | '...') is necessary because '...' is tokenized as
ELLIPSIS
import_from: ('from' (('.'  '...')* dotted_name I ('.'  '...')+)
'import'('*' '(' import_as_names ')' | import_as_names))
import_as_name: NAME ['as'NAME]
dotted_as_name: dotted_name ['as'NAME]
import_as_names: import_as_name (',' import_as_name)* [',']
dotted_as_names: dotted_as_name (',' dotted_as_name)*
dotted_name: NAME ('.' NAME)*
global_stmt: 'global'NAME (','NAME)*
nonlocal_stmt: 'nonlocal'NAME (',' NAME)*
assert_stmt: 'assert' test [',' test]
compound_stmt: if_stmt I while_stmt | for_stmt I try_stmt I with_stmt I
funcdef | classdef | decorated | async_stmt
async_stmt: ASYNC (funcdef | with_stmt | for_stmt)
if_stmt: 'if'namedexpr_test ':'suite ('elif'namedexpr_test ':'suite)* ['
else'':' suite]
while_stmt: 'while' namedexpr_test ':' suite ['else'':' suite]
for_stmt: 'for' exprlist 'in' testlist ':'[TYPE_COMMENT] suite ['else'':'
suite]
try_stmt: ('try' ':' suite
((except_clause ':' suite)+
['else' ':' suite]
['finally'':' suite] |
'finally'':' suite))
with_stmt: 'with' with_item (',' with_item)* ':' [TYPE_COMMENT] suite
with_item: test ['as' expr]
# NB compile.c makes sure that the default except clause is last
except_clause: 'except'[test ['as'NAME]]
suite: simple_stmt | NEWLINE INDENT stmt+ DEDENT
namedexpr_test: test [':=' test]
test: or_test ['if' or_test 'else' test] | lambdef
test_nocond: or_test | lambdef_nocond
lambdef: 'lambda'[varargslist] ':' test
lambdef_nocond:'lambda'[varargslist]':'test_nocond
or_test: and_test ('or'and_test)*
and_test: not_test ('and'not_test)*
not_test: 'not' not_test I comparison
comparison: expr (comp_op expr)*
# <> isn't actually a valid comparison operator in Python. It's here for the
# sake of a __future__ import described in PEP 401 (which really works :-)
comp_op: '<'|'>'|'=='|'>='|'<='|'<>'|'!='|'in'|'not' 'in'|'is'|'is' 'not'
star_expr: '*' expr
expr: xor_expr ('|' xor_expr)*
xor_expr: and_expr ('^' and_expr)*
```

and\_expr: shift\_expr ('&' shift\_expr)\*   
shift\_expr: arith\_expr (('<<'|'>>') arith\_expr)\*   
arith\_expr: term (('+'|'-') term)\*   
term: factor (('\*'|'@'|'/'|'%'|'//') factor)\*   
factor: ('+'|'-'|'\~') factor | power   
power: atom\_expr ['\*\*'factor]   
atom\_expr: [AWAIT] atom trailer\*   
atom: ('('[yield\_expr|testlist\_comp] ')'|   
'['[testlist\_comp]']'|   
'{'[dictorsetmaker] '}'I   
NAME | NUMBER | STRING+ | '...'  'None' | 'True' I 'False')   
testlist\_comp: (namedexpr\_test|star\_expr) ( comp\_for  (','(namedexpr\_testI   
star\_expr))\* [','] )   
trailer: '(' [arglist] ')'| '[' subscriptlist ']'|'.' NAME   
subscriptlist: subscript (',' subscript)\* [',']   
subscript: test | [test] ':'[test] [sliceop]   
sliceop: ':'[test]   
exprlist: (expr|star\_expr) (','(expr|star\_expr))\* [',']   
testlist: test (',' test)\* [',']   
dictorsetmaker: ( ((test ':' test | '\*\*' expr)   
(comp\_for | (','(test ':' test | '\*\*' expr))\* [','])) I   
((test | star\_expr)   
(comp\_for | (','(test | star\_expr))\* [','])) )   
classdef: 'class'NAME ['('[arglist] ')']':' suite   
arglist: argument (',' argument)\* [',']   
# The reason that keywords are test nodes instead of NAME is that using NAME   
# results in an ambiguity. ast.c makes sure it's a NAME.   
# "test '=' test" is really "keyword '=' test", but we have no such token.   
# These need to be in a single rule to avoid grammar that is ambiguous   
# to our LL(1) parser. Even though 'test' includes '\*expr' in star\_expr,   
# we explicitly match '\*'here, too, to give it proper precedence.   
# Illegal combinations and orderings are blocked in ast.c:   
# multiple (test comp\_for) arguments are blocked; keyword unpackings   
# that precede iterable unpackings are blocked; etc.   
argument:( test [comp\_for]I   
test ':=' test |   
test '=' test I   
'\*\*' test |   
'\*' test )   
comp\_iter: comp\_for | comp\_if   
sync\_comp\_for: 'for' exprlist 'in' or\_test [comp\_iter]   
comp\_for: [ASYNC] sync\_comp\_for   
comp\_if:'if' test\_nocond [comp\_iter]   
# not used in grammar, but may appear in "node" passed from Parser to   
Compiler

```python
encoding_decl: NAME
yield_expr: 'yield'[yield_arg]
yield_arg: 'from'test I testlist_star_expr
# the TYPE_COMMENT in suites is only parsed for funcdefs,
# but can't go elsewhere due to ambiguity
func_body_suite: simple_stmt I NEWLINE [TYPE_COMMENT NEWLINE] INDENT stmt+
DEDENT
func_type_input: func_type NEWLINE* ENDMARKER
func_type: '('[typelist] ')''->' test
# typelist is a modified typedargslist (see above)
typelist: (test (',' test)* [','
['*' [test] (',' test)* [',' '**' test] | '**' test]]
'*'[test] (',' test)* [',' '**' test] | '**' test)
```