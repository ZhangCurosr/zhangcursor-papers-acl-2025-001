# Algorithmic Fidelity of Large Language Models in Generating Synthetic German Public Opinions: A Case Study

Bolei Ma\*1,2 Berk Yoztyurk\*1 Anna-Carolina Haensch1,2,3 Xinpeng Wang1,2 Markus Herklotz1 Frauke Kreuter1,2,3 Barbara Plank1,2 Matthias Assenmacher1,2

1LMU Munich, 2Munich Center for Machine Learning,

3University of Maryland, College Park

Equal contribution.

Correspondence: bolei.ma@lmu.de

## Abstract

In recent research, large language models (LLMs) have been increasingly used to investigate public opinions. This study investigates the algorithmic fidelity of LLMs, i.e., the ability to replicate the socio-cultural context and nuanced opinions of human participants. Using open-ended survey data from the German Longitudinal Election Studies (GLES), we prompt different LLMs to generate synthetic public opinions reflective of German subpopulations by incorporating demographic features into the persona prompts. Our results show that Llama performs better than other LLMs at representing subpopulations, particularly when there is lower opinion diversity within those groups. Our findings further reveal that the LLM performs better for supporters of left-leaning parties like The Greens and The Left compared to other parties, and matches the least with the right-party AfD. Additionally, the inclusion or exclusion of specific variables in the prompts can significantly impact the models' predictions. These findings underscore the importance of aligning LLMs to more effectively model diverse public opinions while minimizing political biases and enhancing robustness in representativeness.¹

## 1 Introduction

Recent advances in LLMs have generated significant interest in their potential for synthetic data generation across various domains. A key and widely debated question is whether LLMs can produce synthetic data that accurately represent human opinions (Argyle et al., 2023; Santurkar et al., 2023; Veselovsky et al., 2023; von der Heyde et al., 2025; Long et al., 2024, inter alia).

In social science research, it is usually surveys that provide insights into the attitudes and opinions of a population. Recent studies have explored using information from survey responses for LLM prompts, i.e., creating so-called personas and then using the LLM “responses" to additional questions (Argyle et al., 2023; Dominguez-Olmedo et al., 2024; Durmus et al., 2024, inter alia). Assessing the fidelity of LLMs in capturing and reproducing human opinions deepens our understanding of model behavior while at the same time helping researchers design more reliable models aligned with human values and improving their usability (Ma et al., 2024). Among these studies, Argyle et al. (2023) introduced the algorithmic fdelity, a concept for assessing how effectively LLMs replicate the socio-cultural context and nuanced opinions of diverse human subpopulations. Their work used

![](images/dfc99a01fc53a033e23d0135992273e050f478a5afeefe9db6adb2d1118e352f.jpg)  
Figure 1: The distribution of the top 5 answer categories between November 2019 and November 2021 in the German GLES survey. There is a significant surge in the Health Policy category from November 2019 to April 2020, with it becoming the dominant focus during this period and afterwards, likely due to the outbreak of COVID-19 in early 2020.

LLMs to generate synthetic responses (“silicon samples") informed by demographic and ideological profiles from political science datasets. Their findings suggest that LLMs can closely approximate real-world opinion distributions in certain contexts, such as U.S. elections, although challenges persist in fully aligning the generated data with actual demographic groups.

Recent research on LLM responses to opinion polling have predominantly focused on USbased and English-centric survey data. For example, von der Heyde et al. (2025) evaluated the algorithmic fidelity of GPT-3.5 in predicting German voting behavior. Their findings revealed that while GPT-3.5 accurately represented the voting patterns of center and left-leaning political groups, it struggled to capture those of right-leaning parties. However, like many similar studies, their analysis was limited to closed-ended survey questions with single-choice responses. This highlights a key challenge: while LLMs may perform well in Englishspeaking contexts, less is known about their ability to generate representative opinions in non-Englishspeaking countries and for open-ended questions. This is particularly true for open-ended setups, where scaling and accurately interpreting responses pose significant challenges (Resnik et al., 2024).

To address these challenges, this study explores the algorithmic fidelity of LLMs in generating synthetic public opinions in an open-ended survey question based on German survey data. We use the survey question about the most important problems facing Germany today from the German Longitudinal Election Study (GLES). The survey is a longitudinal panel survey, and the answer distribution can be found in Figure 1. We select variables from the original survey data to represent survey participants with their characteristics as personas. We include three LLMs (Gemma, Llama2, Mixtral) in our study and prompt them to simulate survey participants to answer the open-ended question about the most important political problem in Germany. Finally, we compare the outputs regarding the distributional alignment with original survey answers, predictive performance, and answer diversity. Our most important findings are:

(1) Llama2 is better at modeling group opinions (§4.1).

(2) Llama2's representativeness fluctuates across survey waves, with the model's representativeness of the population decreasing as survey diversity increases; the model represents subpopulation opinions unevenly, with favorable alignment for left-leaning parties (such as the Left, the Greens) over right-parties (such as AfD) (§4.2).

(3) Including more variables in prompts improves performance, with party affiliation being the most influential factor (§4.3).

## 2 Related Work

LLMs for Survey Response Generation. Recent studies have increasingly repurposed survey questionnaires, originally designed for public opinion polling, to assess the opinions generated by LLMs (Ma et al., 2024). For instance, Santurkar et al. (2023) identified significant differences between LLM opinion distributions and US-based survey participants. Similarly, Dominguez-Olmedo et al. (2024) highlighted disparities between LLM and human opinions, emphasizing the sensitivity of model outputs to biases in prompting. Tjuatja et al. (2024) found that LLMs are highly sensitive to prompt perturbations and fail to replicate humanlike behavior. Collectively, these studies suggest that LLMs align more closely with populations holding left-leaning, Western-oriented values.

Opinion Generation in the German Contexts. While most studies on opinions in LLM output are English- and U.S.-centric, some research has explored other contexts, such as the German case. In a recent study, von der Heyde et al. (2025) employed the data of 2017 post-election cross-section of the GLES. Respondents to this study reported their vote choice in the survey. von der Heyde et al. (2025) prompted GPT-3.5 (Brown et al., 2020) with personas to simulate the survey participants. Based on the close-ended choice setup, they found that it does not predict citizens’ vote choice accurately, exhibiting a bias towards the Green and Left parties, similar to previous work in English contexts.

Evaluation of LLM Outputs. Previous studiesprimarily focused on closed-ended multiplechoice questions, often relying on the model's first token prediction (e.g., Santurkar et al., 2023; Dominguez-Olmedo et al., 2024; Tjuatja et al., 2024) or semi-automated extraction of text answers (von der Heyde et al., 2025). Alternatively, Wang et al. (2024a,b) proposed training a classifier directly on LLM responses manually labeled by annotators, finding this method more robust. After the output extraction, a few evaluation metrics have been applied to measure the alignment of human and LLM responses (Ma et al., 2024), such as Cohen's Kappa (Argyle et al., 2023; Hwang et al., 2023), 1-Wasserstein distance (Santurkar et al., 2023; Hwang et al., 2023), KL divergence (Dominguez-Olmedo et al., 2024; Sun et al., 2024), Euclidean distance (Wang et al., 2023), Jensen-Shannon distance (Durmus et al., 2024), etc. and correlation and statistical analyses (Sun et al., 2024; Jiang et al., 2024b). For our case study, we adapt these metrics to examine the fidelity of LLMgenerated synthetic German public opinions.

## 3 Experimental Setups

## 3.1 Data

German Longitudinal Election Study (GLES Panel). We use the GLES Panel dataset from GESIS (2023). The survey consists of 21 waves² and contains socio-demographic information, vote intentions, choices, and political attitudes of participants. The target population is German citizens eligible to vote during the respective elections in Germany. Along with the respondents' sociodemographic data for our prompts, we use respondents’ answers to the question “In your opinion, what is the most important problem facing Germany today?" for comparing human answers and LLM outputs. The answers of participants were collected as free-form texts.

Selected Information. We included six variables from the original survey: age, gender, leaning party, region, education degree, and vocational degree. Details on the sub-groups of the variables are shown in §A.

Coding Scheme. For coding the LLM text responses into categories, we follow the coding scheme proposed in an additional sub-study of the GLES (GESIS, 2024). Like Mellon et al. (2024), who collapsed \~50 classes into a simpler classification, we also set “coarse" classes (n=16). We merged rarely represented classes into an upper class (e.g., "Price Level", "Housing Policy", and “Economic Policy" classes into one “Economic Policy" class). The distribution of GLES survey answers based on the coarse classes is shown in Figure 1. The full list of fine and coarse classes can be found in §B in the Appendix.

## 3.2 Text Generation

Models. We chose three instruction-tuned openweight LLMs: Llama-2-13b-chat-hf (Touvron et al., 2023), Gemma-7b-it (Team, 2024), and Mixtral-8x7B-Instruct-v0.1 (Jiang et al., 2024a).

Prompt Design. After initial trial runs and inspecting the LLM outputs, we used the prompt in Figure 2 in our experiments with LLMs. We chose German as the prompting language, as it is the language in which the GLES survey was conducted, and the generated texts can be compared to the original text. During the experimentation phase, the placeholders are replaced with the respondent's information, i.e., the variables from the survey data.

![](images/7832592a6835432fcf8de03c025a7ac13866e76df81c841665b8379929c75951.jpg)  
Figure 2: Translated prompt in English. The original prompt in German is presented in Figure 9 in Appendix.

## 3.3 LLM Output Classification

To evaluate and compare the LLM outputs with human responses, we needed to categorize the responses into specific classes and trained a classifier to code the responses from the LLMs.

Manual Annotation. Drawing from Wang et al (2024a,b), we manually annotated 1,500 LLM outputs, selecting 500 outputs randomly from each LLM. We then trained a classifier based on the manually developed annotation scheme for the LLM outputs. Details on the scheme can be found in §C.

Classifier Training and Inference. We finetuned the German version of the base BERT (Devlin et al., 2019) classifier on the annotated LLM outputs. The fine-tuned classifier achieves a weighted F1 score of 0.93 on the test set. The classifier is then used to classify all LLM responses.

## 3.4 Evaluation Metrics for Experiments

In the context of generative models, representativeness is the model's ability to recover populationlevel properties of the original data (Eigenschink et al., 2023), i.e., a dimension of algorithmic fidelity. To compare the representativeness of the LLM answers with the original survey data and to measure the association between the variables, we used the following evaluation metrics.

Jensen-Shannon (JS) Divergence. JS divergence is a symmetric and normalized measure of divergence derived from KL divergence (Kullback and Leibler, 1951). It is calculated as:

$$
\mathrm { J S D } ( P \parallel Q ) = \frac { 1 } { 2 } D _ { \mathrm { K L } } ( P \parallel M ) + \frac { 1 } { 2 } D _ { \mathrm { K L } } ( Q \parallel M )\tag{1}
$$

where $M = { \textstyle \frac { 1 } { 2 } } ( P + Q )$ is the mixture distribution of P and $Q$ (Lin, 2006). The JS divergence is bounded between 0 and 1 (when using log2), making it easier to interpret than KL divergence. We use JS Distance, the square root of JS Divergence, as in Durmus et al. (2024), because its bounded range facilitates comparison across different data waves. The JS distance is applied to measure the representativeness of the coded LLM answers compared to the real survey data.

Entropy. Entropy measures the variability or uncertainty in a set of outcomes (Jurafsky and Martin, 2024):

$$
\mathrm { H } ( X ) = - \sum _ { x \in { \mathcal { X } } } p ( x ) \log p ( x )\tag{2}
$$

We use entropy to assess the diversity of text categories in synthetic and survey data. Lower entropy indicates less variability, meaning fewer bits are needed to represent the information in the data.

Conditional Entropy. Conditional entropy measures the remaining uncertainty in variable X when another variable $Y$ is known. It calculates the entropy of X given the distribution of Y:

$$
{ \begin{array} { r l } { H ( X \mid Y ) = \displaystyle \sum _ { y \in A _ { Y } } P ( y ) \left[ \sum _ { x \in A _ { X } } P ( x \mid y ) \log { \frac { 1 } { P ( x \mid y ) } } \right] } & { } \\ { \quad = \displaystyle \sum _ { x \in A _ { X } } \sum _ { y \in A _ { Y } } P ( x , y ) \log { \frac { 1 } { P ( x \mid y ) } } } & { ( 3 ) } \end{array} }
$$

We use conditional entropy to evaluate how much uncertainty remains about responses in the survey when the subpopulation is known. This helps assess whether the synthetic data captures patterns in specific groups within the population.

Information Gain. Also called mutual information. It measures how much information one random variable provides about another. It is calculated as the difference between the entropy of the variable and its conditional entropy given another variable:

$$
I ( X ; Y ) = H ( Y ) - H ( Y \mid X )\tag{4}
$$

It indicates how much knowing one variable (e.g., X) reduces uncertainty about another variable (e.g., Y). A higher information gain indicates that knowing one variable reduces uncertainty about another variable. In our experiments, we calculate the population entropy H (Y) and conditional subpopulation entropy $H ( Y \mid X )$ , where X represents demographic features. We will compute $H ( Y )$ and $H ( Y \mid X )$ for subpopulations and compare the information gained in survey and LLM data.

Cramér's V. This is a measure of association between nominal variables (Cramér, 1999). It is based on Pearson's $\chi ^ { 2 }$ test. However, Cramér's V discounts the value of the $\chi ^ { 2 }$ statistic for both the sample size (N) and the size of the table of counts (minimum of row count or column count minus 1 ) (Holbrook, 2022). It is computed as:

$$
V = { \sqrt { \frac { \chi ^ { 2 } } { N \cdot \operatorname* { m i n } ( r - 1 , c - 1 ) } } }\tag{5}
$$

We use Cramérs’ V to check “pattern correspondence" in LLM outputs. We map each input variable $( X _ { i } )$ to the output variable (Y), and check whether the pairwise correlations in survey data are also present in the LLM-generated data.

## 4 Experiments and Results

Three main experiments were conducted on the GLES data. The first evaluated all three LLMs using a single wave, focusing on dataset statistics and representativeness (§4.1). The second extended this analysis across multiple waves with Llama2 to track performance over time (§4.2). The third involved ablation studies to assess how different variables affect representativeness and response diversity (§4.3).

## 4.1 Experiment 1: Model Pre-Experiment in One Wave

After dropping the observations with missing features, we used the survey data from wave 12 (collected between 05-11-2019 and 19-11-2019, i.e., before COVID-19) for the first experiment. All three LLMs have been prompted to generate synthetic answers. The generated answers are classified as stated in §3.3. We compared dataset statistics and the textual style of the answers and computed JS Distance and entropy.

![](images/2e8ed4d5d1055dc7c2353ba5f63df2b2194ec2bc1ba4aede6a3d99658f7d58ab.jpg)  
Figure 3: Information Gain for leaning party variable (Xi). Left: population entropy (H (Y)), right: subpopulation entropy $( H ( Y \mid X _ { i } ) )$ . A large gap between left and right $( H ( Y ) - H ( Y \mid X _ { i } ) )$ means big information gain when focusing on the samples of the subpopulation group, indicating responses with this group are less diverse.

In Table 1, we give an overview of statistics about labels, textual characteristics, and representativeness (on the population level). 42 % of the Gemma model answers were about COVID-19 (identified using Regex), even though the survey answers were collected before COVID-19, which indicates a very large proportion of hallucinations. Therefore, we did not include the Gemma model in the further subpopulation-level analysis. The detailed JS Distances in each social group category can be found in Table 14 in the Appendix.

<table><tr><td>Metric</td><td>Gemma</td><td>Llama2</td><td>Mixtral</td><td>Survey</td></tr><tr><td>Avg. Labels per Sample</td><td>1.03</td><td>1.20</td><td>1.33</td><td>1.03</td></tr><tr><td>Avg. Samples per Label</td><td>593</td><td>692</td><td>769</td><td>597</td></tr><tr><td>Avg. Word Count</td><td>36.37</td><td>25.65</td><td>43.75</td><td>2.29</td></tr><tr><td>Non-German Answer Rate</td><td>0.02</td><td>0.06</td><td>0.03</td><td></td></tr><tr><td>Non-response Rate</td><td>0</td><td>0</td><td>0.05</td><td>0.04</td></tr><tr><td>LLM Refusal Rate</td><td>1e-4</td><td>0</td><td>1e-4</td><td>-</td></tr><tr><td>COVID Regex Match Rate</td><td>0.42</td><td>0.03</td><td>2e-3</td><td>0</td></tr><tr><td>JS Distance to Survey</td><td>0.62</td><td>0.28</td><td>0.29</td><td></td></tr><tr><td>Answer Entropy</td><td>2.26</td><td>2.90</td><td>2.56</td><td>2.93</td></tr></table>

Table 1: Survey and LLM data statistics in experiment 1

A case study on information gain of the party variables: Llama2 aligns more closely with survey data and maintains subgroup stability. Figure 3 compares population-level answer entropies (left) with conditional entropies (right) for each leaning party value. Information gain, calculated as the difference between these entropies, reflects how much additional insight is provided by knowing the leaning party value. The population-level entropies $( H ( Y ) )$ are close, with the survey (2.93) closely matching Llama2 (2.90), while Mixtral was a bit lower (2.56) (see left of each subplot). After incorporating the leaning party information and only looking at the samples containing the specific party affiliations $( H ( Y \mid$ $X _ { i } ) )$ , Mixtral shows still lower conditional entropy (see right of each subplot), indicating less variation in responses. Especially for “Die Grünen (The Greens)" and $\mathrm { ^ { 6 6 } A f D ^ { \prime } }$ , there are drastic drops of $H ( Y \mid X _ { i } )$ . This suggests that Mixtral may risk reflecting dominant group opinions, reducing diversity, and showcasing stereotypical representations of these subgroups. In contrast, Llama2 exhibits less information gain, i.e., it is more aligned with the survey data.

## 4.2 Experiment 2: Wave Experiment with Llama2

We focused on Llama2 in the second in-depth experiment and repeated the generation process for the most 10 recent panel waves in GESIS (2023) (waves 12-21). Over these two years, we observe large shifts in survey label distributions (see Figure 1). This allows us to evaluate the representativeness of the model under varying label distributions and seek answers to the following questions:

• Do the LLM capabilities at both the population and subpopulation levels vary over time?

![](images/651586fc9d95447394a7ad5ba603be37bf32f22fce12d704a25d3b9b409affa0.jpg)  
Figure 4: JS Distance of six subpopulation groups in Experiment 2. An in-depth presentation of the JS Distance for each group is shown in Figure 13 in the Appendix.

• In which subpopulations are opinions represented more accurately?

Llama2 captures shifts in survey trends, but higher answer diversity correlates with reduced representativeness. For panel waves 12-21, we repeated the text generation process and classified the answers. Table 2 shows the population-level entropy values and the JS distances. We calculated Pearson's correlation coefficient between survey entropy and the JS distance and got r = —0.35, indicating that the model's representativeness of the population decreases as the diversity in answers increases. For wave 13, with data collection between April 2020 and May 2020, the diversity of answers reached its minimum (with an entropy of 0.58). In Table 13, we see that 92.4 % of answers were about “Health Policy" (and about the COVID-19). This shows that LLMs' responses reflect the change in the survey date.

<table><tr><td></td><td>12</td><td>13</td><td>14</td><td>15</td><td>16</td><td>17</td><td>18</td><td>19</td><td>20</td><td>21</td><td>avg.</td></tr><tr><td>LLM entropy</td><td></td><td>2.900.58</td><td></td><td></td><td></td><td></td><td></td><td></td><td>31.671.31 2.12 2.202.272.46 2.46 2.492.04</td><td></td><td></td></tr><tr><td>survey entropy</td><td></td><td>2.93 2.02 2.24 2.31 2.53 2.82 2.75 2.85 2.92 2.19 2.55</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>JS distance</td><td></td><td>0.29 0.29 0.24 0.22 0.20 0.23 0.23 0.22 0.24 0.30 0.24</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 2: Population level entropy values and the JS distance in the Wave Experiment from wave 12-21.

Subpopulation-level findings: JS distances reveal representational variation influenced by group information and complexity. Figure 4 shows the JS distances at the subpopulation level for each variable. We observed the most variation for education and leaning party variables. Although the difference is smaller than the three variables above, gender and region have a consistent

JS difference regardless of panel waves. And no age variable value consistently has a lower JS score. It shows the model can represent the opinions of different social groups at various levels but offers no clear explanation for the variation in representation. This could come from better recognition of certain groups’ views, the training data, or model architecture. Another possibility is that some social groups are more “informative" about this question.

Llama2 closely reflects sociodemographic patterns, with minor deviations from survey influences. Cramér's V values in Figure 14 in Appendix show pairwise patterns between prompting variables and text answers. However, in comparison with the survey's values, we see that the model underestimates the influence of age and education degree on the text answers, the model consistently overestimates the effect of region, and the gender and party variables are both overestimated and underestimated. However, except region, the differences are usually less than 0.05, indicating that the Llama2 closely reflects patterns between sociodemographics and the survey.

A case study on party variables: Llama2 better models groups with left-leaning parties. To check how much of the JS distance can be associated with the modeling difficulty of the variables, we plot the subpopulation entropy and JS distances for leaning party in Figure 5. Since the population entropy is the same for all subpopulations, we can safely assume that lower conditional entropy means higher information gain for that variable. This allows us to examine how representativeness, linked to the available information in a variable, impacts alignment success. When mutual information is high, LLMs can better model subpopulation behavior. However, certain groups do not fall into this trend line; “die Linke (the Left)"and “die Grü- nen (the Greens)" are modeled better, and “AfD" is worse (compared to the information their groups carry). This finding aligns with previous work (e.g. Santurkar et al., 2023; von der Heyde et al., 2025), which shows that LLMs tend to have a more leftleaning feature. We show additional results on other variables and observe similar results showing that LLMs are biased towards left, Western, and educated people in §G (4th paragraph).

![](images/0e52362b5f0a7f4157b7cfb19eac9a74818e4e60752e7ca96c616d23a661d2d9.jpg)  
Figure 5: Subpopulation entropy and JS distance for leaning party (mean values for waves 12-21).

## 4.3 Experiment 3: Ablation Experiment

To further show how individual demographic information affects the LLM output diversity, we conducted a series of ablation experiments with the following variations:

• Including only one social group variable.

• Excluding one social group variable.

• Using no social group variables.

These were compared against the experiment with all variables included.

In the base prompt, the model was only informed that the response was from a German citizen, with the relevant survey time frame. Detailed prompt variations can be found in Table 8 in Appendix. We used Llama2 and wave 12 data to analyze how adding or removing social group single variables impacts representativeness and answer diversity The use of wave 12 is because it took place before COVID (see the dates of the waves in §A), and might have more diverse answer categories (compared to the dominance of health policy responses illustrated in Figure 1).

Variable inclusion and exclusion have an impact on model performance. Figure 6 shows the JS distances in ablation experiments. Including all variables reduces the JS distance by 0.15 compared to the base prompt. Adding a single variable improves predictions. Removing a variable worsens performance, though it is still better than using only one variable, except for the “all except party" case.

![](images/34122f0a770555360b2f5ad99d5066fccbc5f955d94075dfc7d52348b2cb12ee.jpg)  
Experiment Typeno demographicsone variableall except one variableall variables  
Figure 6: JS distances for the ablation experiment.

LLM outputs show stronger correlation with variables when prompted with only one variable. Table 3 compares Cramér's V values between the survey, including only one variable, and all variables included experiments. When only one variable is provided, the generated texts show stronger correlations with the input variable. Although JS distances decrease when more variables are added, this observation suggests that synthetic data patterns are dynamic and can be influenced by the number of prompt variables.

<table><tr><td>Prompt Variable</td><td>Survey</td><td>LLM-one</td><td>LLM-all</td></tr><tr><td>Age</td><td>0.09</td><td>0.09</td><td>0.07</td></tr><tr><td>Education Degree</td><td>0.06</td><td>0.25</td><td>0.05</td></tr><tr><td>Gender</td><td>0.08</td><td>0.20</td><td>0.16</td></tr><tr><td>Leaning Party</td><td>0.16</td><td>0.35</td><td>0.17</td></tr><tr><td>Region</td><td>0.06</td><td>0.42</td><td>0.15</td></tr><tr><td>Vocational Degree</td><td>0.08</td><td>0.12</td><td>0.07</td></tr></table>

Table 3: Cramér's V values for the Ablation Experiment

The inclusion of the party variable has the most significant impact on model performance, with its presence leading to substantial improvements in information gain. As shown in Figure 6, adding either only the party or education variable alone results in the greatest reduction in JS distance compared to the model without demographics; excluding only the party variable leads to a smaller improvement in JS distance, both highlighting the party variable's greater impact. Similar to Figure 3, we plot information gain for party, comparing survey entropy to Llama2 with all variables, with only one party variable, and with no variables in Figure 7. As expected, Llama2-base, which includes no subpopulation variables, produces entropies close to the population entropy, with random variations of ±0.03. However, in the 1VAR-party experiment, information gain ranges from 0.2 to 1.3, significantly above random variation. This, along with Cramér's V values, suggests that the model generates typical responses, reducing the variation in subpopulation opinions. Further detailed experimental results are provided in §G of the Appendix.

![](images/38555b867058dbe6c510fece60c8756b9d141a6a8d6772ece741cc87961f0bc8.jpg)  
Sourcesurvey1VAR\_partyLlama2\_allLlama2\_base  
Figure 7: Information Gain for leaning party variable, comparing survey entropy to Llama2-al1 (with all variables), 1VAR-party (with only party variable), and Llama2-base (with no variables). Left: population entropy (H(Y)), right: subpopulation entropy $( H ( Y \mid X _ { i } ) )$ 1

## 5 Discussion

We next distill key findings from our experiments, compare them to prior research, and offer insights into the role of LLMs in modeling demographic behaviors and their practical insights in surveybased applications based on our German case study.

Algorithmic Fidelity in Modeling the German population. von der Heyde et al. (2025) found that GPT-3.5 vote predictions for the 2017 German election are inaccurate and biased towards the Green and Left parties. We also found that the model is better at modeling the opinions of the Green and the Left parties than the right parties. The subpopulation entropy and in-group diversity can partially explain this finding. Other factors could be the models' training data and the RLHF methods used.

Reduction in in-group diversity. Bisbee et al. (2024) found that while GPT-3.5 could replicate survey averages, its synthetic answers lacked variation compared to real survey data. Similarly, von der Heyde et al. (2025) noted GPT-3.5's difficulty in capturing nuanced subpopulation behaviors. In our analysis, we also observe a reduction in in-group diversity under certain conditions, particularly when only one variable is provided to the model or when using the Mixtral model. This suggests that the ability to represent within-group diversity is limited by the model's input structure and specific architecture.

The role of LLM prompts. Binz and Schulz (2023) highlighted how cognitive biases, such as the framing effect, influenced GPT-3's outputs. In our third study, we also noticed that providing only one variable in a prompt caused Llama2 to focus disproportionately on that variable, possibly interpreting it as more critical than when multiple variables were included. Interestingly, this effect varied by model: Mixtral over-relied on variables even with full prompts, while Llama2 showed stronger biases with fewer variables. This suggests that model-specific differences influence how demographic factors are integrated and highlights the need for careful prompt design. Argyle et al. (2023) motivated the silicon sampling approach on this conditional probability formula:

$$
P ( V , B _ { \mathrm { L L M } } ) = P ( V | B _ { \mathrm { L L M } } ) P ( B _ { \mathrm { L L M } } )\tag{6}
$$

here B is demographic backstories, and V is voting patterns. If the model learned the $P ( V | B )$ , one could correct for the P(B) and obtain:

$$
P ( V , B _ { \mathrm { s u r v e y } } ) = P ( V | B _ { \mathrm { s u r v e y } } ) P ( B _ { \mathrm { s u r v e y } } )\tag{7}
$$

However, in the ablation experiment, we observed that prompting with the social groups is not straightforward, and it does not align LLMs' inner parameters to “solely" consider $P ( B _ { \mathrm { s o c i a l \_ g r o u p } } )$ LLMs might not always be conditioned to sample from the joint distribution of backstories. We propose that demographic variables' order, number, and predictive power have a complex interplay and that this is a further research direction (see, e.g., Shu et al., 2024). Also, insights of vignette experiments from survey methodology (Steiner et al., 2017) could be useful in prompt design.

![](images/c6deffef4308231acdf516de807e4d3d34850ff4b243b81cf9b5bfe3d95058eb.jpg)  
Figure 8: JS distances of answers for the last five waves (17-21) from GLES, comparing each survey's answers to those of the preceding surveys. In most cases, differences between survey responses increase over time. The peak of wave 21 compared to wave 20 corresponds to the drop of the health policy category on 29.09.2021 in Figure 1, possibly due to some effect of COVID related topics in that time.

Practical evaluation of LLMs. Figure 8 illustrates the JS distances of responses in the last five survey waves compared to earlier waves. As shown, responses from older surveys tend to differ more significantly from those in recent surveys.

Consider a hypothetical scenario where LLMs are continuously updated with up-to-date training data while surveys are conducted less frequently due to cost constraints. In such a scenario, Figure 8 suggests that LLMs may also help researchers estimate the answers due to the timeliness of their training data. However, even if the model is assumed to be representative enough, the variety in subgroup answers should also be considered for practical uses. In contrast, recent work by Park et al. (2024) demonstrates how LLMs, when applied to large-scale human participant interviews, can simulate subpopulations’attitudes and behaviors, with surveys as a valuable evaluation tool. This highlights the potential of LLMs not only to provide estimates but also to more accurately reflect diverse human perspectives, emphasizing the importance of incorporating subgroup variation.

## 6 Conclusion and Recommendations

This paper evaluates the algorithmic fdelity of LLMs to represent the opinions of German subpopulations. While von der Heyde et al. (2025) found that GPT-3.5 struggles with the nuances of German subpopulations and the country's multi-party system in closed-ended voting questions, we explore instead free-form open-ended text responses, focusing on how these responses align with survey data. By using free-form text responses rather than multiple-choice questions, we can identify detailed issues in contextual information and the variety of different subpopulations, underscoring the value of this evaluation approach. Our findings show that LLMs, particularly the Llama2 model, are capable of associating text responses with socialdemographic variables, indicating a degree of representativeness. However, the number of variables included in the prompt plays a crucial role in model performance. Despite this, the models still tend to generate stereotypical representations, with a noticeable favor towards left-leaning parties, consistent with previous findings on the limited diversity of opinions reflected in LLMs.

Based on these findings, we recommend that both LLM and social science researchers consider the following steps for future evaluation of LLMgenerated responses in survey-based research:

Improved representation of opinion diversity: LLMs should be further developed to reflect the full spectrum of opinions within subpopulations while harmful contents are cautiously manipulated. This includes addressing biases and avoiding the oversimplification of diverse views into stereotypical categories with certain safety mechanisms.

Cross-national comparison: Due to the current discussion of English-centric biases of LLMs, a more inclusive evaluation with opinion diversity from non-English data or cross-national sources such as GlobalOpinionQA (Durmus et al., 2024) should be conducted and improved.

Timeliness and survey simulation: LLMs can be valuable in situations where real survey data is limited or outdated (Namikoshi et al., 2024; Ma et al., 2024). With continuously updated training data, LLMs could be further evaluated in the case of estimating shifts in public opinion.

## Limitations

As in our setup, we need to manually annotate the LLM responses into 16 classes. This limitation constrains our ability to conduct a more comprehensive exploration within this setup, as manual annotation is both costly and inefficient. Future research could address this limitation by exploring the potential of combining human and LLM annotation, as recently proposed by Choksi et al. (2024).

Another limitation concerns the inherent challenge of scaling and coding open-ended survey responses. While we propose reducing the number of classes to 16 in this paper which has also been proposed by recent social science research (Mellon et al., 2024), there might be instances where this simplification may not fully capture the nuanced nature of the responses. To address this, future studies could incorporate qualitative analyses by subject matter experts as part of the final validation process.

Additionally, our study focused on three openweight LLMs and the German language, which could limit the scope of our findings and might have restricted the generalizability of our findings to other LLMs or languages. Future research should include a broader variety of models and prompting languages to explore their performance and generalizability. Moreover, as our approach primarily relies on zero-shot prompting, a promising next step would be to evaluate how alternative methods, such as fine-tuning (e.g., Cao et al., 2025; Holtdirk et al., 2025), could influence model performance in the German context.

## Ethical Considerations

Throughout the experimentation, we use the publicly available survey dataset from GESIS (2023). The original data contains social-demographics of the human participants of the survey, with the personally identifiable data removed. We solely present the survey results and the responses generated by the LLMs, in an objective manner. We do not intend to express our personal opinions on the questions.

As LLMs are deployed in systems that interact with humans, aligning them with humans' ethical values gains more importance. Although, as of today, LLMs can not reliably support opinion polling surveys, they still carry important information about human opinions, values, and attitudes (Ma et al., 2024). Even if synthetic survey data is not yet used to support surveys, LLMs as LLMs can be important tools for consulting political and social information about countries, populations, subpopulations, and politics in general (Hämäläinen et al., 2023).

When interpreting the opinions reflected in LLM outputs, we advise caution against anthropomorphism. Following recent discussions (Santurkar et al., 2023; Röttger et al., 2024; Durmus et al., 2024; Ma et al., 2024), we use the phrase “opinions reflected in the LLM outputs" rather than suggesting that LLMs “have" opinions, to avoid anthropomorphizing these models.

## Acknowledgments

We thank the members of SODA Lab and MaiNLP labs from LMU Munich, and the members of the Social Data Science Group from University of Mannheim for their constructive feedback. Xinpeng Wang and Barbara Plank are supported by ERC Consolidator Grant DIALECT (101043235). Markus Herklotz and Matthias Assenmacher are supported by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under the National Research Data Infrastructure – NFDI 27/1 - 460037581.

## References

Utkarsh Agarwal, Kumar Tanmay, Aditi Khandelwal, and Monojit Choudhury. 2024. Ethical reasoning and moral value alignment of LLMs depend on the language we prompt them in. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 6330–6340, Torino, Italia. ELRA and ICCL.

Hans-Jürgen Andreß, Katrin Golsch, and Alexander W. Schmidt. 2013. Applied Panel Data Analysis for Economic and Social Surveys. Springer.

Lisa P. Argyle, Ethan C. Busby, Nancy Fulda, Joshua R. Gubler, Christopher Rytting, and David Wingate. 2023. Out of one, many: Using language models to simulate human samples. Political Analysis, 31(3):337–351.

Arnav Arora, Lucie-aimée Kaffee, and Isabelle Augenstein. 2023. Probing pre-trained language models for cross-cultural differences in values. In Proceedings of the First Workshop on Cross-Cultural Considerations in NLP (C3NLP), pages 114–130, Dubrovnik, Croatia. Association for Computational Linguistics.

Marcel Binz and Eric Schulz. 2023. Using cognitive psychology to understand gpt-3. Proceedings of the National Academy of Sciences, 120(6):e2218523120.

James Bisbee, Joshua D. Clinton, Cassy Dorff, Brenton Kenkel, and Jennifer M. Larson. 2024. Synthetic replacements for human survey data? the perils of large language models. Political Analysis, 32(4):401–416.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam Mc-Candlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. Preprint, arXiv:2005.14165.

Yong Cao, Haijiang Liu, Arnav Arora, Isabelle Augenstein, Paul Röttger, and Daniel Hershcovich. 2025. Specializing large language models to simulate survey response distributions for global populations. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3141–3154, Albuquerque, New Mexico. Association for Computational Linguistics.

Yong Cao, Li Zhou, Seolhwa Lee, Laura Cabello, Min Chen, and Daniel Hershcovich. 2023. Assessing cross-cultural alignment between ChatGPT and human societies: An empirical study. In Proceedings of the First Workshop on Cross-Cultural Considerations in NLP (C3NLP), pages 53–67, Dubrovnik, Croatia. Association for Computational Linguistics.

Madiha Zahrah Choksi, Marianne Aubin Le Quéré, Travis Lloyd, Ruojia Tao, James Grimmelmann, and Mor Naaman. 2024. Under the (neighbor)hood: Hyperlocal surveillance on nextdoor. In Proceedings of the CHI Conference on Human Factors in Computing Systems, CHI '24, New York, NY, USA. Association for Computing Machinery.

Jacob Cohen. 1960. A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1):37–46.

Harald Cramér. 1999. Mathematical Methods of Statistics (PMS-9). Princeton University Press.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Ricardo Dominguez-Olmedo, Moritz Hardt, and Celestine Mendler-Dünner. 2024. Questioning the survey responses of large language models. In Advances in

Neural Information Processing Systems, volume 37 pages 45850–45878. Curran Associates, Inc.

Esin Durmus, Karina Nguyen, Thomas Liao, Nicholas Schiefer, Amanda Askell, Anton Bakhtin, Carol Chen, Zac Hatfield-Dodds, Danny Hernandez, Nicholas Joseph, Liane Lovitt, Sam McCandlish, Orowa Sikder, Alex Tamkin, Janel Thamkul, Jared Kaplan, Jack Clark, and Deep Ganguli. 2024. Towards measuring the representation of subjective global opinions in language models. In First Conference on Language Modeling.

Peter Eigenschink, Thomas Reutterer, Stefan Vamosi, Ralf Vamosi, Chang Sun, and Klaudius Kalcher. 2023. Deep Generative Models for Synthetic Data: A Survey. IEEE Access, 11:47304–47320.

GESIS. 2023. Gles panel 2016-2021, wellen 1- 21. GESIS, Köln. ZA6838 Datenfile Version 6.0.0, https://doi.org/10.4232/1.14114.

GESIS. 2024. Gles codierung des wichtigsten politischen problems (2018-2022). GESIS, Köln. ZA7957 Datenfile Version 2.0.0, https://doi.org/10.4232/1.14241.

Perttu Hämäläinen, Mikke Tavast, and Anton Kunnari. 2023. Evaluating large language models in generating synthetic hci research data: a case study. In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems, CHI ’23, New York, NY, USA. Association for Computing Machinery.

Thomas M. Holbrook. 2022. An Introduction to Political and Social Data Analysis Using R. Thomas M. Holbrook.

Tobias Holtdirk, Dennis Assenmacher, Arnim Bleier, and Claudia Wagner. 2025. Addressing Systematic Non-response Bias with Supervised Fine-Tuning of Large Language Models: A Case Study on German Voting Behaviour.

EunJeong Hwang, Bodhisattwa Majumder, and Niket Tandon. 2023. Aligning language models to user opinions. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 5906– 5919, Singapore. Association for Computational Linguistics.

Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2024a. Mixtral of experts. Preprint, arXiv:2401.04088.

Hang Jiang, Xiajie Zhang, Xubo Cao, Cynthia Breazeal, Deb Roy, and Jad Kabbara. 2024b. PersonaLLM: Investigating the ability of large language models to

express personality traits. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 3605–3627, Mexico City, Mexico. Association for Computational Linguistics.

Dan Jurafsky and James H. Martin. 2024. Speech and language processing: an introduction to natural language processing, computational linguistics, and speech recognition.

S. Kullback and R. A. Leibler. 1951. On Information and Sufficiency. The Annals of Mathematical Statistics, 22(1):79 – 86.

Jianhua Lin. 2006. Divergence measures based on the shannon entropy. IEEE Trans. Inf. Theor., 37(1):145–151.

Lin Long, Rui Wang, Ruixuan Xiao, Junbo Zhao, Xiao Ding, Gang Chen, and Haobo Wang. 2024. On LLMs-driven synthetic data generation, curation, and evaluation: A survey. In Findings of the Association for Computational Linguistics ACL 2024, pages 11065–11082, Bangkok, Thailand and virtual meeting. Association for Computational Linguistics.

Bolei Ma, Xinpeng Wang, Tiancheng Hu, Anna-Carolina Haensch, Michael A. Hedderich, Barbara Plank, and Frauke Kreuter. 2024. The potential and challenges of evaluating attitudes, opinions, and values in large language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 8783–8805, Miami, Florida, USA. Association for Computational Linguistics.

Jonathan Mellon, Jack Bailey, Ralph Scott, James Breckwoldt, Marta Miori, and Phillip Schmedeman. 2024. Do ais know what the most important issue is? using language models to code open-text social survey responses at scale. Research & Politics, 11(1):20531680241231468.

Keiichi Namikoshi, Alex Filipowicz, David A. Shamma, Rumen Iliev, Candice L. Hogan, and Nikos Arechiga. 2024. Using llms to model the beliefs and preferences of targeted populations. Preprint, arXiv:2403.20252.

Joon Sung Park, Carolyn Q. Zou, Aaron Shaw, Benjamin Mako Hill, Carrie Cai, Meredith Ringel Morris, Robb Willer, Percy Liang, and Michael S. Bernstein. 2024. Generative agent simulations of 1,000 people. Preprint, arXiv:2411.10109.

Philip Resnik, Bolei Ma, Alexander Hoyle, Pranav Goel, Rupak Sarkar, Maeve Gearing, Anna-Carolina Haensch, and Frauke Kreuter. 2024. Topic-oriented protocol for content analysis of text – a preliminary study. Unpublished manuscript.

Paul Röttger, Valentin Hofmann, Valentina Pyatkin, Musashi Hinck, Hannah Kirk, Hinrich Schuetze, and Dirk Hovy. 2024. Political compass or spinning arrow? towards more meaningful evaluations for values and opinions in large language models. In Proceedings of the 62nd Annual Meeting of the Association

for Computational Linguistics (Volume 1: Long Papers), pages 15295–15311, Bangkok, Thailand. Association for Computational Linguistics.

Guilherme Sanches de Oliveira and Edward Baggs. 2023. Psychology's WEIRD Problems. Elements in Psychology and Culture. Cambridge University Press.

Shibani Santurkar, Esin Durmus, Faisal Ladhak, Cinoo Lee, Percy Liang, and Tatsunori Hashimoto. 2023. Whose opinions do language models reflect? In Proceedings of the 40th International Conference on Machine Learning, ICML'23. JMLR.org.

Bangzhao Shu, Lechen Zhang, Minje Choi, Lavinia Dunagan, Lajanugen Logeswaran, Moontae Lee, Dallas Card, and David Jurgens. 2024. You don't need a personality test to know these models are unreliable: Assessing the reliability of large language models on psychometric instruments. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5263–5281, Mexico City, Mexico. Association for Computational Linguistics.

Peter M. Steiner, Christiane Atzmüller, and Dan Su. 2017. Designing valid and reliable vignette experiments for survey research: A case study on the fair gender income gap. Journal of Methods and Measurement in the Social Sciences, 7:52–94.

Seungjong Sun, Eungu Lee, Dongyan Nan, Xiangying Zhao, Wonbyung Lee, Bernard J. Jansen, and Jang Hyun Kim. 2024. Random silicon sampling: Simulating human sub-population opinion using a large language model based on group-level demographic information. Preprint, arXiv:2402.18144.

Gemma Team. 2024. Gemma: Open models based on gemini research and technology. Preprint, arXiv:2403.08295.

Lindia Tjuatja, Valerie Chen, Tongshuang Wu, Ameet Talwalkwar, and Graham Neubig. 2024. Do LLMs exhibit human-like response biases? a case study in survey design. Transactions of the Association for Computational Linguistics, 12:1011–1026.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models. Preprint, arXiv:2302.13971.

Veniamin Veselovsky, Manoel Horta Ribeiro, Akhil Arora, Martin Josifoski, Ashton Anderson, and Robert West. 2023. Generating faithful synthetic data with large language models: A case study in computational social science. Preprint, arXiv:2305.15041.

Leah von der Heyde, Anna-Carolina Haensch, and Alexander Wenz. 2025. Vox populi, vox ai? using large language models to estimate german vote choice. Social Science Computer Review, 0(0):08944393251337014.

Xinpeng Wang, Chengzhi Hu, Bolei Ma, Paul Rottger, and Barbara Plank. 2024a. Look at the text: Instruction-tuned language models are more robust multiple choice selectors than you think. In First Conference on Language Modeling.

Xinpeng Wang, Bolei Ma, Chengzhi Hu, Leon Weber-Genzel, Paul Röttger, Frauke Kreuter, Dirk Hovy, and Barbara Plank. 2024b. “my answer is C": First-token probabilities do not match text answers in instructiontuned language models. In Findings of the Association for Computational Linguistics ACL 2024, pages 7407–7416, Bangkok, Thailand and virtual meeting. Association for Computational Linguistics.

Xuena Wang, Xueting Li, Zi Yin, Yue Wu, and Jia Liu. 2023. Emotional intelligence of large language models. Journal of Pacific Rim Psychology, 17:1–12.

## A Data and Prompt Template

Table 4 shows the dates for the waves involved in the original GLES survey (GESIS, 2023). The six main social demographic variables and their subgroups in the original survey are presented in Table 5.

<table><tr><td>Wave</td><td>Start Date</td><td>End Date</td></tr><tr><td>10</td><td>06-11-2018</td><td>21-11-2018</td></tr><tr><td>11</td><td>28-05-2019</td><td>12-06-2019</td></tr><tr><td>12</td><td>05-11-2019</td><td>19-11-2019</td></tr><tr><td>13</td><td>21-04-2020</td><td>05-05-2020</td></tr><tr><td>14</td><td>03-11-2020</td><td>17-11-2020</td></tr><tr><td>15</td><td>25-02-2021</td><td>12-03-2021</td></tr><tr><td>16</td><td>06-05-2021</td><td>19-07-2021</td></tr><tr><td>17</td><td>07-07-2021</td><td>20-07-2021</td></tr><tr><td>18</td><td>11-08-2021</td><td>24-08-2021</td></tr><tr><td>19</td><td>15-09-2021</td><td>25-09-2021</td></tr><tr><td>20</td><td>29-09-2021</td><td>12-10-2021</td></tr><tr><td>21</td><td>09-12-2021</td><td>21-12-2021</td></tr></table>

Table 4: Data collection dates of GLES waves

In Figure 9, we show the original prompt in German we used for the LLM response generation. We use the template in German and expect the LLMs to respond in German, in order to mirror the real survey case. The placeholders in the prompt are replaced with the respondents' information and then fed to the LLMs that were experimented with. The prompt templates used in the ablation study in §4.3 are presented in Table 8.

<table><tr><td>Social Groups</td><td>Sub-Groups</td></tr><tr><td rowspan="4">Age</td><td>18-29</td></tr><tr><td>30-44</td></tr><tr><td>45-59</td></tr><tr><td>60+</td></tr><tr><td rowspan="2">Gender</td><td>Male</td></tr><tr><td>Female</td></tr><tr><td rowspan="9"></td><td>AfD</td></tr><tr><td>CDU/CSU</td></tr><tr><td>FDP</td></tr><tr><td></td></tr><tr><td>Grünen</td></tr><tr><td>A minor party</td></tr><tr><td>Linke</td></tr><tr><td>SPD No party</td></tr><tr><td>East Germany</td></tr><tr><td rowspan="3">Region</td><td>West Germany</td></tr><tr><td>High school diploma</td></tr><tr><td>Higher education entrance qualification</td></tr><tr><td rowspan="5">Education Degree</td><td>Secondary school diploma</td></tr><tr><td></td></tr><tr><td>Intermediate school diploma</td></tr><tr><td>Is still student</td></tr><tr><td>No school diploma</td></tr><tr><td rowspan="8"></td><td>Completed vocational internship/volunteer work</td></tr><tr><td>Vocational school diploma</td></tr><tr><td>University of applied sciences degree</td></tr><tr><td>Specialist school diploma</td></tr><tr><td>Completed apprenticeship</td></tr><tr><td>Vocational Degree Master craftsman or technician qualification</td></tr><tr><td>University degree</td></tr><tr><td>In vocational training Commercial or agricultural apprenticeship</td></tr><tr><td>Commercial apprenticeship</td></tr><tr><td>No vocational training completed</td></tr></table>

Table 5: Social-Demographic Groups and Sub-Groups.

Identifizieren Sie das wichtigste Problem, mit dem Deutschland im {month } {year} konfrontiert ist. Geben Sie die Antwort in einem prägnanten Satz an, konzentrieren Sie sich nur auf ein einziges Thema ohne weitere Ausführungen oder Auflistung zusätzlicher Probleme. Wiederholen Sie nicht die Informationen die Ihnen gegeben wurden, und geben Sie Ihre Antwort direkt und ohne einleitende Phrasen. Antworten Sie auf Deutsch und ausschließlich auf Deutsch, verwenden Sie keine Englische Sprache. Antworten Sie aus der Sicht eines Befragten mit deutscher Staatsbürgerschaft und den im nachfolgenden spezifizierten Eigenschaften.   
{artikel} Befragte ist {age} Jahre alt und {gender}. {pronoun} {schulabschluss\_clause } und {berufabschluss\_clause} {pronoun2} lebt in {region} und unterstützt hauptsächlich {party}.

Figure 9: Original prompt template in German.

## B Classes of the Coding Scheme

In Table 9, we show the original classes of the coding scheme based on GESIS (2024) as well as the merged and reduced 16 classes.

## C Annotation Scheme

The annotators followed the coding guidelines provided by (GESIS, 2024) for consistency. We used the coarse classes for annotation to achieve better agreement among annotators. One of the leading authors and another project collaborator were asked to conduct the same annotations of the 1,500 samples as volunteers. There exists disagreement on minor cases that were resolved after discussion. Both annotators are consent about the annotated data use. Figure 10 and 11 show the annotation screen and the annotation instruction given to the annotators respectively.

![](images/527efda4e9bac069f58d81f9ddc68cb25a4ed5b837a14ad5e4c9f1aef726842a.jpg)  
Figure 10: Annotation Screen.

## D Technical Setup

We used Python 3.12.1 and the transformers 3 library (version 4.42.4) by HuggingFace (with Pytorch Framework as the backend) to create two custom classes: the BertClassifier (for the multilabel classification task) and TextGenerator (for generating synthetic answers ).

![](images/8e1913b4d3cac6df2d535d00962c08ace3b97f9cfdb3d648145dec666f8bf851.jpg)  
Figure 11: The instructions at the annotating tool.

Text Generation. To fit models into a single GPU, we have used the 8-bit quantized version of the models. The inference configurations can be found in the study repository. We did the inference with batch\_size of 16 to benefit the parallel computing power and reduce runtime. On average, TextGenerator generated 1.16 answers per second. We performed 25 generation experiments, using ca. 75 GPU hours for the generation task.

Text Classification. As the contexts are all in German, we used the German version of the BERT model4. The BertClassifier training takes around 20 minutes for the setup 5. We trained the with a batch\_size of 32, a learning\_rate of 2e-5, and a fixed\_precision at 16 bits to fasten the convergence. The early stop condition stopped the training after 3rd epoch when no further loss reduction was observed. Table 6 shows the other relevant model parameters and hyperparameters.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>epochs</td><td>15</td></tr><tr><td>learning_rate</td><td>2e-5</td></tr><tr><td>batch_size</td><td>32</td></tr><tr><td>weight_decay</td><td>0.01</td></tr><tr><td>fp16</td><td>True</td></tr><tr><td>max_length</td><td>512</td></tr></table>

Table 6: Hyperparameters for the BERT model

## E Qualitative Analysis

Table 10 shows a few sample responses from LLM experiments. Before discussing models’ representativeness, we point to three qualitative issues observed in the text generations:

Introductory Phrases. We noted that all models use some “introductory phrases" even though models were prompted not to use any. The Gemma model starts the sentence 96 % of the time by listing the social group variables given to it. Llama2 model uses the “The most important problem facing Germany" phrase in 96 % of its generations. The Mixtral model uses the “One of the most important issues" phrase in 75 % of the answers. Even if the text lengths had been comparable, these style characteristics would allow humans to discriminate synthetic responses from actual ones. At this point, we did not put further effort into trying different prompts and making the synthetic responses stylistically similar to survey responses.

Mention of Future Events. Despite specifying the survey month and year (November 2019), responses occasionally referenced events that occurred after the survey date, such as the COVID-19 pandemic (2020) and the Energy Crisis in Germany (2021). For example, the Gemma model contained COVID-19-related words (COVID, corona, coronavirus, COVID-19, sars-cov, etc.) in 42 % of its responses. This problem was observed relatively less in Llama2 and Mixtral models (3 % and 0.2 % of answers, respectively).

Mixed-Language Answers. Although models were instructed to respond in German, a small percentage (up to 3%) of answers had some parts in German, whereas some terms or clauses were in English.

## F Additional Metrics

In this section, we present metrics in addition to those in §3.4. These include the base metrics for the main experimentation, as well as additional metrics used for the additional results in §G.

Proportion Agreement. It is the proportion of two variables exactly matching. Like accuracy, this measure does not consider the probability of matching by chance and should be used as a descriptive quantity (Argyle et al., 2023).

Cohen's Kappa (κ). It is a measure of agreement between two categorical variables and is often used as a measure of inter-rater agreement (Cohen, 1960). Unlike proportional agreement, it corrects for the agreement by chance, and It is defined as $\begin{array} { r } { \kappa = \frac { p _ { 0 } - p _ { e } } { 1 - p _ { e } } } \end{array}$ where

• $p _ { o }$ is the observed agreement ratio • $p _ { e }$ is the expected agreement when annotators assign labels randomly.

Kullback-Leibler Divergence. It also known as relative entropy, is a method used in measuring the statistical distance between two probability distributions (Kullback and Leibler, 1951). For distributions P and Q of a discrete random variable $X = [ X _ { 1 } , . . . , X _ { n } ]$ , the Kullback-Leibler (KL) divergence can be defined as:

$$
D _ { \mathrm { K L } } ( P \parallel Q ) = \sum _ { x \in \mathcal { X } } P ( x ) \log \left( \frac { P ( x ) } { Q ( x ) } \right)\tag{8}
$$

KL Divergence is not a distance measure since it does not satisfy the symmetry requirement of a metric. i.e $\mathrm { K L } ( \mathrm { P , Q } ) : = \mathrm { K L } ( \mathrm { Q , P } )$ , unless P and Q are equal. If not, KL divergence is always greater than 0 and not bounded.

Absolute Percentage Error (APE). JS distance enables us to compare model performance at the question level. We employ the APE to evaluate the accuracy of predictions in each category. APE is calculated by determining the absolute differences between predicted and actual frequencies and then normalizing these differences by the reference survey frequencies. For each label, we compute:

$$
\mathrm { A P E } _ { L } = \left| \frac { y _ { t } - \hat { y } _ { t } } { y _ { t } } \right|\tag{9}
$$

where $y _ { t }$ is survey frequency and $\hat { y } _ { t }$ LLM output frequency for the label l.

## G Additional Results

In this section, we show additional results and figures for the main experimentation in $\ S 4$

Label Distribution on LLM Outputs. Figure 12 shows the label distribution on the three LLM outputs based on the coarse labels.

Detailed JS Distances of Subpopulation in Experiment 1: Llama2 achieves better performance in most categories. We show the detailed JS Distances in each social group category in Experiment 1 in Table 14 for all three experimented LLMs. Among the three LLMs, we notice Llama2 has the least JS Distances across the most subcategories compared to the other two LLMs, showing more alignment with the real survey data.

Additional JS Distances of Subpopulation Variables in Experiment 2: Further indication of WEIRD bias of LLMs. We show additional results of the survey's subpopulation entropy and the JS Distance between the Llama output and survey results across 10 waves in experiment 2 in Figure 15, i.e., results for four other variables in addition to the variable leaning\_party in Figure 5 from §4.2. For the vocational degree variable, groups with a completed vocational internship are the least well-represented on average. For the education degree variable, groups with no degree or only a secondary school diploma are less represented compared to those with higher educational qualifications. Among age groups, older cohorts are less well-represented than younger ones. For the regional variable, which includes only two groups, there is a greater discrepancy in representing East Germany compared to West Germany. Overall, these findings align with prior evidence that LLMs exhibit biases favoring Western, younger, and more educated subpopulations, commonly referred to as WEIRD bias⁵, as highlighted in studies such as Santurkar et al. (2023), Cao et al. (2023), Arora et al. (2023), and Agarwal et al. (2024).

Label-Level Percentage Errors in Experiment 3. Table 7 shows label-level percentage errors. We compared the best-performing models in the 1-var-\* and w/o.-\* experiments. In 7 out of 14 labels, using all variables produced the lowest percentage errors. Both JS distances and percentage errors decreased with the inclusion of more variables, reinforcing the representational accuracy of the Llama2 model for the German population.

Proportional Agreement (PA) and κ Scores across Waves: LLMs face in achieving consistent agreement with survey data over time, particularly in representing complex social phenomena. The scores in Table 11 compare Llama2- generated responses with resampled survey data across different waves. The PA for Llama2 fluctuates significantly, reaching as high as 0.56 in wave 13 but dropping in later waves, especially after wave 17, indicating inconsistencies in the model's ability to align with the original survey data. By contrast, the survey resample maintains relatively stable PA scores, ranging between 0.34 and 0.55, indicating better agreement with the original survey. The κ scores, however, are low for both Llama2 and the survey resample, with Llama2 performing particularly poorly (0.01–0.04). This suggests that while the model can capture some high-level agreement (as seen in PA scores), it struggles to replicate the nuanced variability and structure of human responses across waves, especially as the diversity of responses increases. These findings underscore the challenges LLMs face in representing complex social phenomena in survey data, especially over time.

<table><tr><td colspan="5">Labels LLM-base min(1var. *) min(w/o. *) LLM-all</td></tr><tr><td>Political System, Processes</td><td>7.92</td><td>-3.62</td><td>0.92</td><td>9.71</td></tr><tr><td>Social Policy</td><td>-23.76</td><td>-21.94</td><td>-15.16</td><td>-7.45</td></tr><tr><td>Health Policy</td><td>22.02</td><td>3.32</td><td>2.84</td><td>2.08</td></tr><tr><td>Family and Gender Equality</td><td>-0.27</td><td>-0.25</td><td>-0.14</td><td>-0.10</td></tr><tr><td>Education Policy</td><td>-2.20</td><td>-2.15</td><td>-1.88</td><td>-1.83</td></tr><tr><td>Environmental Policy</td><td>4.43</td><td>0.42</td><td>0.15</td><td>2.52</td></tr><tr><td>Economic Policy</td><td>8.15</td><td>7.92</td><td>11.49</td><td>9.21</td></tr><tr><td>Security</td><td>-6.26</td><td>-7.37</td><td>-7.06</td><td>-6.88</td></tr><tr><td>Foreign Policy</td><td>6.58</td><td>1.93</td><td>1.59</td><td>0.54</td></tr><tr><td>Media and Communication</td><td>0.09</td><td>-0.01</td><td>0.00</td><td>0.03</td></tr><tr><td>Others</td><td>-3.11</td><td>-3.12</td><td>-3.12</td><td>-3.00</td></tr><tr><td>Migration and Integration</td><td>-17.16</td><td>-21.40</td><td>-19.80</td><td>-14.61</td></tr><tr><td>East Germany</td><td>0.01</td><td>-0.18</td><td>-0.17</td><td>0.14</td></tr><tr><td>Values, Polit. Culture, Social</td><td>3.56</td><td>1.31</td><td>9.90</td><td>9.65</td></tr></table>

Table 7: Percentage errors in the ablation experiment. LLM-base denotes no demographics; min(1var. \*) denotes best-performing experiment with one variable; min(w/o. \*) denotes best-performing experiment with all except one variable; LLM-all denotes all variables.

Label-level Breakdown with APE: LLM predictions show over- and underrepresentation of certain political topics compared to survey data. Table 12 and 13 show the comparison of predicted label percentages between survey and LLM answers in experiment 1 and 2 respectively. In Table 12, we observe that “Security", “Migration and Integration" and “Social Policy" topics are less represented than the survey in all LLM-texts and “Values, political culture and general social criticism" represented much more, with a mean APE=218.1. the Mixtral model emphasized “Environment Politics", whereas Llama2 focused on “Political Systems and Processes" more than others. Table 13 shows “Health Policy", “Values, political culture and general social criticism" and “Economic Policy" are consistently more represented as the text answer categories, whereas “Migration and Integration" and “Security" are less represented. We also calculated the mean APE per label to compare errors on average on which categories the Llama2 represented the political topics more accurately.

Factual Knowledge of Llama2 without Demographic Prompts. We also check actual knowledge of Llama2 model without giving any survey contexts in Table 15. It shows Llama2's general accuracy in providing vote percentages for major elections, though minor errors and formatting issues occur, especially in the 2020 U.S. election. For Germany's most important problem in 2019, Llama2 identifies key issues like climate change and economic security but gives rough percentage estimates rather than precise data. These results suggest that while the model captures broad trends, it struggles with exact figures when not given specific demographic or contextual prompts.

<table><tr><td>Experiment Name</td><td>Prompt Information</td></tr><tr><td>1_var_region</td><td>Der/Die Befragte lebt in {eastwest}. [/INST]</td></tr><tr><td>1_var_party</td><td>Der/Die Befragte unterstützt hauptsächlich {party }. [/INST]</td></tr><tr><td>1_var_education_degree</td><td>Der/Die Befragte {schulabschluss_clause} [/INST]</td></tr><tr><td>1_var_age</td><td>Der/Die Befragte ist {age} Jahre alt. [/INST]</td></tr><tr><td>1_var_gender</td><td>{artikel} Befragte ist {gender} [/INST]</td></tr><tr><td>1_var_vocational_degree</td><td>Der/Die Befragte {berufabschluss_clause} [/INST]</td></tr><tr><td>without_age</td><td>{artikel} Befragte ist {gender}. {pronoun } {schulabschluss_clause} und {berufabschluss_clause} {pronoun2} lebt in {eastwest} und unterstützt hauptsächlich</td></tr><tr><td>without_region</td><td>{party}. [/INST] {artikel} Befragte ist {age} Jahre alt und {gender}. {pronoun} {schulabschluss_clause} und {berufabschluss_clause} {pronoun2} unterstützt hauptsächlich {party}. [/INST]</td></tr><tr><td>without_vocational_degree</td><td>{artikel} Befragte ist {age} Jahre alt und {gender}. {pronoun} {schulabschluss_clause} {pronoun2} lebt in {eastwest} und unterstützt hauptsächlich {party}. [/INST]</td></tr><tr><td>without_education_degree</td><td>{artikel} Befragte ist {age} Jahre alt und {gender}. {pronoun} und {berufabschluss_clause} {pronoun2} lebt in {eastwest} und unterstützt hauptsächlich</td></tr><tr><td>without_party</td><td>{party}. [/INST] {artikel} Befragte ist {age} Jahre alt und {gender}. {pronoun} {schulabschluss_clause} und {berufabschluss_clause} {pronoun2} lebt in {eastwest}. [/INST]</td></tr><tr><td>without_gender</td><td>Der/Die Befragte ist {age} Jahre alt. Er/Sie {schulabschluss_clause} und {berufabschluss_clause} Er/Sie lebt in {eastwest} und unterstützt hauptsächlich {party }. [/INST]</td></tr></table>

Table 8: Ablation experiments and the modified prompt contents. 1\_var\_\* denotes the experimentation of prompting with only one variation\*. wi thout\_\* denotes the experimentation of prompting with all variables except \*.

<table><tr><td>Fine Labels</td><td>Coarse Labels</td></tr><tr><td>Election Campaign and Government Formation, Political Struc- tures and Processes, Democracy, Bureaucracy, Lobbyism, Cor- ruption</td><td>Political System and Processes</td></tr><tr><td>Values, political culture and general social criticism Social Policy, Social Justice, Poverty, Unemployment and Basic Social Policy</td><td>Values, political culture and general social criticism</td></tr><tr><td>Security, Pensions and Demographic Change</td><td></td></tr><tr><td>Health Policy, Nursing, Corona Pandemic</td><td>Health Policy</td></tr><tr><td>Family Policy, Gender Equality</td><td>Family and Gender Equality Policy</td></tr><tr><td>Education Policy, School Policy</td><td>Education Policy</td></tr><tr><td>Energy Policy, Environmental Policy, Climate Policy, Natural Environmental Policy Disasters</td><td></td></tr><tr><td>Economic Policy, Price Level, Infrastructure, Digital Infrastruc- Economic Policy ture, Transport Policy, Housing Policy</td><td></td></tr><tr><td>Terrorism, Internal Security, Crime and Violence, Radicalization Security and Extremism, Law and Justice, Defense Foreign Policy, Europe and European Union, (Germany&#x27;s Rela- Foreign Policy</td><td></td></tr><tr><td>tions with) Russia, (Germany&#x27;s Relations with) Turkey, Interna- tional Conflicts and Peace, War in Ukraine</td><td></td></tr><tr><td>Media Others</td><td>Media and Communication</td></tr><tr><td>Migration and Integration</td><td>Others</td></tr><tr><td></td><td>Migration and Integration</td></tr><tr><td>East Germany</td><td>East Germany</td></tr><tr><td>Not specified</td><td>Not specified</td></tr><tr><td>Don&#x27;t know</td><td>Don&#x27;t know</td></tr></table>

Table 9: Description of the Classes, originally in German, translated to English

<table><tr><td>Model</td><td>Original Answer</td><td>Translation</td></tr><tr><td>Gemma</td><td>Als 61-jährige Frau mit deutscher Staatsbürgerschaft, die in Westdeutschland lebt und hauptsächlich keine Partei unterstützt, ist das wichtigste Problem, mit dem Deutschland im November 2019 konfrontiert ist, die weiterhin bestehende Situation der COVID-19-Pandemie.</td><td>As a 61-year-old woman with German citizenship, living in West Germany and mainly not supporting any political party, the most important issue facing Germany in November 2019 is the ongoing situation of the COVID-19 pandemic.</td></tr><tr><td>Llama2</td><td>Das wichtigste Problem, das Deutschland im November 2019 konfrontiert, ist die sich erhöhte Unsicherheit und Angst in der Gesellschaft, insbesondere bei den Menschen, die sich nicht in der Mitte der Gesellschaftskontaktiven Gruppenzugehörigkeit finden.</td><td>The most important problem facing Germany in November 2019 is the growing insecurity and fear in society, especially among people who do not belong to the centre of society.</td></tr><tr><td>Mixtral</td><td>Eines der wichtigsten Probleme, mit denen Deutschland im November 2019 konfrontiert ist, ist der Umgang mit den Folgen des Klimawandels und die Reduzierung von Treibhausgasemissionen, was auch entscheidend für die zukünftige Wirtschaft ist.</td><td>One of the most important issues facing Germany in November 2019 is dealing with the consequences of climate change and reducing greenhouse gas emissions, which is also crucial for the future economy.</td></tr></table>

Table 10: Sample Answers from the Model Experiment

![](images/2e07ef32653432d7d713a4c299c95eda8bfef824ead318818bf60dbda14b2814.jpg)  
Figure 12: Label distributions of three experimented LLMs

<table><tr><td>wave</td><td>12</td><td>13</td><td>14</td><td>15</td><td>16</td><td>17</td><td>18</td><td>19</td><td>20</td><td>21</td></tr><tr><td>PA survey resample</td><td>0.38</td><td>0.55</td><td>0.52</td><td>0.46</td><td>0.42</td><td>0.37</td><td>0.38</td><td>0.36</td><td>0.34</td><td>0.52</td></tr><tr><td>κ survey resample</td><td>0.27</td><td>0.27</td><td>0.29</td><td>0.21</td><td>0.22</td><td>0.23</td><td>0.24</td><td>0.24</td><td>0.23</td><td>0.25</td></tr><tr><td>PA Llama2</td><td>0.14</td><td>0.56</td><td>0.41</td><td>0.46</td><td>0.31</td><td>0.24</td><td>0.23</td><td>0.21</td><td>0.19</td><td>0.25</td></tr><tr><td>κ Llama2</td><td>0.02</td><td>0.01</td><td>0.03</td><td>0.03</td><td>0.03</td><td>0.02</td><td>0.02</td><td>0.04</td><td>0.03</td><td>0.02</td></tr></table>

Table 11: Proportional Agreement (PA) and (κ) Scores. The original survey is the first annotator, and the second annotator is the survey resample (comparison to stratified sampling from the original survey) or Llama2 model

<table><tr><td>Category</td><td>Source</td><td>Gemma</td><td>Llama2</td><td>Mixtral</td><td>Mean APE</td></tr><tr><td rowspan="2">East Germany</td><td>LLM</td><td>0.5</td><td>0.7</td><td>1.8</td><td rowspan="2">368.07</td></tr><tr><td>Survey</td><td>0.2</td><td>0.2</td><td>0.2</td></tr><tr><td rowspan="2">Economic Policy</td><td>LLM</td><td>4.9</td><td>20.2</td><td>14.8</td><td rowspan="2">78.02</td></tr><tr><td>Survey</td><td>9.0</td><td>9.0</td><td>9.0</td></tr><tr><td rowspan="2">Education Policy</td><td>LLM</td><td>0.1</td><td>0.5</td><td>0.0</td><td rowspan="2">91.38</td></tr><tr><td>Survey</td><td>2.4</td><td>2.4</td><td>2.4</td></tr><tr><td rowspan="2">Environmental Policy</td><td>LLM</td><td>1.2</td><td>14.8</td><td>35.3</td><td rowspan="2">78.09</td></tr><tr><td>Survey</td><td>14.6</td><td>14.6</td><td>14.6</td></tr><tr><td rowspan="2">Family and Gender Equality Policy</td><td>LLM</td><td>0.1</td><td>0.4</td><td>0.1</td><td rowspan="2">56.62</td></tr><tr><td>Survey</td><td>0.3</td><td>0.3</td><td>0.3</td></tr><tr><td rowspan="2">Foreign Policy</td><td>LLM</td><td>28.8</td><td>2.0</td><td>0.3</td><td rowspan="2">753.35</td></tr><tr><td>Survey</td><td>1.3</td><td>1.3</td><td>1.3</td></tr><tr><td rowspan="2">Health Policy</td><td>LLM</td><td>41.6</td><td>3.3</td><td>0.4</td><td rowspan="2">1344.72</td></tr><tr><td>Survey</td><td>1.1</td><td>1.1</td><td>1.1</td></tr><tr><td rowspan="2">Media and Communication</td><td>LLM</td><td>0.1</td><td>0.1</td><td>0.0</td><td rowspan="2">83.89</td></tr><tr><td>Survey</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td rowspan="2">Migration and Integration</td><td>LLM</td><td>7.2</td><td>8.6</td><td>14.9</td><td rowspan="2">57.41</td></tr><tr><td>Survey</td><td>24.1</td><td>24.1</td><td>24.1</td></tr><tr><td rowspan="2">Others</td><td>LLM</td><td>0.0</td><td>0.2</td><td>6.1</td><td rowspan="2">97.60</td></tr><tr><td>Survey</td><td>3.0</td><td>3.0</td><td>3.0</td></tr><tr><td rowspan="2">Political System and Processes</td><td>LLM</td><td>1.4</td><td>15.7</td><td>2.6</td><td rowspan="2">84.93</td></tr><tr><td>Survey</td><td>7.5</td><td>7.5</td><td>7.5</td></tr><tr><td rowspan="2">Security</td><td>LLM</td><td>2.0</td><td>1.8</td><td>3.3</td><td rowspan="2">70.10</td></tr><tr><td>Survey</td><td>7.9</td><td>7.9</td><td>7.9</td></tr><tr><td rowspan="2">Social Policy</td><td>LLM</td><td>1.2</td><td>16.0</td><td>12.2</td><td rowspan="2">60.48</td></tr><tr><td>Survey</td><td>24.8</td><td>24.8</td><td>24.8</td></tr><tr><td rowspan="2">Values, Political Culture, and Social Criticism</td><td>LLM</td><td>10.8</td><td>15.6</td><td>8.3</td><td rowspan="2">207.01</td></tr><tr><td>Survey</td><td>3.8</td><td>3.8</td><td>3.8</td></tr><tr><td>APE</td><td></td><td>150.0</td><td>70.0</td><td>73.0</td><td></td></tr></table>

Table 12: Comparison of Predicted Label Percentages in Experiment 1. Colors indicate differences between LLM and survey: green (LLM > Survey + 1%), red (LLM < Survey − 1%), black (|LLM − Survey| < 1%).

<table><tr><td>Category</td><td>src</td><td>12</td><td>13</td><td>14</td><td>15</td><td>16</td><td>17</td><td>18</td><td>19</td><td>20</td><td>21</td><td>mean APE</td></tr><tr><td>Foreign Policy</td><td>llm survey</td><td>1.8 1.2</td><td>0.1 0.3</td><td>0.5 0.3</td><td>0.3 0.3</td><td>1.0 0.3</td><td>0.8 0.4</td><td>1.0 0.6</td><td>1.2 0.7</td><td>1.4 0.4</td><td>1.2 0.4</td><td>112.94</td></tr><tr><td>Education Policy</td><td>llm survey</td><td>0.5 2.2</td><td>0.2 1.0</td><td>0.2 1.1</td><td>0.4 1.4</td><td>0.6 2.0</td><td>0.4 2.4</td><td>0.4 1.7</td><td>0.4 2.0</td><td>0.4 2.2</td><td>0.6 1.1</td><td>75.13</td></tr><tr><td>Family and Gender Equality Policy</td><td>llm survey</td><td>0.2 0.3</td><td>0.0 0.1</td><td>0.1 0.1</td><td>0.0 0.3</td><td>0.1 0.2</td><td>0.0 0.3</td><td>0.0 0.2</td><td>0.1 0.2</td><td>0.1 0.2</td><td>0.1 0.2</td><td>61.72</td></tr><tr><td>Health Policy</td><td>llm survey</td><td>3.1 1.0</td><td>92.4 59.5</td><td>68.9 54.2</td><td>78.5 54.2</td><td>55.6 47.4</td><td>50.6 33.7</td><td>48.3 33.5</td><td>33.9 26.0</td><td>34.0 20.2</td><td>32.0 57.9</td><td>60.4</td></tr><tr><td>LLM Refusal</td><td>llm survey</td><td>0.0 0.0</td><td>0.0 0.0</td><td>0.0 0.0</td><td>0.0 0.0</td><td>0.0 0.0</td><td>0.0 0.0</td><td>0.0 0.0</td><td>0.0 0.0</td><td>0.0 0.0</td><td>0.0 0.0</td><td>nan</td></tr><tr><td>Media and Communication</td><td>llm survey</td><td>0.1 0.0</td><td>0.0 0.0</td><td>0.0 0.1</td><td>0.0 0.1</td><td>0.0 0.1</td><td>0.0 0.0</td><td>0.0 0.0</td><td>0.0 0.0</td><td>0.0 0.0</td><td>0.0 0.1</td><td>98.33</td></tr><tr><td>Migration and Integration</td><td>llm survey</td><td>9.6 23.0</td><td>0.5 9.5</td><td>1.6 11.1</td><td>1.4 6.4</td><td>2.1 6.5</td><td>2.3 10.9</td><td>2.4 10.3</td><td>2.2 11.7</td><td>2.1 10.3</td><td>1.8 7.1</td><td>77.65</td></tr><tr><td>East Germany</td><td>llm survey</td><td>0.3 0.2</td><td>0.1 0.0</td><td>0.4 0.0</td><td>0.1 0.0</td><td>0.2 0.0</td><td>0.4 0.0</td><td>0.3 0.0</td><td>0.3 0.0</td><td>0.3 0.1</td><td>0.4 0.1</td><td>949.73</td></tr><tr><td>Political System and Processes</td><td>llm survey</td><td>17.3 7.2</td><td>0.3 2.7</td><td>1.5 3.0</td><td>1.1 5.2</td><td>1.5 5.4</td><td>1.1 4.6</td><td>1.4 4.6</td><td>1.5 4.5</td><td>1.5 6.7</td><td>1.9 3.2</td><td>75.8</td></tr><tr><td>Security</td><td>llm survey</td><td>1.0 7.5</td><td>0.2 1.8</td><td>0.2 4.0</td><td>0.3 2.4</td><td>0.6 2.7</td><td>0.7 3.0</td><td>0.5 2.2</td><td>0.4 2.2</td><td>0.4 2.1</td><td>0.4 2.3</td><td>83.5</td></tr><tr><td>Others</td><td>llm survey</td><td>0.1 3.0</td><td>0.0 2.1</td><td>0.0 2.1</td><td>0.0 3.2</td><td>0.0 3.1</td><td>0.0 2.6</td><td>0.0 2.5</td><td>0.0 2.2</td><td>0.0 2.3</td><td>0.0 1.8</td><td>99.37 56.05</td></tr><tr><td>Social Policy</td><td>llm survey</td><td>17.5 23.7 17.2</td><td>1.2 7.5</td><td>9.5 9.1</td><td>5.2 8.6</td><td>14.0 8.8</td><td>16.3 12.2</td><td>15.1 10.7 12.7</td><td>17.0 13.4 23.4</td><td>17.4 14.8 22.6</td><td>26.3 8.0 14.8</td><td>30.98</td></tr><tr><td>Environmental Policy</td><td>llm survey</td><td>13.9</td><td>0.5 2.0</td><td>2.3 3.2</td><td>3.2 4.2</td><td>8.1 7.4</td><td>10.5 11.5</td><td>18.1</td><td>20.5 4.7</td><td>21.1 4.5</td><td>7.8 5.8</td><td>191.96</td></tr><tr><td>Values, Political Culture and General Social Criticism</td><td>llm survey</td><td>13.4 3.6 17.9</td><td>1.3 0.8</td><td>5.8 1.4</td><td>2.6 1.4</td><td>4.5 2.0</td><td>3.7 1.5 13.2</td><td>4.7 1.4 13.2</td><td>1.4 14.9</td><td>1.4 15.3</td><td>2.1 14.8</td><td>59.31</td></tr><tr><td>Economic Policy</td><td>llm survey llm</td><td>8.3 0.0</td><td>3.1 8.9</td><td>9.0 6.2</td><td>6.9 8.4</td><td>11.8 9.4</td><td>11.8 0.0</td><td>9.2 0.0</td><td>10.2 0.0</td><td>12.5 0.0</td><td>4.9 0.0</td><td>100.0</td></tr><tr><td>Not Specified</td><td>survey llm</td><td>3.7 0.0</td><td>0.0 3.0 0.0</td><td>0.0 3.5 0.0</td><td>0.0 3.1 0.0</td><td>0.0 3.8 0.0</td><td>4.0 0.0</td><td>4.0 0.0</td><td>4.1 0.0</td><td>4.9 0.0</td><td>2.4 0.0</td><td>100.0</td></tr><tr><td>Do Not Know APE</td><td>survey</td><td>1.1</td><td>0.7 66.0</td><td>0.8</td><td>0.8</td><td>0.9</td><td>1.1 50.0</td><td>0.9 53.0</td><td>1.0</td><td>0.8 51.0</td><td>0.7 80.0</td><td></td></tr><tr><td></td><td></td><td>71.0</td><td></td><td>46.0</td><td>50.0</td><td>38.0</td><td></td><td></td><td>47.0</td><td></td><td></td><td></td></tr></table>

Table 13: Comparison of Predicted Label Percentages between survey and LLM-answers in Experiment 2. Colors indicate differences between LLM and survey: green $( \mathrm { L L M } > \mathrm { S u r v e y } + 1 \% )$ , red $( \mathrm { L L M } < \mathrm { S u r v e y } - 1 \% )$ , black (|LLM − Survey| < 1%).

![](images/1b03dd9455423812c73cb43a593425e5e5915e26e9464559dadc95fb5a5846dc.jpg)

![](images/fe29852181875066314320149ba47b067debba7abfd829e83c7714a05ac1dcf1.jpg)

![](images/21a4b56354b6bf2d49d6222cd08a2ead760cab52893b5baaaab78f11d2845074.jpg)  
Figure 13: The mean and ±1 standard deviation of JS Distances for social groups in Experiment 2.

![](images/d30b58453b1ab82dfe2b334a84d007250f8b018fe46154893a7359c67f2d1ac3.jpg)  
Figure 14: Cramér's Values for pairwise patterns between the six prompting variables and text answers from the survey and LLMs in Experiment 2

<table><tr><td>Social Group Category</td><td>Social Group</td><td>Gemma</td><td>Llama2</td><td>Mixtral</td></tr><tr><td>Population</td><td>Population</td><td>0.617</td><td>0.287</td><td>0.295</td></tr><tr><td rowspan="4">Age Groups</td><td>18-29</td><td>0.638</td><td>0.233</td><td>0.246</td></tr><tr><td>30-44</td><td>0.613</td><td>0.310</td><td>0.310</td></tr><tr><td>45-59</td><td>0.627</td><td>0.307</td><td>0.309</td></tr><tr><td>60+</td><td>0.610</td><td>0.286</td><td>0.299</td></tr><tr><td rowspan="13">Vocational Degree</td><td>Completed vocational internship/volunteer work</td><td>0.640</td><td>0.491</td><td>0.278</td></tr><tr><td>Vocational school diploma</td><td>0.600</td><td>0.314</td><td>0.264</td></tr><tr><td>University of applied sciences degree</td><td>0.626</td><td>0.272</td><td>0.339</td></tr><tr><td>Specialist school diploma</td><td>0.603</td><td>0.269</td><td>0.355</td></tr><tr><td>Completed apprenticeship</td><td>0.618</td><td>0.304</td><td>0.305</td></tr><tr><td>Master craftsman or technician qualification</td><td>0.626</td><td>0.297</td><td>0.374</td></tr><tr><td>University degree</td><td>0.618</td><td>0.287</td><td>0.334</td></tr><tr><td>In vocational training</td><td>0.648</td><td>0.238</td><td>0.306</td></tr><tr><td>Commercial or agricultural apprenticeship</td><td>0.629</td><td>0.335</td><td>0.287</td></tr><tr><td>Commercial apprenticeship</td><td>0.630</td><td>0.302</td><td>0.320</td></tr><tr><td>No vocational training completed</td><td>0.619</td><td>0.289</td><td>0.242</td></tr><tr><td>Male</td><td>0.617</td><td>0.300</td><td>0.325</td></tr><tr><td rowspan="2">Gender</td><td>Female</td><td>0.619</td><td>0.279</td><td>0.279</td></tr><tr><td>AfD</td><td>0.618</td><td>0.329</td><td>0.351</td></tr><tr><td rowspan="8">Leaning Party</td><td>CDU/CSU</td><td>0.612</td><td>0.325</td><td>0.301</td></tr><tr><td>FDP</td><td>0.648</td><td>0.374</td><td>0.423</td></tr><tr><td>Grünen</td><td>0.639</td><td>0.232</td><td>0.556</td></tr><tr><td>A minor party</td><td>0.606</td><td>0.307</td><td>0.323</td></tr><tr><td>Linke</td><td>0.594</td><td>0.256</td><td>0.267</td></tr><tr><td>SPD</td><td>0.619</td><td>0.275</td><td>0.318</td></tr><tr><td>No party</td><td>0.651</td><td>0.340</td><td>0.352</td></tr><tr><td>East Germany</td><td></td><td></td><td></td></tr><tr><td rowspan="2">Region</td><td>West Germany</td><td>0.599 0.623</td><td>0.300 0.285</td><td>0.294 0.299</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td rowspan="6">Education Degree</td><td>High school diploma</td><td>0.617</td><td>0.279</td><td>0.326</td></tr><tr><td>Higher education entrance qualification</td><td>0.622</td><td>0.293</td><td>0.343</td></tr><tr><td>Secondary school diploma</td><td>0.625</td><td>0.320</td><td>0.241</td></tr><tr><td>Intermediate school diploma</td><td>0.619</td><td>0.295</td><td>0.302</td></tr><tr><td>Student</td><td>0.833</td><td>0.680</td><td>0.379</td></tr><tr><td>No school diploma</td><td>0.667</td><td>0.385</td><td>0.348</td></tr></table>

Table 14: Detailed JS Distances in each social group category in Experiment 1

![](images/097d984f5b789acfaf0251105b8900ddb1155acaf71af80c5c24c73824a89ad8.jpg)

![](images/566dc1a0a4aa6768b56eeedbabf50617af3eddcff67928952e2841ce8e33712e.jpg)

![](images/e683fa121056f55daf9061d19a73fa147339f535e7578a16e63c7e103af9fadf.jpg)

![](images/7150daf0d25908cc3be3cabce68cafb82170a17b7d136c4407e40cba74d0827b.jpg)  
Figure 15: Survey's Subpopulation Entropy and the JS Distance in Experiment 2 (mean values for waves 12-21).

<table><tr><td>Prompt</td><td>Llama2 Answer</td></tr><tr><td>What were the vote percentages for the major candidates in the 2016 US</td><td>In the 2016 US presidential election, the vote percentages for the major candidates were:</td></tr><tr><td>presidential election?</td><td>Donald Trump (Republican): 46.1%</td></tr><tr><td></td><td>Hillary Clinton (Democratic): 48.2%</td></tr><tr><td></td><td>Gary Johnson (Libertarian): 3.3%</td></tr><tr><td></td><td>Jill Stein (Green): 1.1%</td></tr><tr><td></td><td>Note: These figures represent the national popular vote, excluding Maine and Nebraska, which allocate electoral votes proportionally.</td></tr><tr><td>What were the vote percentages for</td><td>In the 2020 United States presidential election, the vote percentages for the major</td></tr><tr><td>the major candidates in the 2020 US presidential election?</td><td>candidates were as follows: * Joe Biden (Democratic Party): 51.1% (665,8533,295 votes)*</td></tr><tr><td></td><td>Donald Trump (Republican Party): 47.5% (62,2045,637 votes)*</td></tr><tr><td></td><td>Other candidates: 1.4% (19,362,8822 votes) Note: These figures are based on the official results from the Federal Election</td></tr><tr><td></td><td>Commission and may not reflect final totals or percentages.</td></tr><tr><td>What were the vote percentages in the</td><td></td></tr><tr><td>2017 German federal election?</td><td>The vote percentages in the 2017 German federal election were as follows:</td></tr><tr><td></td><td>Chancellor Angela Merkel&#x27;s Christian Democratic Union (CDU): 32.9%</td></tr><tr><td></td><td>Social Democratic Party (SPD): 20.5%</td></tr><tr><td></td><td>Alternative for Germany (AfD): 12.6%</td></tr><tr><td></td><td>Free Democratic Party (FDP): 10.7%</td></tr><tr><td></td><td>Green Party: 9.2%</td></tr><tr><td></td><td></td></tr><tr><td></td><td>Note: These figures represent national vote share. Actual seat distribution may differ due to proportional representation.</td></tr><tr><td>What is the most important problem in Germany in November 2019? List the</td><td>It is difficult to determine the most important problem in Germany in November 2019, as opinions and priorities can vary greatly among individuals and groups. However, based on current events and societal trends, here are some of the issues</td></tr><tr><td>answers and their percentages for the German population. Do your best guess.</td><td>that might be considered important by the German population: 1. Climate change and environmental protection (25-30%) - Concerns about climate</td></tr><tr><td></td><td>change and the environment have been growing in Germany, with increased activism and calls for stronger government action to reduce carbon emissions and</td></tr><tr><td></td><td>protect natural resources. 2. Economy and job security (20-25%) - The German economy has been facing</td></tr><tr><td></td><td>challenges, including a slowdown in growth and job insecurity, particularly in the</td></tr><tr><td></td><td>automotive and manufacturing sectors. 3. Immigration and integration (15-15%) - Germany has been a destination for</td></tr><tr><td></td><td>many refugees and migrants, and the integration of these individuals into society has been a persistent challenge, with debates around issues such as employment,</td></tr><tr><td></td><td>education, and social cohesion.</td></tr><tr><td></td><td>4. Healthcare and social welfare (5-10%) - Germany&#x27;s healthcare system has been</td></tr><tr><td></td><td>under strain, with debates over funding and access to certain medical services,</td></tr><tr><td></td><td>particularly for low-income families. 5. European and international politics (&lt; 5%) - Germany has played a key role in</td></tr></table>

Table 15: Checking the factual knowledge of Llama2 model without giving any survey contexts