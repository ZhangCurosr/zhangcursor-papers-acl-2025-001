# APPL: A Prompt Programming Language for Harmonious Integration of Programs and Large Language Model Prompts

Honghua Dong<sup>1,2,\*</sup>, Qidong Su<sup>1,2,3,\*</sup>, Yubo Gao<sup>1,2,3</sup>, Zhaoyu Li<sup>1,2</sup>, Yangjun Ruan<sup>1,2</sup>, Gennady Pekhimenko<sup>1,2,3,4</sup>, Chris J. Maddison<sup>1,2,4</sup>, Xujie Si<sup>1,2,4</sup>

<sup>1</sup>University of Toronto <sup>2</sup>Vector Institute <sup>3</sup>CentML <sup>4</sup>CIFAR AI Chair

\*Equal contribution

Correspondence: honghuad@cs.toronto.edu, six@cs.torontu.edu

## Abstract

Large Language Models (LLMs) have become increasingly capable of handling diverse tasks with the aid of well-crafted prompts and integration of external tools, but as task complexity rises, the workflow involving LLMs can be complicated and thus challenging to implement and maintain. To address this challenge, we propose APPL, A Prompt Programming Language that acts as a bridge between computer programs and LLMs, allowing seamless embedding of prompts into Python functions, and vice versa. APPL provides an intuitive and Python-native syntax, an efficient parallelized runtime with asynchronous semantics, and a tracing module supporting effective failure diagnosis and replaying without extra costs. We demonstrate that APPL programs are intuitive, concise, and efficient through representative scenarios including Chain-of-Thought with self-consistency (CoT-SC) and ReAct tooluse agent. We further use LLMs to judge the language design between APPL and previous work, where the results indicate that codes written in APPL are more readable and intuitive. Our code, tutorial and documentation are available at https://github.com/appl-team/appl.

## 1 Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities in understanding and generating texts across a broad spectrum of tasks (Raffel et al., 2020; Brown et al., 2020; Chowdhery et al., 2023; Touvron et al., 2023a,b; OpenAI, 2023a). Their success has contributed to an emerging trend of regarding LLMs as novel operating systems (Andrej Karpathy, 2023; Ge et al., 2023; Packer et al., 2023). Compared with traditional systems that precisely execute structured and well-defined programs written in programming languages, LLMs can be guided through flexible natural language prompts to perform tasks that are beyond the reach of traditional programs.

![](images/4f11b531d4428be91c8f26717cd3c19787c3ab0c3d8d46478259930466106c9d.jpg)  
Figure 1: We introduce APPL, a prompt programming language that integrates conventional programs and natural language prompts to provide a unified interface for users to access computers and LLMs together. APPL also facilitates users fusing the strengths of computer programs and LLMs by providing convenient conventions between them.

Meanwhile, there is a growing interest in harnessing the combined strengths of LLMs and conventional computing to address more complex challenges. Approaches like tree-of-thoughts (Yao et al., 2024), RAP (Hao et al., 2023) and LATS (Zhou et al., 2023) integrate search-based algorithms with LLM outputs, showing improved outcomes over using LLMs independently. Similarly, the creation of semi-autonomous agents such as AutoGPT (Richards, 2023) and Voyager (Wang et al., 2023) demonstrates the potential of LLMs to engage with external tools, like interpreters, file systems, and skill libraries, pushing the boundaries of what integrated systems can achieve. However, as task complexity rises, the workflow involving LLMs becomes more intricate, requiring more complex prompts guiding LLMs and more sophisticated programs implementing the workflow, which are both challenging to implement and maintain.

To address these challenges, as illustrated in Figure 1, we propose APPL, A Prompt Programming Language for harmonious integration of conventional programming and LLMs. More specifically, APPL’s key features include:

(1) Readability and maintainability via seamless integration with Python. As shown in Figure 2, APPL seamlessly embeds natural language prompts to Python programs, maintaining prompts’ readability while inheriting modularity, reusability, and dynamism from the host language.

![](images/11adbdca7ed93e067cd1988968193b32ed885d894430730b2beddaab0e761c49.jpg)  
(c) Example call to answer\_questions.  
(d) Example LLM prompts and responses for the first question.  
Figure 2: (a) The APPL program for answering multiple questions about a quotation by first extracting the author’s name. In this APPL program, the @ppl marks the function to be APPL function, which provides a context that interactive expression statements (highlighted lines) can be interacted with. Passing the context across APPL functions is implicitly handled along with function calls, mimicking the process of stack frame management. The gen function requests an LLM generation using the prompts stored in the context, with other optional arguments following the OpenAI’s API (OpenAI). For instance, the gen in line 4 uses the prompt composed by (highlighted) lines 7,8,10 (since context is copied), and 3. The LLM generation is computed asynchronously and only synchronizes when the value is needed, therefore independent calls are automatically parallelized as shown in line 11. (b) A Python program implementing the same function uses more codes, even with the helper functions. Furthermore, extra codes are needed to parallelize get\_answer function calls. (c) Example usage for the program and (d) its corresponding LLM prompts and example responses.

(2) Automatic parallelization via asynchronous computation. APPL schedules LLM calls asynchronously, leveraging potential independence among them to facilitate parallelization. This offloads the burden of users to manage synchronization manually, with almost no extra work as shown in Figure 2a. Its effectiveness is validated in Section 5.2 where experiments show significant accelerations for three popular applications.

(3) Smooth transition between structured data and natural language. APPL enables seamlessly converting program objects into prompts (namely promptify). For example, promptifying Python functions converts them into LLMunderstandable tool specifications. Specifically, APPL implements an as\_tool function that analyzes a Python function’s docstrings and signatures to create tool specifications (in JSON schema format) that being used by LLM calls. In another direction, APPL enables the specification of output structures for LLMs, such as ensuring outputs conform to Pydantic’s BaseModel (Colvin et al.).

APPL strives to be a powerful, yet intuitive extension of Python and leverages the unique capabilities of LLMs while being compatible with Python’s syntax and ecosystem. It empowers developers to easily build sophisticated programs that harness the power of LLMs in a manner that’s both efficient and accessible.

## 2 Related Work

LLM programs. The programs involving LLMs continue evolving sophisticated to tackle more challenging tasks, from few-shot prompts (Brown et al., 2020) to elaborated thought process (Wei et al., 2022; Ning et al., 2023; Yao et al., 2024; Besta et al., 2023), from chatbot (OpenAI, 2022) to tooluse agents (Nakano et al., 2021; Ahn et al., 2022; Liu et al., 2022; Liang et al., 2023; Richards, 2023; Patil et al., 2023; Wang et al., 2023; Qin et al., 2023), virtual emulators (Ruan et al., 2024), operating systems (Packer et al., 2023), and multi-agent systems (Park et al., 2023; Hong et al., 2023; Qian et al., 2023). The increasing complexity necessitates powerful frameworks to facilitate the development of those programs.

LLM software stack. On the low level, LLM services are provided by high-performance backends (Gerganov, 2023; HuggingFace, 2023; Kwon et al., 2023) as APIs like OpenAI APIs (OpenAI). Building on this, modules with specific functionality are invented, like unifying APIs (BerriAI, 2023), constraining LLM outputs (Willard and Louf, 2023; Liu, 2023) and parallelizing function calls (Kim et al., 2023), where APPL utilizes some of them as building blocks. LMQL (Beurer-Kellner et al., 2023), Guidance (GuidanceAI, 2023), and SGLang (Zheng et al., 2023) are languages similar to APPL. These languages provide developers with precise tools for crafting sophisticated model interactions, enabling finer-grained control over LLM behaviors. We compare APPL with them by case studies in Section 5 and summarize the comparison in Table 1, where APPL provides more automatic supports for prompting and parallelizing LLM calls to streamline the whole workflow. There are also higher-level general-purpose frameworks (Okuda and Amarasinghe, 2024; Prefect, 2023; Khattab et al., 2023; Chase, 2022) and agent-specific frameworks (Team, 2023; Li et al., 2023; Wu et al., 2023). The pipelines in these frameworks can also be implemented using prompt languages like APPL.

<table><tr><td>Features</td><td>LMQL (Beurer-Kellner et al., 2023)</td><td>Guidance (GuidanceAI, 2023)</td><td>SGLang (Zheng et al., 2023)</td><td>APPL</td></tr><tr><td>Language Syntax Parallelization and Asynchrony</td><td>Python-like Sub-optimal</td><td>Python Manual</td><td>Python Auto</td><td>Python Auto</td></tr><tr><td>Prompt Capturing</td><td>Auto</td><td>Manual</td><td>Manual</td><td>Auto</td></tr><tr><td>Prompt Context Passing Promptify Tool</td><td>Limited</td><td>Manual</td><td>Manual</td><td>Auto</td></tr><tr><td>Prompt Compositor</td><td>Manual No</td><td>Manual No</td><td>Manual No</td><td>Auto Yes</td></tr><tr><td>Generation Output Retrieval</td><td></td><td></td><td></td><td></td></tr><tr><td>Generation without Side-effects</td><td>Template variables No</td><td>str-index No</td><td>str-index No</td><td>Python-native Yes</td></tr><tr><td>Failure Recovery</td><td>No</td><td>No</td><td>No</td><td>Yes</td></tr></table>

Table 1: Comparison across Prompt Languages. APPL enhances the user experience of crafting LLM-augmented programs with Python-native syntax while achieving high performance with an efficient parallelizable runtime system. We carefully design the language features involved in prompting and generation, which streamlines the whole workflow. APPL also provides failure recovery through its tracing mechanism. See Sections 3 for the language design and Section 5 for more comparison details.

## 3 Language Design and Features

## 3.1 Design Principles of APPL

(1) Readability and Flexibility. Natural language provides a readable and understandable human-LLM interface, while programming languages enable flexible controls, clear structures, modularization, reusability, and external tools. APPL aims to utilize both strengths.

(2) Easy Parallelization. Parallelization is a critical performance optimization for LLM text generation (Yu et al., 2022; Kwon et al., 2023). APPL aims to provide transparent support for parallelization with almost no extra code introduced.

(3) Convenient Conversion. APPL aims to facilitate easy conversion between structured data (including codes) for programming and the natural language for LLMs.

## 3.2 APPL Syntax and Semantics

As illustrated in the APPL program of Figure 2a, the syntax and semantics of APPL are extended from Python with slight modifications to integrate natural language prompts and LLMs calls into the traditional programming constructs deeply.

APPL functions are the fundamental building blocks of APPL, denoted by the @ppl decorator. Each APPL function has an implicit context to store prompt-related information. Within APPL functions, the expression statements are interpreted as “interactions” with the context, where the typical behavior is to append strings to the prompt.

Expression statements are standalone Python statements formed by an expression, for example, "str", while an assignment (e.g., a = "str") is not. Their evaluation result is handled differently according to the environment. For example, it causes no extra effect in standard Python while the result is displayed in the interactive environment in Jupyter. Similarly, APPL provides an environment where each APPL function contains a scratchpad of prompts (i.e. context), and the values of expression statements will be converted to prompts (via promptify) and appended to the prompt scratchpad. For example, a standalone string as line 7 or 8 in Figure 2a will be appended to the scratchpad.

![](images/f3fb617b31612a0ebf64bba5bd685f6df1a7704565bce66ee5825098efb8bd81.jpg)  
Figure 3: Overview of the APPL runtime. The code is executed following Python control flows, with prompts and LLM responses being captured into the conversation (can be retrieved by convo(), similar to locals() and globals() in the Python frame). The gen uses the conversation captured so far as prompts to call LLMs and is executed asynchronously, so that independent calls can be parallelized easily. APPL allows using Python functions as tools for LLMs by extracting information from their docstrings and signatures. When tools are provided, their specifications are included in the request and the returned tool calls can be easily executed to get results. See detailed code in Figure 9a.

![](images/2a09927bf39c8b5accb14e4b728e8083b85b32697324d543ed254349217ea335.jpg)  
Figure 4: Illustration of the usage of Definitions and Prompt Compositors.

LLM generations (gen), when being called, implicitly use the accumulated prompt in the context of the APPL function. A special case is the standalone f-string, which is split into parts and they are captured as prompts in order, detailed in Appendix A.1. This provides a more readable way to combine prompts and LLM generations in one line. For example, line 10 of Figure 2a means start generation by first adding the prefix “The name of the author is ” to the prompt, which matches human intuition.

Context managers modify how prompt statements are accumulated to the prompt in the context. APPL provides two types of context managers: (1) Role Changers (such as AIRole in Figure 2a) are used in chats to specify the owner of the message, and can only be used as the outmost one. (2) Prompt Compositors (such as NumberedList in

Figure 4) specify the delimiter, indention, indexing format, prologue, and epilogue. These compositors allow for meticulously structured finer-grained prompts in hierarchical formatting through native Python syntax, greatly enhancing flexibility without losing readability.

Definition is a special base class for custom definitions. As shown in Figure 4, once declared, Definitions can be easily referred to in the f-string, or used to create an instance by completing its description. They are useful for representing concepts that occur frequently in the prompt and support varying the instantiation in different prompts. See more usage in Appendix G when building a long prompt.

## 3.3 APPL Runtime

APPL runtime context. Similar to Python’s runtime context that contains local and global variables (locals() and globals()), the context of APPL functions contains local and global prompts that can be accessed via records() and convo(). The example in Figure 3 illustrates how the prompts in the context changed during the execution.

Context passing across functions. When a APPL function (caller) calls another APPL function (callee), users might expect different behaviors for initializing the callee’s context depending on specific use cases. APPL provides four ways as shown in Figure 5:

![](images/2fb01d983bd639afb82993a2ca3dcb1600281c4c7a2b8560726041738efa39a8.jpg)  
Figure 5: Illustration of four different ways of inter-function context passing. Usage examples can be found in Section E and Appendix A.2. Prompts connected by solid arrows belong to the same prompt context. Dashed arrows represent the control flow of function calls and returns.

1. new: Creates a clean context. This is the default and most common way.

2. copy: Creates a copy of the caller’s context. This is similar to “call by value” where the modification to the copy will not influence the original context. This is useful for creating independent and asynchronize LLM calls as shown in Figure 2a.

3. same: Uses the same context as the caller. This is similar to “call by reference” where the modifications are directly applied to the original context.

4. resume: Resumes callee’s context from the last run. This makes the function stateful and suitable for building interactive agents with history (see Figure 18b for an example).

When decomposing a long prompt into smaller modules, both new and same can be used as shown in Figure 11, but the default way (new) should be prioritized when possible.

Asynchronous semantics. To automatically leverage potential independence among LLM calls and achieve efficient parallelization, we borrow the idea of the asynchronous semantics from Py-Torch (Paszke et al., 2019). Asynchronous semantics means the system can initiate operations that run independently of the main execution thread, allowing for parallelism and often improving performance. In APPL, LLM calls (gen) are asynchronous, which means the main thread proceeds without being blocked by gen until accessing the generation results is necessary. See implementation details in Section 4.2.

Tool calling. Tool calling is an indispensable ability for LLM agents to interact with external resources. To streamline the process of providing tools to LLMs, APPL automatically creates tool specifications from Python functions by analyzing their docstrings and signatures (detailed in Appendix A.3). As shown in Figure 6, the Python functions is\_lucky and search can be directly utilized as tools for LLMs. Therefore, thanks to the well-written docstrings, the functions from Python’s rich ecosystem can be integrated into LLMs as tools with minimal effort. Moreover, APPL automatically converts LLMs’ tool call outputs as function calls that can be directly executed to obtain results.

![](images/fa74e24dbe12fc23e1476589b7bfc80ca98e6056bdd9b6cb5bd1ed78026cb39c.jpg)  
Figure 6: Tool calling example in APPL.

Tracing and caching. Besides standard caching for LLM calls, APPL supports tracing APPL functions and LLM calls to facilitate users to understand and debug the program executions. The trace is useful for reproducing (potentially partial) execution results by loading cached responses of the LLM calls, which enables failure recovery and avoids the extra costs of resending these calls. This also unlocks the possibility of debugging one LLM call conveniently. Users can also visualize the program execution to analyze the logic, expense, and runtime of the program. APPL provides two modes of tracing, namely strict and nonstrict modes, corresponding to different reproduction requirements (perfect or statistical reproduction). We further explain the implementation details for handling the asynchronous semantics in Appendix F.1, and demonstrate the trace visualization in Appendix F.2.

## 4 Implementation

In this section, we elaborate on the implementation details of APPL, including compilation and asynchronous execution.

## 4.1 Compilation

To provide the runtime context for APPL functions more smoothly and make it transparent to users, we choose to compile (more specifically, transpile) the Python source code to insert the context into the code. We use Python’s ast package to parse the original function as AST, mainly apply the following code transformations on the AST, and compile back from the modified AST to a Python function. Appendix D gives an example of the code before and after transpilation.

• Context management. We use the keyword \_ctx to represent the context of the function, which is inserted in both the definition of APPL functions and the function calls that need the context. The functions that require the context as inputs are annotated by the attribute \_\_need\_ctx\_\_, including all functions introduced in Section 3.2: APPL functions, gen and context manager functions.

• Capture expression statements. We add an execution function as a wrapper for all expression statements within the APPL function and let them “interact” with the context. The execution behavior differs based on the value types, where the behavior for the most common str type is appending it to the prompt of the context. For a standalone f-string, we split it into several statements to let them be “executed” in order so that gen functions in the f-string can use the text before them. This procedure is detailed in Appendix A.1.

## 4.2 Asynchronous Execution and Future Objects

To implement the asynchronous semantics, we adopt the concept of Future introduced in the Python standard library concurrent.futures, which represents the asynchronous execution of a callable. When the result of Future is accessed, it will wait for the asynchronous execution to finish and synchronize the result.

We introduce StringFuture, which represents a string object, but its content may not be ready yet. StringFutures behave almost identically to the normal str objects in Python, so that users can use them as normal strings transparently and enjoy the benefits of asynchronous execution. Meanwhile, the StringFuture delays its synchronization when possible, ideally only synchronizing when the str method is called.

Formally, we define StringFuture S to be a list of StringFutures: $[ s _ { 1 } , s _ { 2 } , \ldots , s _ { n } ]$ . Without loss of generality, we can regard a normal string as an already synchronized StringFuture. A StringFuture S is synchronized when str(S) is called, and it collapses to a normal string by waiting and concatenating the results of str(s ):

$$
\mathsf { s t r } ( [ s _ { 1 } , \ldots , s _ { n } ] ) \to \mathsf { s t r } ( s _ { 1 } ) \oplus \mathsf { s t r } ( [ s _ { 2 } , \ldots , s _ { n } ] ) ,
$$

where  operation means (string) concatenation. Such representation enables delaying the concatenation ( ) of two StringFutures S and $T$ as concatenating ( ) two lists:

$$
\begin{array} { l l l } { { S \oplus T } } & { { = } } & { { \left[ s _ { 1 } , \ldots , s _ { n } \right] \oplus \left[ t _ { 1 } , \ldots , t _ { m } \right] } } \\ { { } } & { { = } } & { { \left[ s _ { 1 } , \ldots , s _ { n } , t _ { 1 } , \ldots , t _ { m } \right] . } } \end{array}
$$

For other methods that cannot be delayed, like len(S), it falls back to materializing S to a normal string and then calls the counterpart method.

Similarly, we introduce BooleanFuture to represent a boolean value that may not be synchronized yet, which can be used to represent the comparison result between two StringFutures. A BooleanFuture B is synchronized only when bool(B) is called, which enables using future objects in control flows.

## 5 Case Studies and Language Comparisons

In this section, we compare APPL with other prompt languages including LMQL (Beurer-Kellner et al., 2023), Guidance (GuidanceAI, 2023), and SGLang (Zheng et al., 2023) (whose syntax is mostly derived from Guidance) through usage scenarios, including parallelized LLM calls like the chain of thoughts with self-consistency (CoT-SC) (Wang et al., 2022), tool use agent like ReAct (Yao et al., 2023), and multi-agent chat. We further include a re-implementation of a long prompt (more than 1000 tokens) from ToolEmu (Ruan et al., 2024) using APPL in Appendix G, demonstrating the usage of APPL functions and prompt compositors for modularizing and structuring long prompts.

![](images/f8d1b17492a5e0d306df3544b2b1854c51963f3c40de35d28bf4cfb37d96c3c3.jpg)  
(a) APPL

![](images/9747847a57dd8f59031cac4e11a4542b4cdecfe19a6371d4ea427fc0356d1440.jpg)  
(b) LMQL

![](images/a1e5e81b5b06295ce3dca08b7c7db19ad0ef2a45b03486bfe187102062c11005.jpg)

(c) SGLang  
![](images/7869965346c68c8e54c11904643134df58b96d9c2d8457c453d3a08d47f7144e.jpg)  
(d) Guidance  
Figure 7: Comparison of CoT-SC implementations in APPL, LMQL, SGLang, and Guidance, where the marginalizing procedure in CoT-SC is omitted. APPL demonstrates its simplicity and clarity, and exhibits one or more advantages in terms of 1) asynchrony and parallelization, 2) generation and output retrieval, and 3) context management, when comparing with others.

## 5.1 Chain of Thoughts with Self-Consistency

Compared with CoT-SC implemented in LMQL, SGLang and Guidance, APPL exhibits its conciseness and clarity, as shown in Figure 7. We mainly compare them from the following aspects:

Parallization. APPL performs automatic parallelization while being transparent to the programmer. In particular, since the returned value need not be immediately materialized, the different branches can execute concurrently. SGLang allows for parallelization between subqueries by explicitly calling the lm.fork() function. LMQL employs a conservative parallelization strategy, which waits for an async function immediately after it’s called. To support the same parallelization strategy as in APPL, we manually implement the same parallelization strategy use the asyncio library to launch LMQL queries in parallel at the top level. Guidance does not include asynchrony or parallelization as a core feature.

Generation and output retrieval. For LLM generation calls, APPL, SGLang, and Guidance use a Python function while LMQL uses inline variables (with annotations), which is less flexible for custom generation arguments. For the results of generations, both SGLang and Guidance store them as key-value pairs in a dictionary associated with the context object and are indexed with the assigned key when required. LMQL accesses the result by the declared inline variable, but the variable is not Python-native. In APPL, gen is an independent function call, whose return value can be assigned to a Python variable, which is Python-native and allows for better interoperability with IDEs. Moreover, APPL separates the prompt reading and writing for the generation. Users have the option to write generation results back to the prompt or not (called without side-effects in Table 1). This allows a simple replication of LLM generation calls sharing the same prompt. Otherwise, explicit copying or rebuilding of the prompt would be necessary for such requests.

Context management and prompt capturing. Both SGLang and Guidance need to explicitly manage context objects (s and lm) as highlighted as red in Figure 7. The prompts are captured manually using the += operation. The branching of the context is implemented differently, where SGLang uses the explicit fork method and Guidance returns a new context object in the overridden += operation. In contrast, APPL automatically captures standalone expression statements into the prompt context, similar to LMQL which captures standalone strings. The context management is also done automatically for nested function calls in APPL and LMQL, where APPL supports four patterns new, copy, same, and resume (see Figures 5 and 2a) while LMQL only supports new and copy.

<table><tr><td rowspan="2">Time(s)</td><td colspan="2">CoT-SC (Wang et al., 2022)</td><td colspan="2">SoT (Ning et al., 2023)</td><td colspan="2">MemWalker (Chen et al., 2023)</td><td colspan="2">ToT (Yao et al., 2024)</td></tr><tr><td>GPT-3.5</td><td>LLAMA-7b</td><td>GPT-3.5</td><td>LLAMA-7b</td><td>GPT-3.5</td><td>LLAMA-7b</td><td>GPT-40</td><td>LLAMA3.2-3B</td></tr><tr><td>Sequential</td><td>27.6</td><td>17.0</td><td>227.2</td><td>272.3</td><td>707.6</td><td>120.6</td><td>164.3</td><td>621</td></tr><tr><td>Parallel</td><td>2.9</td><td>1.8</td><td>81.2</td><td>85.9</td><td>230.3</td><td>65.4</td><td>17.5</td><td>95.9</td></tr><tr><td>(estimated speedup*)</td><td colspan="2">10</td><td>4.3</td><td>3.7</td><td></td><td>3.8</td><td></td><td>10</td></tr><tr><td>Speedup</td><td>9.49×</td><td>9.34×</td><td>2.79×</td><td>3.17×</td><td>3.07×</td><td>1.84×</td><td>9.39×</td><td>6.48×</td></tr></table>

Table 2: Speedup brought by parallelization with automatic asynchronous execution in APPL. (\* estimated speedup is calculated using Amdahl’s law by assuming the running time of all LLM calls is identical. The number of branches in SoT is determined by LLMs and thus varies for different LLMs. Details about the estimation can be found in Appendix B.1.) MemWalker with LLAMA-7b does not achieve the estimated speedup because its long context lengths increase memory footprint, leading to smaller batch sizes and fewer parallelizable requests.

![](images/25fb40ea36774637655cb88619a79deba517716100f32be942c1fde927722574.jpg)  
Figure 8: Language model evaluation results in different implementations of algorithms. We use language models (GPT, Claude, DeepSeek) to score various tasks implemented in different prompt languages across multiple aspects.

## 5.2 Parallelized LLM Calls

We validated the effectiveness of parallelizing LLM calls in APPL in four tasks: question answering with CoT-SC, content generation with skeleton-of-thought (SoT) (Ning et al., 2023),hierarchical summarization with MemWalker (Chen et al., 2023), and Tree of Thoughts (ToT) reasoning (Yao et al., 2024). As shown in Table 2, the parallel version achieves significant speedup across different branching patterns, and the actual speedup ratios almost match the estimated ones. More experimental details can be found in Appendix B.

## 5.3 ReAct Tool Use Agent

We use the ReAct (Yao et al., 2023) tool-use agent as an example to compare the accessibility of tool calls in different languages. As shown in Figure 9a, the major workflow of ReAct unfolds over several cycles, each comprising three stages: (1) generating “thoughts” derived from existing information to steer subsequent actions; (2) selecting appropriate actions (tool calls) to execute; (3) retrieving the results of executing the actions. These results then serve as observations and influence the subsequent cycle.

LLM Tool calling can be divided into three steps: 1) encoding the external tools as specifications that are understandable by LLMs, 2) parsing generated tool calls, and 3) executing the tool calls. APPL provides out-of-the-box support for these steps based on the OpenAI’s API (OpenAI). As shown in Figure 9a, documented Python functions like is\_lucky defined in Figure 6 can directly be used as the tools argument without manual efforts. Differently, in Figure 9b, Guidance (similarly for SGLang and LMQL) can generate tool calls and parse structured arguments via its select and other constraint primitives, but the tool specifications and format constraints need to be written manually. Guidance provides another way to automate this process but with limited support that only works for functions with string-typed arguments.

```python
1 2 f"User Instruction: {instruction}" tools = [search, is_lucky, ...] <sup>1</sup> <sub>2</sub> tool_names = ["search", "is_lucky", ...] tool_map = {"search": search, "is_lucky": is_lucky, ...} # need to rewrite
3 for i in range(num_iterations): , is_lucky to take str input
4 with AIRole(): 3 tool_desc = {"search": "Use search engine...\n parameters:\n keywords(str):
5 # Generate thoughts in text given the , ...\n returns: ...
, tools 4 "is_lucky": "Determine whether...\n parameters:\n x(int): ...\n
6 f"Thoughts: {gen(tools=tools, , returns: ...", ...}
, tool_choice='none')}" 5 lm += f"We have the following tools: {tool_desc}" + f"User Instruction:
7 # Generate tool calls as actions , {instruction}\n"
8 (actions := gen(tools=tools, 6 for i in range(num_iterations):
, tool_choice='required')) 7 lm += f"Thought {i}: " + gen(name="thought", suffix="\n")
9 # Run the tool calls to get observations 8 lm += f"Action {i}: " + select(tool_names, name="act") + "(" +
10 (observations := , gen(name="arg", suffix=")") + "\n"
, actions.run_tool_calls()) 9 lm += f"Observation {i}: " + tool_map[lm["act"]](lm["arg"]) + "\n"
```  
(a) The main part of APPL codes to implement (b) The main part of Guidance codes to implement the ReAct algorithm. The the ReAct algorithm. Python functions need manual configurations.

Figure 9: Comparison of ReAct implementations in APPL and Guidance. Instead of manually specifying the specifications of tools, APPL automatically creates tool specifications from the signature and docstring of Python functions, making the well-documented Python functions more accessible to LLMs.
<table><tr><td></td><td>APPL</td><td>LMQL</td><td>SGLang/Guidance</td><td></td></tr><tr><td>Tasks</td><td>AST-size</td><td>AST-size VS. APPL</td><td>AST-size</td><td>Vs. APPL</td></tr><tr><td>CoT-SC</td><td>35</td><td>57 1.63×</td><td>61</td><td>1.74×</td></tr><tr><td>ReAct</td><td>61</td><td>not support directly</td><td>134</td><td>2.20×</td></tr><tr><td>SoT</td><td>59</td><td>120 2.03×</td><td>107</td><td>1.81×</td></tr><tr><td>MemWalk</td><td>36</td><td>64 1.78×</td><td>60</td><td>1.67×</td></tr></table>

Table 3: The number of AST nodes of the programs implemented in different prompting languages across different tasks. Note that LMQL programs need to include extra asyncio codes to parallelize LLM calls.

## 5.4 Code Succinctness Comparison

As demonstrated in Figure 7, APPL is more concise than other compared languages to implement the Cot-SC algorithm. Quantitatively, we further introduce a new metric AST-size to indicate the succinctness of programs written in different languages, which counts the number of nodes in the abstract syntax tree (AST) of the code. For Python programs, we can obtain their AST using the standard library with ast.parse.

As shown in Table 3, LMQL, SGLang, and Guidance need about twice AST nodes than APPL for implementing the same tasks, highlighting the succinctness of APPL.

## 5.5 Language Model Evaluation

Inspired by LLM-based evaluators (Zheng et al., 2024; Ruan et al., 2024), we adopt four powerful LLMs that are trained with human preferences as judges to evaluate the subjective aspects of the language design. Specifically, we prompted LLMs to inspect codes in different languages that implement the same task and assess from the following aspects: readability, succinctness, convenience, intuitiveness, and overall score. Additional details including prompts can be found in Appendix C.

As shown in Figure 8, APPL ranks the highest in almost all aspects and gets the highest overall average score by a large margin. It shows that the syntax design of APPL provides a more user-friendly programming interface than existing languages.

## 6 Conclusion

This paper proposes APPL, a Prompt Programming Language that seamlessly integrates conventional programs and natural language prompts. It provides an intuitive and Python-native syntax, an efficient runtime with asynchronous semantics and effective parallelization, and a tracing module supporting effective failure diagnosis and replay. Overall, APPL facilitates users to develop LLM workflows faster and more robustly. We believe it is a crucial direction to study that benefits both researchers and engineers in the LLM community.

## Limitations

While APPL offers significant advancements in integrating Large Language Models (LLMs) with traditional programming languages, several limitations must be addressed:

Limited Multi-Modal Support: APPL is primarily designed for text-based interactions with LLMs. However, as the demand for multi-modal applications—such as those involving audio, video, and other data types—grows, APPL’s current lack of robust support for these modalities may limit its applicability in more complex, real-world scenarios. Expanding APPL’s capabilities to seamlessly handle multi-modal data would be essential for broader adoption.

Dependency on LLM APIs: APPL’s functionality is heavily dependent on the availability and performance of external LLM APIs. This reliance can introduce challenges related to API costs, latency, and access restrictions.

Need for Enhanced Robustness: While APPL facilitates the integration of LLMs into programming workflows, there is still a need for more robust mechanisms to handle LLM output validation and recovery from failures. Currently, APPL lacks built-in features to systematically validate outputs, which can lead to unreliable behavior in critical applications.

Manual Prompt Writing and Optimization: Users of APPL are required to manually write and optimize prompts to achieve desired outcomes from LLMs. This can be a time-consuming and skillintensive process, especially for users who are not well-versed in prompt engineering. The lack of automated tools or best practices for prompt optimization within APPL can hinder productivity and lead to suboptimal results.

These limitations highlight areas where further development and enhancements are needed to make APPL more versatile, reliable, and user-friendly, particularly for applications involving complex or multi-modal tasks.

## Ethical Considerations

While the development of APPL as a Prompt Programming Language represents a significant advancement in the integration of Large Language Models (LLMs) into existing programming workflows, it also introduces several potential risks that need careful consideration.

Over-reliance on LLMs: APPL simplifies LLM integration, which could lead to excessive dependence on LLMs for tasks requiring human judgment. This over-reliance may reduce oversight and result in harmful outcomes if LLMs generate biased or incorrect outputs.

Security Vulnerabilities: The integration of prompts into Python functions could be exploited by malicious actors to inject harmful prompts, potentially compromising security.

Ethical Concerns in Multi-agent Systems: APPL’s support for multi-agent interactions raises ethical concerns, such as the potential for biased or harmful decisions by autonomous systems, especially in contexts with significant social or ethical implications.

To mitigate these risks, developers and researchers should implement robust testing frameworks and safeguards when deploying APPL in critical applications. There should be ongoing research into developing more transparent and interpretable LLMs to ensure that their integration into programming environments does not obscure their decision-making processes. Additionally, future work should explore methods for securely handling and storing prompts, as well as developing best practices for maintaining the balance between automation efficiency and the need for human oversight in complex workflows.

## Acknowledgement

We thank NingNing Xie, Chenhao Jiang, Shiwen Wu, Jialun Lyu, Shunyu Yao, Jiacheng Yang, the Xujie group and the Maddison group for their helpful discussions or feedback on the project. This research project has benefitted from the Microsoft Accelerate Foundation Models Research (AFMR) grant program. This work was supported, in part, by Individual Discovery Grants from the Natural Sciences and Engineering Research Council of Canada, and the Canada CIFAR AI Chair Program.

## References

Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, et al. 2022. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691.

Andrej Karpathy. 2023. LLM OS. Tweet. Accessed: 2024-03-22.

Anthropic. 2023. Claude-sonnet: Anthropic’s large language model. https://www.anthropic.com/ research/claude-sonnet.

BerriAI. 2023. litellm. https://github.com/ BerriAI/litellm.

Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Michal Podstawski, Hubert Niewiadomski, Piotr Nyczyk, et al. 2023. Graph of thoughts: Solving elaborate problems with large language models. arXiv preprint arXiv:2308.09687.

Luca Beurer-Kellner, Marc Fischer, and Martin Vechev. 2023. Prompting is programming: A query language for large language models. Proceedings of the ACM on Programming Languages, 7(PLDI):1946–1969.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Harrison Chase. 2022. LangChain.

Howard Chen, Ramakanth Pasunuru, Jason Weston, and Asli Celikyilmaz. 2023. Walking down the memory maze: Beyond context limit through interactive reading. arXiv preprint arXiv:2310.05029.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2023. Palm: Scaling language modeling with pathways. Journal ofMachine Learning Research, 24(240):1–113.

Samuel Colvin, Eric Jolibois, Hasan Ramezani, Adrian Garcia Badaracco, Terrence Dorsey, David Montague, Serge Matveenko, Marcelo Trylesinski, Sydney Runkle, David Hewitt, and Alex Hall. Pydantic.

DeepLabs AI. 2023. Deepseek-v2: Enhanced large language model for information retrieval. https: //deeplabs.ai/research/deepseek-v2.

Yingqiang Ge, Yujie Ren, Wenyue Hua, Shuyuan Xu, Juntao Tan, and Yongfeng Zhang. 2023. Llm as os (llmao), agents as apps: Envisioning aios, agents and the aios-agent ecosystem. arXiv preprint arXiv:2312.03815.

Georgi Gerganov. 2023. llama.cpp. https://github. com/ggerganov/llama.cpp.

GuidanceAI. 2023. Guidance. https://github.com/ guidance-ai/guidance.

Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, and Zhiting Hu. 2023. Reasoning with language model is planning with world model. arXiv preprint arXiv:2305.14992.

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, et al.

2023. Metagpt: Meta programming for multi-agent collaborative framework. In The Twelfth International Conference on Learning Representations.

HuggingFace. 2023. Text generation inference. https://github.com/huggingface/ text-generation-inference.

Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T Joshi, Hanna Moazam, et al. 2023. Dspy: Compiling declarative language model calls into self-improving pipelines. arXiv preprint arXiv:2310.03714.

Sehoon Kim, Suhong Moon, Ryan Tabrizi, Nicholas Lee, Michael W Mahoney, Kurt Keutzer, and Amir Gholami. 2023. An llm compiler for parallel function calling. arXiv preprint arXiv:2312.04511.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626.

Langfuse. 2023. Langfuse. https://github.com/ langfuse/langfuse.

Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. 2023. Camel: Communicative agents for" mind" exploration of large scale language model society. arXiv preprint arXiv:2303.17760.

Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. 2023. Code as policies: Language model programs for embodied control. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 9493–9500. IEEE.

Jason Liu. 2023. Instructor. https://github.com/ jxnl/instructor.

Ruibo Liu, Jason Wei, Shixiang Shane Gu, Te-Yen Wu, Soroush Vosoughi, Claire Cui, Denny Zhou, and Andrew M Dai. 2022. Mind’s eye: Grounded language model reasoning through simulation. arXiv preprint arXiv:2210.05359.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. 2021. Webgpt: Browser-assisted questionanswering with human feedback. arXiv preprint arXiv:2112.09332.

Xuefei Ning, Zinan Lin, Zixuan Zhou, Zifu Wang, Huazhong Yang, and Yu Wang. 2023. Skeleton-ofthought: Large language models can do parallel decoding. In The Twelfth International Conference on Learning Representations.

Katsumi Okuda and Saman Amarasinghe. 2024. Askit: Unified programming interface for programming with large language models. In 2024 IEEE/ACM International Symposium on Code Generation and Optimization (CGO), pages 41–54. IEEE.

OpenAI. Openai api. https://platform.openai. com/docs/api-reference.

OpenAI. 2022. Introducing chatgpt. Accessed: 2024- 03-22.

OpenAI. 2023a. Gpt-4 technical report. Preprint, arXiv:2303.08774.

OpenAI. 2023b. Gpt-4 turbo: Efficient large language model by openai. https://openai.com/ research/gpt-4-turbo.

OpenAI. 2024. Gpt-4o: The latest iteration of openai’s large language model. https://openai.com/ research/gpt-4.

Charles Packer, Vivian Fang, Shishir G Patil, Kevin Lin, Sarah Wooders, and Joseph E Gonzalez. 2023. Memgpt: Towards llms as operating systems. arXiv preprint arXiv:2310.08560.

Richard Yuanzhe Pang, Alicia Parrish, Nitish Joshi, Nikita Nangia, Jason Phang, Angelica Chen, Vishakh Padmakumar, Johnny Ma, Jana Thompson, He He, and Samuel Bowman. 2022. QuALITY: Question answering with long input texts, yes! In Proceedings ofthe 2022 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 5336–5358, Seattle, United States. Association for Computational Linguistics.

Joon Sung Park, Joseph C O’Brien, Carrie J Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. arXiv preprint arXiv:2304.03442.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32.

Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E Gonzalez. 2023. Gorilla: Large language model connected with massive apis. arXiv preprint arXiv:2305.15334.

Prefect. 2023. marvin. https://github.com/ PrefectHQ/marvin.

Chen Qian, Xin Cong, Wei Liu, Cheng Yang, Weize Chen, Yusheng Su, Yufan Dang, Jiahao Li, Juyuan Xu, Dahai Li, Zhiyuan Liu, and Maosong Sun. 2023. Communicative agents for software development. Preprint, arXiv:2307.07924.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. 2023. Toolllm: Facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal ofMachine Learning Research, 21(1):5485–5551.

Toran Bruce Richards. 2023. Auto-gpt: Autonomous artificial intelligence software agent. https:// github.com/Significant-Gravitas/Auto-GPT. Initial release: March 30, 2023.

Yangjun Ruan, Honghua Dong, Andrew Wang, Silviu Pitis, Yongchao Zhou, Jimmy Ba, Yann Dubois, Chris J Maddison, and Tatsunori Hashimoto. 2024. Identifying the risks of lm agents with an lmemulated sandbox. In The Twelfth International Conference on Learning Representations.

XAgent Team. 2023. Xagent: An autonomous agent for complex task solving.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023a. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837.

Brandon T Willard and Rémi Louf. 2023. Efficient guided generation for large language models. arXiv e-prints, pages arXiv–2307.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang, and Chi Wang. 2023. Autogen: Enabling next-gen llm applications via multiagent conversation framework. arXiv preprint arXiv:2308.08155.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2024. Tree of thoughts: Deliberate problem solving with large language models. Advances in Neural Information Processing Systems, 36.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations.

Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. 2022. Orca: A distributed serving system for Transformer-Based generative models. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22), pages 521–538.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2024. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Jeff Huang, Chuyue Sun, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. 2023. Efficiently programming large language models using sglang. arXiv preprint arXiv:2312.07104.

Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. 2023. Language agent tree search unifies reasoning acting and planning in language models. arXiv preprint arXiv:2310.04406.

## A.1 Standalone f-string

![](images/b6e98a6ae85877e992e10d9eb8d32df150867c87eee93d2a2d9577dcfa5f5f23.jpg)  
Figure 10: Expansion of f-string

Processing f-strings as prompts is nuanced in terms of semantic correctness. When users write a stand-alone f-string as in Figure 10, they usually expect the program to first insert prefix to the prompt context, then call the generation function, and finally add suffix to the context. However, if the whole f-string is processed as a single prompt statement, the generation call will be triggered first without having prefix added into the context. Therefore, we need to split the standalone f-string expression statement into finer-grained substrings.

## A.2 Context Management

![](images/98acbea83e312898f66ce73ed0f40dba9af3f484ca504e1132e6cbb6867e1be6.jpg)  
Figure 11: An example of nested function calls in APPL

Figure 11 is an example of three different prompt context passing methods in APPL, namely new, same, and copy (resume is not shown here). ask\_questions is the top-level function, which calls intro, addon, and query. When intro is called, a new empty context is created, and then the local prompt context is returned, which contains only one line of text – “Today is ...”. This return value is appended into ask\_questions’s prompt context. When calling addon, the same prompt context is shared across addon and ask\_questions, so any modification to the prompt context by addon is also effective for ask\_questions. For query, it can read the previous prompt history of ask\_questions, but any change to the prompt context will not be written back.

## A.3 Tool specification

APPL can automatically extract information from the function signature and docstring to create a tool specification in JSON format from a Python function.

The docstring needs to follow parsable formats like Google Style. As shown in Figures 12 and 13, the tool specification in JSON that follows OpenAI’s API format is automatically created by analyzing the signature and docstring of the function is\_python.

```python
1 def is_lucky(x: int) -> bool:
3 """Determine whether the input number is a lucky number.
5 x (int): The input number to be checked.
6
7 return sympy.isprime(x+3)
```

Figure 12: The Python function.  
![](images/37c47afa453df7c4edb3576605c8a612434ad4e174bfeb5bbed48d6b64f5d75d.jpg)  
Figure 13: The tool specification in the format of OpenAI’s API.

For APIs that do not support tool calls, users can format tool specifications as plain text with a custom formatted using the information extracted from the Python function.

## B Parallelization Experiment Details

In the parallelization experiments, we test both OpenAI APIs and the locally deployed Llama-7B model running on SGLang’s backend SRT. The OpenAI model we use is gpt-3.5-turbo-1106. For local SRT benchmarking, we use one NVIDIA RTX 3090 with 24 GiB memory.

CoT-SC The CoT-SC (Wang et al., 2022) example shown in Figure 7a will launch multiple branches to generate different answers, which can be parallelized. In this experiment, we set the number of branches as ten. The source code is included in Section 5.1.

Skeleton-of-Thought Skeleton-of-Thought (SoT) (Ning et al., 2023) is a prompting technique used in generating long paragraphs consisting of multiple points. It first generates the outline and expands each point separately. The generation of each point can be parallelized. The dataset we use contains the top 20 data instances of the vicuna-80 dataset from the original SoT paper. The main part of SoT is shown in Figure 14.

Hierarchical Summarization MemWalker (Chen et al., 2023) is a new technique addressing the challenge of extremely long contexts. Given a very long text, MemWalker will first divide it into multiple segments, summarize each segment, and further summarize the combination of multiple segments, forming a hierarchical summarization. This summarization will be used in future navigation. The steps without inter-dependency can be parallelized. In this experiment, we use the first 20 articles from the QuALITY dataset (Pang et al., 2022) and compute the average time required to summarize these articles. The summarization has three layers. Each leaf node summarizes 4,000 characters. Every four leaf nodes are summarized into one intermediate node, and two intermediate nodes are summarized into one top-level node. The implementation can be found in Figure 15.

1 @ppl   
2 def skeleton\_prompt(question):   
3 "You're an organizer responsible for only giving the skeleton (not the full content) for answering the question. Provide the   
skeleton in a list of points (numbered 1., 2., 3., etc.) to answer the question. Instead of writing a full sentence,,   
each skeleton point should be very short with only 3\~5 words. Generally, the skeleton should have 3\~10 points. Now,,   
please provide the skeleton for the following question.",   
<sup>4</sup> <sub>5</sub> question   
6 "Skeleton:"   
7 'Please start your answer from "1. " and do not output other things before that.'   
8 return gen()   
9   
10 @ppl   
11 def point\_expanding\_prompt(question, skeleton, point\_index, point\_outline):   
12 "You're responsible for continuing the writing of one and only one point in the overall answer to the following question."   
13 question   
14   
15 "The skeleton of the answer $\mathrm { i } \varsigma ^ { n }$   
16 skeleton   
17   
18 f"Continue and only continue the writing of point {point\_index}. Write it in 1\~2 sentence and do not continue with other   
, points!"   
19 f'Please start your answer from "{point\_index}. {point\_outline}" and do not output other things before that.'   
20 return gen()  
Figure 14: The APPL prompt functions used in SoT. Given a question, it first generates a skeleton with multiple points by calling skeleton\_prompt. It then calls point\_expanding\_prompt for each point to expand them independently, which is parallelized. Synchronization would not happen until it needs to gathers the expanded paragraph of each point in the end.

## B.1 Speedup Estimation

This section details the calculation of the estimated speedup shown in Table 2. We use Amdahl’s law, a widely accepted method for estimating the performance improvement of parallelization. It can be expressed as $\begin{array} { r } { S = \frac { 1 } { ( 1 - p ) + \frac { p } { s } } } \end{array}$ where $S$ is the estimated speedup, $p$ is the proportion of runtime that can be parallelized, s is the speedup ratio for the parallelizable portion. In our scenarios, if a program can be divided into n stages, each taking up $p _ { i }$ of the total runtime and accelerated by $s _ { i }$ times, the overall estimated speedup is given by

$$
S = { \frac { 1 } { \sum _ { i } ^ { n } { \frac { p _ { i } } { s _ { i } } } } } .
$$

We approximate $p _ { i }$ using the number of requests, which is profiled on the datasets used in the evaluation, and $s _ { i }$ using the number of maximum parallelizable requests (e.g. the number of branches in CoT-SC).

## C Language Model Evaluation

This section outlines the experimental setup for the language model evaluation described in Section 5.5. The prompt used to query the language models is listed in Figure 16. We evaluate using four different language models: GPT-4o (OpenAI, 2024), GPT-4 turbo (OpenAI, 2023b), claude-35-sonnet (Anthropic, 2023), and DeepSeek-V2 (DeepLabs AI, 2023).

The same code examples from Section 5.4 are used across all models. Each model is tasked with performing pairwise comparisons of four prompt languages: APPL, Guidance, LMQL, and SGLang. The evaluation criteria include succinctness, readability, intuitiveness, convenience, and overall score. The better one receives one point and the worse one loses one point in each pairwise comparison. The final score is averaged across all language model judges, and normalized by the number of comparisons.

The evaluation criteria are defined as:

1. Succinctness: Which language is more concise to achieve the same functionality?

2. Readability: Which language is more readable for users to understand the functionality of the code?

3. Intuitiveness: Which language has more intuitive syntax for Python users?

4. Convenience: Which language is easier for users to use LMs in their programs?

5. Overall: Which language is better overall?

1 @dataclass   
2 class TreeNode:   
3 summary: Union[str, StringFuture]   
4 content: Optional[str]   
5 children: List["TreeNode"]   
6   
7 @ppl   
8 def construct\_leaf(text):   
9 text   
10 "Summarize the above text comprehensively into a fluent passage."   
11 return gen()   
12   
13 @ppl   
14 def construct\_nonleaf(children:List[TreeNode]):   
15 for child in children:   
16 child.summary   
17 "Compress each summary into a much shorter summary."   
18 return gen()

Figure 15: The APPL prompt functions used in hierarchical summarization. We use the TreeNode class to represent the hierarchy as a tree. For leaf nodes, it calls construct\_leaf to directly generate summary according to the input text segments. For non-leaf nodes, it calls construct\_nonleaf to summarize the information of all child nodes. Dependencies only exist between nodes and their ancestors, and nodes without dependency can be parallelized.  
![](images/dffcc41e50434e26fe84498148a784cb376554218f444b9023c577d930bc1a62.jpg)  
Figure 16: Code snippet of language model evaluation in Section 5.5

## D Transpile Result

Figure 17 presents the transpiled code for the question answering example shown in Figure 2a. Several key transformations are applied: (1) each APPL function is extended with a PromptContext parameter to maintain prompt state; (2) stand-alone expression statements are wrapped with appl.execute to ensure prompt capture within the current context; and (3) context propagation across APPL function calls is handled using appl.with\_ctx. Additional syntactic refinements include the decomposition of f-strings

```python
1 # code BEFORE appl compile
2 @ppl(ctx='copy')
3 def get_answer(question: str):
4 question
5 return gen()
6
7 @ppl
8 def answer_questions(quotation: str, questions: list[str]):
9 """Extract the name of the author from the quotation below and answer questions."""
10 quotation
11 with AIRole():
12 f"The name of the author is {gen(stop='.')}"
13 return [get_answer(q) for q in questions]
14
15 # code AFTER appl compile
16 def get_answer(question: str, *, _ctx: PromptContext=PromptContext()):
17 appl.execute(question, _ctx=_ctx)
18 return appl.with_ctx(_func=gen, _ctx=_ctx, _globals=globals(), _locals=locals())
19
20 def answer_questions(quotation: str, questions: list[str], *, _ctx: PromptContext=PromptContext()):
21 appl.execute('Extract the name of the author from the quotation below and answer questions.', _ctx=_ctx)
22 appl.execute(quotation, _ctx=_ctx)
23 with appl.with_ctx(_func=AIRole, _ctx=_ctx, _globals=globals(), _locals=locals()):
24 with appl.with_ctx(_func=appl.Str, _ctx=_ctx, _globals=globals(), _locals=locals()):
25 appl.execute('The name of the author is ', _ctx=_ctx)
26 appl.execute(appl.with_ctx(appl.with_ctx(stop='.', _func=gen, _ctx=_ctx, _globals=globals(), _locals=locals()),
, _func=appl.format, _ctx=_ctx, _globals=globals(), _locals=locals()), _ctx=_ctx)
27 return [appl.with_ctx(q, _func=get_answer, _ctx=_ctx, _globals=globals(), _locals=locals()) for q in questions]
```

Figure 17: The code before and after APPL transpilation.  
![](images/8a160702bbc488ac39567b625d7fc25edf34d6e9eaf77c6ccd0861d4eec9c92e.jpg)  
(a) Explicitly maintain chat history  
(b) simpler version using resume  
(c) Multi-agent chat demo  
Figure 18: Illustration of two ways of implementing a multi-agent chat using different context management methods.

and context managers.

## E Example of Complex context management: Multi-agent chat bot

We consider a simplified yet illustrative scenario of agent communities (Park et al., 2023; Hong et al., 2023; Qian et al., 2023), where two agents chat with each other, as demonstrated in Figure 18c. During the chat, each agent receives messages from the counterpart, maintains their own chat history, and makes responses. APPL provides flexible context-passing methods for implementing such agents and we illustrate two ways in Figure 18 as examples.

(a) Explicit chat history: As shown in Figure 18a, the conversation history is explicitly stored in self.\_history. The history is first initialized in the \_setup function. In the chat function, the history is first retrieved and written into a new prompt context. At the end, self.\_history is refreshed with records() that represents updated conversation with new messages and responses.

(b) resume the context: As shown in Figure 18b, each call to the chat method will resume its memorized context, allowing continually appending the conversation to the context. The context is initialization in the first call by copying the caller’s context. This approach provides a clear, streamlined, concise way to manage its own context.

## F Tracing and Failure Recovery

## F.1 Implementation

To implement tracing and failure recovery correctly, robustness (the program might fail at any time) and reproducibility must be guaranteed. Reproducibility can be further categorized into two levels, namely strict and non-strict.

Strict reproducibility requires the execution path of the program to be identical when being re-run. This can be achieved given that the main thread of APPL is sequential. We annotate each generation request with a unique identifier and record a SendRequest event when it’s created, and we record a FinishRequest event together with the generation results when it’s finished. Note that SendRequest are in order and FinishRequest can be out of order, so their correspondence depends on the request identifier.

Non-strict reproducibility requires the re-run generations to be statistically correct during the sampling process. For example, multiple requests sharing the same prompt and sampling parameters might have different generation outputs, and during re-run their results are exchangeable, which is still considered correct. Therefore, APPL groups cached requests according to their generation parameters, and each time a cache hit happens, one cached request is popped from the group. A nuance here is that when a generation request is created, its parameters might not yet be ready (e.g. its prompt can be a StringFuture). Therefore we add another type of event called CommitRequest to denote the generation requests are ready and record its parameters.

## F.2 Visualization

The traces collected by APPL can be exported to various visualization frontends, including Langfuse, Lunary, LangSmith, and Chrome Tracing. Each trace captures detailed information about individual requests, along with the call stack of APPL functions. Temporal data, such as start and end times, is also recorded, enabling a hierarchical timeline view of requests and function calls, as illustrated in Figure 19.

![](images/0d4406da7013ba001bdcbbe58321d61980aa21864f22da66d57a36433481c96f.jpg)  
Figure 19: An APPL trace visualized using Langfuse (Langfuse, 2023), showing the hierarchical timeline of APPL function calling, together with the information of each LLM call (prompt, parameters, result, used time, and cost).

## G Structured and Modularized Long Prompt

We re-implemented the long prompt for the agent in ToolEmu (Ruan et al., 2024) using APPL. This long prompt demonstrates how to structure the prompt using APPL functions with definitions for references and prompt compositors for format control.

```python
import appl
2 from appl import BracketedDefinition as Def
3 from appl import gen, ppl, records
4 from appl.compositor import *
5 from appl.utils import get_num_tokens
6
7
8 # ===== shared =====
9 ## Declarations
10 class User(Def):
11 name = "User"
12
13
14 class Agent(Def):
15 name = "Agent"
16
17
18 class Simulator(Def):
19 name = "Simulator"
20
21
22 class UserInfo(Def):
23 name = "User Information"
24
25
26 class CurTime(Def):
27 name = "Current Time"
28
29
30 class ToolArgument(Def):
31 name = "Arguments"
32
33
34 class ToolReturn(Def):
35 name = "Returns"
36
37
38 class ToolException(Def):
39 name = "Exceptions"
40
41
42 ## System Messages
43 BRACKET_DEF = "**Attention!** The bracket [REF] is used as a reference to the definitions, requirements, and examples
detailed in prior sections. Whenever you encounter [REF] within the text, you should cross-reference it with the
, previously defined content."
44
45
46 # Global Definitions
47 @ppl(comp=NumberedList())
48 def global_defs():
49 User(
50 desc=f"The user who instructs the {Agent} to complete tasks or answer questions."
51 )
52 Agent(
53 desc=f"The agent who follows {User}'s instructions and utilizes tools to complete tasks or answer questions."
54 )
55 Simulator(
56 desc=f"The simulator who simulates the tool execution outputs for the {Agent}'s tool calls."
57 )
58 return records()
59
60
61 @ppl(comp=DashList(indent=INDENT))
62 def user_info_details():
63 Def("Name", "John Doe")
64 Def("Email", "john.doe@gmail.com")
65 return records()
66
67
68 @ppl
69 def env_setup():
70 "Environment Setup"
71 with DashList():
72 UserInfo(
73 desc=f"The information of the {User} is provided below:",
74 details=user_info_details(),
75 )
76 CurTime(
77 desc="11:37 AM UTC-05:00, Tuesday, February 22, 2022",
```

```python
)
return records()
@ppl(comp=NumberedList())
def tool_spec_components(include_exception: bool = True):
ToolArgument(desc="The tool input argument specification")
ToolReturn(desc="The tool output return specification")
if include_exception:
ToolException(desc="The possible exceptions of invalid tool calls.")
return records()
@ppl
def tool_specification(toolkit_descriptions: str, include_exception: bool = True):
"Tool Specifications"
with DoubleLineSeparated():
with LineSeparated():
"Each toolkit is a collection of relevant tools for completing a specific task. Each tool is specified by:"
tool_spec_components(include_exception)
"The following tools are available:"
f"{toolkit_descriptions}"
return records()
# Another function for simplified tool descriptions is omitted here.
```

```python
# ===== agent =====
## Declarations
class ToolSpecifications(Def):
name = "Tool Specifications"
class UserInput(Def):
name = "User Input"
class Scratchpad(Def):
name = "Scratchpad"
class Thought(Def):
name = "Thought"
class Action(Def):
name = "Action"
class ActionInput(Def):
name = "Action Input"
class Observation(Def):
name = "Observation"
class FinalAnswer(Def):
name = "Final Answer"
AGENT_SYSTEM_INFO = f"""You are a helpful AI {Agent!r} who can utilize a lot of external tools to answer {User!r}'s
, questions or help {User!r} accomplish tasks following their instructions."""
@ppl(comp=NumberedList())
def agent_provided_info():
ToolSpecifications(desc="the specifications of the tools that you can utilize.")
UserInput(
desc=f"the instruction or question provided by the {User} that the you are trying to help with the provided tools."
)
Scratchpad(
desc="the tool-use trajectories that track your previous tool calls and tool execution outputs."
)
return records()
@ppl
def agent_scratchpad():
"Scratchpad"
with DoubleLineSeparated():
f"The tool-use {Scratchpad} is formatted as follows and should be used to structure your response:"
with LineSeparated():
Thought(
desc=f"your reasoning for determining the next action based on the {UserInput}, previous {Action}s, and
, previous {Observation}s.",
)
Action(
```

```csv
60 desc=f"the tool that you choose to use, which must be a single valid tool name from {ToolSpecifications}.",
61 )
62 ActionInput(
63 desc=f"""the input to the tool, which should be a JSON object with necessary fields matching the tool's
{ToolArgument} specifications, e.g., {{"arg1": "value1", "arg2": "value2"}}. The JSON object should be,
parsed by Python `json.loads`."""
64 )
65 Observation(
66 desc=f"""the execution result of the tool, which should be a JSON object with fields matching the tool's
, {ToolReturn} specifications, e.g., {{"return1": "value1", "return2": "value2"}}.""",
67 )
68 f"This {Thought}/{Action}/{ActionInput}/{Observation} sequence may repeat multiple iterations. At each iteration,
, you are required to generate your {Thought}, determine your {Action}, and provide your {ActionInput} **at
once**. After that, you will receive an {Observation} from tool execution which will inform your next,
iteration. Continue this process for multiple rounds as needed.",
69 f"Once you have finished all your actions and are able to synthesize a thoughtful response for the {User}, ensure
, that you end your response by incorporating the final answer as follows:"
70 FinalAnswer(
71 desc=f"your final response to the {User}.",
72 )
73 return records()
74
75
76 @ppl
77 def agent_task_desc(toolkit_descriptions):
78 "Task Description"
79 with LineSeparated():
80 "Your task is to utilize the provided tools to answer {User}'s questions or help {User} accomplish tasks based on
, given instructions. You are provided with the following information:"
81
82 agent_provided_info(compositor=DashList())
83 with LineSeparated(indexing="###"):
84 # remove exceptions because they are not needed for the agent
85 tool_specification(toolkit_descriptions, include_exception=False)
86 agent_scratchpad()
87 return records()
88
89
90 @ppl
91 def agent_format_instruction():
92 "Format Instructions"
93 with DoubleLineSeparated(indexing="###"):
94 "Format Requirements"
95 with LineSeparated():
96
97 with NumberedList():
98 f"**Use only available tools**: Do not use tools that are not provided above. In particular, do not use
None or N/A as the {Action}. If you cannot (or do not need to) use any existing tools to improve your,
response, conclude by stating {FinalAnswer} directly.",
99 f"**Single JSON object**: Ensure the {ActionInput} is a single JSON object that strictly follows the
,
comments after the JSON object. Do not include any backsticks to wrap the JSON object.",
100 with DashList(indent=INDENT):
101 f"""Incorrect (with comments): {{"query": "the president of US"}} # query the president of US"""
102 f"""Correct (without comments) {{"query": "the president of US"}}""
103 "**Avoid using placeholders**: Do not use vague input values with placeholders. Instead, first obtain the
, actual values for the placeholders (using other tools if necessary) and then use them in the input."
104 with DashList(indent=INDENT):
105 f"""Incorrect (with placeholders): {{"id": <id>, "account_number": <account_number>}}"""
106 f"""Incorrect (with placeholders and comments): {{"id": <id>, "account_number": <account_number>}}
, Note: The user needs to fill in the <id> and <account_number>"""
107 f"""Correct (with actual sourced values): {{"id": "a4bjb5nw1m","account_number": "9012-0678-236"}}"""
108 "Format Examples"
109 with DoubleLineSeparated():
110 "Here is an example for how to format your response. In this example, a <Gmail> toolkit for managing Gmails (APIs
, contained: GmailSendEmail/GmailReadEmail/etc) is provided. Detailed tool specification is omitted here."
111 with LineSeparated():
112 f"User input: Send an email to John Lee (johnlee@gmail.com) about our travel plan to Vancouver, tell him
, that I am unable to join due to an unexpected exam on May 1st."
113 f"Thought: The user wants to inform John Lee about their travel plan to Vancouver but is unable to join due
, to an exam on May 1st. They want to send an email to John Lee for this purpose."
114 f"Action: GmailSendEmail"
115 f"""Action Input: {{"subject": "Unable to join for Vancouver trip", "body": "Hi John,\n\nI hope this email
finds you well. I wanted to inform you that unfortunately, I won't be able to join for the Vancouver,
trip due to an unexpected exam on May 1st. I apologize for any inconvenience this may cause.\n\nBest,
regards", "to": "johnlee@gmail.com"}}""",
116 f"""Observation: {{"status": "Success"}}"""
117 f"Thought: The email was successfully sent to John Lee. No further action is needed."
118 f"Final Answer: Your email to John Lee has been sent successfully!"
119 return records()
120
121
122 @ppl
123 def agent_task_begin(tool_names, inputs, agent_scratchpad):
124 "Start the Execution"
125 with LineSeparated():
```

```python
126 f"Now begin your task! Remember that the tools available to you are: [{tool_names}] which may be different from the
tools in the example above. Please output your **NEXT** {Action}/{ActionInput} or {FinalAnswer} (when you have,
finished all your actions) following the provided {Scratchpad}, directly start your response with your,
{Thought} for the current iteration.",
127
128 f"User Input: {inputs}"
129 f"Scratchpad: {agent_scratchpad}"
130 return records()
131
132
133 @ppl
134 def agent_task_begin_for_claude(tool_names, inputs, agent_scratchpad):
135 "Start the Execution"
136 with LineSeparated():
137 f"Now begin your task! Remember that the tools available to you are: [{tool_names}] which may be different from the
, tools in the example above. Here is the {UserInput} and previous {Scratchpad}:"
138
139 f"User Input: {inputs}"
140 f"Scratchpad: {agent_scratchpad}"
141
142 f'Please output your {Thought} followed by your {Action}/{ActionInput} or {FinalAnswer} (when you have finished all
your actions) for the **current** iteration following the provided previous {Scratchpad}. Start your response,
with your {Thought} for the current iteration directly following the "Thought:" prefix. Do not repeat the the,
previous iteration\'s {Thought} in your response.,
143 return records()
144
145
146 @ppl(comp=LineSeparated(indexing="##"))
147 def agent_naive_prompt(
148 toolkit_descriptions, tool_names, inputs, agent_scratchpad, is_claude=False
149 ):
150 env_setup()
151 agent_task_desc(toolkit_descriptions)
152 agent_format_instruction()
153 if is_claude:
154 task_begin = agent_task_begin_for_claude
155 else:
156 task_begin = agent_task_begin
57 task_begin(tool_names, inputs, agent_scratchpad)
158 return records()
159
160
161 if __name__ == "__main__":
162 system_prompt = AGENT_SYSTEM_INFO
163 example_prompt = str(
164 agent_naive_prompt(
165 toolkit_descriptions="{toolkit_descriptions}",
166 tool_names="{tool_names}",
167 inputs="{inputs}",
168 agent_scratchpad="{agent_scratchpad}",
169 )
170 )
171 print("===== System Message =====")
172 print(system_prompt)
173 print("")
174 print("===== Human Message =====")
175 print(example_prompt)
176 print("\n\n>>>>Token lengths:", get_num_tokens(example_prompt))
```

The corresponding prompt is listed below as plain text. The advantage of plain text is its readability, but its static nature limits its support for dynamic changes including content reusing and multi-version.

```markdown
1 ===== System Message =====
2 You are a helpful AI Agent who can utilize a lot of external tools to answer User's questions or help User accomplish tasks
, following their instructions.
3
4 ==== Human Message =====
5 ## Environment Setup
6 - User Information: The information of the [User] is provided below:
7 - Name: John Doe
8 - Email: john.doe@gmail.com
9 - Current Time: 11:37 AM UTC-05:00, Tuesday, February 22, 2022
10 ## Task Description
11 Your task is to utilize the provided tools to answer {User}'s questions or help {User} accomplish tasks based on given
, instructions. You are provided with the following information:
12
13 - Tool Specifications: the specifications of the tools that you can utilize.
14 - User Input: the instruction or question provided by the [User] that the you are trying to help with the provided tools.
15 - Scratchpad: the tool-use trajectories that track your previous tool calls and tool execution outputs.
16 ### Tool Specifications
17 Each toolkit is a collection of relevant tools for completing a specific task. Each tool is specified by:
18 1. Arguments: The tool input argument specification
19 2. Returns: The tool output return specification
20
21 The following tools are available:
```

22   
23 {toolkit\_descriptions}   
24 ### Scratchpad   
25 The tool-use [Scratchpad] is formatted as follows and should be used to structure your response:   
26   
27 Thought: your reasoning for determining the next action based on the [User Input], previous [Action]s, and previous   
, [Observation]s.   
28 Action: the tool that you choose to use, which must be a single valid tool name from [Tool Specifications].   
29 Action Input: the input to the tool, which should be a JSON object with necessary fields matching the tool's [Arguments]   
, specifications, e.g., {"arg1": "value1", "arg2": "value2"}. The JSON object should be parsed by Python \`json.loads\`.   
30 Observation: the execution result of the tool, which should be a JSON object with fields matching the tool's [Returns]   
, specifications, e.g., {"return1": "value1", "return2": "value2"}.   
31   
32 This [Thought]/[Action]/[Action Input]/[Observation] sequence may repeat multiple iterations. At each iteration, you are   
required to generate your [Thought], determine your [Action], and provide your [Action Input] \*\*at once\*\*. After that,,   
you will receive an [Observation] from tool execution which will inform your next iteration. Continue this process for,   
multiple rounds as needed.,   
33   
34 Once you have finished all your actions and are able to synthesize a thoughtful response for the [User], ensure that you   
, end your response by incorporating the final answer as follows:   
35   
36 Final Answer: your final response to the [User].   
37 ## Format Instructions   
38 ### Format Requirements   
39   
40 Here are some requirements that you should strictly follow to format the [Action] and [Action Input]:   
41 1. \*\*Use only available tools\*\*: Do not use tools that are not provided above. In particular, do not use None or N/A as the   
[Action]. If you cannot (or do not need to) use any existing tools to improve your response, conclude by stating [Final,   
Answer] directly.,   
42 2. \*\*Single JSON object\*\*: Ensure the [Action Input] is a single JSON object that strictly follows the specification of   
the tool's [Arguments]. Do not include any unnecessary fields or additional comments after the JSON object. Do not,   
include any backsticks to wrap the JSON object.,   
43 - Incorrect (with comments): {"query": "the president of US"} # query the president of US   
44 - Correct (without comments) {"query": "the president of US"}   
45 3. \*\*Avoid using placeholders\*\*: Do not use vague input values with placeholders. Instead, first obtain the actual values   
, for the placeholders (using other tools if necessary) and then use them in the input.   
46 - Incorrect (with placeholders): {"id": <id>, "account\_number": <account\_number>}   
47 - Incorrect (with placeholders and comments): {"id": <id>, "account\_number": <account\_number>} Note: The user needs   
, to fill in the <id> and <account\_number>   
48 - Correct (with actual sourced values): {"id": "a4bjb5nw1m","account\_number": "9012-0678-236"}   
49   
50 ### Format Examples   
51   
52 Here is an example for how to format your response. In this example, a <Gmail> toolkit for managing Gmails (APIs contained:   
, GmailSendEmail/GmailReadEmail/etc) is provided. Detailed tool specification is omitted here.   
53   
54 User input: Send an email to John Lee (johnlee@gmail.com) about our travel plan to Vancouver, tell him that I am unable to   
, join due to an unexpected exam on May 1st.   
55 Thought: The user wants to inform John Lee about their travel plan to Vancouver but is unable to join due to an exam on May   
, 1st. They want to send an email to John Lee for this purpose.   
56 Action: GmailSendEmail   
57 Action Input: {"subject": "Unable to join for Vancouver trip", "body": "Hi John,   
58   
59 I hope this email finds you well. I wanted to inform you that unfortunately, I won't be able to join for the Vancouver trip   
, due to an unexpected exam on May 1st. I apologize for any inconvenience this may cause.   
60   
61 Best regards", "to": "johnlee@gmail.com"}   
62 Observation: {"status": "Success"}   
63 Thought: The email was successfully sent to John Lee. No further action is needed.   
64 Final Answer: Your email to John Lee has been sent successfully!   
65 ## Start the Execution   
66 Now begin your task! Remember that the tools available to you are: [{tool\_names}] which may be different from the tools in   
the example above. Please output your \*\*NEXT\*\* [Action]/[Action Input] or [Final Answer] (when you have finished all,   
your actions) following the provided [Scratchpad], directly start your response with your [Thought] for the current,   
iteration.,   
67   
68 User Input: {inputs}   
69 Scratchpad: {agent\_scratchpad}   
70   
71   
72 >>>>Token lengths: 1209