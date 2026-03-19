# helloworld


https://github.com/shareAI-lab/shareAI-skills

https://geektime-docs.netlify.app/%E5%90%8E%E7%AB%AF-%E6%9E%B6%E6%9E%84/python%E6%A0%B8%E5%BF%83%E6%8A%80%E6%9C%AF%E4%B8%8E%E5%AE%9E%E6%88%98/04%20-%20%E5%AD%97%E5%85%B8%E3%80%81%E9%9B%86%E5%90%88%EF%BC%8C%E4%BD%A0%E7%9C%9F%E7%9A%84%E4%BA%86%E8%A7%A3%E5%90%97%EF%BC%9F/#_1


深入了解memsearch ，deep-searcher 等开源项目

AI 搜索与 AI 记忆在工业界落地的两个核心难题：信息过载（Information Overload） 和 上下文窗口限制（Context Limit）。

基于 LLM 的重排序机制

使用传统的 BGE-Reranker 等专门的 rerank 模型。


我要了解ai工程领域的一些常用词，如memsearch ，deep-searcher，上下文窗口限制（Context Limit，，rerank 。

交叉编码器

重排模型 (Reranker)


Embedding 模型，

模型类型	角色定位	它的工作 (Task)	特点
Embedding 模型	前台索引员	把书名和内容变成一串数字（向量），快速从 100 万本书里找出最像的 50 本。	快！ 毫秒级处理海量数据。
Rerank 模型	部门主管	把前台找来的 50 本书逐本翻开，精读并打分，选出最准的 3-5 本。	精！ 速度比前台慢，但看人（语义）极准。
大模型 (LLM)	终极老板	读完主管选出的 3-5 本书，根据这些信息写出一份专业的诊断报告。	灵！ 懂人话，能推理，会总结。

From <https://gemini.google.com/app/6ba0bf5fc26d27b4?utm_source=deepmind.google&utm_medium=referral&utm_campaign=gdm&utm_content=> 




“在 Python 端利用 FlagModel 封装了 BGE-Reranker-v2-m3。针对硬件告警日志的特殊性，通过手动调整 max_length 至 512，并开启 FP16 半精度推理，在保证 0.9 以上重排准确率的同时，将单次请求的推理延迟控制在 100ms 以内。”

PyTorch



在构建 Rerank 服务时，为了解决 CPU 在执行 Cross-Encoder 计算时的瓶颈，我利用 PyTorch 替换了传统的 NumPy 处理逻辑。通过将数据封装为 Tensors 并部署于 CUDA 环境下，利用 PyTorch 的并行计算能力，将 50 条候选日志的重排耗时从秒级降低到了百毫秒级。


 Cross-Encoder

LangGraph


RAGFlow 

FastMCP 框架



BGE-M3 BGE-Reranker




BGE-M3 

Milvus


HNSW

Pinecone

FlagModel 

PyTorch


BGE-Reranker



Sentence-Transformers  FlagEmbedding 

AI Agent 的记忆系统（如结合 MemSearch）

N8N 






资深 AI 工程师 / RAG 架构师 深入理解大模型全链路检索增强（RAG）技术，精通基于 BGE-M3 与 Hybrid Search 的高性能检索系统设计。具备构建 AI Agent 记忆系统 的实战经验，能够熟练运用 LangGraph 和 FastMCP 实现复杂的多智能体协同工作流。擅长优化高维向量索引（HNSW），在处理海量文档理解与上下文窗口优化方面有深度实践。

2. 技能矩阵（Skills）—— 分层分类
不要乱排，要体现出你的知识体系结构：
• 智能体框架（Agentic Frameworks）： LangGraph (状态机管理)、FastMCP (标准化工具调用接口)、Multi-Agent 工作流设计。
• 检索增强（RAG & Retrieval）： RAGFlow (深度文档解析)、BGE-M3 (混合搜索)、Cross-Encoder (重排序)、BGE-Reranker。
• 向量工程（Vector Engineering）： Milvus & Pinecone (分布式存储)、HNSW 索引算法原理、MemSearch (轻量化嵌入式记忆)。
• 深度学习底层（Core AI）： PyTorch、Sentence-Transformers、FlagEmbedding (模型微调与部署)。

3. 简历项目描述（Project Experience）—— 话术模板
这是最关键的部分，要用 “背景 -> 问题 -> 方案 -> 结果” 的逻辑。
例子 A：针对“AI 记忆系统”
项目名称： 企业级私有化 AI 知识库与长效记忆系统 核心技术： BGE-M3, MemSearch, RAGFlow, Cross-Encoder 个人贡献：
	○ 解决信息过载： 利用 RAGFlow 深度文档理解能力，解决复杂 PDF 中表格与跨页内容的结构化解析难题。
	○ 优化检索精度： 采用 BGE-M3 构建向量与稀疏向量的混合检索（Hybrid Search），并引入 BGE-Reranker (Cross-Encoder) 实现二阶段重排序，将 Top-5 召回准确率提升了 40%。
	○ 突破窗口限制： 设计基于 MemSearch 的嵌入式记忆层，将用户历史上下文动态压缩为结构化 Markdown 记忆片段，大幅降低 Token 消耗并提升 Agent 回复的一致性。
例子 B：针对“智能体工作流”
项目名称： 基于 FastMCP 与 LangGraph 的自动化业务 Agent 核心技术： LangGraph, FastMCP, PyTorch 个人贡献：
	○ 复杂逻辑建模： 使用 LangGraph 构建带循环和条件判断的智能体状态机，解决了传统 Linear Chain 无法处理的复杂业务纠错逻辑。
	○ 标准化集成： 基于 FastMCP 框架构建标准化的工具连接层，实现 Agent 与企业内部私有数据库及第三方 API 的快速解耦集成。

4. 突出这些“高级”关键词
面试官看到这些词会觉得你很专业：
1. “多路召回与融合 (Hybrid Search)”：证明你懂 BGE-M3 的核心价值。
2. “语义衰退管理”：描述 Agent 记忆时，提到如何处理过期信息。
3. “冷热存储分离”：提到 Milvus (冷/海量) 与 MemSearch (热/实时) 的配合。
4. “Cost-Effective RAG”：强调你通过 Reranker 减少了发给 LLM 的数据量，从而降低了成本。

5. 建议补充的亮点
如果你还有余力，可以加一句话：
“熟悉 FlagEmbedding 底层源码，能够针对特定业务领域对 BGE-Reranker 进行微调（Fine-tuning），以适配垂直行业的术语理解。”



构建端到端自动化智能体





RAGAS 或 TruLens 等自动化评估工具。

From <https://github.com/datawhalechina/all-in-rag/blob/main/docs/chapter1/01_RAG_intro.md> 



1. 技能清单 (Skills)
在技能栏，你可以区分“框架”与“原理”：
	• NLP & RAG: 熟悉 RAG 全链路优化，包括文档解析（DeepDoc 逻辑）、多级分块策略（Recursive/Parent-Child）、混合检索与 Rerank (Cross-Encoder) 精排。
	• 模型部署: 掌握使用 Ollama 部署本地大模型，以及 Sentence-Transformers/BGE 系列嵌入模型的集成应用。
	• 工具链: 熟练使用 LangChain/LlamaIndex，精通 jieba/NLTK 等分词处理及 FAISS/ChromaDB 向量数据库应用。

2. 项目经验 (Project Experience) —— 案例示范
项目名称：基于私有知识库的智能问答系统 (RAG)
	• 核心实现：
		○ 针对不同结构文档（PDF/Markdown/Manual），设计并实现了递归字符分割与父子索引策略，有效解决了长文档检索中的上下文丢失问题。
		○ 构建了**“粗排+精排”双级检索架构**：初排采用轻量级 Embedding 模型（BGE/Text2Vec），精排引入 Cross-Encoder 重排序模型，将检索准确率显著提升。
		○ 优化了 Prompt 构造逻辑，采用 XML 标签与双换行符 (\n\n) 进行上下文拼接，增强了 LLM 对多源参考资料的理解边界，降低了幻觉率。
	• 工程优化：
		○ 针对中文语境，集成 jieba/HanLP 进行前置处理，并利用 BGE-M3 模型解决了长文本嵌入窗口（8k Token）限制问题。
		○ 采用 Ollama 实现生成模型的独立服务化部署，与 Python 业务逻辑解耦，实现了高效的资源隔离。

3. 专业话术（面试加分项）
如果面试官问起，你可以用今天学到的逻辑来回答：
	“在处理 RAG 时，我非常关注数据清洗的质量。我了解到像 RAGFlow 这种框架通过视觉解析（Layout Analysis）来保证表格和多栏文本的顺序，这让我意识到简单的字符分割是不够的。因此在实际操作中，我会根据文档类型灵活切换分块策略，并配合 Rerank 来弥补向量搜索在语义细微差别上的不足。”
	
	
	
	“深入理解 Embedding 模型底层的 Tokenization 机制，能够根据模型词表限制优化输入文本的预处理，确保长文本在进入 Transformer 编码器前完成高效的子词切分与特殊标记注入。”
	
	
	在“专业技能”中：
		“深入理解 Transformer 架构体系，能够准确区分 Encoder-only (如 BERT/BGE) 与 Decoder-only (如 GPT 系列) 的原理差异，并能根据任务场景（语义检索 vs. 内容生成）灵活进行模型选型与优化。”
	在“项目经验”中（强调优化）：
		“利用 Encoder 模型天然的双向注意力机制，针对特定业务语料进行 Embedding 微调，解决了 Decoder 类模型在处理短文本相似度时容易出现的语义漂移问题，提升了检索系统的初始召回精度。”
		
		
		
		
		如果你能把这段话转化到简历中，将极大地提升你的架构视野：
		在“核心能力”或“个人总结”中：
			“深谙多模态语义对齐演进路线，理解从 Word2Vec 静态表征到 BERT 动态上下文编码、再到 CLIP 对比学习下的跨模态对齐原理。能够灵活运用 Transformer 架构处理异构数据（图、文、音）的统一嵌入与高效检索。”
		在“项目经验”中（若涉及多模态 RAG）：
			“设计并实现了一套跨模态检索系统，借鉴 CLIP 对比学习思路，将非结构化图像数据与技术文档在共享向量空间内进行对齐，实现了‘以图搜文’与‘语义搜图’功能，极大地扩展了传统 RAG 的知识边界。”
		
		
		
		“这段代码利用了 BGE-Visualized-M3 的统一嵌入能力。在 no_grad 推理模式下，我们将用户的自然语言查询转换为多模态共享空间中的特征向量。由于模型在预训练阶段已经完成了图文对齐，这个向量可以直接用于在向量数据库中检索匹配的图像块或图文混排文档，实现了高效的跨模态语义召回。”
		
		
		
		在“项目经验”中：
			“在系统原型阶段采用 ChromaDB/FAISS 进行快速验证；随着数据量增至千万级，迁移至 Milvus 分布式架构。利用其存算分离的特性优化了资源利用率，并通过 标量过滤 (Scalar Filtering) 实现了业务层面对特定属性的精准检索，解决了纯语义检索在特定场景下的召回漂移问题。”
		在“专业技能”中：
			“深入理解向量检索底层原理，熟悉 FAISS 各类索引（Flat, IVF, HNSW）的优缺点；具备大规模向量数据库 Milvus 的生产部署与性能调优经验。”
			
			
			• 选 FAISS/ChromaDB：做个 Demo、个人知识库、数据量在 100 万以下、追求部署简单（几行代码搞定）。
			• 选 Milvus：做商业产品、海量数据、需要高可用、需要复杂的元数据过滤、有多人协作开发需求。
			
			
			
			• 维度匹配：强调 dimension 必须与你的嵌入模型（如 BGE 的 1024 维）完全一致。
			• 内存管理：提到 FAISS 索引是驻留在 RAM 中的。如果数据量大到单机内存装不下，你会提到使用 IndexIVFFlat（倒排索引）来压缩空间，或者迁移到 Milvus。
			• 持久化：你会强调手动保存的重要性：
Python

faiss.write_index(index, "my_vector_index.faiss") # 数据库会自动做，FAISS 必须手动写
		总结 FAISS 的“库”特征：
			1. 没有 Server：它没有 localhost:11434 之类的地址，它只是你 Python 进程里的一个对象。
			2. 没有 Schema：它存不下你的“文本内容”，它只存 indices（数字 ID）。你得自己拿这个 ID 去你的 MySQL 或 JSON 文件里找对应的文字。
			
			
			如果你在简历中提到优化了 RAG 系统，使用“句子窗口检索”是一个极佳的技术亮点，建议这样描述：
				“针对复杂技术文档中语义细碎、代词指代不明导致的生成质量问题，引入了 LlamaIndex 的句子窗口检索（Sentence Window Retrieval）策略。通过‘窄检索、宽生成’的逻辑分离，在保持检索精确度的同时，为 LLM 提供了完整的上下文支撑，将答案的语义连贯性（Coherence）提升了 [X]%，并有效解决了‘信息断层’引发的幻觉问题。”
			
			
			
			
			句子窗口检索只是 “解耦检索与生成单位” 这一思想的一种实现。顺着这个思路，你还可以了解到：
				• 父子索引（Parent-Child Recursive Retriever）：检索子块（小），返回父块（大）。
				• 文档摘要索引：检索文档的摘要（精炼），返回全文（详实）。
			总结
			你提到的这项技术，本质上是在做 “语义精度”与“逻辑完整性”的解耦。这显示出你已经不再满足于跑通 Demo，而是在思考如何解决生产环境下的真实痛点。
			
			
			“针对垂直领域（如医疗/法律/工业）专有名词多、缩写密集的痛点，引入了基于 BGE-M3 的混合检索（Hybrid Search）架构。通过融合 BM25 稀疏向量的精确匹配能力与稠密向量的语义表征能力，并采用 RRF 算法进行结果重融合，将系统对长尾关键词（Long-tail Keywords）的检索精度提升了 [X]%，显著降低了 RAG 系统的幻觉率。”
			
			
			
			
			
	
	混合检索是处理**“专业知识库”**的刚需。它解决了纯向量检索在大规模工业数据中容易产生的“词不达意”问题。
	
	
	
	
	
	版本一：技术专家型（强调底层原理与混合架构）
		适用场景： 投递搜索后端、RAG 算法工程师、向量数据库研发。
		• 精通混合检索（Hybrid Search）架构设计： 能够熟练运用 RRF（Reciprocal Rank Fusion） 算法融合 BM25 词法检索与稠密向量检索，有效平衡搜索的“精准度”与“语义泛化能力”。
		• 深谙 NLP 底层机制： 深入理解 Tokenizer（WordPiece/BPE） 及其对 OOV（未登录词） 的处理逻辑；能独立针对业务场景优化分词策略，确保索引与搜索侧的词法一致性。
		• 大模型工程化落地： 熟悉主流向量数据库（如 Milvus/Elasticsearch/Qdrant）的底层原理，能够根据数据规模调优相似度模型与分词算子，解决复杂语境下的语义偏移问题。
	
	版本二：工程实战型（强调性能、合规与上海创业背景）
		适用场景： 投递初创公司技术负责人、高级开发、大模型应用架构师。
		• 高效 RAG 系统构建者： 具备从零构建工业级知识库的能力，擅长通过 RRF 融合逻辑 解决搜索结果“打架”问题，在不引入繁琐权重调优的前提下，实现搜索效果的自动化提升。
		• 严谨的工程合规意识： 深刻理解“一人有限公司”及“超级个体”模式下的技术成本控制，能利用 Elasticsearch/Lucene 等开源生态快速搭建高可用搜索集群。
		• 复杂场景处理经验： 能够应对词典动态更新带来的重索引挑战，熟悉 BM25 饱和度曲线 调优及 $k$ 常数 对 RRF 排名结果的影响，提升长尾词条的检索召唤率。
	
	版本三：超级个体/方案架构型（强调业务逻辑与避坑能力）
		适用场景： 投递技术合伙人、独立开发项目展示、技术顾问。
		• 全栈检索方案设计： 拒绝死记硬背，能够从**数学本质（如 RRF 排名倒数融合）**出发解释搜索逻辑，为业务提供低成本、高回报的语义搜索技术选型。
		• 敏锐的业务洞察： 关注技术与商业的结合，如在上海徐汇区等政策高地背景下，利用 AI 技术栈降低初创企业的获客成本与知识管理成本。
		• 技术抗风险能力： 熟知搜索系统中的常见“坑位”（如分词不一致导致的召回失败、一人公司财产混同的技术选型风险），确保系统架构的稳健性与可扩展性。
	
	From <https://gemini.google.com/app/ddeabf598f22af77?utm_source=deepmind.google&utm_medium=referral&utm_campaign=gdm&utm_content=> 
	
	
	
	
	
	• 精通查询构建（Query Construction）技术： 擅长利用 LLM（如 GPT-4, Claude 3.5）的自然语言理解能力，将模糊的业务需求精准“翻译”为结构化查询。能够熟练处理包含复杂过滤条件（Filtering）、**聚合操作（Aggregation）及多表关联（Join）**的跨数据源查询任务。
	• 异构数据源集成专家： 不局限于单一向量库，具备处理 SQL（结构化）、Metadata（半结构化）及图数据库（关系型）的综合实战经验。能够通过 Self-Querying 等技术，自动从自然语言中提取元数据过滤条件，解决纯语义搜索在处理数值、时间范围、类别过滤时的失效问题。
	• 闭环 RAG 流程优化： 深入理解从“自然语言”到“结构化 DSL（如 Elasticsearch DSL, Cypher, SQL）”的转换链路，能有效设计 Prompt Engineering 与 Few-shot 示例，显著提升复杂查询条件下的检索召回率（Recall）与精确度（Precision）。
	• 混合检索深度实践： 能够灵活组合 RRF 排序算法 与 结构化预过滤（Pre-filtering），在上海徐汇区等高浓度 AI 创业背景下，为企业提供高性能、低幻觉的知识库问答与决策支持方案。

🛠 技能清单（关键词优化）
	• 检索增强： Query Construction, Self-Querying Retrieval, Hybrid Search (RRF), Metadata Filtering.
	• 数据工程： SQL/NoSQL, Vector DB (Milvus/Qdrant), Graph DB, JSON-LD.
	• 模型应用： LLM-based DSL Generation (Text-to-SQL, Text-to-Cypher), Intent Classification.

如果你要把这段代码转化成简历上的优势，不要只写“我会用 Chroma”，建议这样写：
	“熟练运用 BGE 系列预训练模型实现高性能中文语义建模，通过 Chroma 向量数据库构建支持元数据过滤（Metadata Filtering）的知识库索引。在实际项目中，通过优化 Embedding 批处理策略与持久化索引结构，实现了在大规模文档场景下的毫秒级语义检索响应。”

QAnything 源码

From <https://zhuanlan.zhihu.com/p/697031773> 

你的痛点/需求
推荐框架
理由
文档全是垃圾格式（PDF表格多）
Ragflow
文档解析能力是它的杀手锏。
追求“开箱即用”且要给业务用
Dify
UI 友好，权限管理和 API 集成非常成熟。
对中文检索精度有“执念”
Qanything
网易自研的 Embedding 和 Reranker 对中文语境优化极好。
数据量极大（千万级）
LlamaIndex
配合外部向量数据库（如 Milvus），底层控制力最强。
完全私有化部署且资源有限
MaxKB / FastGPT
架构相对轻量，适合单机或小规模集群。

AWS Multi-agent Orchestrator

From <https://zhuanlan.zhihu.com/p/1893252158327063392> 

各大厂商都开始布局多智能体框架，如OpenAI的Swarm、微软的AutoGen、以及亚马逊的Multi-Agent Orchestrator。

From <https://zhuanlan.zhihu.com/p/1893252158327063392> 
<img width="766" height="7764" alt="image" src="https://github.com/user-attachments/assets/5ff774cc-18de-4e5f-ba75-13c7e06272d0" />

