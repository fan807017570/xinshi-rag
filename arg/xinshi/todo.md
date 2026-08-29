1. 检索与入库（对准确率影响最大）
方向	说明
混合检索（稠密 + 稀疏/BM25）	口语问法（「在哪」「多大」）和正文用词（「位于」「占地面积」）不完全重合时，关键词检索很有效。Milvus 支持稀疏向量或双路召回再融合，适合你们这种宣传稿 + 大纲混在一起的文档。
更强多语/混合嵌入	在中文场景可试 BAAI/bge-m3（稠密+稀疏若配合框架），或略大一档的 zh 模型，在延迟可接受前提下提升召回。
切块策略	现在固定 chunk_size=500：事实句（地址、面积）和长段荣誉混在一个块里会被稀释。可对「1.1 基本信息」用更小 chunk，或对 FAQ 单独建短文档；大纲占位段（只有「- xxx」）可考虑不入库或打标签降权。
父文档检索	小块用于向量命中，返回时带上父段或整节 markdown，减少「命中半句话、上下文不够」的情况。
BGE 向量归一化	若 Milvus 里用内积/余弦，建议在 HuggingFaceEmbeddings 里打开与官方一致的 normalize_embeddings（若该类支持），与索引度量一致，避免距离失真。
2. 流水线与成本（延迟 + 稳定性）
方向	说明
改写（query_rewrite）按需调用	每问都调一次 LLM：多一次网络延迟和费用。简单短句可直接检索；或规则命中（含「在哪」「面积」等）再改写。
复用 LLM 客户端	main() 里每次循环 get_llm() 新建连接不划算，可在进程内单例复用（与 vectorstore / reranker 一样）。
Rerank 批推理	CrossEncoder.predict 可设 batch_size，20 对一次算完，通常比默认更快。
Reranker 候选数收敛	当前最多对 RERANK_CANDIDATE_LIMIT=16 条候选做 CrossEncoder 打分。可先尝试 RERANK_CANDIDATE_LIMIT=8，rerank 耗时通常会随候选数明显下降；代价是召回排序覆盖面略降。
Reranker 输入长度收敛	当前 RERANK_DOC_CHAR_LIMIT=800、RERANK_MAX_LENGTH=256。可尝试 RERANK_DOC_CHAR_LIMIT=400、RERANK_MAX_LENGTH=192，减少 CrossEncoder 编码长度；若 chunk 本身较短，质量影响通常可控。
明确事实问题跳过 rerank	地址、校长、面积、电话、学费等强事实问答，向量召回前几条通常已经足够。可按关键词命中直接返回 Milvus 前 TOP_K_RERANK 条，跳过 reranker；收益高，但需要维护少量规则。
降低召回量与角色放大倍数	当前 TOP_K_RETRIEVE=20，teacher/student 问题会乘 ROLE_FILTER_MULTIPLIER=3，最多先召回 60 条。可尝试 TOP_K_RETRIEVE=12、ROLE_FILTER_MULTIPLIER=2，减少召回与后续候选池压力；代价是覆盖面降低。
更小 reranker 模型	当前使用 bge-reranker-large，质量好但慢。可评估 bge-reranker-base 或更小中文 reranker；对学校招生知识库这类窄领域场景，小模型可能足够。
扩大 rerank score cache	当前 RERANK_SCORE_CACHE_SIZE=4096，只对重复 query-doc 生效。若用户问题重复率高，可调到 20000；代价是内存占用增加，且服务重启后缓存丢失。
启动即加载 vs 懒加载	现在 vectorstore + BGEReranker 在 import 时就初始化，首次启动慢但后续提问快；若希望 CLI 秒开，可改成第一次提问时再加载（需权衡）。
3. 提示词与生成质量
方向	说明
Chat 消息结构	若底层是 Chat 模型，用 SystemMessage + HumanMessage 比一大段字符串拼接更清晰，也便于后续加「禁止编造」等系统约束。
要求引用依据	例如：「回答中必须包含资料中的原词或短句（如地名、数字）」可降低胡编；或要求文末列 [章节] 出处。
合规与语气	招生场景可加一条：涉及升学率、承诺性表述时一律按资料原文、不得发挥。
4. 工程与可维护性
方向	说明
配置与密钥	llm.py 里 API Key 硬编码风险高，应改为环境变量（如 DEEPSEEK_API_KEY），config 里也不要提交真实密钥。
ingest 路径	DirectoryLoader("./data/docs") 依赖当前工作目录。建议用 Path(__file__).resolve().parent / "data" / "docs"，在任意目录执行 python ingest.py 都一致。
依赖迁移	日志里已提示：langchain_community.vectorstores.Milvus、HuggingFaceEmbeddings 的迁移路径；中长期可迁到 langchain_milvus + 官方推荐写法，减少破窗。
空块与脏数据	ingest 时过滤 strip() 后为空的 sub_chunks，避免无效向量占用索引。
5. 可观测与调试
方向	说明
调试开关	环境变量 DEBUG_RAG=1 时打印：改写后 query、每条 doc 的 section + 前 120 字、（若有）相似度/rerank 分数。排「召回不到」会快很多。
简单指标	记录：检索耗时、rerank 耗时、LLM 耗时，便于以后换模型或调 TOP_K。
6. 产品体验
方向	说明
多轮对话	当前 while True 无历史，「那食堂呢？」接不上。可用 ConversationBufferMemory 或手动拼最近一轮 Q/A。
流式输出	llm.stream 打印答案更跟手。
退出与异常	KeyboardInterrupt、Milvus 连不上时的友好提示与重试（你们 milvus.py 里已有连接重试，可抽到公共工具）。
建议优先级（务实顺序）
安全：API Key 进环境变量（立刻做）。
路径：ingest 用基于 __file__ 的路径（减少「本地能跑、换目录挂」）。
检索：条件化 query_rewrite +（有余力）混合检索或更好的切块/FAQ 段。
工程：调试输出 + rerank batch_size + 复用 get_llm()。
Reranker 延迟优先优化：先调 RERANK_CANDIDATE_LIMIT=8，再调 RERANK_DOC_CHAR_LIMIT=400 / RERANK_MAX_LENGTH=192；若仍慢，再加明确事实问题跳过 rerank，最后考虑换小模型。
中长期：langchain_milvus、父文档检索、多轮与流式。
如果你愿意下一步只改一两处，我可以按你选的条目直接改仓库里的代码（例如：环境变量 + ingest 路径 + 复用 LLM + rerank batch 这一组改动小、性价比高）。
