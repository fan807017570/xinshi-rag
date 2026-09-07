# 统一 RAG 与成绩查询意图编排方案评审

| 项目 | 内容 |
|---|---|
| 评审对象 | [统一RAG成绩查询改造方案.md](统一RAG成绩查询改造方案.md) |
| 评审日期 | 2026-09-01 |
| 评审结论 | 通过（已完成代码级复审） |
| 阻断问题 | 0 个 |

## 1. 结论摘要

方案已按“单 Graph、双渠道复用、微信协议隔离、现有能力兼容”的核心要求实施。代码采用一张统一聊天 Graph；成绩查询命中后调用独立参数 LLM，缺参时追问并按渠道合并上下文，参数完整后经校验和 URL 编码拼接固定入口。

实施后代码级复审确认：聊天运行时只有一个 Graph 编译点；成绩和普通 RAG 在同一条件路由内；Web 无法注入微信身份；同步 completions、微信适配及 SSE 兼容路径均有定向测试。成绩意图采用“规则确认查询、规则确认非查询、无法确认时调用 LLM”的三级识别。全量单元测试 60 项通过。

## 2. 需求覆盖评审

| 需求 | 方案覆盖 | 评审 |
|---|---|---|
| 主 RAG 和成绩意图放入同一 Graph | 成绩节点直接接入主 Graph，禁止嵌套或串联第二张 CompiledGraph | 通过 |
| completions 支持成绩意图 | Web 注入内部 `channel=WEB`，支持参数提取、history 补充和链接回复 | 通过 |
| `/api/chat` 保留微信报文处理 | 验签、解密、ACK、队列、客服 API 均留在 Graph 外 | 通过 |
| index.html 不需要 H5 超链接 | 保持 `textContent` 纯文本渲染，不增加链接、二维码或跳转 | 通过 |
| 非确定意图调用 LLM | 两组简单规则都无法确定时才进入严格 JSON LLM 分类 | 通过 |
| 不改变原有能力 | 保留公共模型、普通 RAG 节点、SSE、CLI、微信异步机制和配置兼容 | 通过 |
| 改造方案需要 review | 本文覆盖架构、兼容性、安全、异常和测试门禁 | 通过 |

## 3. 关键风险与处理结论

### 3.1 P0：Web 缺少可信 OpenID

风险：直接复用 `ScoreContextClient` 会迫使网页端伪造 OpenID，造成渠道身份混淆和上下文串用。

处理：Web 只使用请求 history 合并参数，不调用微信上下文后端；微信只使用解密得到的可信 OpenID。内部渠道字段由服务端构造，公共请求模型不暴露 OpenID。

结论：风险已消除，伪造 Web OpenID/MsgId 的接口测试通过。

### 3.2 P0：统一 Graph 变成“Graph 中调用 Graph”

风险：如果只是新增顶层路由 Graph，而节点内继续调用 `_rag_graph.invoke()` 或 `ScoreQueryGraph.invoke()`，表面统一但运行时仍是多 Graph。

处理：成绩节点与现有 RAG 节点必须注册到同一个 Builder；`score_query_graph.py` 不得再执行 `compile()`；聊天链路静态检查只允许一个编译点。

结论：已落实，静态检查确认只有一个聊天 Graph 编译点。

### 3.3 P0：微信安全边界被业务 Graph 吞并

风险：把签名、AES 或 OpenID 提取放进 Graph，会使网页端绕过微信可信边界，也会让 Graph 状态携带原始密文。

处理：微信适配层完成全部协议和安全处理，只把解密后的文本、可信 OpenID、MsgId 传给内部执行上下文。

结论：通过。

### 3.4 P1：completions 在 Graph 前强制校验 LLM

风险：当前 `create_chat_completion()` 先调用 `get_llm_runtime_status()`。若保留，确定性的成绩分支仍会因 LLM 未配置返回 503，削弱新增能力。

处理：把回答模型运行时校验移动到统一 Graph 的普通 RAG 分支。普通问答的错误行为不变；能够被简单规则高置信确认的成绩查询不依赖意图 LLM，其他启用渠道消息按新要求调用意图 LLM。

结论：通过；高置信成绩快路径绕过 LLM、非快路径执行意图分类、普通 RAG 分支校验回答模型的测试均已覆盖。

### 3.5 P1：Web history 是客户端可控数据

风险：客户端可伪造 history，影响成绩链接判断或产生授权错觉。

处理：history 只用于参数 LLM 重新提取最近一次追问窗口，不产生授权结论；参数完整后仍由 H5 独立完成身份认证和实际成绩查询。

结论：在当前业务边界内可接受。未来若 Graph 直接查询成绩，必须改为服务端会话与强认证，不能沿用本方案。

### 3.6 P1：SSE 返回协议可能被成绩短路破坏

风险：`stream_answer_question()` 当前假设 Graph 结束时一定存在 retrieval 和 prompt；成绩分支提前 END 会导致 KeyError 或事件结构变化。

处理：本次 `STREAM` 上下文显式关闭成绩路由，继续执行原 RAG 分支。后续若要支持流式成绩意图，需要单独定义 direct-answer 的 `meta/delta/done` 映射。

结论：通过，属于回归门禁。

### 3.7 P1：配置名仍带 WECHAT 前缀

风险：能力扩展到 Web 后，`WECHAT_SCORE_QUERY_ENABLED` 和 `WECHAT_SCORE_H5_URL` 语义不准确；直接改名又会破坏部署配置。

处理：新共享名称优先、旧名称回退，第一阶段不删除旧变量；微信上下文 API 配置继续保留原名称。

结论：通过。

### 3.8 P1：非快路径统一调用意图 LLM 的延迟与误降级

风险：不确定表达需要调用意图 LLM，会增加一次模型调用；模型超时、输出格式错误或低置信时不能误进入成绩分支。

处理：第一组规则确认成绩查询并进入参数分支；第二组规则确认普通问题、成绩规则、录入操作、否定表达和录取分数线并直接进入 RAG；剩余请求使用零温度、严格 JSON 枚举、置信度阈值和独立超时配置。异常或无效输出安全降级到普通 RAG，历史 `WECHAT_SCORE_INTENT_*` 配置继续兼容。

结论：符合本次调整。测试验证确定查询和确定非查询均不调用意图模型，只有不确定表达调用模型，并覆盖真实统一 Graph 路由。

### 3.9 P1：参数 LLM 输出污染链接

风险：模型输出或上下文后端返回非法类型、额外字段、越界年份或恶意字符串，可能造成错误链接或 query 注入。

处理：参数 LLM 使用独立零温度模型和严格 JSON 字段集合；服务端校验类型、年份、学年连续性、学期和考试枚举、文本长度，并在持久化合并后再次校验。链接只在必填参数完整时生成，使用 `urlencode` 拼接，不接受模型生成 URL。

结论：代码级风险已控制；学生姓名等参数进入 URL 后仍可能被代理或浏览器历史记录，部署环境应优先使用 HTTPS 并限制访问日志中的 query 采集。

### 3.10 P1：Graph 状态包含敏感信息

风险：OpenID、学生姓名和消息正文可能被完整 state 日志或 tracing 采集。

处理：禁止打印 state；日志只记录渠道、意图、耗时和不可逆身份摘要；外部 tracing 必须关闭输入采集或脱敏。

结论：通过；静态扫描未发现消息正文、完整 OpenID 或完整 Graph state 日志。

## 4. 实施门禁

以下任一项不满足，不得判定改造完成：

1. **单 Graph 门禁**：聊天运行时只有一个 `StateGraph.compile()` 结果，节点内不调用其他 CompiledGraph。
2. **普通 RAG 门禁**：普通问答的答案结构、sources、retrieval metrics、rewrite 和 standalone query 行为通过回归。
3. **成绩能力门禁**：Web 与微信的三级意图判断、参数 LLM、缺参追问、多轮合并、链接编码和功能关闭场景全部通过。
4. **微信安全门禁**：验签、AES、AppId、可信 OpenID、立即 ACK、队列容量和客服消息测试全部通过。
5. **渠道隔离门禁**：Web 无法注入 OpenID、MsgId 或触发微信上下文持久化，只使用 history 补参。
6. **SSE 兼容门禁**：`POST /api/chat/stream` 的 `meta/delta/done/error` 行为不变。
7. **配置与日志门禁**：旧配置可用，固定 H5 URL 校验不放宽，日志无正文、OpenID、学生姓名和完整 Graph state。

### 4.1 实施复审结果

| 门禁 | 代码与测试证据 | 结论 |
|---|---|---|
| 单 Graph | `rag.application` 仅有一个 `compile(name="xinshi-unified-chat")`；成绩模块无 `StateGraph`/`compile` | 通过 |
| 普通 RAG | 真实统一 Graph 测试覆盖 answer、sources、metrics、rewrite、standalone query | 通过 |
| 成绩能力 | 覆盖三级意图判断、参数 LLM、缺参追问、多轮补参、链接编码和 LLM 失败降级 | 通过 |
| 微信安全 | 可信身份只取解密 `FromUserName`，缺失身份丢弃；协议层保持在 Graph 外 | 通过 |
| 渠道隔离 | 公共 `ChatRequest` 不包含 OpenID/MsgId；Web 与微信使用独立参数上下文策略 | 通过 |
| SSE 兼容 | STREAM 禁用成绩路由；测试验证 `meta/delta/delta/done`，并消除重复 LLM 预检 | 通过 |
| 配置与日志 | 新变量优先、旧变量回退；H5 仍校验固定 HTTP `/h5/score?appid=...`；日志不输出正文和完整身份 | 通过 |

## 5. 建议的代码评审检查点

- 检查统一 Graph 条件边是否只有一个明确目标，避免条件边与普通边叠加导致重复执行；
- 检查成绩分支 END 前是否已设置统一 `result_type` 和 `answer`；
- 检查普通 RAG wrapper 是否仍错误地假设所有 Graph 结果都有 retrieval；
- 检查 `ScoreConversationContextProvider` 是否按内部 channel 选择实现，而不是按请求字段猜测；
- 检查 Web history 合并是否只处理最近的成绩追问窗口，避免跨话题污染；
- 检查后端上下文失败时是否保持 current-message fallback，而不是误报持久化成功；
- 检查 H5 URL 是否仍为固定配置且只含 appid；
- 检查 `get_llm_runtime_status()` 是否只移动到 RAG 分支，没有被完全删除；
- 检查 stream 和 CLI 是否显式设置兼容模式；
- 检查删除独立 Graph 后原有测试是否迁移，而不是简单移除。

## 6. 最终评审意见

实现与设计一致，未发现阻断或未处理的高风险遗漏。意图识别已收敛为确定查询、确定非查询和不确定时调用 LLM 的三级判断；成绩查询命中后调用参数 LLM，缺参追问，完整后编码拼接链接。测试已覆盖真实 CompiledStateGraph、同步接口、普通 RAG 响应结构、多轮补参、渠道隔离、配置兼容和 SSE 协议。评审结论为**通过**，保留 URL 中携带学生参数的隐私风险提示。

部署前仍需在目标环境使用真实 DeepSeek、Milvus 和微信回调配置完成联调；该项属于外部依赖验收，不影响本次代码评审结论。
