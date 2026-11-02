# Research Agent

基于 Python 3.11、LangGraph 与 LangChain 的多智能体调研系统。本仓库按增量方式实现；当前已完成 **Step 7：可靠性、成本与可观测性优化**。

## 当前能力

- 已建立 Planner、Supervisor、Researcher、Verifier、Writer、Graph、Memory、RAG、Tool、Guardrail 与 Evidence 的模块边界。
- 已实现统一 `AgentState`、强类型任务模型、DAG 依赖/环路校验和任务状态机。
- Supervisor 会在每轮选出全部可运行任务，并通过 LangGraph `Send` 并行路由到对应 Agent 节点。
- 已实现 Requirement Parser、动态 Planner、Supervisor、Researcher、Verifier 与 Report Writer Agent。
- Agent 使用 Pydantic schema 约束模型输出；支持 OpenAI API 和兼容 OpenAI API 的模型服务。
- 长文本结果写入线程安全的 Blackboard，LangGraph State 只传递 `task_id`、`result_id` 与引用 URI。
- 支持 LangGraph InMemory/PostgreSQL Checkpoint，并可通过 `thread_id` 从中断位置继续执行。
- 支持持久化多轮研究会话：每轮使用独立 `thread_id`，通过 `conversation_id` 与 `parent_thread_id` 继承上一轮报告并恢复完整对话链。
- 支持双水位上下文压缩：在自然任务边界依次执行可再生引用、工具结果裁剪、旧轮次粗粒度化、分段摘要和状态快照；硬水位事件随 Checkpoint 保存，并通过 API/UI 展示压缩层级、节省字符数与保留引用。
- PostgreSQL + pgvector 长期记忆保存用户偏好、历史研究和高价值证据；Planner 会读取相关历史。
- Episodic Memory 记录需求解析、Planner 决策、Supervisor 路由、任务结果与失败原因。
- 记忆生命周期包含价值评分、长度压缩、幂等更新、访问增值、过期/低价值/容量淘汰。
- PostgreSQL Blackboard 可在进程重启后继续解析 Checkpoint 中已有的结果引用。
- 已实现 Query Rewrite、BGE-M3 Dense/Sparse Encoding、Hybrid Retrieval、RRF、BGE-Reranker 与 Top-K 上下文组装。
- RAG 支持网页、知识库、数据库三类来源；PostgreSQL 后端使用 pgvector、BGE-M3 lexical weights 与全文检索回退。
- Planner 根据用户明确要求的研究模块和关键问题，动态拆成 N 个可并行的 Researcher 子课题，并为每个子课题定义 5 个初始查询和完成标准；运行时复用一个无状态 Researcher 执行器严格执行任务。每轮结果由任务级 Verifier 评分，未达标时只针对原任务内的证据缺口补搜，全部任务通过后再进入全局 Verifier 和 Writer。
- 已实现 MCP 风格 Tool Registry，支持工具注册、元数据发现、延迟加载，以及通过官方 MCP ClientSession 动态发现 stdio Server 工具。
- 已实现 Tavily Search、RAG Knowledge Base 与只读 PostgreSQL Database Tool，并可通过本地 FastMCP Server 对外暴露。
- 小模型根据子课题和动态工具目录生成受 schema 约束的调用计划；Researcher 只加载本次任务需要的工具，并可根据证据缺口进行有界补充检索。
- Pre-execution Guardrail 会在外部调用前执行参数 schema、权限、SQL 风险、人工批准标记和每次运行的搜索配额检查。
- 分级恢复策略会将瞬态工具异常映射为指数退避 Retry，将空检索和非法工具参数映射为局部 Re-planning，将权限、高风险操作和重复空检索映射为 Human Intervention。
- SAGA Coordinator 为报告历史等副作用维护幂等 journal；Checkpoint 重放时跳过已完成步骤，失败时按逆序执行补偿动作。支持内存和 PostgreSQL journal。
- Human-in-the-Loop 支持节点后暂停、进度查询、审批恢复、目标与约束修改；Planner 只重写直接受影响任务及其下游依赖，保留无关已完成结果。
- Dynamic Model Router 将需求抽取、Query Rewrite、影响分析和工具选择路由到小模型，将规划、验证、研究推理与报告生成路由到大模型。
- Agent Observability 统一记录 Supervisor 决策、Agent/Tool/Model spans、耗时、失败归因、证据数量和估算 Token；启用 LangSmith 后同步导出 trace。
- Web 工作台可按轮选择自动路由或指定已配置模型，并展示不包含私有思维链的安全执行轨迹（规划、调度、模型 schema、工具调用和 Token 估算）。
- 保留 Step 2 离线执行器，因此未配置模型密钥时仍可运行核心 DAG 测试。
- Worker 异常会转为结构化错误，失败依赖的任务会自动标记为 `blocked`。
- 已提供 FastAPI 应用工厂以及 `/health`、`/ready`、`/research` 接口。
- 已提供 PostgreSQL 16 + pgvector 与 API 服务的 Docker Compose 配置。
- 已预留 OpenAI 兼容接口、Tavily 和 LangSmith 的环境变量；健康检查无需任何密钥即可运行。
- Tavily 仅在配置 `TAVILY_API_KEY` 且启用工具后调用；未配置时不会注册或伪造实时搜索结果。

## 目录

```text
research_agent/
├── agents/            # 结构化 Agent、模型工厂与运行时适配器
├── api/               # FastAPI 入口
├── core/              # 配置与通用基础设施
├── evidence/          # Evidence Store
├── frontend/          # 本地调研工作台（HTML/CSS/JavaScript）
├── graph/             # State、动态 DAG 与 LangGraph 工作流
├── guardrail/         # 工具调用前置治理
├── initdb/            # PostgreSQL 初始化脚本
├── memory/            # Checkpoint、Long-term 与 Episodic Memory
├── rag/               # 检索与重排
├── tests/             # 自动化测试
├── tools/             # Tool Registry 与工具适配器
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

## 本地运行

需要 Python 3.11。

```powershell
Copy-Item .env.example .env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
uvicorn api.main:app --reload
```

启用 BGE-M3 与 BGE-Reranker 需要安装可选 RAG 依赖：

```powershell
python -m pip install -e ".[dev,rag]"
```

打开：

- Web 调研工作台：http://localhost:8000/
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health
- 配置就绪信息：http://localhost:8000/ready
- 调研任务：`POST http://localhost:8000/research`

### Web 调研工作台

浏览器打开 `http://127.0.0.1:8000/` 后，可以直接完成以下操作：

- 配置 OpenAI 或 OpenAI 兼容 API 的 Base URL、API Key，以及大小模型名称。
- 配置 Tavily 搜索和 LangSmith tracing，并开启外部工具与动态模型路由。
- 创建调研任务，查看动态 DAG、Agent 状态、证据数量和执行错误。
- 在报告后继续追问；页面会沿父 checkpoint 恢复历史轮次，并把上一轮报告作为下一轮的有界上下文。
- 每轮可选择自动模型路由、轻量模型或高能力模型；执行中可查看安全的模型思考与 Agent 调度摘要。
- 查看每次真实工具调用的查询、状态、耗时，以及 Tavily 返回的来源 URL 和原始片段。
- 处理中断审批、修改目标或约束，并从 Checkpoint 增量恢复。
- 运行中点击“暂停执行”，系统会在当前 Agent 批次结束后的 Supervisor 安全边界停止；点击“继续执行”使用同一 `thread_id` 从 Checkpoint 续跑，不会重复执行已完成节点。
- 查看、复制和下载最终 Markdown 报告；最近任务保存在当前浏览器中。

页面提交的密钥只保存在当前 API 服务进程的内存中，不会通过接口回显，也不会写入浏览器存储；服务重启后需要重新配置。运行中的任务使用启动时已创建的 Runtime，因此请在开始新任务前完成配置。

当前版本没有用户登录和租户隔离。请仅在本机或可信内网使用，不要将服务直接暴露到公网。Swagger API 文档仍可通过 `http://127.0.0.1:8000/docs` 访问。

默认启用 `REQUIRE_EXTERNAL_EVIDENCE=true` 和 `RESEARCH_QUALITY_GATE_ENABLED=true`。在此模式下，新任务若未配置 Tavily 或未启用工具会直接拒绝执行。Planner 为每个 Researcher 子课题定义 5 个互补初始查询与完成标准；任务级 Verifier 对覆盖度、证据、来源多样性和冲突进行评分，未达到门槛时在单任务预算内继续补搜。全局 Verifier 随后统一验证跨模块结论，Writer 在报告末尾追加由 Evidence Store 生成的“可核验来源”。系统不会把模型内置知识标记成搜索结果。若只想进行离线开发测试，需显式关闭外部证据要求和质量门禁，但这种结果不属于实时调研。

内部 PostgreSQL 查询是可选数据源，默认 `DATABASE_TOOL_ENABLED=false`，因此没有业务数据库时不会尝试连接，也不会阻塞调研。显式启用后，连接默认最多等待 3 秒（`DATABASE_CONNECT_TIMEOUT_SECONDS=3`）；连接失败、表不存在或查询零行都会记录为跳过，工作流继续使用网页证据完成报告。

Web 工作台默认采用分段执行：每段运行到下一个 Supervisor 边界并持久化 Checkpoint，未收到暂停请求时浏览器自动触发下一段。这样既能在节点间安全打断，也能在刷新页面或人工调整后继续执行。API 客户端同样可以用 `resume=true` 和原 `thread_id` 执行下一段；若明确需要服务端一次跑到底，可在请求中传 `continuous=true`。

OpenAI 官方快速入门建议通过 `OPENAI_API_KEY` 环境变量提供密钥。本项目还支持用 `OPENAI_BASE_URL` 指向兼容 OpenAI API 的服务。不要把 `.env` 提交到版本控制。

至少需要配置：

```dotenv
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://api.openai.com/v1
SMALL_MODEL=your-small-model-id
LARGE_MODEL=your-large-model-id
STRUCTURED_OUTPUT_METHOD=json_schema
CHECKPOINT_BACKEND=postgres
BLACKBOARD_BACKEND=postgres
LONG_TERM_MEMORY_BACKEND=postgres
EPISODIC_MEMORY_BACKEND=postgres
RAG_ENABLED=true
RAG_BACKEND=postgres
BGE_DEVICE=cpu
TOOLS_ENABLED=true
TAVILY_API_KEY=your-tavily-key
RESEARCH_QUALITY_GATE_ENABLED=true
RESEARCH_INITIAL_SEARCHES=5
RESEARCH_FOLLOWUP_BATCH_SIZE=2
RESEARCH_MAX_SEARCHES_PER_TASK=10
RESEARCH_COMPLETION_SCORE=80
RESEARCH_MIN_UNIQUE_SOURCES=5
RESEARCH_MIN_INDEPENDENT_DOMAINS=3
RESEARCH_MAX_TOOL_ROUNDS=3
```

如果兼容服务不支持 JSON Schema，可将 `STRUCTURED_OUTPUT_METHOD` 改为 `function_calling` 或 `json_mode`。

调用调研接口：

```powershell
$body = @{ user_goal = "调研当前大模型 Agent 平台的发展趋势" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/research -ContentType application/json -Body $body
```

响应中的 `thread_id` 是恢复令牌。恢复同一任务：

```powershell
$resumeBody = @{
  resume = $true
  thread_id = "previous-thread-id"
  user_id = "user-123"
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/research -ContentType application/json -Body $resumeBody
```

完成一轮后发起后续问题时，应创建新的 `thread_id`，并把上一轮作为父节点。`conversation_id` 在整段对话中保持不变；`model` 可省略（自动路由）或指定已配置的具体模型：

```powershell
$followUpBody = @{
  user_goal = "只比较企业级产品，并补充价格与部署方式"
  user_id = "user-123"
  thread_id = "new-turn-id"
  conversation_id = "conversation-1"
  parent_thread_id = "previous-thread-id"
  model = "your-large-model-id"
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/research -ContentType application/json -Body $followUpBody
```

响应中的 `reasoning_steps` 是可审计的高层执行轨迹，不包含模型隐藏提示词、原始 prompt 或私有思维链。

本地默认使用内存后端；Docker 部署时应按上面的配置将 Checkpoint、长期记忆、Episodic Memory 和 Blackboard 切换为 PostgreSQL。

## 多阶段 RAG

检索链路为：

```text
Query Rewrite → BGE-M3 Dense/Sparse → Hybrid Retrieval → RRF → BGE-Reranker → Top-K Context
```

`RAG_ENABLED` 默认关闭，避免开发测试期间自动下载模型。设为 `true` 后，Researcher 可通过 `knowledge_base.search` 检索 `knowledge_chunks` 中已经索引的网页、知识库和数据库内容。首次实际检索会从配置的模型仓库下载 `BAAI/bge-m3` 与 `BAAI/bge-reranker-v2-m3`，请预留模型缓存空间。

Docker 中启用时还需设置 `INSTALL_RAG=true` 后重新构建镜像；本地 Python 安装不需要该变量。

常用配置：

```dotenv
RAG_ENABLED=true
INSTALL_RAG=true
RAG_BACKEND=postgres
BGE_M3_MODEL=BAAI/bge-m3
BGE_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
BGE_DEVICE=cpu
BGE_USE_FP16=false
RAG_CANDIDATE_K=30
RAG_TOP_K=8
RAG_RRF_K=60
RAG_CONTEXT_CHARS=12000
```

可通过 `RAGPipeline.index_sources()` 写入 `SourceDocument`。启用 Step 6 工具后，Agent 也可通过 Knowledge Base Tool 调用该检索链路。

## MCP 工具与 Guardrail

开启 Agent 工具调用：

```dotenv
TOOLS_ENABLED=true
TOOL_PERMISSIONS=search:execute,knowledge_base:read,database:read
TAVILY_API_KEY=your-tavily-key
SEARCH_MAX_CALLS_PER_RUN=100
SEARCH_MAX_CALLS_PER_TASK=10
DATABASE_STATEMENT_TIMEOUT_MS=10000
DATABASE_MAX_ROWS=100
```

工具执行链路：

```text
Agent → Tool Selector → Tool Registry → Guardrail → Local/MCP Tool → External Service
```

Database Tool 只允许单条 `SELECT`、`WITH`、`EXPLAIN` 或 `SHOW`，在只读事务中执行，并拒绝 `DROP`、`DELETE`、`UPDATE`、`INSERT`、管理语句和高风险 PostgreSQL 函数。SQL 参数、最大返回行数和 statement timeout 均受限制。

启动本地 stdio MCP Server：

```powershell
python -m tools.mcp_server
```

其他进程可使用 `MCPStdioToolProvider` 连接任意 stdio MCP Server，发现结果先以 `ToolSpec` 注册，实际代理在首次调用时才加载。远端 MCP 工具同样经过本地 Guardrail。

## Step 7：恢复、SAGA 与人机协同

恢复与 SAGA 配置：

```dotenv
RECOVERY_ENABLED=true
RECOVERY_MAX_TOOL_RETRIES=2
RECOVERY_MAX_TASK_REPLANS=1
RECOVERY_BACKOFF_SECONDS=0.25
SAGA_BACKEND=postgres
SAGA_AUTO_SETUP=true
```

需要暂停查看 Planner 结果时，可在启动请求中指定节点：

```json
{
  "user_goal": "调研大模型 Agent 平台",
  "user_id": "user-123",
  "pause_after": ["planner"]
}
```

查询 Checkpoint 中的最新进度：

```text
GET /research/{thread_id}
```

修改目标并只重新执行受影响节点：

```json
{
  "resume": true,
  "thread_id": "previous-thread-id",
  "user_id": "user-123",
  "human_feedback": {
    "revised_goal": "仅研究企业级 Agent 平台",
    "additional_constraints": ["只采用 2026 年资料"],
    "affected_task_ids": ["find_sources"]
  }
}
```

高风险工具返回的 `human_requests[].approval_token` 可在恢复请求中通过 `human_feedback.approvals` 提交。未被影响的已完成节点保持 `completed`，Blackboard 结果继续复用。

## 动态模型路由与可观测性

```dotenv
DYNAMIC_MODEL_ROUTING=true
MODEL_ROUTER_COMPLEXITY_CHARS=4000
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=research-agent-local
```

即使未启用 LangSmith，进程内仍会记录结构化 Trace。OpenAI 兼容服务未返回 usage 时，Token 字段明确使用字符长度估算，不伪装为供应商实际计费值。

## Lead 路由、预算与计划审批

Lead 会根据研究类型、关键问题数量和比较/行业类关键词，在 Single-Agent 与 Multi-Agent 之间确定性路由。Planner 为每个任务生成包含搜索、来源、Token、时长和重试上限的任务合同；服务端会把模型提出的额度夹紧到配置上限。所有 Worker 共享全局并发、总任务、搜索、Token、费用和运行时长断路器。

```dotenv
MAX_CONCURRENT_WORKERS=5
MAX_TOTAL_TASKS=12
MAX_TOTAL_SEARCHES=100
MAX_TOTAL_INPUT_TOKENS=500000
MAX_TOTAL_OUTPUT_TOKENS=200000
MAX_TOTAL_COST_USD=0
MAX_RUN_SECONDS=1200
TASK_MAX_SEARCHES=10
TASK_MAX_SOURCES=12
TASK_MAX_INPUT_TOKENS=60000
TASK_MAX_OUTPUT_TOKENS=20000
TASK_MAX_SECONDS=300
PLAN_APPROVAL_MODE=auto
PLAN_APPROVAL_TASK_THRESHOLD=8
PLAN_APPROVAL_TOKEN_THRESHOLD=200000
PLAN_APPROVAL_COST_THRESHOLD_USD=5
```

`MAX_TOTAL_COST_USD=0` 表示不启用费用断路；配置各模型的每百万 Token 单价后，API 才会返回可用于断路的估算费用。`/research` 响应中的 `lead_decision`、`plan_estimate` 和 `budget` 分别给出路由理由、执行前估算和运行中实际估算。高成本计划会使用 `plan:<run_id>` 审批令牌暂停，确认后才分发 Worker。

Researcher、Retriever 和 Database Worker 只能发现并执行搜索、读取、知识库和只读数据库能力；任何声明 `write`、`send`、`deploy`、`delete`、`mutate` 或 `admin` 能力的工具都会在角色边界被拒绝。

## 上下文压缩

压缩在 Agent Prompt 组装和多轮继承两个自然边界执行。低于软水位时上下文保持不变；超过软水位后按五层阶梯逐层压缩，降到软水位即停止；超过硬水位时会强制生成包含目标、约束、决策理由、未完成事项、精确引用、验证结果和下一步行动的状态快照。

```dotenv
MAX_AGENT_CONTEXT_CHARS=16000
CONTEXT_SOFT_WATERMARK=0.70
CONTEXT_HARD_WATERMARK=0.85
CONTEXT_TOOL_RESULT_MAX_CHARS=2400
CONTEXT_SUMMARY_MAX_CHARS=1200
CONTEXT_COMPRESSION_MAX_EVENTS=1000
```

每次压缩记录会出现在 `/research` 响应的 `context_compressions` 中。记录包含触发水位、最高压缩层级、压缩前后字符数、内容哈希、保留引用数、Prompt Cache 前缀是否变化，以及是否要求硬水位 Checkpoint。

## Docker 运行

```powershell
Copy-Item .env.example .env
docker compose up --build
```

首次启动 PostgreSQL 时会自动启用 `vector` 扩展。停止服务：

```powershell
docker compose down
```

如需同时删除本地数据库卷，请在确认数据可丢弃后显式运行 `docker compose down -v`。

## 验证

```powershell
pytest
ruff check .
docker compose config
```

无需模型密钥即可运行一次离线 DAG：

```powershell
python -c "from graph.workflow import run_research_workflow; s = run_research_workflow('研究新能源汽车市场'); print(s['workflow_status']); print(s['final_report'])"
```

## 下一步

增量开发步骤已全部完成。后续可继续扩展真实通知/报告仓储的 SAGA Step、HTTP MCP Transport、离线评测集与生产级鉴权。

## 参考

- [OpenAI API Developer quickstart](https://developers.openai.com/api/docs/quickstart)
- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
