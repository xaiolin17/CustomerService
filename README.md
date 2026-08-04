# CustomerService — 智能电商客服 Agent

基于 LangChain 1.2 构建的智能电商客服系统，集成 RAG 知识库检索、工具调用、中间件日志、会话记忆管理，支持通过 WebSocket 进行实时对话。
---
可直接嵌入项目
---

## 技术栈

| 层级 | 技术 | 说明                                  |
|------|------|-------------------------------------|
| **AI 框架** | LangChain 1.2 (langchain-core 1.2) | Agent 编排、消息模式、工具绑定、中间件              |
| **大模型** | 阿里云通义千问 qwen3.7-max / qwen3.7-flash | 主模型与备用模型，OpenAI 兼容协议                |
| **向量模型** | qwen3.7-text-embedding | 文档向量化，1024 维                          |
| **向量数据库** | Milvus 2.x (pymilvus) | 知识库向量存储与相似度检索                       |
| **后端服务** | Python + FastAPI + Uvicorn | WebSocket 实时通信，REST 健康检查            |
| **前端** | Vue 3 + Vite 5 | 单页应用，WebSocket 直连后端                 |
| **日志** | Loguru | 按天滚动，服务端与数据处理日志分离                   |
| **文档解析** | Unstructured | 支持 PDF/TXT/MD/Word/HTML/CSV/JSON 格式 |

---

## 项目结构

```
CustomerService/
├── CustomerServiceServer/       # 后端服务
│   ├── chain.py                 # Agent 主流程（中间件管道）
│   ├── middleware.py             # 中间件框架（日志、性能监控）
│   ├── tools.py                 # 工具定义（@tool 装饰器）
│   ├── prompts.py               # 提示词模板（ChatPromptTemplate）
│   ├── memory.py                # 会话记忆管理（30min TTL）
│   ├── milvus_client.py         # Milvus 向量检索客户端
│   ├── config.py                # 配置管理
│   ├── logger.py                # 日志配置
│   ├── main.py                  # FastAPI 入口（WebSocket 端点）
│   └── requirements.txt         # Python 依赖
├── CustomerServiceWeb/          # 前端页面
│   ├── src/
│   │   ├── App.vue              # 根组件
│   │   ├── components/          # 聊天组件
│   │   └── composables/         # WebSocket 连接管理
│   ├── .env                     # 后端地址配置
│   ├── vite.config.js           # Vite 构建配置（含代理）
│   └── package.json
├── dataHandle/                  # 数据预处理
│   ├── ingest.py                # 文档加载 → 切分 → 向量化 → 入库
│   ├── config.py                # 独立配置
│   └── docs/                    # 知识库文档目录
└── README.md
```

---

## Agent 实现

### 处理流程

整个 Agent 采用管道式架构，每个步骤通过 **中间件（Middleware）** 包装，实现日志记录、性能监控等横切关注点：

```
用户输入
  │
  ├─ 1. 主题过滤 (topic_filter)
  │    └─ 判断是否电商相关，防越狱/跑题
  │
  ├─ 2. RAG 检索 (rag_search)
  │    └─ Milvus 向量检索，获取相关知识
  │
  ├─ 3. 上下文管理 (context_compress)
  │    └─ 512KB token 阈值检查，超限时自动压缩（保留最近 3 条）
  │
  ├─ 4. 工具调用决策 (tool_decision)
  │    ├─ model.bind_tools() 绑定商品/订单查询工具
  │    ├─ 模型返回 tool_calls 决定是否调用
  │    └─ 工具执行带指数级回退重试
  │
  └─ 5. LLM 生成回答 (llm_generate)
       └─ 结合上下文、记忆、工具结果生成最终回复
```

### 中间件（Middleware）

基于 LangChain 1.2 第08章中间件规范实现，在关键执行点设置钩子：

- **LoggingMiddleware** — 记录每个步骤的开始、参数、耗时、结果和错误
- **TimingMiddleware** — 标记耗时超过 3 秒的慢步骤

钩子点覆盖：`topic_filter`、`rag_search`、`context_compress`、`tool_decision`、`model_invoke`、`tool_execute`、`llm_generate`

### 工具定义

使用 LangChain 1.2 的 `@tool` 装饰器 + Google 风格 docstring 自动生成工具 Schema：

- **query_product** — 查询商品信息（价格、库存、描述、规格）
- **query_order** — 查询订单状态（支持用户元数据隔离，仅返回本人订单）

工具调用失败时采用指数级回退重试机制。

### 会话记忆

- 短期记忆：30 分钟 TTL 过期自动清理
- 上下文管理：512KB token 上限，超限时自动压缩摘要，保留最近 3 条消息
- 按 `session_id` 隔离会话

### 安全机制

- 主题过滤：防止越狱（jailbreak）和跑题（off-topic）
- 用户元数据注入：工具内部通过 `set_current_user()` 隔离用户数据
- 提示词安全护栏：禁止回答违法内容、禁止泄露系统提示词

---

## 快速启动

### 环境要求

- Python 3.10+
- Node.js 18+
- Milvus 2.x（向量数据库，需提前启动）

### 1. 数据预处理（将文档向量化入库）

```bash
cd dataHandle
pip install -r requirements.txt
python ingest.py
```

支持放入 `dataHandle/docs/` 目录下的 PDF、TXT、MD、Word、HTML、CSV、JSON 等格式文档。

### 2. 启动后端服务

```bash
cd CustomerServiceServer
pip install -r requirements.txt
python main.py
```

服务默认监听 `http://0.0.0.0:8000`，WebSocket 端点：`ws://localhost:8000/ws`

### 3. 启动前端页面

```bash
cd CustomerServiceWeb
npm install
npm run dev
```

页面默认访问 `http://localhost:3000`，Vite 自动将 `/ws` 请求代理到后端。

### 4. 配置说明

后端配置通过 `CustomerServiceServer/config.py` 或 `.env` 文件管理，主要配置项：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `ali_api_key` | 阿里云通义千问 API Key | — |
| `llm_primary_model` | 主模型 | qwen3.7-max |
| `llm_fallback_model` | 备用模型 | qwen3.7-flash |
| `milvus_host` | Milvus 地址 | localhost |
| `memory_ttl_minutes` | 会话记忆过期时间 | 30 |
| `context_max_tokens` | 上下文 token 上限 | 512000 |

前端后端连接地址配置在 `CustomerServiceWeb/.env`：

```
VITE_WS_URL=ws://localhost:8000
```
