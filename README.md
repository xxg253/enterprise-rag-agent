# 企业文档智能问答系统（RAG 多智能体）

一个**可写进简历**的 AI Agent 项目：用 LangGraph 编排多智能体，结合 RAG 检索、  
短期/长期记忆，构建一个能"读懂企业文档 + 记得住上下文"的问答助手。向量存储使用
**Milvus**（生产级、支持持久化，索引不随进程重启而丢失）。

**两种使用方式**：

- 🌐 **Web 应用**（推荐）：本地起一个服务，浏览器打开就能用，有前端界面
- 💻 **命令行**：`python main.py demo` / `chat` / `ask`

> 项目默认 **mock 模式**，无需任何 API Key 即可完整跑通整条流水线（便于学习与演示）；  
> 切换到真实模型只需在 `.env` 里改几行，**不用动代码、也不用把 Key 提交进仓库**。

---

## 一、技术栈（简历关键词）

| 技术            | 在本项目中的落地                                                     |
| ------------- | ------------------------------------------------------------ |
| **LangGraph** | `StateGraph` 编排 4 个 Agent 节点；条件边实现"生成↔审稿"闭环；`TypedDict` 共享状态 |
| **LangChain** | `BaseChatModel` / `Embeddings` 抽象、消息对象、提示词工程                 |
| **RAG**       | 文档加载 → 语义切块（小文件整篇保留）→ 向量化 → 相似度检索 → 上下文注入                  |
| **向量数据库**    | **Milvus**（HNSW 索引 + 持久化 + 幂等 upsert）+ numpy 手写实现作降级兜底，同一接口可替换       |
| **上下文/记忆**    | 短期滑动窗口 + 长期语义召回，两类记忆统一进提示词                                   |
| **Web 服务**    | FastAPI 提供 REST + SSE 流式接口；原生 HTML/JS 前端（零构建）                |
| **文件管理**      | 前端上传/列表/删除，后端热重建索引；文件名清洗 + 扩展名白名单                            |
| **工程化**       | pydantic 强类型配置、依赖注入、工厂模式、pytest 单元测试、CLI                     |

## 二、架构图

```
                          ┌─────────────────────────────┐
                          │        main.py (CLI)         │
                          └──────────────┬──────────────┘
                                         │
                          ┌──────────────▼──────────────┐
                          │   EnterpriseAssistant        │
                          │  (记忆 + RAG + LangGraph 编排) │
                          └───────┬──────────────┬───────┘
                                  │              │
              ┌───────────────────▼──┐     ┌─────▼─────────────────┐
              │   Memory (记忆层)     │     │  Retriever (RAG 层)    │
              │  short_term 短期窗口  │     │  loader → splitter     │
              │  long_term  语义召回  │     │  vectorstore(Milvus)     │
              └──────────────────────┘     └───────────────────────┘
                                  │
                          ┌───────▼────────────────────────────────┐
                          │      LangGraph 多智能体流程图            │
                          │                                        │
                          │   START                                │
                          │     │                                  │
                          │  [route 路由]──DIRECT──►[direct]──►END  │
                          │     │RAG                               │
                          │  [retrieve 检索]                        │
                          │     │                                  │
                          │  [generate 生成]◄────┐                  │
                          │     │               │需要改进           │
                          │  [critique 审稿]────┘                  │
                          │     │不需要改进                        │
                          │    END                                 │
                          └────────────────────────────────────────┘
```

## 三、目录结构

```
enterprise_rag_agent/
├── config.yaml                 # 全局配置（模型/检索/记忆/图参数，密钥只写占位符）
├── .env.example                # 环境变量模板（可复制的模板，进仓库）
├── .env                        # 真实密钥（本地私有，已被 .gitignore 忽略，**不进仓库**）
├── .gitignore
├── requirements.txt
├── run_web.py                  # Web 服务启动入口（推荐）
├── main.py                     # CLI 入口：ask / demo / chat
├── web/
│   └── index.html              # 前端页面（左对话 + 右文件库，含流式与上传）
├── data/                       # 知识库目录（前端上传的文件也存这里）
│   ├── company_policy.md       # 示例：内部制度
│   └── product_faq.md          # 示例：产品问答
├── src/
│   ├── config.py               # pydantic 强类型配置 + .env 加载 + ${VAR} 展开
│   ├── assistant.py            # 顶层编排：记忆 + RAG + 图
│   ├── web_server.py           # FastAPI 后端（REST + SSE 流式接口）
│   ├── llms/
│   │   ├── chat.py             # 聊天模型（Mock + OpenAI 兼容）
│   │   └── embeddings.py       # 嵌入模型（Mock + OpenAI 兼容）
│   ├── memory/
│   │   ├── short_term.py       # 短期记忆（滑动窗口）
│   │   └── long_term.py        # 长期记忆（向量召回 + JSONL 落盘）
│   ├── rag/
│   │   ├── loader.py           # 加载 + 切块（支持 txt/md/csv/json/pdf/docx）
│   │   ├── vectorstore.py      # Milvus 向量库（+ numpy 降级兜底）
│   │   ├── cache.py            # 嵌入向量磁盘缓存（省 token）
│   │   └── retriever.py        # 统一检索入口（来源均衡 + 邻块扩展）
│   └── agents/
│       ├── state.py            # LangGraph 共享状态（TypedDict）
│       ├── nodes.py            # 路由/检索/生成/审稿 四个节点
│       └── graph.py            # StateGraph 编排
└── tests/
    └── test_pipeline.py        # pytest 单元测试
```

## 四、快速开始

### 方式零：从零clone（新人建议照这个顺序）

```bash
# 1) 拉代码
git clone https://github.com/xxg253/enterprise-rag-agent.git
cd enterprise-rag-agent

# 2) 建虚拟环境（务必用虚拟环境，别装进系统 Python）
python -m venv .venv

# Windows 下激活：
.venv\Scripts\activate
# macOS / Linux 下激活：
# source .venv/bin/activate

# 3) 装依赖
pip install -r requirements.txt

# 4) 跑起来
python run_web.py
#    浏览器打开 http://127.0.0.1:8000
```

**关于向量库（新人最常问的两个问题）**

> **Q：仓库里为什么看不到 Milvus 的数据文件？**
> 因为向量数据属于**运行期生成物**，和数据库文件、缓存、`.venv` 一样，不该进 git。
> 本项目用 **Milvus Lite**——它是 Milvus 官方的嵌入式单机版，纯 Python 进程内运行，
> **不需要装 Docker、不需要起服务**。首次 `run_web.py` 时会自动创建 `milvus/rag.db`
> 并把 `data/` 目录里的文档切块、向量化、建索引。
>
> 想清空重来？直接删掉 `milvus/` 目录重启即可，会自动重建。
> 想换成自己的知识库？把你的 `.pdf/.docx/.md/.txt` 丢进 `data/`，或者直接在网页右侧上传。

> **Q：一定要联网吗？**
> 不配 `.env` 也能跑。此时是 **mock 模式**：用假模型生成检索结果，
> 用来演示流程和跑测试。想接真实模型（阿里云百炼 / OpenAI 等 OpenAI 兼容接口），
> 见下一节「接入真实大模型」。

### 方式一：Web 应用（浏览器界面）

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2)（可选）配置真实模型；不配就是 mock 模式，照样能跑
cp .env.example .env

# 3) 启动 Web 服务
python run_web.py

# 4) 浏览器打开
#    http://127.0.0.1:8000
```

界面功能：

- 输入框提问，回车发送
- **实时显示智能体执行轨迹**（路由 → 检索 → 生成 → 审稿）
- 答案**流式逐字返回**
- **右侧「知识库文件」面板**：上传文件、查看文件列表、**预览文件内容**、删除文件
- 上传/删除后**索引自动重建**，无需重启服务，新内容立即可检索
- 顶部状态栏显示当前模式/模型/向量库参数

**支持的知识库文件格式**：`.txt` `.md` `.markdown` `.csv` `.json` `.pdf` `.docx`  
（PDF 用 pypdf 抽取文本，DOCX 用 python-docx 解析段落，CSV/JSON 会扁平化成可检索文本）

### 方式二：命令行

```bash
pip install -r requirements.txt

python main.py demo      # 一键演示（mock 模式，无需 Key）
python main.py ask "你们的退款政策是什么？"
python main.py chat      # 交互式多轮对话
pytest -v                # 跑测试
```

## 五、接入真实大模型（密钥管理）

### 为什么密钥不在 config.yaml 里

把 API Key 直接写进 `config.yaml` 再提交，是最常见也最致命的泄密方式——
**git 历史会永久保留，事后删文件也删不干净**。本项目的做法是：

```
config.yaml   →  只写占位符： api_key: ${LLM_API_KEY}     （进仓库）
.env          →  真实密钥                                     （不进仓库，已被 .gitignore 忽略）
.env.example  →  变量模板                                     （进仓库，告诉别人要配什么）
```

启动时 `src/config.py` 会自动加载同目录的 `.env`，把 `${LLM_PROVIDER}` 这类占位符
替换成真实值。占位符支持 shell 语法 `${VAR:-默认值}`，所以**没配 `.env` 也能直接跑**
（自动退回 mock 模式）。

### 使用步骤

```bash
# 1) 复制模板
cp .env.example .env          # Windows: copy .env.example .env

# 2) 编辑 .env 填入自己的 Key
```

```dotenv
# ---- 大模型 ----
LLM_PROVIDER=openai
LLM_MODEL=qwen-plus
# 任何 OpenAI 兼容端点都可用（阿里云 MaaS / DeepSeek / 腾讯混元 / vLLM 自建…）
LLM_BASE_URL=https://your-endpoint/compatible-mode/v1
LLM_API_KEY=sk-your-key-here

# ---- 嵌入模型 ----
EMBEDDINGS_PROVIDER=openai
EMBEDDINGS_MODEL=text-embedding-v3
EMBEDDINGS_BASE_URL=https://your-endpoint/compatible-mode/v1
EMBEDDINGS_API_KEY=sk-your-key-here
```

也可以直接用系统环境变量（`load_dotenv` 不会覆盖已存在的环境变量，优先级更高）。

> 配了 `provider=openai` 却忘了填 Key，会在**启动阶段**直接报错并指名是哪个变量缺失，
> 而不是等到第一次提问才抛 401 —— 见 `AppConfig.validate_runtime()`。

> ⚠️ 仓库里的 `data/` 只放了示例文档（公司制度、产品 FAQ）。
> 你自己的简历、论文等资料**不会**被提交，请把文件放进本地 `data/` 目录，
> 或直接在网页右侧「上传文件」里传。

## 接口一览（简历可写）

| 方法     | 路径                          | 说明                            |
| ------ | --------------------------- | ----------------------------- |
| GET    | `/`                         | 前端页面                          |
| POST   | `/api/chat`                 | 普通问答，返回 JSON                  |
| POST   | `/api/chat/stream`          | **流式问答（SSE）**，前端实时渲染          |
| GET    | `/api/info`                 | 返回运行配置，供前端状态栏                 |
| GET    | `/api/files`                | **列出知识库文件**（名称/大小/时间/是否可索引）   |
| GET    | `/api/files/{name}/preview` | **预览文件解析后的文本内容**              |
| POST   | `/api/upload`               | **上传文件**（multipart），保存后自动重建索引 |
| DELETE | `/api/files/{name}`         | **删除文件**并重建索引                 |

> 安全设计：文件名经过清洗（防路径穿越 `../`），扩展名白名单校验（与解析器  
> 支持的格式统一维护，从根上避免"能上传却索引不了"的不一致 bug）。

## 六、面试话术（可直接背）

1. **为什么用 LangGraph 而不是普通链？**
   > 因为业务流程里有**循环**（生成→审稿→不通过就重写）和**条件分支**（要不要检索）。  
   > 普通 Chain 是线性的，表达不了回边；LangGraph 用状态图 + 条件边天然支持循环与多智能体协作，  
   > 而且每一步状态可观测、可追踪。
2. **RAG 关键点在哪？**
   > 三个环节：切块策略（**小文件整篇不切、大文件按段落与句子边界切**，避免跨块信息丢失）、  
   > 向量化（语义嵌入）、检索（Milvus 向量库 + 来源均衡 + 邻块扩展）。  
   > 工程上还要控 top_k 和做相关性过滤，防止把噪声喂给模型造成幻觉。
3. **向量库为什么用 Milvus？**
   > 三个诉求：**持久化**（重启不重建索引）、**索引效率**（HNSW，召回率与延迟均衡）、**扩展性**（亿级规模）。  
   > schema 上把文本/来源/块号都存成标量字段，检索结果可直接还原成文档对象，不用在内存留副本；  
   > 主键用内容哈希 + `upsert`，所以索引重建是幂等的，重复上传不会堆出脏数据。  
   > 向量库是接口（Protocol）+ 工厂，Milvus 挂了就降级到 numpy 兜底，业务代码一行不改。
4. **上下文/记忆怎么设计的？**
   > 分短期和长期：短期是滑动窗口保留最近几轮，保证"刚才聊了什么"；  
   > 长期是把历史问答向量化、按语义召回，并落盘 JSONL，保证重启后仍记得。  
   > 两类记忆和检索片段一起拼进提示词，这就是上下文工程。
5. **怎么保证项目可维护/可扩展？**
   > 配置与代码分离（pydantic 强类型校验）、LLM/Embedding 用工厂+依赖注入（换厂商不改业务代码）、  
   > 向量库面向接口编程（Milvus 与 numpy 可无缝替换）、核心逻辑有 pytest 覆盖。
6. **API Key 怎么管理？**
   > `config.yaml` 里只放 `${LLM_API_KEY}` 占位符，真实值在 `.env`（已 gitignore），
   > 加载时展开；占位符支持 `${VAR:-默认值}` 语法，所以没配 Key 时自动退回 mock 模式，
   > 保证"公开仓库 + 零配置可运行"。另外配了 `provider=openai` 却漏填 Key 会在**启动阶段**
   > 直接报错指名变量，而不是等到第一次提问才抛 401。  
   > 还有一条自动化测试专门扫描 `config.yaml` 和 `src/**/*.py`，一旦有人把 `sk-` 开头的
   > 真实密钥写回去，测试立刻失败——把安全约定变成了可执行的约束。

## 七、可扩展方向

- 把 Mock 嵌入换成真实嵌入，检索质量会真正体现语义相似性
- 增加 Rerank 环节（检索后用交叉编码器重排）
- 接入 PDF/Word 解析、多知识库隔离（多租户）
- 把 LangGraph 换成 `astream_events` 做流式输出
- 增加工具调用节点（联网搜索、计算器、数据库查询）
