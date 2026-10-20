# NanoCoder

轻量级终端 AI 编程助手（约 3k 行 Python），架构灵感来自 Claude Code。  
对接任意 **OpenAI 兼容** API（OpenAI / DeepSeek / UniAPI / 本地 Ollama 等），可选 LiteLLM 路由更多厂商。

```bash
git clone https://github.com/HaoyuZhang0214/NanoCoder
cd NanoCoder
pip install -e .          # 或: uv pip install -e .
cp .env.example .env      # 填入 API Key 与模型
nanocoder                 # 进入交互 REPL
```

---

## 它做什么

在终端里用自然语言完成读改代码、跑命令、搜仓库等任务。Agent 会：

1. 读系统提示 + 对话历史，调用 LLM（带工具 schema）
2. 若模型发起 tool call → 鉴权 / 钩子 → 执行 → 把结果塞回历史 → 再调模型
3. 若模型只返回文本 → 把回复流式打印给用户

一次用户输入可能包含多轮「模型 ↔ 工具」循环，直到模型给出最终回答或达到轮次上限。

---

## 快速开始

### 1. 环境变量

复制 `.env.example` 为 `.env`：

```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1   # 兼容网关务必带 /v1
NANOCODER_MODEL=gpt-4o-mini                 # 或 deepseek-chat 等
# NANOCODER_PROVIDER=litellm                # 可选：走 LiteLLM
# NANOCODER_EMBED_MODEL=text-embedding-3-small  # codebase_search 用
```

也可用 `NANOCODER_API_KEY` / `DEEPSEEK_API_KEY` 代替 `OPENAI_API_KEY`。

> **常见坑**：`OPENAI_BASE_URL` 写成 `https://api.xxx.io`（缺 `/v1`）时，SDK 会请求 `/chat/completions`，网关常返回 `404 page not found`。应写成 `https://api.xxx.io/v1`。

### 2. 运行方式

| 命令 | 说明 |
|------|------|
| `nanocoder` | 交互 REPL（改动类工具会询问授权） |
| `nanocoder -p "列出当前目录"` | 一次性模式（无交互，改动类工具默认拒绝） |
| `nanocoder --yes -p "..."` | 自动批准所有工具（脚本 / CI） |
| `nanocoder --demo` | 离线脚本演示，无需 API Key |
| `nanocoder -r <session_id>` | 恢复已保存会话 |
| `nanocoder -m <model> --base-url <url>` | CLI 覆盖环境变量 |

### 3. REPL 斜杠命令

| 命令 | 作用 |
|------|------|
| `/help` | 帮助 |
| `/reset` | 清空对话与 todo |
| `/model` / `/model <name>` | 查看 / 切换模型 |
| `/tokens` | Token 用量与粗算费用 |
| `/compact` | 手动触发上下文压缩 |
| `/diff` | 本会话改过的文件 |
| `/undo` | 回滚最近一次文件改动（checkpoint） |
| `/plan` | 开关 Plan Mode（只读调查 → 出计划） |
| `/save` / `/sessions` | 存盘 / 列出会话 |
| `/mcp` / `/mcp reconnect <name>` | MCP 状态 / 复活挂掉的 server |
| `quit` | 退出 |

`Enter` 提交；`Esc+Enter` 换行（方便粘贴代码）。

---

## 核心逻辑

```
用户输入
   │
   ▼
┌──────────── cli.py ────────────┐
│ 解析命令 / 流式回调 / 权限询问  │
└──────────────┬─────────────────┘
               ▼
┌──────────── agent.py ──────────┐
│  LangGraph:                     │
│    call_llm ──► 有 tool?        │
│       │              │ yes      │
│       │ no           ▼          │
│       │         execute_tools   │
│       │              │          │
│       │◄─────────────┘          │
│       ▼                         │
│     返回最终文本                 │
└──────────────┬─────────────────┘
               │
     ┌─────────┼─────────┐
     ▼         ▼         ▼
  llm.py   tools/*    permissions
  (流式)   (+ MCP)    / hooks
```

### Agent 主循环（`nanocoder/agent.py`）

- **对话历史**放在 `Agent.messages`（user / assistant / tool），不进图状态。
- **LangGraph** 只携带控制量：`rounds`、`has_tools`、`final_response`。
- 图结构：`START → call_llm ⇄ execute_tools → END`。
- 多个 tool call 时用线程池并行执行；权限确认与 PreHook 仍在主线程，避免终端提示交错。
- `plan_mode` 开启时，非只读工具直接拒绝（优先于 `--yes`）。
- 工具异常、权限拒绝都会变成 **tool 结果字符串** 回给模型，循环不崩。

### 工具执行管线

```
模型发起 tool call
  → on_tool（CLI 打印）
  → PreToolUse hooks（exit 2 可拦截）
  → Permission（只读放行；改动需确认 / --yes / always）
  → tool.execute(...)
  → PostToolUse hooks（只观测）
  → 结果写入 messages，回到 call_llm
```

**只读工具**（无需确认）：`read_file`、`glob`、`grep`、`codebase_search`、`todo_write`  
**改动类工具**（默认询问）：`write_file`、`edit_file`、`bash`、`agent`（子 Agent）、以及 MCP 工具

「始终允许」会记到 `~/.nanocoder/permissions.json`。

---

## 内置工具

| 工具 | 作用 |
|------|------|
| `bash` | 跑 shell；危险命令闸；追踪 `cd` |
| `read_file` / `write_file` / `edit_file` | 读、写、唯一匹配搜索替换 |
| `glob` / `grep` | 按文件名 / 内容搜索 |
| `codebase_search` | AST 分块 + FAISS 稠密 + BM25 + RRF 混合检索 |
| `todo_write` | Agent 自维护任务清单（每轮注入 system） |
| `agent` | 派生子 Agent 做子任务 |

MCP：在 `~/.nanocoder/mcp.json` 配置 stdio server，启动时加载为 `mcp__<server>__<tool>`。示例见 [`examples/mcp_filesystem.md`](examples/mcp_filesystem.md)。

---

## 模块地图

包名是 `nanocoder/`（约 2.9k 行）：

```
nanocoder/
├── cli.py           REPL、斜杠命令、一次性模式、流式展示
├── agent.py         LangGraph 主循环、并行工具、plan mode
├── llm.py           OpenAI 兼容流式客户端、重试、成本粗算；可选 LiteLLM
├── config.py        环境变量 / .env
├── prompt.py        system prompt + plan mode 附加说明
├── context.py       三层上下文压缩（snip → summarize → collapse）
├── permissions.py   改动类工具授权
├── hooks.py         ~/.nanocoder/hooks.json 前后钩子
├── mcp.py           MCP stdio 客户端与 reconnect
├── session.py       会话存盘 / 续聊
├── checkpoints.py   文件改动快照与 /undo
├── retrieval.py     代码索引与混合检索（codebase_search）
├── shell.py         POSIX shell 路由（Windows 走 Git Bash）
├── demo.py          离线脚本演示
└── tools/           上表各工具实现 + base.Tool
```

### 其它能力一览

| 能力 | 说明 |
|------|------|
| 上下文压缩 | 50% snip 长工具输出 → 70% LLM 摘要旧轮次 → 90% 硬折叠 |
| Plan Mode | `/plan` 后只读调查，给出编号计划；用户输入 `approve` 再执行 |
| Hooks | `PreToolUse` / `PostToolUse`，stdin 喂 JSON；pre 用 exit 2 否决 |
| Checkpoints | 写文件前快照，`/undo` 回滚最近一次 |
| 会话 | `/save` 到磁盘，`-r` 或 `/sessions` 恢复 |
| 检索 | 增量索引缓存在 `~/.nanocoder/code_index` |

用户级配置目录：`~/.nanocoder/`（`permissions.json`、`hooks.json`、`mcp.json`、会话、代码索引）。

---

## 编程方式调用

```python
from nanocoder import Agent, Config, LLM
from nanocoder.tools import ALL_TOOLS
from nanocoder.permissions import Permission

cfg = Config.from_env()
llm = LLM(model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url)
agent = Agent(llm=llm, tools=ALL_TOOLS, permission=Permission(allow_all=True))
print(agent.chat("用一句话介绍当前工作目录里有什么"))
```

---

## 开发与测试

```bash
pip install -e ".[dev]"
pytest
nanocoder --demo                    # 无需 Key 的冒烟
python examples/plan_hooks_demo.py  # plan + hooks 离线演示
```

可选依赖：`pip install -e ".[litellm]"` 后设置 `NANOCODER_PROVIDER=litellm`。

---

## 设计原则（读代码时有用）

1. **历史在 Agent，控制在 Graph** — 状态轻，compress / reset 直接改 `messages`。
2. **错误变成 tool 文本** — 工具炸了、权限拒了，模型都能看到并改路。
3. **兼容优先** — 默认 OpenAI SDK；换模型只改 `BASE_URL` + `API_KEY` + `MODEL`。
4. **安全默认** — REPL 改动需确认；`-p` 非交互拒绝改动；`--yes` 才全自动。

License: MIT
