<div align="center">

# NanoCoder

**编程 agent 里的 nanoGPT。**


```bash
git clone https://github.com/HaoyuZhang0214/NanoCoder
cd NanoCoder
pip install -e .
```

```
corecoder/
├── agent.py        agent 主循环 + 并行工具执行       240 行   
├── llm.py          流式客户端 + 重试 + 成本统计       349 行
├── context.py      三层上下文压缩                     220 行
├── session.py      会话存盘 / 续聊 + 路径穿越防护      97 行
├── permissions.py  改动类工具的用户授权                75 行
├── hooks.py        工具调用前后的用户 shell 钩子        87 行
├── shell.py        POSIX shell 路由（Windows 走 Git Bash） 61 行
├── mcp.py          MCP stdio 客户端，接外部工具       237 行
├── prompt.py       系统提示词                          41 行
├── cli.py          REPL + 斜杠命令 + 一次性模式        393 行
├── config.py       环境变量配置                        55 行
├── checkpoints.py  /undo 快照与回滚                      93 行
├── demo.py         离线端到端演示                       101 行
└── tools/
    ├── bash.py       shell + 危险命令闸 + cd 追踪      203 行
    ├── edit.py       唯一匹配搜索替换 + diff            99 行
    ├── grep.py       内容搜索                           93 行
    ├── glob_tool.py  文件名匹配                         52 行
    ├── read.py       文件读取                           56 行
    ├── write.py      文件写入                           46 行
    ├── todo.py       agent 自维护的任务清单             79 行
    ├── agent.py      子 agent 派生                      72 行
    └── base.py       工具基类                           32 行
examples/
└── plan_hooks_demo.py  离线 plan mode + hooks 演示（免 API key）
```