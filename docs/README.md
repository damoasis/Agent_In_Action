# Agent In Action ：教程总目录

> 基于 Jupyter Notebook 和示例项目的 AI 智能体完整教程系列

---

## 教程列表

| 模块 | 教程文件 | 主要内容 | 关键技术 |
|------|---------|---------|---------|
| **模块一** | [01-agent-tool-mcp-tutorial.md](./01-agent-tool-mcp-tutorial.md) | 智能体基础与工具集成 | GAME 框架、Function Calling、MCP 协议 |
| **模块二** | [02-agent-multi-role-tutorial.md](./02-agent-multi-role-tutorial.md) | 多角色智能体系统 | LangGraph、子图、Map-Reduce、Human-in-the-Loop |
| **模块三** | [03-agent-build-docker-deploy-tutorial.md](./03-agent-build-docker-deploy-tutorial.md) | 企业级系统搭建与部署 | FastAPI、异步任务、Docker、Compose 编排 |
| **模块四** | [04-agent-evaluation-tutorial.md](./04-agent-evaluation-tutorial.md) | 智能体监控与评估 | Langfuse、链路追踪、LLM-as-Judge、AI 安全 |
| **模块五** | [05-agent-model-finetuning-tutorial.md](./05-agent-model-finetuning-tutorial.md) | 模型微调与推理优化 | LoRA、PEFT、LlamaFactory、vLLM |

---

## 学习路径

```
模块一：理解 Agent 基础，掌握工具调用和 MCP 协议
   ↓
模块二：学习 LangGraph，构建多节点工作流和多智能体协作系统
   ↓
模块三：将 Agent 系统包装为 Web 服务，掌握 Docker 容器化部署
   ↓
模块四：接入 Langfuse，实现全链路监控和自动化质量评估
   ↓
模块五：通过 LoRA 微调定制专属模型，使用 vLLM 高性能部署
```

---

## 技术栈总览

| 类别 | 技术 |
|------|------|
| **Agent 框架** | LangChain、LangGraph、LangFuse |
| **协议工具** | MCP（Model Context Protocol）、Function Calling |
| **Web 服务** | FastAPI、Uvicorn、Streamlit |
| **容器化** | Docker、Docker Compose |
| **监控评估** | Langfuse、LLM-as-Judge |
| **模型微调** | LoRA、PEFT、LlamaFactory |
| **推理优化** | vLLM、PagedAttention |
| **LLM 提供商** | OpenAI、DeepSeek、Qwen |

---

## 代码结构

```
Agent_In_Action/
├── 01-agent-tool-mcp/          # 模块一：Agent 基础 + MCP
│   ├── mcp-demo/               # MCP 服务器和客户端示例
│   └── *.ipynb                 # Notebook 教程
├── 02-agent-multi-role/        # 模块二：LangGraph 多角色
│   ├── langgraph/              # 基础和进阶 LangGraph
│   └── deepresearch/           # 深度研究助手实战项目
├── 03-agent-build-docker-deploy/ # 模块三：旅小智旅行规划系统
│   ├── backend/                # FastAPI 后端
│   ├── frontend/               # Streamlit 前端
│   └── docker-compose.yml      # 多容器编排
├── 04-agent-evaluation/        # 模块四：Langfuse 监控评估
│   └── langfuse/code/          # 集成和评估 Notebooks
├── 05-agent-model-finetuning/  # 模块五：模型微调
│   ├── lora/                   # LoRA 从零实现 + PEFT
│   ├── llama_factory/          # LlamaFactory 企业微调
│   └── easy-dataset/           # 数据集构建工具
└── docs/                       # 📖 本教程系列（你在这里）
```
