# Cyber React Agent

基于 ReAct（Reasoning + Acting）范式、**不依赖 LangChain 等框架从零实现**的网络安全问答智能体。
支持流式输出、工具调用循环、本地知识库检索（RAG）与来源引用。

## 特性

- **ReAct 主循环** —— 以 `Thought / Action / Input / Answer` 文本协议驱动「推理 → 调用工具 → 观察结果」闭环
- **流式输出** —— 逐 token 推送到标准输出，或交给自定义回调（便于接 WebSocket / 前端）
- **成本可控** —— 可配置步数上限，预算耗尽时自动收敛为收尾回答，不会无限循环
- **健壮解析** —— 行首锚定的字段解析（兼容中英文名与 Markdown 标记）；工具参数五级降级解析
- **容错** —— 按异常类型与 HTTP 状态码区分重试与快速失败，指数退避；工具异常降级为 Observation 回灌模型自纠
- **RAG** —— Markdown 结构感知切分 + 本地 ONNX 嵌入 + ChromaDB 持久化检索，结果附来源引用
- **可验证** —— 65 个离线单元测试，不联网、不依赖向量库

## 架构

```
main.py                    CLI 入口（REPL）
agent/
  react_agent.py           ReAct 主循环：流式调用 / 解析 / 工具调度 / 步数上限 / 重试
  prompt.py                系统提示词与输出契约
  memory.py                短期对话记忆（窗口裁剪不破坏消息配对）
tools/
  __init__.py              工具注册表 + 参数按名绑定与校验
  search_tool.py           web_search  互联网搜索
  file_tool.py             read_file   读取本地文件
  rag_tool.py              rag_search  知识库检索
rag/
  paths.py                 路径解析（锚定项目根目录，不依赖工作目录）
  loader.py                文档加载（.md / .txt）
  splitter.py              结构感知切分（标题 / 段落边界 + 重叠）
  embedding.py             嵌入模型（Chroma 内置本地 ONNX）
  chroma_db.py             向量库读写
  retriever.py             检索结果整理（文本 + 来源 + 距离）
  ingest.py                知识库导入（幂等，可重复执行）
knowledge_base/            示例知识库（5 篇安全文档）
tests/                     65 个离线单元测试
storage/chroma_db/         向量库（派生产物，不入库）
```

一次问答的消息流：

```
用户提问
   │
   └─> 模型（流式）─> Thought / Action / Input
         │
         ├─ 未命中工具 ──> 输出 Answer，结束
         │
         └─ 命中工具 ──> 执行工具 ──> Observation 回填 ──> 回到模型
```

## 快速开始

要求 Python 3.11+

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

在项目根目录创建 `.env`：

```
DEEPSEEK_API_KEY=你的key
```

导入知识库（首次运行，或文档更新后重新执行）：

```powershell
python -m rag.ingest
```

运行：

```powershell
python main.py                  # 流式输出
python main.py --no-stream      # 等整段返回后再打印
python main.py --quiet          # 只输出最终答案
python main.py --max-steps 12   # 调整单轮步数上限
```

## 测试

```powershell
python -m unittest discover -s tests -t . -v
```

全部用例离线运行：伪造模型客户端，按脚本依次返回字符串、流式分片或异常实例，
因此不消耗 API 额度、不依赖网络与向量库。

## 设计要点

**为什么用文本 ReAct 协议而不是原生函数调用**
文本协议兼容任意 chat 模型，代价是解析脆弱、模型可能伪造工具结果。
生产环境更推荐原生 `tool_calls`（带 `tool_call_id` 与 JSON Schema）。

**Observation 为什么用 `user` 角色回填**
DeepSeek 接口要求 `tool` 角色消息必须携带 `tool_call_id`，而文本协议并不存在真实的
函数调用记录，实测会直接失败：

```
Failed to deserialize the JSON body into the target type:
messages[3]: missing field `tool_call_id`
```

因此改用带 `Observation: ` 前缀的 `user` 消息回填。

**为什么传 `stop` 序列**
文本 ReAct 下模型会顺着提示词里的示例自己续写 `Observation:`，相当于编造工具结果。
在生成层截断比事后正则清洗可靠。

**为什么把路径锚定到项目根目录**
原先向量库使用相对路径 `./chroma_db`，工作目录不同就会读写到不同的库，
曾导致「知识库检索无结果」。现在统一由 `rag/paths.py` 解析，可用
`CHROMA_DB_PATH` 环境变量覆盖。

**为什么统一嵌入路径**
入库与查询若走不同运行时（PyTorch / ONNX），即使模型同名，向量也可能存在偏差。
现在两者共用 Chroma 内置的本地 ONNX 模型，检索结果才可信。

## 已知限制

- 工具集只有 3 个；`web_search` 直接抓取 DuckDuckGo HTML 且未解析为纯文本；`read_file` 未限制根目录
- 记忆为进程内短期窗口，无持久化与摘要压缩
- 单 Agent，无任务规划与多智能体协作
- 未使用原生 function calling
