# 后训练数据集资源清单

> **说明**：
> - 许可：除个别注明外均为 **apache-2.0**；数据均为模型蒸馏生成（合成数据），可能存在事实错误，训练前建议自行清洗。
> - 用途不限于单一阶段：可作 **SFT** 冷启动语料，也可用于 **OPD（离线后训练，如 DPO）**，含验证答案 / 评分信号的还可直接构造 **RLVR / GRPO** 奖励与偏好对。
> - **题目来源**：蒸馏流程的输入——原始问题 / 指令取自哪个数据集或如何构建（只提供"题目"）。
> - **回答模型**：为这些题目生成答案（含推理链）的教师模型（负责"答题"）。最终数据 = 题目来源的问题 + 回答模型的答案。

---

## 速查总表

| # | 数据集 | 数据量 | 题目来源 | 回答模型 | 类别 |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | Chinese-DeepSeek-V3.2-Exp-chat-example | 6,655 轮 | 自建真实中文对话 | DeepSeek-V3.2-Exp | 中文对话 |
| 2 | Chinese-Qwen3-235B-Thinking-2507-Distill-100k | 106,950 条 | 自建中文问题（约 110K） | Qwen3-235B-A22B-Thinking | 中文推理 |
| 3 | Competitive-Programming-python-blend | 10K–100K | Nemotron 竞赛编程等多源 | 多源混合（上游自带答案） | 竞赛编程 |
| 4 | DeepSeek-V3.2-Exp-reasoning-example | 208 题 | Nemotron-Post-Training-v2 数学题 | DeepSeek-V3.2-Exp（+R1-0528 对照） | 数学推理 |
| 5 | DeepSeek-v3.1-reasoner-Distilled-math-samples | <1K | Nemotron-Post-Training-v2 数学子集 | DeepSeek-V3.1 Reasoner（+R1-0528 对照） | 数学推理 |
| 6 | GPT-OSS-120B-Distilled-Reasoning-math | 1K–10K | 数学题（未明示来源） | gpt-oss-120b | 数学推理 |
| 7 | GPT-OSS-20B-Distilled-Reasoning-Mini | 1,990 条 | Qwen3-4B 按 7 类模板生成 + natural_reasoning / GSM8K | openai/gpt-oss-20b | 通用推理 |
| 8 | IELTS-writing-feedback-reasoning | 8,971 条 | IELTS Task 2 题目 + 学生作文 | GLM-4.7 | 写作评分 |
| 9 | LogicMind-Chat-Reasoning-SFT-300K | 296,168 条 | Nemotron-Post-Training-v2 chat 部分 | qwen-3-32b | 通用推理 |
| 10 | LogosForge-scored-sft-v1 | 63,852 条 | facebook/natural_reasoning | gpt-oss-120B-high（qwen3-235b 打分） | 带评分推理 |
| 11 | MultiReason-ChatAlpaca | 18,300 条 | ChatAlpaca 风格多轮提问 | gpt-oss-120B-high | 多轮对话 |
| 12 | Natural-Reasoning-gpt-oss-120B-S1 | 1K–10K（S1 批次） | facebook/natural_reasoning 前 100K 题 | gpt-oss-120-high | 通用推理 |
| 13 | Qwen3-235B-A22B-Instruct-2507-Distilled-chat | 6,535 条 | 跨语种对话问题（未明示来源） | Qwen3-235B-A22B-Instruct | 多语对话 |
| 14 | Qwen3.5-reasoning-700x | ~700 条 | Alibaba-Superior-Reasoning-Stage2 | qwen3.5-27b | 综合推理 |
| 15 | ShareGPT-Qwen3-235B-A22B-Instuct-2507 | 未公布 | ShareGPT 风格多轮提问 | Qwen3-235B-A22B-Instruct | 多轮对话 |
| 16 | ShareGPT-gpt-oss-120B-reasoning | 未公布 | ShareGPT 风格多轮提问 | gpt-oss-120B（medium） | 多轮推理 |
| 17 | financial-economics-reasoning | 122,378 条 | BAAI/IndustryInstruction 金融经济指令 | Qwen3-235B-A22B-Thinking | 金融经济 |
| 18 | glm-4.7-Superior-Reasoning-stage1 | 1,192 条 | Alibaba-Superior-Reasoning stage1 数学题 | glm-4.7 | 数学推理 |
| 19 | glm-4.7-multiturn-CoT | 3,725 条 | ChatAlpaca / ShareGPT 风格多轮提问 | glm-4.7 | 多轮推理 |
| 20 | gpt-oss-120B-distilled-math-OpenAI-Harmony | 1K–10K | 数学题（未明示来源） | gpt-oss-120b | 数学推理 |
| 21 | gpt-oss-120B-distilled-reasoning | 1K–10K | 数学题（未明示来源） | gpt-oss-120b | 数学推理 |
| 22 | gpt-oss-120b-Reasoning-Instruction | 未公布 | DeepSeek 指令（文件名暗示） | gpt-oss-120b | 推理指令 |
| 23 | gpt-oss-120b-reasoning-STEM-5K | ~5K–10K | 部分取自 Nemotron-Post-Training-v1 | gpt-oss-120b-high | STEM 推理 |
| 24 | qwen3-coder-480b-distill-mini | 9,543 条 | microsoft/rStar-Coder 代码题 | Qwen3-Coder-480B-A35B | 代码推理 |
| 25 | nohurry/Opus-4.6-Reasoning-3000x-filtered | 27B、35B（各采样 3,900） | ORCA / minimax / drive 等数学与交互题 | Opus 4.6 | 数学推理 |
| 26 | Roman1111111/claude-opus-4.6-10000x | 27B（9,633）、35B（10,000） | GSM8K / MATH + 逻辑谜题 + 多步指令 | Claude Opus 4.6 | 数学 + 逻辑推理 |
| 27 | stepfun-ai/Step-3.5-Flash-SFT | 9B（buffer 采样 ≤100K） | 混合（上游自带答案） | Step-3.5-Flash | 通用 SFT |
| 28 | unsloth/OpenMathReasoning-mini | GRPO 冷启动 SFT（cot split） | OpenMath 系数学题 | 上游自带 | 数学推理 |
| 29 | open-r1/DAPO-Math-17k-Processed | GRPO 强化学习（en） | DAPO-Math-17k | 上游自带 | 数学推理 |

---

## 详细清单

### 一、中文数据

#### 1. Chinese-DeepSeek-V3.2-Exp-chat-example

- **数据量**：6,655 轮真实中文对话（含少量混合语种）
- **题目来源**：作者构建的真实中文对话问题（自建）
- **回答模型**：deepseek/deepseek-v3.2-exp（官方 API，8K 上下文）
- **下载地址**：https://huggingface.co/datasets/Jackrong/Chinese-DeepSeek-V3.2-Exp-chat-example
- **内容例子**：对话消息格式；典型特征为「短问长答」——平均 Prompt 67 字符、平均 Output 1,460 字符：

```json
{"conversation": [
  {"from": "human", "value": "请解释什么是边际效应递减。"},
  {"from": "gpt", "value": "边际效应递减是指……（约 1,400 字符的详细解释）"}
]}
```

- **适用范围**：中文对话 / 解释归纳类任务 SFT；8K 上下文内短问长答风格对齐；不适合需要强推理链的任务。

#### 2. Chinese-Qwen3-235B-Thinking-2507-Distill-100k

- **数据量**：106,950 条（原始问题约 110K，清洗后保留）
- **题目来源**：作者构建的中文问题，覆盖数学 / 通用知识 / 技术编程 / 语言写作 / 商业经济
- **回答模型**：Qwen3-235B-A22B-Thinking-2507（OpenRouter，32K 上下文）
- **下载地址**：https://huggingface.co/datasets/Jackrong/Chinese-Qwen3-235B-Thinking-2507-Distill-100k
- **内容例子**：每条含 `Generator` 字段（回答模型名）+ 消息 + 推理链，推理呈长尾分布（部分超 5,000 字符）：

```json
{"Generator": "qwen-3-235b-a22b-thinking-2507", "messages": [
  {"role": "user", "content": "中文问题"},
  {"role": "assistant", "content": "<think>完整中文推理链……</think>\n最终答案"}
]}
```

- **适用范围**：中文推理能力蒸馏 SFT；领域分布为数学 48,020 / 通用知识 15,934 / 技术编程 12,042 / 语言写作 9,181 / 商业经济 7,766，可按领域抽样；支持长上下文（32K）训练。

---

### 二、数学推理

#### 3. GPT-OSS-120B-Distilled-Reasoning-math

- **数据量**：1K–10K 条
- **题目来源**：数学题（README 未明示具体来源数据集）
- **回答模型**：gpt-oss-120b（MXFP4）
- **下载地址**：https://huggingface.co/datasets/Jackrong/GPT-OSS-120B-Distilled-Reasoning-math（README 引用块确认）
- **内容例子**：字段为 `Generator / Category / Input / CoT_Native_Reasoning / Reasoning / Answer`；推理复杂度比 39.19（推理长度 ÷ 输入长度），答案收敛比 0.67：

```json
{"Generator": "gpt-oss-120b", "Category": "math",
 "Input": "Given that 2^x = 8, find x.",
 "Output": "<think>First, divide both sides by 12. 36 / 12 = 3. So x = 3.</think> The answer is 3."}
```

- **适用范围**：数学问题求解 SFT / CoT 蒸馏；推理与答案分离便于分别控制；与 DeepSeek / Qwen3 系 `<think>` 格式兼容。

#### 4. gpt-oss-120B-distilled-reasoning

- **数据量**：1K–10K 条
- **题目来源**：数学题（README 未明示具体来源数据集）
- **回答模型**：gpt-oss-120b
- **下载地址**：https://huggingface.co/datasets/Jackrong/gpt-oss-120B-distilled-reasoning
- **内容例子**：与 #3 同源，提供三种标注模板：`CoT`（推理+答案一体）、`CoT_Native`（原生分字段）、`Think`（Qwen3/DeepSeek 风格 `<think>` 包裹）。
- **适用范围**：数学推理 SFT；按目标基座模型的模板选格式，减少格式清洗成本。

#### 5. gpt-oss-120B-distilled-math-OpenAI-Harmony

- **数据量**：1K–10K 条
- **题目来源**：数学题（README 未明示具体来源数据集）
- **回答模型**：gpt-oss-120b
- **下载地址**：https://huggingface.co/datasets/Jackrong/gpt-oss-120B-distilled-math-OpenAI-Harmony（README 引用块确认）
- **内容例子**：OpenAI 通道（channel）格式，analysis 通道放推理、final 通道放答案：

```json
{"output": "<|channel|>analysis<|message|>We have right triangle ABC with right angle at C……<|end|>\n<|start|>assistant<|channel|>final<|message|>### The circle ω …… \\boxed{4}<|return|>"}
```

- **适用范围**：面向 OpenAI 系（gpt-oss 家族）格式的数学 SFT；多通道推理-答案分离结构，适合做 RL 时的格式约束奖励。

#### 6. DeepSeek-v3.1-reasoner-Distilled-math-samples

- **数据量**：<1K 条（小样本研究用）
- **题目来源**：nvidia/Nemotron-Post-Training-Dataset-v2 数学子集
- **回答模型**：DeepSeek-V3.1 Reasoner（另附 R1-0528 对照答案）
- **下载地址**：https://huggingface.co/datasets/Jackrong/DeepSeek-v3.1-reasoner-Distilled-math-samples
- **内容例子**：含 `question / reasoning / answer` + R1-0528 对照答案；平均推理链 17,236 字符、答案 2,143 字符，超长 CoT 长尾显著：

```json
{"question": "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}...",
 "reasoning": "（平均 17K 字符的分步推理）",
 "answer": "\\boxed{...}",
 "seed_answer": "（R1-0528 对照答案）"}
```

- **适用范围**：超长推理链（long CoT）SFT / RL 研究；50% 样本去掉了格式指令以提升多样性；用于分析师生推理长度迁移。

#### 7. DeepSeek-V3.2-Exp-reasoning-example

- **数据量**：208 题（V3.2-Exp 与 R1-0528 对比样本）
- **题目来源**：nvidia/Nemotron-Post-Training-Dataset-v2 数学题（代数 / 数论 / 微积分 / 组合等）
- **回答模型**：DeepSeek-V3.2-Exp（另附 R1-0528 对照答案）
- **下载地址**：https://huggingface.co/datasets/Jackrong/DeepSeek-V3.2-Exp-reasoning-example
- **内容例子**：数学题 + "Step N" 分步推导 + `\boxed{}` 结论；V3.2-Exp 平均 4.35 步、数学符号密度 25.10（R1-0528 无显式 CoT）：

```json
{"problem": "In triangle ABC with BC=3, CA=4, AB=5 ……",
 "answer": "Step 1: …… Step 2: …… \\boxed{4}"}
```

- **适用范围**：结构化分步解答教学 / 主题化课程设计（数论、组合 vs 微积分偏好互补）；小样本，仅作参考与对照实验。

#### 8. glm-4.7-Superior-Reasoning-stage1

- **数据量**：1,192 条（domain 主要为 math）
- **题目来源**：Alibaba-Superior-Reasoning 管线 stage1 数学题
- **回答模型**：glm-4.7（低温蒸馏，temp 0.6）
- **下载地址**：https://huggingface.co/datasets/Jackrong/glm-4.7-Superior-Reasoning-stage1
- **内容例子**：带训练元信息：

```json
{"id": "问题哈希", "conversation": [
  {"from": "human", "value": "数学问题"},
  {"from": "gpt", "value": "<think>……</think>\n答案"}],
 "input": "……", "output": "<think>……</think>\n……",
 "domain": "math",
 "meta": {"training_stage": "stage1", "sampling_temperature": 0.6, "teacher_model": "glm-4.7"}}
```

- **适用范围**：stage1 推理冷启动 SFT、数学 CoT 行为对齐、回答模型替换消融实验（对比 Alibaba 原管线）。

---

### 三、STEM 推理

#### 9. gpt-oss-120b-reasoning-STEM-5K

- **数据量**：1K–10K（约 5K，卡面 pretty_name 为 STEM-10K）
- **题目来源**：部分取自 nvidia/Nemotron-Post-Training-Dataset-v1
- **回答模型**：gpt-oss-120b-high
- **下载地址**：https://huggingface.co/datasets/Jackrong/gpt-oss-120b-reasoning-STEM-5K
- **内容例子**：推理链与答案强制分离（`CoT_Native——reasoning` / `Answer`），覆盖选择、简答、计算证明、概念题：

```json
{"Generator": "gpt-oss-120b", "Category": "stem",
 "Input": "Prompt/Question Text",
 "CoT_Native——reasoning": "完整多步推理链",
 "Answer": "最终答案（与推理解耦）"}
```

- **适用范围**：STEM（数理化生、计算机、工程、生命科学）COT/SFT；「过程对 + 答案对」双维度评测；长尾学科样本偏少，建议混合其他数据源；license 卡面注明 CC-BY-4.0。

---

### 四、代码数据

#### 10. Competitive-Programming-python-blend

- **数据量**：10K–100K 条
- **题目来源**：nvidia/Nemotron-SFT-Competitive-Programming-v2（Python 87.54% + C++ 2.50%）+ nohurry/Opus-4.6-Reasoning-3000x-filtered（5.83%）+ nvidia/Nemotron-SFT-Instruction-Following-Chat-v2（2.50%）+ Qwen3.5-reasoning-700x（1.58%）+ nvidia/Nemotron-SFT-SWE-v2 agentless（0.05%）
- **回答模型**：多源混合——各上游数据集自带答案，非统一教师蒸馏
- **下载地址**：https://huggingface.co/datasets/Jackrong/Competitive-Programming-python-blend
- **内容例子**：统一为 `messages` 格式，来源归一化为 user/assistant 角色，每条带 SHA-256 `id`：

```json
{"id": "e3f7b0d4…", "messages": [
  {"role": "user", "content": "竞赛编程题目"},
  {"role": "assistant", "content": "<think>解题思路……</think>\n```python\n代码\n```"}]}
```

- **适用范围**：Python 竞赛编程 / 代码推理 SFT；多 license 混合（apache-2.0/cc-by-4.0/odc-by/mit 等）。

#### 11. qwen3-coder-480b-distill-mini

- **数据量**：9,543 条（原始题目 10,000，清洗后保留）
- **题目来源**：microsoft/rStar-Coder 代码题（含输入输出验证和测试用例）
- **回答模型**：Qwen3-Coder-480B-A35B-Instruct（32K 上下文）
- **下载地址**：https://huggingface.co/datasets/Jackrong/qwen3-coder-480b-distill-mini
- **内容例子**：含 `question / answer / reasoning` 字段：

```json
{"question": "代码题（带测试用例）",
 "reasoning": "Qwen3-Coder-480B-A35B 的推理链",
 "answer": "代码解答"}
```

- **适用范围**：代码推理 SFT；经过去重、完整性、括号配平、模板化过滤四道清洗；Apache-2.0，可商用。

---

### 五、通用推理 / 指令数据

#### 12. LogicMind-Chat-Reasoning-SFT-300K

- **数据量**：296,168 条
- **题目来源**：nvidia/Nemotron-Post-Training-Dataset-v2 的 chat 部分
- **回答模型**：qwen-3-32b
- **下载地址**：https://huggingface.co/datasets/Jackrong/LogicMind-Chat-Reasoning-SFT-300K
- **内容例子**：chat 风格指令 + 显式推理链 + 最终解，预计算长度字段：

```json
{"problem": "……", "qwen3-reasoning": "显式推理过程", "qwen3-solution": "最终答案"}
```

- **适用范围**：大规模通用推理 SFT（含 RL 数据、数学/代码/推理/聊天多任务）；适合做推理长度控制与响应压缩基准；语言与领域分布非均匀，建议分层抽样。

#### 13. LogosForge-scored-sft-v1

- **数据量**：63,852 条
- **题目来源**：facebook/natural_reasoning
- **回答模型**：gpt-oss-120B-high 作答，qwen-3-235b-a22b-instruct-2507 打分（temp 0.2）
- **下载地址**：https://huggingface.co/datasets/Jackrong/LogosForge-scored-sft-v1
- **内容例子**：蒸馏回答 + 自动评分信号，带四档质量标签：

```json
{"conversation": [{"from": "human", "value": "问题"}, {"from": "gpt", "value": "<think>……</think>答案"}],
 "model_answer_score": 9, "CoT_score": 8, "average_score": 8.5,
 "reference_answer_similarity": 85.0,
 "judgment": "Venti"}
```

- **适用范围**：质量感知过滤 / 排序 / 课程学习（curriculum learning）；`Venti > Grande > Tall > No` 四档可直接筛数据；正交分离「答案对不对」与「推理好不好」，适合构造 RL 奖励信号。

#### 14. Natural-Reasoning-gpt-oss-120B-S1

- **数据量**：1K–10K（S1 首批；基于 natural_reasoning 前 100K 题，含 81.7% 参考答案）
- **题目来源**：facebook/natural_reasoning 前 100,000 题（经过去基准污染：MATH/GPQA/MMLU-Pro）
- **回答模型**：gpt-oss-120-high
- **下载地址**：https://huggingface.co/datasets/Jackrong/Natural-Reasoning-gpt-oss-120B-S1
- **内容例子**：

```json
{"generator": "gpt-oss-120-high",
 "question": "多步推理问题（平均 55 词）",
 "CoT_reasoning": "回答模型的完整思维链",
 "anwser": "最终答案",
 "reference_answer": "原始参考答案（可为空）"}
```

- **适用范围**：通用复杂推理 CoT 蒸馏；参考答案字段可用于答案校验与 **DPO 偏好对构造**；样本效率高。

#### 15. GPT-OSS-20B-Distilled-Reasoning-Mini

- **数据量**：1,990 条（从 1,994 条中筛选）
- **题目来源**：unsloth/Qwen3-4B-Instruct-2507 按 7 类思维模板生成的 300 条初始问题，并混合 facebook/natural_reasoning、openai/gsm8k 等
- **回答模型**：openai/gpt-oss-20b（High）
- **下载地址**：https://huggingface.co/datasets/Jackrong/GPT-OSS-20B-Distilled-Reasoning-Mini
- **内容例子**：

```json
{"question": "……", "answer": "<think>详细推理过程</think>\n最终回答"}
```

- **适用范围**：小模型推理能力提升、指令跟随 SFT；样本经 1,946 条人工评分验证（8.0–10.0 区间集中），质量高但规模小，适合实验与消融。

#### 16. glm-4.7-multiturn-CoT

- **数据量**：3,725 条多轮对话
- **题目来源**：ChatAlpaca / ShareGPT 风格多轮提问
- **回答模型**：glm-4.7
- **下载地址**：https://huggingface.co/datasets/Jackrong/glm-4.7-multiturn-CoT
- **内容例子**：每轮回复为 `<think>` + 最终答案：

```json
{"conversation": [
  {"from": "human", "value": "……"},
  {"from": "gpt", "value": "<think>……</think>\n……"},
  {"from": "human", "value": "追问"},
  {"from": "gpt", "value": "<think>……</think>\n……"}]}
```

- **适用范围**：多轮推理对话 SFT；蒸馏工作流带断点续传 / 审计 / 拒收跟踪，可复用其流水线。

#### 17. MultiReason-ChatAlpaca

- **数据量**：18,300 条对话（157,910 条消息，human/gpt 各半）
- **题目来源**：ChatAlpaca 风格多轮提问
- **回答模型**：gpt-oss-120B-high
- **下载地址**：https://huggingface.co/datasets/Jackrong/MultiReason-ChatAlpaca
- **内容例子**：100% 的 gpt 回复含 `<think>`（平均 549 词推理 + 413 词答案）：

```json
{"conversation": [
  {"from": "human", "value": "……"},
  {"from": "gpt", "value": "<think>...reasoning...</think> The product is 40."}]}
```

- **适用范围**：多轮指令跟随与对话建模；推理/答案长度可单独统计与裁剪；不需要 `<think>` 时可批量移除该片段。

#### 18. gpt-oss-120b-Reasoning-Instruction

- **数据量**：未公布（README 为空，仅含 `DeepSeek-if-gpt-result.jsonl`）
- **题目来源**：DeepSeek 指令（由文件名 `DeepSeek-if` 推断）
- **回答模型**：gpt-oss-120b
- **下载地址**：https://huggingface.co/datasets/Jackrong/gpt-oss-120b-Reasoning-Instruction
- **内容例子**：未提供 schema，需下载后查看。
- **适用范围**：推理指令数据补充，使用前需自行检查格式与质量。

#### 19. ShareGPT-gpt-oss-120B-reasoning

- **数据量**：未公布
- **题目来源**：ShareGPT 风格多轮提问
- **回答模型**：gpt-oss-120B（medium reasoning；卡面注明多轮 high 会超出上下文窗口，故用 medium）
- **下载地址**：https://huggingface.co/datasets/Jackrong/ShareGPT-gpt-oss-120B-reasoning
- **内容例子**：ShareGPT 风格多轮蒸馏（`from`/`value` 消息列表）。
- **适用范围**：多轮推理对话 SFT；与 #17 风格相近，可混合使用。

---

### 六、多语对话

#### 20. Qwen3-235B-A22B-Instruct-2507-Distilled-chat

- **数据量**：6,535 条（清洗后，单一 chat 类别）
- **题目来源**：跨语种对话问题（README 未明示具体来源数据集）
- **回答模型**：Qwen3-235B-A22B-Instruct-2507
- **下载地址**：https://huggingface.co/datasets/Jackrong/Qwen3-235B-A22B-Instruct-2507-Distilled-chat
- **内容例子**：非推理 chat 数据（不含 CoT），`messages` 格式可直接模板拼接：

```json
{"messages": [
  {"role": "user", "content": "……"},
  {"role": "assistant", "content": "……（平均 630 词 / 1,037 token）"}]}
```

- **适用范围**：SFT/DPO 聊天能力对齐；语言分布 英 90.1% / 中 5.5% / 俄 2.7% / 韩 0.9% / 日 0.6%，刻意混语以增强跨语言迁移；按 `language` 或 `word_count` 分桶抽样可控制输出长度分布。

#### 21. ShareGPT-Qwen3-235B-A22B-Instuct-2507

- **数据量**：未公布
- **题目来源**：ShareGPT 风格多轮提问
- **回答模型**：Qwen3-235B-A22B-Instruct-2507
- **下载地址**：https://huggingface.co/datasets/Jackrong/ShareGPT-Qwen3-235B-A22B-Instuct-2507
- **内容例子**：ShareGPT 风格多轮对话（`from`/`value` 消息列表）。
- **适用范围**：多轮对话 SFT；README 极简，使用前需下载确认字段与规模。

---

### 七、垂直领域

#### 22. financial-economics-reasoning

- **数据量**：122,378 条（题目规模；清洗后 10K–100K）
- **题目来源**：BAAI/IndustryInstruction_Finance-Economics（中英双语金融 / 经济 / 商业指令）
- **回答模型**：qwen-3-235b-a22b-thinking-2507（32K 上下文）
- **下载地址**：https://huggingface.co/datasets/Jackrong/financial-economics-reasoning
- **内容例子**：保留完整推理链：

```json
{"instruction": "金融/经济/商业指令",
 "output": "<think>逐步推理链</think>\n蒸馏输出"}
```

- **适用范围**：金融、经济、商业领域对话模型微调；Q&A、摘要与推理密集任务；不适用于医疗/法律等高风险领域（未经验证）。

#### 23. IELTS-writing-feedback-reasoning

- **数据量**：8,971 条
- **题目来源**：IELTS 写作 Task 2 题目 + 学生作文
- **回答模型**：GLM-4.7（低温蒸馏，temp 0.3）
- **下载地址**：https://huggingface.co/datasets/Jackrong/IELTS-writing-feedback-reasoning
- **内容例子**：显式保留评分推理过程：

```json
{"input": "<IELTS Task 2 Prompt + Student Essay>",
 "output": "<think>逐维度评分推理</think> + 最终 Band 分（0–9，0.5 步长）与反馈",
 "reasoning": "<think>逐步评分推理</think>",
 "evaluation": "<最终评分解释与专业反馈>"}
```

- **适用范围**：雅思写作自动评分（回归）、写作反馈生成；严格遵循官方评分标准多维分析，可做「推理式评分」教学数据；若只需要分数可只取 `evaluation` 字段。


#### 24. nohurry/Opus-4.6-Reasoning-3000x-filtered

- **数据量**：1K–10K 条（HF 标签）
- **题目来源**：ORCA / drive-minimax / 竞赛编程等多源数学与交互题
- **回答模型**：Opus 4.6（原数据集 `crownelius/Opus-4.6-Reasoning-3000x` 蒸馏，约 $409 USD、1.6M 补全 tokens、平均 758 tokens/行；清洗后 3,305 → 2,160 行）
- **与上游的关系**：本数据集是对原版的再过滤——删除了 979 条拒答；⚠️ 上游 README 注明"原数据集已有更好的过滤版本，建议用原版"
- **下载地址**：https://huggingface.co/datasets/nohurry/Opus-4.6-Reasoning-3000x-filtered
- **内容例子**：`problem / thinking / solution` 三字段分离；`thinking` 为**裸 CoT（无 `<think>` 包裹）**，notebook 里按 `<think>{thinking}</think>\n{solution}` 包装：

```json
{"id": "orca_3452", "problem": "252 fifth-grade students and 8 teachers……",
 "thinking": "Let me work through this problem step by step. First……",
 "solution": "# Solution: Calculating Bus Rental……",
 "difficulty": "medium", "category": "math", "timestamp": "2026-02-10T04:41:14Z", "hash": "fc4f25418b13ce49"}
```

- **适用范围**：数学 CoT SFT；与 Claude Opus 4.6 系（#25）混用即 27B/35B notebook 的主数据配方；注意 `thinking` 无标签包裹，训练管道需自行补 `<think>`。

#### 25. Roman1111111/claude-opus-4.6-10000x

- **数据量**：1K–10K 条（HF 标签；约 10,000）
- **题目来源**：高难度数学题（GSM8K / MATH）+ 逻辑谜题 + 多步指令
- **回答模型**：Claude Opus 4.6（官方 API，$87.20 USD、27.2M tokens）
- **下载地址**：https://huggingface.co/datasets/Roman1111111/claude-opus-4.6-10000x
- **内容例子**：`messages` 对话列表（assistant 可带 `reasoning` 字段，放隐藏思维链）+ `metadata` 元信息：

```json
{"messages": [
  {"role": "system", "content": "You are a helpful AI assistant."},
  {"role": "user", "content": "Ken created a care package……"},
  {"role": "assistant", "reasoning": "……", "content": "最终答案"}],
 "metadata": {"model": "claude-opus-4.6", "difficulty": "medium", "category": "simple logic and math"}}
```

- **适用范围**：数学 + 符号逻辑 + 通用问题求解 SFT；README 注明面向 Qwen3.5 系列（27B/25B-A3B/9B/4B/2B/0.8B）提升 BigBench-Hard / GSM8K；`reasoning` 字段需与 `content` 合并成 `<think>` 格式后训练。

#### 26. stepfun-ai/Step-3.5-Flash-SFT

- **数据量**：超大（notebook 配置采样上限 100,000）
- **题目来源**：混合指令（上游自带答案）
- **回答模型**：Step-3.5-Flash（上游官方 SFT 数据）
- **下载地址**：https://huggingface.co/datasets/stepfun-ai/Step-3.5-Flash-SFT
- **内容例子**：字段为 `conversations` 消息列表，但存在畸形内部结构（消息非 dict、字段缺失），需边扫边清洗：

```json
{"conversations": [{"role": "user", "content": "……"}, {"role": "assistant", "content": "……"}]}
```
- **适用范围**：通用指令 SFT 的大规模语料补充；自带官方格式，脏数据比例需靠过滤函数兜底。

#### 27. unsloth/OpenMathReasoning-mini

- **数据量**：未核实（notebook 用 `split="cot"`）
- **题目来源**：OpenMath 系数学题
- **回答模型**：上游自带（Unsloth 团队整理）
- **下载地址**：https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini
- **内容例子**：`problem / expected_answer / generated_solution` 三字段；notebook 用 pandas `to_numeric` 保留 `expected_answer` 为纯数字的行（供奖励校验）：

```json
{"problem": "……", "expected_answer": "42", "generated_solution": "……"}
```

- **适用范围**：GRPO notebook 的**冷启动 SFT 语料**——先把答案包装成 `<think>...</think><answer>\boxed{...}</answer>` 格式做监督训练，再进入 RL；纯数字答案方便后续写格式奖励函数。

#### 28. open-r1/DAPO-Math-17k-Processed

- **数据量**：~17K（config `"en"`，train split）
- **题目来源**：DAPO-Math-17k（数学推理）
- **回答模型**：上游自带（Open R1 团队整理）
- **下载地址**：https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed
- **内容例子**：含 `prompt`（已模板化的题目）与 `solution`（标准答案）：

```json
{"prompt": "Solve……", "solution": "……"}
```

- **适用范围**：GRPO notebook 的 **RL 阶段语料**；notebook 预处理为：模板包 format → 算 token 长度 `N` → 过滤 `N ≤ max_seq_length × 0.8`（给 `max_completion_length` 留余量）；`solution` 仅用于奖励函数里的 `\boxed{}` 答案校验。

---

## 使用提示


1. **格式兼容**：`<think>` 包裹风格兼容 Qwen3 / DeepSeek 系；OpenAI-Harmony 用通道格式；messages / ShareGPT / 字段分离式需按目标模板转换。
2. **质量风险**：全部为合成蒸馏数据——推理链可能含错误、冗长或与答案不一致；重要训练前建议二次清洗（长度截断、去重、评分过滤）。
3. **RL 关联**：含参考答案（#14）、评分信号（#13）、Band 评分（#23）的数据集可直接构造可验证奖励（RLVR）或偏好对；多数可作为 RL 训练前的冷启动 SFT 语料。
