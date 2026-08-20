# ZZC's Cheap SFT Tutorial

## 项目结构

```text
cheap_sft/
├── README.md                     ← 本导读
├── cook_qwopus_sft.ipynb         ← 上篇：27B SFT 省钱五件套（19 个 code cell，原理讲解 + 可运行代码）
├── cook_r1_grpo.ipynb            ← 中篇：R1-Zero 冷启动 SFT + GRPO 多奖励函数
├── cook_merge_heal.ipynb         ← 下篇：18B 缝合 + QLoRA 愈合（官方教程对齐实现）
└── reference/                    官方材料
    ├── MERGE_PROCESS.md          ← 官方合并+愈合完整技术流程（HF Jackrong/Qwopus-GLM-18B-Merged-GGUF 公布）
    ├── merge_layerweighted.py    ← 官方后续 Fusion 项目公开的完整合并脚本（下篇工程骨架来源）
    ├── Qwopus-GLM-18B-Technical-Report.pdf
    └── Qwopus3-5-27b-Colab_complete_guide_to_llm_finetuning.pdf
```

三个 notebook 要点：

- **上篇（省钱 SFT）**：显存账单拆解 → 五件套（4bit QLoRA / LoRA r=64 / Unsloth 梯度检查点 / adamw_8bit / 梯度累积）逐件展开，复用 27B notebook 原代码；多源数据归一化管道（nohurry / Jackrong 700x / Roman 三种 schema → `<think>` 契约）已 CPU 验证，含 Roman 数据集 `Feature 'Json' not found` 的 parquet 绕过方案
- **中篇（R1-Zero GRPO）**：冷启动 SFT 教格式契约（`<think>…</think><answer>\boxed{…}</answer>`）→ GRPO 多奖励函数（结构塑造 + boxed 答案校验 + 重复惩罚 + 采样监控），复用 GRPO notebook 原代码与奖励权重
- **下篇（缝合+愈合）**：jackrong 未公开脚本——以官方 MERGE_PROCESS.md 为主线，工程骨架对齐官方 merge_layerweighted.py（safe_open + ShardWriter 5GB 分片），本沙箱合成 checkpoint 六项断言通过；愈合部分为官方配置表（r=64/α=32、1000 步、346M 参数）的完整 Unsloth 实现

## cheap_sft 一句话

> **不是少训练，而是把训练账单砍到免费 Colab 单卡付得起**：4bit 量化 + LoRA 让 27B 可训，
> 8bit 优化器 + 梯度累积让 6×6 的等效 batch 不涨显存；先 SFT 教格式再 GRPO 教推理，
> 两阶段组合是 R1 系性价比之王；把两个 9B 缝成 18B 再加 1000 步 QLoRA 愈合，
> 用 35B MoE 一半的显存拿到 40/44 的分数。

## 各篇实验配置

- 🧠 **上篇**：`unsloth/Qwen3.5-27B` 4bit + LoRA r=64，三源数据 ~14K 条，batch 6×6、lr 2e-4、2 epochs，Colab 单卡
- 🧪 **中篇**：`unsloth/Llama-3.2-3B-Instruct` LoRA 16bit r=64；冷启动 `unsloth/OpenMathReasoning-mini`（cot），RL `open-r1/DAPO-Math-17k-Processed`（en）；GRPO lr 5e-6、num_generations=6、1500 步，Kaggle 单卡
- 🧬 **下篇**：缝合不用 GPU（40GB RAM、15 分钟、33GB 输出）；愈合 RTX 5090 ~13GB 显存、~14 小时、1383 条 70/15/15 数据

## 已完成的正确性验证（本沙箱，CPU）

三个 notebook 中可离线验证的核心逻辑均已跑出真实输出（notebook cell 内可见）：

1. **缝合逻辑**（cook_merge_heal.ipynb 第 4 节）：Qwen3.5 多模块 + hybrid attention 合成 checkpoint，
   六项断言全部通过——前缀重编号（`model.language_model.layers`）、多模块张量取 A、
   linear_attn 随层平移、layer_types 翻倍、5GB 分片与 index.json、B 非层张量丢弃：
   ```
   ✅ 缝合完成：2 层 (A) + 2 层 (B) = 4 层
      A 保留 9 张量，B 重编号 5 张量，输出 1 分片 + index.json
   ✅ 六项断言全部通过
   ```
2. **多源归一化管道**（cook_qwopus_sft.ipynb 第 6 节）：真实 Jackrong/DeepSeek 数据 208 行
   + 三种 schema 合成样本 + 1 条格式反例，端到端跑通：138 条通过、反例被正确拦截、
   长度过滤与 `<think>` 格式校验行为符合预期。
3. **格式契约**（cook_r1_grpo.ipynb 第 5 节）：Llama-3.x Jinja 模板渲染、
   `_extract_think_from_generated_solution` 抽取、兜底话逻辑均在本机跑出正确产物。

GPU cell（Unsloth 训练 / vLLM 评估 / GGUF 转换）以 `gpu` 标签标注，保持干净待 Colab/Kaggle 运行。

## 使用步骤

1. 打开 [Google Colab](https://colab.research.google.com/)（中篇原 notebook 为 Kaggle，两者皆可），运行时选 **T4/A100 GPU**；
2. 上传对应 notebook（文件 → 上传笔记本），从上到下运行；带 `gpu` 标签的 cell 在 Colab 正常运行，本机跳过；
3. 训练完成后每个 notebook 都有 Qwopus 风格收尾：保存 LoRA → merged 16bit → GGUF 三档量化（q4_k_m / q8_0 / bf16）；
4. 下篇的真实缝合先下载两个源模型（Jackrong/Qwopus3.5-9B-v3.5 与 Qwen3.5-9B-GLM5.1-Distill-v1），再运行第 4 节。

## 参考

- [jackrong / Jackrong-llm-finetuning-guide](https://github.com/R6410418/Jackrong-llm-finetuning-guide) —— 三个教程的代码来源（27B Colab / GRPO / 技术报告）
- `reference/MERGE_PROCESS.md`（官方合并+愈合教程，Kyle Hessling 著）—— 下篇主线
- `reference/merge_layerweighted.py` —— 官方完整合并脚本（下篇工程骨架）
- `food/post-training.md` —— 24 个 Jackrong 数据集 + 5 个 train_code 第三方数据集清单
- 数据归一化与缝合逻辑的验证结论即本沙箱运行结果（notebook 内 cell 输出）
