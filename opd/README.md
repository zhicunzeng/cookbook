# ZZC's OPD Tutorial



## 项目结构

```text
opd/
├── README.md                   ← 本导读
├── src/
│   ├── OPD_Colab.ipynb         ← 主教程：27 个 cell（原理讲解 + 可运行代码）
│   └── opd_core_verified.py    ← OPD 核心算法 CPU 验证脚本（无 GPU 也能跑，先理解机制再上 Colab）
└── reference/                  论文与参考材料（On-Policy Distillation、MiniLLM 等）
```

Notebook 要点：

- **算法核心约 30 行**：学生 on-policy 采样（记录逐 token logprob）→ 教师一次前向打分 → 逐 token reverse KL 取负作 advantage → PPO clip 更新。即 TML 所说的"把 RL 的 KL 正则参考模型换成教师"的一行改动
- **附录**：TRL GKDTrainer 替代方案、T4/L4/A100 配置表、OOM 等故障排查、进阶方向（top-k logit 蒸馏、OPD+可验证奖励混合）

## OPD 一句话

> 学生模型**自己回答**（on-policy），教师模型对学生的回答**逐 token 打分**
>（reverse KL 作为稠密奖励），学生在"自己会走到的状态"上向教师对齐。

结合了 SFT 蒸馏的稠密信号与 RL 的 on-policy 采样，Qwen3 官方报告用 RL **十分之一**的算力
把 AIME'24 从 67.6% 推到 74.4%。与 Qwopus 经典的 off-policy SFT 蒸馏相比，OPD **只需要 prompt
不需要教师预写答案**，且天然解决"学生犯错后不会恢复"的复合错误问题。

## 本教程的实验配置

- 🧑‍🎓 学生：`Qwen/Qwen3-0.6B` + LoRA（r=16）
- 🧑‍🏫 教师：`Qwen/Qwen3-4B-Instruct-2507`，4bit 量化冻结，只做前向打分——同家族 tokenizer 一致，满足逐 token 对齐前提
- 📚 数据：`HuggingFaceH4/no_robots`（通用指令）+ `openai/gsm8k`（数学）——只取 prompt，答案由学生当场生成、教师当场批改
- 📏 评估：GSM8K test 50 题 greedy 准确率（训练前后同题对比）+ 3 条定性指令题，附 KL/教师认可度曲线
- ⏱️ Colab 免费 T4：demo 配置约 40-60 分钟；附录给出 L4/A100 完整配置表

## 已完成的正确性验证（本沙箱，CPU）

notebook 中的核心算法（采样 → 教师打分 → reverse KL → PPO 式更新）已用两个微型 Qwen3
在 CPU 上端到端跑通（`opd_core_verified.py`，约 1 分钟）：

```
step | reverse KL | 教师偏好行为占比(基线 25%)
   0 |     2.3197 |        26.1%
  10 |     0.0276 |        97.9%   ← 10 步内基本收敛
  59 |     0.0083 |        97.9%
reverse KL 相对下降 96.0%，解析 KL 与采样估计一致（0.026 vs 0.033）✅
```

这验证了 notebook 代码的四个易错点全部正确：位置对齐切片、EOS 监督掩码、
重要性比率方向、PPO clip + advantage 标准化的稳定性。

## 使用步骤

1. 打开 [Google Colab](https://colab.research.google.com/)，运行时选 **T4 GPU**；
2. 上传 `OPD_Colab.ipynb`（文件 → 上传笔记本）；
3. 从上到下依次运行 cell；第 8 节是算法核心（约 30 行），建议精读；
4. 训练完成后可在 notebook 内保存 LoRA、合并 16bit、转 GGUF 或推送到 HuggingFace Hub。

## 参考

- Thinking Machines Lab《On-Policy Distillation》(2025) —— 算法来源
- Qwen3 Technical Report (2025) —— OPD vs RL 算力对比
- MiniLLM (Gu et al., 2023) —— reverse KL 蒸馏；DAgger (Ross et al., 2010) —— 思想源头
- [jackrong / Jackrong-llm-finetuning-guide](https://github.com/R6410418/Jackrong-llm-finetuning-guide)（Qwopus）—— 工程风格来源
