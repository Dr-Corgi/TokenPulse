<div align="center">

# TokenPulse

**LLM 模型诊断工具 - 给大模型做体检**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](#english) | 简体中文

</div>

---

## 📖 项目简介

TokenPulse 是一套全面的 LLM 模型诊断工具，**类似于医院体检系统**。它能够对大语言模型进行多维度的健康检查，发现潜在问题并生成诊断报告。

### 为什么选择 TokenPulse？

| 特性 | 说明 |
|------|------|
| 🔍 **多维诊断** | 覆盖词表利用率、概率稳定性、置信度校准等多个维度 |
| 🏥 **体检式设计** | 模块化检查项，支持快速体检/全套体检/自定义体检 |
| 📊 **可视化报告** | 自动生成 Markdown 格式的健康报告 |
| 🔌 **解耦架构** | 推理过程与诊断逻辑分离，易于扩展 |
| ⚡ **高效复用** | 一次推理，多次诊断 |

---

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/TokenPulse.git
cd TokenPulse

# 安装依赖
pip install -e .
```

### 基础用法

```python
from tokenpulse import TransformerLensProvider, ClinicRunner

# 1. 创建模型提供者
provider = TransformerLensProvider("gpt2-small")

# 2. 创建体检运行器
runner = ClinicRunner(provider, preset="quick")

# 3. 执行体检
report = runner.run(["What is AI?", "Explain quantum computing."])

# 4. 生成报告
print(runner.generate_report_markdown())
```

### 命令行演示

```bash
# 快速体检
python examples/mvp_demo.py --model gpt2-small --preset quick

# 全套体检
python examples/mvp_demo.py --model gpt2-small --preset full --output report.md
```

---

## 🏥 体检套餐

| 套餐 | 包含检查项 | 适用场景 |
|------|-----------|---------|
| `quick` | 词表利用率、Token 稳定性 | 快速健康检查 |
| `full` | 词表利用率、Token 稳定性、置信度校准 | 完整体检 |
| `calibration` | 置信度校准 | 校准专项检查 |

### 健康状态说明

| 状态 | 含义 | 建议 |
|:----:|------|------|
| 🟢 | 健康 | 无需处理 |
| 🟡 | 需关注 | 建议进一步分析 |
| 🔴 | 需处理 | 必须进行优化 |
| ⚪ | 未检测 | 模型能力不支持 |

---

## 📋 检查项详情

### 🔬 常规检查（Blood）

#### 1. 词表利用率检查 `VocabularyUtilizationCheck`

检测模型是否存在词表坍缩问题。

```python
from tokenpulse.checks.blood import VocabularyUtilizationCheck

check = VocabularyUtilizationCheck()
result = check.run(provider, ["Hello world!"])

print(f"词表利用率: {result.raw_metrics['utilization_ratio']:.1%}")
print(f"状态: {result.health_status.value}")
```

**核心指标**：
- 词表利用率（Utilization Ratio）
- 类型 Token 比（Type-Token Ratio）
- Gini 系数

#### 2. Token 稳定性检查 `TokenStabilityCheck`

检测模型输出的概率一致性。

```python
from tokenpulse.checks.blood import TokenStabilityCheck

check = TokenStabilityCheck(n_samples=5, temperature=0.7)
result = check.run(provider, ["What is AI?"])

print(f"变异系数: {result.raw_metrics['coefficient_of_variation']['mean']:.3f}")
```

**核心指标**：
- 变异系数（Coefficient of Variation）
- Top-Token 一致性
- Jaccard 相似度

### 💉 免疫检查（Immune）

#### 3. 置信度校准检查 `CalibrationCheck`

评估模型置信度与实际准确率的匹配程度。

```python
from tokenpulse.checks.immune import CalibrationCheck

check = CalibrationCheck()
result = check.run(provider)

print(f"ECE: {result.raw_metrics['ece']:.3f}")
print(f"准确率: {result.raw_metrics['accuracy']:.1%}")
```

**核心指标**：
- 期望校准误差（ECE）
- Brier Score
- 校准曲线数据

---

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│  L5: 报告与编排层 (Report & Orchestration)                      │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ ClinicRunner    │ │ HealthScorer    │ │ ReportGenerator │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  L4: 诊断模块层 (Checks)                                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Blood    │ │ Neural   │ │ Bone     │ │ Immune   │           │
│  │ 概率分布 │ │ 敏感鲁棒 │ │ 表征权重 │ │ 校准置信 │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
├─────────────────────────────────────────────────────────────────┤
│  L3: 核心计算引擎 (Core Engine)                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │ InfoTheory  │ │ LinAlg      │ │ Calibration │               │
│  │ 信息论工具  │ │ 线性代数    │ │ 校准工具    │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
├─────────────────────────────────────────────────────────────────┤
│  L2: 数据与缓存层 (Data & Cache)                                │
├─────────────────────────────────────────────────────────────────┤
│  L1: 模型适配层 (Model Adapters)                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │ HFProvider  │ │ VLLMProvider│ │ APIProvider │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
TokenPulse/
├── src/tokenpulse/
│   ├── providers/          # L1: 模型适配层
│   │   ├── base.py              # 抽象基类
│   │   ├── data_structures.py   # 数据结构
│   │   └── transformer_lens_provider.py
│   ├── core/               # L3: 核心计算引擎
│   │   ├── info_theory.py       # 信息论计算
│   │   └── calibration.py       # 校准计算
│   ├── checks/             # L4: 诊断模块
│   │   ├── base_check.py        # 检查项基类
│   │   ├── check_result.py      # 结果数据结构
│   │   ├── blood/               # 概率分布检查
│   │   │   ├── vocab_utilization.py
│   │   │   └── token_stability.py
│   │   └── immune/              # 校准检查
│   │       └── calibration_check.py
│   └── clinic/             # L5: 编排与报告
│       ├── runner.py            # 体检编排器
│       ├── scorer.py            # 健康评分器
│       ├── report.py            # 报告生成器
│       └── data_structures.py   # 报告数据结构
├── examples/               # 示例脚本
│   └── mvp_demo.py
├── tests/                  # 单元测试
├── docs/                   # 文档
│   ├── requirements.md
│   └── technical_design.md
└── README.md
```

---

## 🗺️ 开发路线

### 已完成 ✅

<details>
<summary><b>P0 阶段：奠基之作</b></summary>

- [x] TransformerLensProvider 适配器
- [x] 核心计算函数 (entropy, kl_divergence, ece)
- [x] 单元测试覆盖

</details>

<details>
<summary><b>P1 阶段：MVP 闭环</b></summary>

- [x] 词表利用率检查 (VocabularyUtilizationCheck)
- [x] Token 稳定性检查 (TokenStabilityCheck)
- [x] 置信度校准检查 (CalibrationCheck)
- [x] MVP 演示脚本
- [x] ClinicRunner 编排器
- [x] ReportGenerator 报告生成器

</details>

### 进行中 🚧

<details>
<summary><b>P2 阶段：专科检查</b></summary>

- [ ] 微扰敏感度检查 (PerturbationSensitivityCheck)
- [ ] 注意力沉检查 (AttentionSinkCheck)
- [ ] 表征坍缩检查 (RepresentationCollapseCheck)
- [ ] L2 缓存完善

</details>

### 计划中 📋

<details>
<summary><b>P3 阶段：生态扩展</b></summary>

- [ ] API 黑盒校准 (OpenAI/Anthropic)
- [ ] 知识熵检查
- [ ] 训练动态对比
- [ ] 基线数据库

</details>

---

## 🧪 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_checks.py -v
pytest tests/test_clinic.py -v
```

当前测试覆盖：**156 个测试用例**，100% 通过率

---

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

本项目参考了以下优秀开源项目：

- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) - 模型内部激活提取
- [llms-calibration](https://github.com/explodinggradients/llms-calibration) - 校准计算参考

---

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐️ Star！**

Made with ❤️ by TokenPulse Team

</div>

---

<a name="english"></a>

## English

**TokenPulse** is a comprehensive diagnostic tool for Large Language Models, designed like a hospital health check system. It performs multi-dimensional health checks on LLMs, identifies potential issues, and generates diagnostic reports.

### Quick Start

```python
from tokenpulse import TransformerLensProvider, ClinicRunner

provider = TransformerLensProvider("gpt2-small")
runner = ClinicRunner(provider, preset="quick")
report = runner.run(["What is AI?"])
print(runner.generate_report_markdown())
```

### Available Checks

| Check | Description |
|-------|-------------|
| `VocabularyUtilizationCheck` | Detects vocabulary collapse issues |
| `TokenStabilityCheck` | Analyzes token probability consistency |
| `CalibrationCheck` | Evaluates confidence calibration (ECE) |

### License

MIT License - see [LICENSE](LICENSE) for details.
