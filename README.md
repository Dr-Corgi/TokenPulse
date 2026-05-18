# TokenPulse

> LLM 模型诊断工具 - 给大模型做体检

## 项目简介

TokenPulse 是一套全面的 LLM 模型诊断工具，类似于医院体检系统。它能够对大语言模型进行多维度的健康检查，发现潜在问题并生成诊断报告。

### 核心理念

**构建"体检中心"而非"医学研究所"**

- 先基建后业务，先浅层后深层
- 先低成本高收益，后高成本长周期
- 推理过程与诊断逻辑解耦，扩展性强

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│  L5: 报告与编排层 (Report & Orchestration)              │
│  ClinicRunner, ReportGenerator, HealthScorer            │
├─────────────────────────────────────────────────────────┤
│  L4: 诊断模块层 [Checks] - 血液/神经/骨骼/免疫          │
│  VocabularyUtilizationCheck, TokenStabilityCheck, etc.  │
├─────────────────────────────────────────────────────────┤
│  L3: 核心计算引擎 (Core Engine) - 熵/KL/ECE/SVD         │
│  InfoTheoryCalculator, CalibrationCalculator            │
├─────────────────────────────────────────────────────────┤
│  L2: 数据与缓存层 (Data & Cache)                        │
├─────────────────────────────────────────────────────────┤
│  L1: 模型适配层 (Model Adapters) - TransformerLens/vLLM │
│  TransformerLensProvider                                │
└─────────────────────────────────────────────────────────┘
```

## 开发路线

### P0 阶段：奠基之作 ✅
- [x] TransformerLensProvider 适配器
- [x] 核心计算函数 (entropy, kl_divergence, ece)
- [x] 单元测试覆盖

### P1 阶段：MVP 闭环 ✅
- [x] 词表利用率检查 (VocabularyUtilizationCheck)
- [x] Token 稳定性检查 (TokenStabilityCheck)
- [x] 置信度校准检查 (CalibrationCheck)
- [x] MVP 演示脚本
- [x] ClinicRunner 编排器
- [x] ReportGenerator 报告生成器

### P2 阶段：专科检查
- [ ] 微扰敏感度检查
- [ ] 注意力沉检查
- [ ] 表征坍缩检查

### P3 阶段：生态扩展
- [ ] API 黑盒校准
- [ ] 知识熵检查
- [ ] 训练动态对比

## 快速开始

### 安装

```bash
# 从源码安装
git clone https://github.com/your-repo/TokenPulse.git
cd TokenPulse
pip install -e .
```

### Python API 使用

```python
from tokenpulse import TransformerLensProvider, ClinicRunner

# 创建模型提供者
provider = TransformerLensProvider("gpt2-small")

# 创建体检运行器
runner = ClinicRunner(provider, preset="quick")

# 执行体检
report = runner.run(["What is AI?", "Explain quantum computing."])

# 生成报告
print(runner.generate_report_markdown())
```

### 命令行使用

```bash
# 快速体检
python examples/mvp_demo.py --model gpt2-small --preset quick

# 全套体检
python examples/mvp_demo.py --model gpt2-small --preset full

# 保存报告
python examples/mvp_demo.py --model gpt2-small --output report.md
```

### 单独使用检查项

```python
from tokenpulse import TransformerLensProvider
from tokenpulse.checks.blood import VocabularyUtilizationCheck

# 加载模型
provider = TransformerLensProvider("gpt2-small")

# 运行词表利用率检查
check = VocabularyUtilizationCheck()
result = check.run(provider, ["Hello world!", "How are you?"])

print(f"Utilization: {result.raw_metrics['utilization_ratio']:.1%}")
print(f"Status: {result.health_status.value}")
```

## 体检套餐

| 套餐 | 包含检查项 | 适用场景 |
|------|-----------|---------|
| `quick` | 词表利用率、Token稳定性 | 快速健康检查 |
| `full` | 词表利用率、Token稳定性、置信度校准 | 完整体检 |
| `calibration` | 置信度校准 | 校准专项检查 |

## 健康状态说明

| 状态 | 含义 | 建议 |
|------|------|------|
| 🟢 GREEN | 健康 | 无需处理 |
| 🟡 YELLOW | 需关注 | 建议进一步分析 |
| 🔴 RED | 需处理 | 必须进行优化 |
| ⚪ SKIP | 未检测 | 模型能力不支持 |

## 文档

- [需求文档](docs/requirements.md)
- [技术设计文档](docs/technical_design.md)

## 依赖

- Python >= 3.9
- PyTorch >= 2.0
- TransformerLens >= 2.0
- NumPy >= 1.24

## 许可证

MIT License
