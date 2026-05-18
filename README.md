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
├─────────────────────────────────────────────────────────┤
│  L4: 诊断模块层 [Checks] - 血液/神经/骨骼/免疫          │
├─────────────────────────────────────────────────────────┤
│  L3: 核心计算引擎 (Core Engine) - 熵/KL/ECE/SVD         │
├─────────────────────────────────────────────────────────┤
│  L2: 数据与缓存层 (Data & Cache)                        │
├─────────────────────────────────────────────────────────┤
│  L1: 模型适配层 (Model Adapters) - TransformerLens/vLLM │
└─────────────────────────────────────────────────────────┘
```

## 开发路线

### P0 阶段：奠基之作 ✅
- [x] TransformerLensProvider 适配器
- [x] 核心计算函数 (entropy, kl_divergence, ece)
- [x] 单元测试覆盖

### P1 阶段：MVP 闭环
- [ ] 词表利用率检查
- [ ] Token 稳定性检查
- [ ] 置信度校准检查
- [ ] MVP 演示脚本

### P2 阶段：专科检查
- [ ] 微扰敏感度检查
- [ ] 注意力沉检查
- [ ] 表征坍缩检查

### P3 阶段：生态扩展
- [ ] API 黑盒校准
- [ ] 知识熵检查
- [ ] 训练动态对比

## 快速开始

```bash
# 安装
pip install tokenpulse

# 命令行使用
tokenpulse check --model gpt2-small --preset quick

# Python API
from tokenpulse import ClinicRunner, TransformerLensProvider

provider = TransformerLensProvider("gpt2-small")
runner = ClinicRunner(provider, preset="quick")
report = runner.run(["What is AI?"])
print(report)
```

## 文档

- [需求文档](docs/requirements.md)
- [技术设计文档](docs/technical_design.md)

## 许可证

MIT License
