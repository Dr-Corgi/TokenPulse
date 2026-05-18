# TokenPulse - 技术设计文档

> **项目名称**: TokenPulse - LLM模型诊断工具
>
> **核心理念**: 构建"体检中心"而非"医学研究所"，先基建后业务，先浅层后深层，先低成本高收益后高成本长周期。

## 1. 架构概览

### 1.1 分层架构图

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
│  ┌─────────────┐                                               │
│  │ Perturbation│                                               │
│  │ 扰动生成器  │                                               │
│  └─────────────┘                                               │
├─────────────────────────────────────────────────────────────────┤
│  L2: 数据与缓存层 (Data & Cache)                                │
│  ┌─────────────────────┐ ┌─────────────────────┐               │
│  │ FeatureStore        │ │ TensorProcessor     │               │
│  │ 特征缓存管理        │ │ 张量对齐与裁剪      │               │
│  └─────────────────────┘ └─────────────────────┘               │
├─────────────────────────────────────────────────────────────────┤
│  L1: 模型适配层 (Model Adapters)                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │ HFProvider  │ │ VLLMProvider│ │ APIProvider │               │
│  │ HuggingFace │ │ vLLM        │ │ OpenAI/etc  │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 数据流向

```
用户请求 → ClinicRunner(编排) → Check.run()
                                    ↓
                            ModelProvider.generate()
                                    ↓
                            FeatureStore(查缓存/存结果)
                                    ↓
                            CoreEngine(计算)
                                    ↓
                            CheckResult(结果)
                                    ↓
                            HealthScorer(评分)
                                    ↓
                            ReportGenerator(报告)
```

---

## 2. 集成路线图

### 2.1 集成优先级原则

| 原则 | 说明 |
|------|------|
| 先基建后业务 | 先搭好 L1/L3 骨架，再填充具体检查项 |
| 先浅层后深层 | Logits 提取成本低，优先于中间层激活 |
| 先低成本高收益 | 核心计算函数不到100行，支撑80%检查项 |
| 解耦设计 | 每个集成项独立封装，不互相依赖 |

### 2.2 P0 阶段：奠基之作（基建与核心引擎）

> **目标**: 搭好骨架，不产出具体体检项，但为后续所有检查提供底座。

#### 2.2.1 模型适配层集成：TransformerLens

| 项目 | 内容 |
|------|------|
| **集成来源** | [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) |
| **优先级** | P0 - 最高 |
| **定位** | 整个系统的"采血针"，无痛苦提取注意力、隐层、Logits |
| **集成方式** | 编写 `TransformerLensProvider` 继承 L1 `ModelProvider` 基类 |
| **输出转换** | 将 TransformerLens 输出转为统一 `ModelOutput` 格式 |

**核心能力提取**:
```python
# TransformerLens 的核心优势
model = HookedTransformer.from_pretrained("gpt2-small")

# 一行获取所有缓存
logits, cache = model.run_with_cache(prompt)

# cache 包含:
# - blocks.{i}.hook_resid_pre     # 残差流
# - blocks.{i}.attn.hook_pattern  # 注意力模式
# - blocks.{i}.hook_mlp_out       # MLP输出
```

**适配器设计要点**:
- 利用 `run_with_cache()` 一次性提取所有中间结果
- 支持按需提取，避免不必要的计算开销
- 处理 TransformerLens 与 HuggingFace 的 token mapping 差异

#### 2.2.2 核心计算引擎搭建 (L3层基础)

| 函数 | 用途 | 复用场景 |
|------|------|----------|
| `calc_entropy` | 信息熵计算 | 概率分布检查、词表利用率 |
| `calc_kl_divergence` | KL散度计算 | 微扰敏感度、分布对比 |
| `calc_ece` | 期望校准误差 | 置信度校准检查 |

**实现策略**: PyTorch 手写，不依赖外部大型库，总计不到 100 行代码，支撑 80% 体检项。

---

### 2.3 P1 阶段：高价值、低阻力的"常规体检"

> **目标**: 只需 Logits 输出，计算速度快，快速回答核心问题。
>
> **MVP 闭环**: P0 + P1 = 可演示的最小产品

#### 2.3.1 词表利用率与无条件概率检查

| 项目 | 内容 |
|------|------|
| **集成来源** | [tokenizer-analysis-suite](https://github.com/huggingface/tokenizers) (TokEval) |
| **提取内容** | 词表覆盖率计算脚本，适配到 `VocabularyUtilizationCheck` |
| **所需能力** | 仅 Logits |
| **收益** | 立刻检测"词表冻死"现象 |
| **可视化** | 空输入时的词表概率分布直方图 |

**检查项设计**:
```python
class VocabularyUtilizationCheck(BaseCheck):
    """
    体检流程:
    1. 给模型空输入（或极简prompt）
    2. 提取 logits 分布
    3. 统计高频 token 占比
    4. 画出概率分布直方图
    """
    check_name = "vocabulary_utilization"
    check_category = "blood"
    required_capabilities = ["logits"]
```

#### 2.3.2 Token 概率稳定性（非确定性）检查

| 项目 | 内容 |
|------|------|
| **集成来源** | 论文 *Beyond Reproducibility* 的分析逻辑 |
| **提取内容** | 同一 prompt 多次采样，统计 token 概率标准差 |
| **所需能力** | 仅 Logits |
| **收益** | 发现模型"精神分裂"（概率乱跳）问题 |

**检查项设计**:
```python
class TokenStabilityCheck(BaseCheck):
    """
    体检流程:
    1. 同一 prompt 重复 N 次
    2. 收集每次的 token 概率分布
    3. 计算各位置的概率标准差
    4. 标记高波动区域
    """
    check_name = "token_stability"
    check_category = "blood"
    required_capabilities = ["logits"]
```

#### 2.3.3 基础置信度校准检查

| 项目 | 内容 |
|------|------|
| **集成来源** | [llms-calibration](https://github.com/explodinggradients/llms-calibration) 或 [large-model-calibration-and-uncertainty](https://github.com/ICLR2023/large-model-calibration) |
| **提取内容** | 核心的 ECE (期望校准误差) 计算流水线 |
| **所需能力** | 仅 Logits + 简单多选题数据集 |
| **收益** | 直观暴露模型"盲目自信"程度 |
| **可视化** | Reliability Diagram（校准曲线） |

**检查项设计**:
```python
class ConfidenceCalibrationCheck(BaseCheck):
    """
    体检流程:
    1. 跑一组常识题/多选题
    2. 收集置信度与实际准确率
    3. 计算 ECE
    4. 绘制 Reliability Diagram
    """
    check_name = "confidence_calibration"
    check_category = "immune"
    required_capabilities = ["logits"]
```

**MVP 演示闭环**:
```
┌─────────────────────────────────────────────────────────┐
│  TokenPulse MVP 演示                                    │
├─────────────────────────────────────────────────────────┤
│  1. 常规抽血: 空输入 → 词表概率分布直方图               │
│     → 立刻看出是否有病态高概率 token                    │
│                                                         │
│  2. 心电图: 常识题 → ECE → 校准曲线                     │
│     → 立刻看出模型是否过度自信                          │
└─────────────────────────────────────────────────────────┘
```

---

### 2.4 P2 阶段：深度机理的"专科检查"

> **前提**: P0/P1 已跑通，L2 缓存支持完善
>
> **特点**: 需要中间层激活，计算成本高

#### 2.4.1 微扰敏感度（鲁棒性）检查

| 项目 | 内容 |
|------|------|
| **集成来源** | [SPUQ](https://github.com/language-models/SPUQ) (Semantic-Preserving Uncertainty Quantification) |
| **提取内容** | Prompt 扰动生成逻辑（同义词替换、加废话等） |
| **封装位置** | L3 层 `PerturbationGenerator` |
| **组合使用** | 结合 P0 的 `calc_kl_divergence` |
| **收益** | 检测模型是否"神经衰弱"（换个说法就崩） |

**集成代码提取要点**:
```python
# 从 SPUQ 提取的扰动生成逻辑
class PerturbationGenerator:
    def synonym_replace(self, text, ratio=0.1):
        """同义词替换"""
        pass

    def add_noise(self, text, noise_type="whitespace"):
        """添加无关内容"""
        pass

    def paraphrase(self, text):
        """改写"""
        pass
```

#### 2.4.2 注意力沉与注意力退化检查

| 项目 | 内容 |
|------|------|
| **集成来源** | [Awesome-Attention-Sink](https://github.com/Awesome-Attention-Sink) 列表 |
| **推荐代码** | *What are you sinking?* 或 *Quantizable transformers* 的统计代码 |
| **提取内容** | 首 token 注意力占比、注意力熵计算 |
| **所需能力** | Attention Weights |
| **收益** | 验证长文本失效是否因注意力"死掉" |

**检查指标**:
- 首 token (BOS) 注意力占比
- 注意力熵（分布均匀程度）
- 各层注意力模式对比

#### 2.4.3 表征坍缩检查

| 项目 | 内容 |
|------|------|
| **集成来源** | [SimVQ](https://github.com/simvq/simvq) / [SimSMoE](https://github.com/simsmoe/simsmoe) |
| **提取内容** | SVD 分解与有效秩计算代码 |
| **所需能力** | Hidden States |
| **收益** | 发现模型"脑子空空"（隐层全挤在一起） |

**核心指标**:
- 有效秩 (Effective Rank)
- 稳定秩 (Stable Rank)
- 各层奇异值分布

---

### 2.5 P3 阶段：长线生态（锦上添花）

> **特点**: 学术探索性质强，或需特定数据格式

#### 2.5.1 黑盒校准

| 项目 | 内容 |
|------|------|
| **集成来源** | [APRICOT](https://github.com/apricot-llm/apricot) |
| **适用场景** | 经常需要测闭源 API 时再集成 |
| **特点** | 无需模型内部信息 |

#### 2.5.2 知识熵

| 项目 | 内容 |
|------|------|
| **集成来源** | [Knowledge-Entropy](https://github.com/knowledge-entropy/ke) |
| **绑定项** | OLMo 架构 + Dolma 数据集 |
| **适用场景** | 研究微调遗忘问题时借鉴 |
| **迁移成本** | 较高，通用性受限 |

#### 2.5.3 训练动态检查

| 项目 | 内容 |
|------|------|
| **集成来源** | [Pythia](https://github.com/EleutherAI/pythia) |
| **借鉴内容** | 多 Checkpoint 对比思路（非代码搬运） |
| **适用场景** | 有训练过程访问权限时 |

---

### 2.6 集成来源汇总表

| 阶段 | 集成项 | 来源仓库 | 提取内容 | 所需能力 |
|------|--------|----------|----------|----------|
| P0 | TransformerLens | `TransformerLensOrg/TransformerLens` | HookedTransformer 封装 | - |
| P0 | 核心计算 | 自实现 | entropy, KL, ECE | - |
| P1 | 词表利用率 | `huggingface/tokenizers` | 覆盖率计算 | Logits |
| P1 | Token稳定性 | *Beyond Reproducibility* | 概率标准差 | Logits |
| P1 | 置信度校准 | `explodinggradients/llms-calibration` | ECE流水线 | Logits |
| P2 | 微扰敏感度 | `SPUQ` | 扰动生成器 | Logits |
| P2 | 注意力沉 | `Awesome-Attention-Sink` | 注意力统计 | Attention |
| P2 | 表征坍缩 | `SimVQ`/`SimSMoE` | SVD/有效秩 | Hidden |
| P3 | 黑盒校准 | `APRICOT` | API校准方法 | Logprobs |
| P3 | 知识熵 | `Knowledge-Entropy` | 知识熵计算 | Hidden |
| P3 | 训练动态 | `Pythia` | Checkpoint对比思路 | All |

---

## 3. 核心数据结构

### 2.1 L1层：模型输出结构

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch

@dataclass
class ModelOutput:
    """统一的模型输出结构"""
    # 基本信息
    model_id: str
    prompt: str
    generated_text: str

    # 概率分布
    logits: Optional[torch.Tensor] = None          # [seq_len, vocab_size]
    logprobs: Optional[torch.Tensor] = None        # [seq_len, vocab_size]
    top_logprobs: Optional[List[Dict[int, float]]] = None

    # 中间激活
    hidden_states: Optional[Dict[int, torch.Tensor]] = None  # layer_id -> [seq_len, hidden_dim]
    attention_weights: Optional[Dict[int, torch.Tensor]] = None  # layer_id -> [heads, seq_len, seq_len]

    # 元数据
    tokens: Optional[List[int]] = None
    token_strings: Optional[List[str]] = None

@dataclass
class ModelCapabilities:
    """模型能力声明"""
    supports_logits: bool = True
    supports_hidden_states: bool = True
    supports_attention_weights: bool = True
    max_batch_size: int = 1
    supports_streaming: bool = False
```

### 2.2 L2层：缓存结构

```python
from dataclasses import dataclass
from typing import Any
import hashlib

@dataclass
class CacheKey:
    """缓存键"""
    model_id: str
    dataset_hash: str
    extraction_config_hash: str

    def to_string(self) -> str:
        content = f"{self.model_id}|{self.dataset_hash}|{self.extraction_config_hash}"
        return hashlib.sha256(content.encode()).hexdigest()

@dataclass
class FeatureEntry:
    """缓存条目"""
    cache_key: str
    model_output: ModelOutput
    timestamp: float
    size_bytes: int
```

### 2.3 L3层：计算结果结构

```python
from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np

@dataclass
class ComputeResult:
    """计算引擎通用结果"""
    metric_name: str
    value: float
    details: Dict[str, Any] = None

@dataclass
class EntropyResult(ComputeResult):
    """熵计算结果"""
    metric_name: str = "entropy"
    value: float = 0.0  # 香农熵
    per_token_entropy: np.ndarray = None  # 每个token的熵
    normalized_entropy: float = 0.0  # 归一化熵

@dataclass
class SVDResult(ComputeResult):
    """SVD分解结果"""
    metric_name: str = "svd"
    singular_values: np.ndarray = None
    effective_rank: float = 0.0
    explained_variance_ratio: np.ndarray = None
```

### 2.4 L4层：诊断结果结构

```python
from dataclasses import dataclass
from typing import Dict, List, Any
from enum import Enum

class HealthStatus(Enum):
    GREEN = "green"      # 健康
    YELLOW = "yellow"    # 需关注
    RED = "red"          # 需处理
    SKIP = "skip"        # 未检测

@dataclass
class CheckResult:
    """单个检查项结果"""
    check_name: str
    check_category: str  # blood/neural/bone/immune

    # 原始指标
    raw_metrics: Dict[str, float]

    # 健康评估
    health_status: HealthStatus
    health_score: float  # 0-100

    # 详细信息
    details: Dict[str, Any] = None
    visualization_data: Dict[str, Any] = None  # 用于绘图的数据

    # 对比基准
    baseline_name: str = None
    baseline_deviation: float = None  # 相对基准的偏差百分比

    # 元信息
    execution_time: float = 0.0
    samples_used: int = 0
    error_message: str = None
```

### 2.5 L5层：报告结构

```python
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime

@dataclass
class ClinicReport:
    """完整体检报告"""
    # 基本信息
    report_id: str
    created_at: datetime
    model_info: Dict[str, Any]  # 模型ID、参数量等

    # 体检配置
    check_preset: str  # quick/full/custom
    checks_executed: List[str]
    checks_skipped: List[str]

    # 总体评估
    overall_health_score: float
    overall_health_status: HealthStatus

    # 各项结果
    check_results: List[CheckResult]

    # 建议
    recommendations: List[str]

    # 可视化
    charts_base64: Dict[str, str] = None  # 图表名称 -> base64编码
```

---

## 3. 详细模块设计

### 3.1 L1层：模型适配器

#### 3.1.1 抽象基类

```python
from abc import ABC, abstractmethod
from typing import List, Union
from .data_structures import ModelOutput, ModelCapabilities

class BaseProvider(ABC):
    """模型提供者抽象基类"""

    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id
        self.config = kwargs
        self._capabilities = None

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """返回模型能力声明"""
        pass

    @abstractmethod
    def generate(
        self,
        prompts: Union[str, List[str]],
        return_logits: bool = True,
        return_hidden_states: bool = False,
        return_attention_weights: bool = False,
        hidden_state_layers: List[int] = None,  # None表示所有层
        **kwargs
    ) -> Union[ModelOutput, List[ModelOutput]]:
        """生成文本并提取特征"""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """分词"""
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """解码"""
        pass

    def check_capability(self, requirement: str) -> bool:
        """检查是否支持某能力"""
        return getattr(self.capabilities, requirement, False)
```

#### 3.1.2 HuggingFace适配器

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class HFProvider(BaseProvider):
    """HuggingFace Transformers适配器"""

    def __init__(self, model_id: str, device: str = "auto", **kwargs):
        super().__init__(model_id, **kwargs)
        self.device = device

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=kwargs.get("torch_dtype", torch.float16),
            output_hidden_states=True,
            output_attentions=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # 确定能力
        self._capabilities = ModelCapabilities(
            supports_logits=True,
            supports_hidden_states=True,
            supports_attention_weights=True,
            max_batch_size=kwargs.get("max_batch_size", 1),
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def generate(self, prompts, return_logits=True, return_hidden_states=False,
                 return_attention_weights=False, hidden_state_layers=None, **kwargs):
        # 实现细节...
        pass
```

#### 3.1.3 vLLM适配器

```python
class VLLMProvider(BaseProvider):
    """vLLM适配器 - 高性能推理"""

    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        # vLLM特定初始化
        from vllm import LLM
        self.llm = LLM(model=model_id, **kwargs)

        # vLLM默认不返回中间层，需要特殊配置
        self._capabilities = ModelCapabilities(
            supports_logits=True,
            supports_hidden_states=kwargs.get("enable_hidden_states", False),
            supports_attention_weights=False,  # vLLM通常不支持
            max_batch_size=kwargs.get("max_batch_size", 32),
            supports_streaming=True,
        )
```

#### 3.1.4 API适配器

```python
class APIProvider(BaseProvider):
    """OpenAI/Anthropic API适配器"""

    def __init__(self, model_id: str, api_key: str, api_base: str = None, **kwargs):
        super().__init__(model_id, **kwargs)
        self.api_key = api_key
        self.api_base = api_base

        # API只能获取logprobs
        self._capabilities = ModelCapabilities(
            supports_logits=False,  # 只有logprobs
            supports_hidden_states=False,
            supports_attention_weights=False,
            max_batch_size=10,
        )

    def generate(self, prompts, return_logits=True, **kwargs):
        if return_logits:
            # 对于API，return_logits映射为请求logprobs
            kwargs["logprobs"] = True
        # 调用API...
```

### 3.2 L2层：数据与缓存

#### 3.2.1 特征存储

```python
import os
import json
import zarr
from typing import Optional, List
from .data_structures import ModelOutput, CacheKey, FeatureEntry

class FeatureStore:
    """特征缓存管理器"""

    def __init__(self, cache_dir: str = ".llm_clinic_cache", max_memory_cache_size: int = 1e9):
        self.cache_dir = cache_dir
        self.max_memory_cache_size = max_memory_cache_size

        # 内存缓存
        self._memory_cache: Dict[str, ModelOutput] = {}

        # 磁盘缓存（使用Zarr格式）
        self._disk_store = None
        self._init_disk_store()

    def _init_disk_store(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        self._disk_store = zarr.open(os.path.join(self.cache_dir, "features.zarr"), mode="a")

    def get(self, key: CacheKey) -> Optional[ModelOutput]:
        """获取缓存的模型输出"""
        key_str = key.to_string()

        # 先查内存
        if key_str in self._memory_cache:
            return self._memory_cache[key_str]

        # 再查磁盘
        if key_str in self._disk_store:
            # 从Zarr加载并反序列化
            return self._load_from_disk(key_str)

        return None

    def put(self, key: CacheKey, output: ModelOutput):
        """缓存模型输出"""
        key_str = key.to_string()

        # 检查内存缓存大小，必要时淘汰
        self._check_memory_limit()

        # 存入内存
        self._memory_cache[key_str] = output

        # 异步写入磁盘
        self._save_to_disk_async(key_str, output)

    def _check_memory_limit(self):
        """检查内存限制，必要时淘汰旧缓存"""
        # LRU淘汰策略
        pass

    def _save_to_disk_async(self, key: str, output: ModelOutput):
        """异步保存到磁盘"""
        # 使用线程池异步写入
        pass

    def _load_from_disk(self, key: str) -> ModelOutput:
        """从磁盘加载"""
        pass

    def clear(self):
        """清空缓存"""
        self._memory_cache.clear()
        # 清空磁盘缓存
```

#### 3.2.2 张量处理器

```python
import torch
import numpy as np

class TensorProcessor:
    """张量对齐与预处理"""

    @staticmethod
    def align_logits(logits: torch.Tensor, target_vocab_size: int) -> torch.Tensor:
        """对齐logits到目标词表大小"""
        current_size = logits.shape[-1]
        if current_size < target_vocab_size:
            # Pad with zeros
            padding = torch.zeros(*logits.shape[:-1], target_vocab_size - current_size)
            return torch.cat([logits, padding], dim=-1)
        elif current_size > target_vocab_size:
            # Truncate
            return logits[..., :target_vocab_size]
        return logits

    @staticmethod
    def remove_padding(tensor: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """移除padding位置的值"""
        # 根据attention_mask提取有效部分
        pass

    @staticmethod
    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """安全转换为numpy数组"""
        if tensor.requires_grad:
            tensor = tensor.detach()
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.numpy()

    @staticmethod
    def streaming_extract(model, prompts, layers, offload_dir: str):
        """流式提取中间层，避免显存爆炸"""
        for layer_idx in layers:
            # 提取该层
            hidden_states = model.get_layer_output(layer_idx, prompts)
            # Offload到CPU
            hidden_states = hidden_states.cpu()
            # 写入磁盘
            np.save(f"{offload_dir}/layer_{layer_idx}.npy", hidden_states.numpy())
            # 清理GPU显存
            torch.cuda.empty_cache()
```

### 3.3 L3层：核心计算引擎

#### 3.3.1 信息论工具

```python
import torch
import numpy as np
from typing import Union

class InfoTheoryCalculator:
    """信息论计算工具"""

    @staticmethod
    def entropy(prob_dist: Union[np.ndarray, torch.Tensor], dim: int = -1) -> Union[float, np.ndarray]:
        """
        计算香农熵 H(X) = -Σ p(x) * log(p(x))

        Args:
            prob_dist: 概率分布，shape [..., vocab_size]
            dim: 计算熵的维度

        Returns:
            熵值（标量或数组）
        """
        if isinstance(prob_dist, torch.Tensor):
            prob_dist = prob_dist.float()
            # 避免log(0)
            log_probs = torch.log(prob_dist + 1e-10)
            return -torch.sum(prob_dist * log_probs, dim=dim)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                log_probs = np.log(prob_dist + 1e-10)
                return -np.sum(prob_dist * log_probs, axis=dim)

    @staticmethod
    def kl_divergence(p: Union[np.ndarray, torch.Tensor],
                      q: Union[np.ndarray, torch.Tensor]) -> float:
        """
        计算KL散度 D_KL(P || Q) = Σ P(x) * log(P(x) / Q(x))

        Args:
            p: 概率分布P
            q: 概率分布Q

        Returns:
            KL散度值
        """
        if isinstance(p, torch.Tensor):
            p, q = p.float(), q.float()
            log_ratio = torch.log(p + 1e-10) - torch.log(q + 1e-10)
            return torch.sum(p * log_ratio).item()
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                log_ratio = np.log(p + 1e-10) - np.log(q + 1e-10)
                return np.sum(p * log_ratio)

    @staticmethod
    def cross_entropy(p: Union[np.ndarray, torch.Tensor],
                      q: Union[np.ndarray, torch.Tensor]) -> float:
        """计算交叉熵 H(P, Q) = -Σ P(x) * log(Q(x))"""
        if isinstance(p, torch.Tensor):
            return -torch.sum(p * torch.log(q + 1e-10)).item()
        else:
            return -np.sum(p * np.log(q + 1e-10))

    @staticmethod
    def top_p_mass(prob_dist: Union[np.ndarray, torch.Tensor],
                   p: float = 0.9) -> int:
        """
        计算Top-P概率质量对应的token数量

        Args:
            prob_dist: 概率分布
            p: 目标概率质量（如0.9）

        Returns:
            达到p概率质量所需的token数量
        """
        if isinstance(prob_dist, torch.Tensor):
            sorted_probs, _ = torch.sort(prob_dist, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            return torch.sum(cumsum <= p).item() + 1
        else:
            sorted_probs = np.sort(prob_dist)[::-1]
            cumsum = np.cumsum(sorted_probs)
            return np.sum(cumsum <= p) + 1

    @staticmethod
    def perplexity(prob_dist: Union[np.ndarray, torch.Tensor]) -> float:
        """计算困惑度 = exp(entropy)"""
        return np.exp(InfoTheoryCalculator.entropy(prob_dist))
```

#### 3.3.2 线性代数工具

```python
import numpy as np
from scipy import linalg
from typing import Tuple

class LinAlgCalculator:
    """线性代数计算工具"""

    @staticmethod
    def svd(matrix: np.ndarray, full_matrices: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        奇异值分解

        Args:
            matrix: 输入矩阵
            full_matrices: 是否返回完整矩阵

        Returns:
            U, S, Vt
        """
        return np.linalg.svd(matrix, full_matrices=full_matrices)

    @staticmethod
    def effective_rank(matrix: np.ndarray, threshold: float = 1e-4) -> float:
        """
        计算有效秩（Effective Rank）

        基于奇异值分布的熵计算，比直接阈值更鲁棒

        Effective Rank = exp(H), where H = -Σ p_i * log(p_i)
        p_i = s_i / Σ s_j (归一化奇异值)
        """
        _, s, _ = LinAlgCalculator.svd(matrix)

        # 归一化奇异值
        s_normalized = s / np.sum(s)

        # 计算熵
        s_normalized = s_normalized[s_normalized > 0]  # 避免log(0)
        entropy = -np.sum(s_normalized * np.log(s_normalized))

        return np.exp(entropy)

    @staticmethod
    def stable_rank(matrix: np.ndarray) -> float:
        """
        计算稳定秩（Stable Rank）

        = ||A||_F^2 / ||A||_2^2 = Σ s_i^2 / s_1^2
        """
        _, s, _ = LinAlgCalculator.svd(matrix)
        return np.sum(s ** 2) / (s[0] ** 2)

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    @staticmethod
    def condition_number(matrix: np.ndarray) -> float:
        """计算条件数"""
        return np.linalg.cond(matrix)

    @staticmethod
    def explained_variance(matrix: np.ndarray, n_components: int) -> float:
        """计算前n个主成分解释的方差比例"""
        _, s, _ = LinAlgCalculator.svd(matrix)
        total_var = np.sum(s ** 2)
        explained = np.sum(s[:n_components] ** 2)
        return explained / total_var
```

#### 3.3.3 校准工具

```python
import numpy as np
from typing import List, Tuple

class CalibrationCalculator:
    """置信度校准计算工具"""

    @staticmethod
    def expected_calibration_error(
        confidences: np.ndarray,
        accuracies: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        计算期望校准误差（ECE）

        ECE = Σ (n_i / N) * |acc_i - conf_i|

        Args:
            confidences: 置信度数组 [0, 1]
            accuracies: 准确率数组（0或1）
            n_bins: 分箱数量

        Returns:
            ece: 期望校准误差
            bin_accs: 每个箱的平均准确率
            bin_confs: 每个箱的平均置信度
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        bin_accs = np.zeros(n_bins)
        bin_confs = np.zeros(n_bins)

        for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

                bin_accs[i] = accuracy_in_bin
                bin_confs[i] = avg_confidence_in_bin

        return ece, bin_accs, bin_confs

    @staticmethod
    def brier_score(confidences: np.ndarray, accuracies: np.ndarray) -> float:
        """
        计算Brier Score

        BS = (1/N) * Σ (conf - acc)^2
        """
        return np.mean((confidences - accuracies) ** 2)

    @staticmethod
    def reliability_diagram_data(
        confidences: np.ndarray,
        accuracies: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """生成可靠性图数据"""
        _, bin_accs, bin_confs = CalibrationCalculator.expected_calibration_error(
            confidences, accuracies, n_bins
        )
        return bin_confs, bin_accs
```

#### 3.3.4 扰动生成器

```python
import random
import re
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class PerturbationConfig:
    """扰动配置"""
    synonym_ratio: float = 0.1  # 同义词替换比例
    typo_ratio: float = 0.05    # 拼写错误比例
    blank_ratio: float = 0.1    # 空白注入比例
    seed: int = 42

class PerturbationGenerator:
    """文本扰动生成器"""

    def __init__(self, config: PerturbationConfig = None):
        self.config = config or PerturbationConfig()
        random.seed(self.config.seed)

        # 同义词词典（示例，实际应从文件加载）
        self.synonym_dict = {
            "happy": ["joyful", "pleased", "glad"],
            "sad": ["unhappy", "sorrowful", "dejected"],
            # ... 更多同义词
        }

        # 常见拼写错误映射
        self.typo_map = {
            "the": ["teh", "hte"],
            "and": ["adn", "nad"],
            # ... 更多拼写错误
        }

    def perturb_synonym(self, text: str) -> Tuple[str, List[Tuple[int, str, str]]]:
        """
        同义词替换扰动

        Returns:
            扰动后的文本, [(位置, 原词, 新词), ...]
        """
        words = text.split()
        n_replace = int(len(words) * self.config.synonym_ratio)
        changes = []

        indices = random.sample(range(len(words)), min(n_replace, len(words)))

        for idx in indices:
            word = words[idx].lower().strip(".,!?")
            if word in self.synonym_dict:
                synonym = random.choice(self.synonym_dict[word])
                original = words[idx]
                words[idx] = synonym + original[len(word):]  # 保留标点
                changes.append((idx, original, words[idx]))

        return " ".join(words), changes

    def perturb_typo(self, text: str) -> Tuple[str, List[Tuple[int, str, str]]]:
        """拼写错误注入扰动"""
        words = text.split()
        n_typo = int(len(words) * self.config.typo_ratio)
        changes = []

        # 随机选择要注入typo的位置
        indices = random.sample(range(len(words)), min(n_typo, len(words)))

        for idx in indices:
            word = words[idx].lower().strip(".,!?")
            if word in self.typo_map:
                typo = random.choice(self.typo_map[word])
                original = words[idx]
                words[idx] = typo
                changes.append((idx, original, typo))
            else:
                # 随机字符替换
                if len(word) > 2:
                    pos = random.randint(0, len(word) - 1)
                    chars = list(word)
                    chars[pos] = random.choice("abcdefghijklmnopqrstuvwxyz")
                    typo = "".join(chars)
                    original = words[idx]
                    words[idx] = typo
                    changes.append((idx, original, typo))

        return " ".join(words), changes

    def perturb_blank(self, text: str) -> Tuple[str, List[Tuple[int, str]]]:
        """无关上下文注入扰动（在句子中插入空白/填充词）"""
        fillers = ["[BLANK]", "...", "[MASK]", "_____"]
        words = text.split()
        n_blank = int(len(words) * self.config.blank_ratio)
        changes = []

        indices = random.sample(range(len(words)), min(n_blank, len(words)))

        for idx in sorted(indices, reverse=True):
            filler = random.choice(fillers)
            words.insert(idx, filler)
            changes.append((idx, filler))

        return " ".join(words), changes

    def generate_perturbations(self, text: str, types: List[str] = None) -> dict:
        """生成多种类型的扰动"""
        types = types or ["synonym", "typo", "blank"]
        results = {}

        if "synonym" in types:
            results["synonym"] = self.perturb_synonym(text)
        if "typo" in types:
            results["typo"] = self.perturb_typo(text)
        if "blank" in types:
            results["blank"] = self.perturb_blank(text)

        return results
```

### 3.4 L4层：诊断模块

#### 3.4.1 基类设计

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class CheckConfig:
    """检查项配置"""
    sample_size: int = 100
    batch_size: int = 1
    seed: int = 42
    custom_params: Dict[str, Any] = None

class BaseCheck(ABC):
    """诊断模块基类"""

    # 子类必须定义
    check_name: str = ""
    check_category: str = ""  # blood/neural/bone/immune
    required_capabilities: List[str] = []  # ["logits", "hidden_states", ...]

    def __init__(self, config: CheckConfig = None):
        self.config = config or CheckConfig()
        self._result: Optional[CheckResult] = None

    @abstractmethod
    def run(
        self,
        provider: "BaseProvider",
        dataset: List[str],
        cache: "FeatureStore" = None
    ) -> "CheckResult":
        """
        执行检查

        Args:
            provider: 模型提供者
            dataset: 测试数据集
            cache: 特征缓存

        Returns:
            检查结果
        """
        pass

    def check_provider_capability(self, provider: "BaseProvider") -> bool:
        """检查提供者是否具备所需能力"""
        for cap in self.required_capabilities:
            if not provider.check_capability(f"supports_{cap}"):
                return False
        return True

    def get_result(self) -> Optional["CheckResult"]:
        """获取最近的检查结果"""
        return self._result

    @abstractmethod
    def interpret_result(self, result: "CheckResult") -> str:
        """解释检查结果的含义"""
        pass

    @abstractmethod
    def get_health_status(self, result: "CheckResult") -> "HealthStatus":
        """根据结果判断健康状态"""
        pass
```

#### 3.4.2 具体检查项示例

```python
import numpy as np
from typing import List
from .base_check import BaseCheck, CheckConfig
from ..core.info_theory import InfoTheoryCalculator
from ..data_structures import CheckResult, HealthStatus

class VocabularyUtilizationCheck(BaseCheck):
    """词表利用率检查"""

    check_name = "vocabulary_utilization"
    check_category = "blood"
    required_capabilities = ["logits"]

    # 健康阈值
    HEALTHY_THRESHOLD = 0.3  # 利用率 > 30% 为健康
    WARNING_THRESHOLD = 0.1  # 利用率 > 10% 为警告

    def __init__(self, config: CheckConfig = None, top_k: int = 100):
        super().__init__(config)
        self.top_k = top_k

    def run(self, provider, dataset, cache=None) -> CheckResult:
        # 1. 获取模型输出
        outputs = provider.generate(dataset, return_logits=True)

        # 2. 统计词表使用情况
        all_used_tokens = set()
        for output in outputs:
            if output.logits is not None:
                # 获取top-k预测
                top_k_indices = np.argsort(output.logits, axis=-1)[:, -self.top_k:]
                all_used_tokens.update(top_k_indices.flatten())

        # 3. 计算利用率
        vocab_size = outputs[0].logits.shape[-1]
        utilization_ratio = len(all_used_tokens) / vocab_size

        # 4. 构建结果
        result = CheckResult(
            check_name=self.check_name,
            check_category=self.check_category,
            raw_metrics={
                "utilization_ratio": utilization_ratio,
                "unique_tokens_used": len(all_used_tokens),
                "vocab_size": vocab_size,
            },
            health_status=self.get_health_status_from_value(utilization_ratio),
            health_score=utilization_ratio * 100,  # 0-100
            details={
                "top_k_per_token": self.top_k,
                "samples_analyzed": len(dataset),
            }
        )

        self._result = result
        return result

    def get_health_status_from_value(self, utilization: float) -> HealthStatus:
        if utilization >= self.HEALTHY_THRESHOLD:
            return HealthStatus.GREEN
        elif utilization >= self.WARNING_THRESHOLD:
            return HealthStatus.YELLOW
        else:
            return HealthStatus.RED

    def interpret_result(self, result: CheckResult) -> str:
        ratio = result.raw_metrics["utilization_ratio"]
        if ratio >= self.HEALTHY_THRESHOLD:
            return f"词表利用率正常（{ratio:.1%}），模型输出多样性良好"
        elif ratio >= self.WARNING_THRESHOLD:
            return f"词表利用率偏低（{ratio:.1%}），可能存在输出多样性不足的风险"
        else:
            return f"词表利用率严重不足（{ratio:.1%}），模型可能存在词表坍缩问题"

    def get_health_status(self, result: CheckResult) -> HealthStatus:
        return result.health_status


class RepresentationCollapseCheck(BaseCheck):
    """表征坍缩检查"""

    check_name = "representation_collapse"
    check_category = "bone"
    required_capabilities = ["hidden_states"]

    # 健康阈值
    COLLAPSE_THRESHOLD = 0.5  # 有效秩下降超过50%视为坍缩

    def __init__(self, config: CheckConfig = None, collapse_threshold: float = 0.5):
        super().__init__(config)
        self.collapse_threshold = collapse_threshold

    def run(self, provider, dataset, cache=None) -> CheckResult:
        from ..core.linalg import LinAlgCalculator

        # 1. 获取hidden states
        outputs = provider.generate(
            dataset[:self.config.sample_size],
            return_hidden_states=True
        )

        # 2. 计算各层的有效秩
        layer_effective_ranks = {}
        first_layer_rank = None

        for layer_idx, hidden_states in outputs[0].hidden_states.items():
            # hidden_states: [seq_len, hidden_dim]
            effective_rank = LinAlgCalculator.effective_rank(
                hidden_states.cpu().numpy()
            )
            layer_effective_ranks[layer_idx] = effective_rank

            if layer_idx == 0:
                first_layer_rank = effective_rank

        # 3. 检测坍缩层
        collapsed_layers = []
        for layer_idx, rank in layer_effective_ranks.items():
            if first_layer_rank and rank < first_layer_rank * (1 - self.collapse_threshold):
                collapsed_layers.append(layer_idx)

        # 4. 计算健康分数
        if first_layer_rank:
            min_rank_ratio = min(layer_effective_ranks.values()) / first_layer_rank
        else:
            min_rank_ratio = 1.0

        result = CheckResult(
            check_name=self.check_name,
            check_category=self.check_category,
            raw_metrics={
                "layer_effective_ranks": layer_effective_ranks,
                "first_layer_rank": first_layer_rank,
                "min_rank_ratio": min_rank_ratio,
            },
            health_status=HealthStatus.RED if collapsed_layers else HealthStatus.GREEN,
            health_score=min_rank_ratio * 100,
            details={
                "collapsed_layers": collapsed_layers,
                "num_layers": len(layer_effective_ranks),
            },
            visualization_data={
                "type": "line_chart",
                "x": list(layer_effective_ranks.keys()),
                "y": list(layer_effective_ranks.values()),
                "xlabel": "Layer Index",
                "ylabel": "Effective Rank",
            }
        )

        self._result = result
        return result

    def interpret_result(self, result: CheckResult) -> str:
        collapsed = result.details.get("collapsed_layers", [])
        if collapsed:
            return f"检测到表征坍缩，涉及第 {collapsed} 层，建议检查训练过程"
        return "各层表征正常，未检测到坍缩现象"

    def get_health_status(self, result: CheckResult) -> HealthStatus:
        return result.health_status
```

### 3.5 L5层：编排与报告

#### 3.5.1 体检流水线

```python
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

@dataclass
class ClinicConfig:
    """体检配置"""
    preset: str = "quick"  # quick / full / custom
    checks: List[str] = None  # 自定义检查项列表
    parallel: bool = True
    max_workers: int = 4

class ClinicRunner:
    """体检流水线执行器"""

    # 预设体检套餐
    PRESETS = {
        "quick": [
            "vocabulary_utilization",
            "confidence_calibration",
        ],
        "full": [
            "vocabulary_utilization",
            "unconditional_entropy",
            "perturbation_sensitivity",
            "representation_collapse",
            "attention_sink",
            "confidence_calibration",
        ],
    }

    def __init__(self, provider: "BaseProvider", config: ClinicConfig = None):
        self.provider = provider
        self.config = config or ClinicConfig()
        self.cache = FeatureStore()
        self._checks: Dict[str, BaseCheck] = {}
        self._results: List[CheckResult] = []

        # 注册检查项
        self._register_checks()

    def _register_checks(self):
        """注册所有可用的检查项"""
        # 自动发现并注册checks目录下的所有检查类
        from ..checks.blood.vocab_utilization import VocabularyUtilizationCheck
        from ..checks.bone.representation_collapse import RepresentationCollapseCheck
        # ... 其他检查项

        for check_class in [VocabularyUtilizationCheck, RepresentationCollapseCheck]:
            instance = check_class()
            self._checks[instance.check_name] = instance

    def get_available_checks(self) -> List[str]:
        """获取所有可用的检查项"""
        return list(self._checks.keys())

    def get_compatible_checks(self) -> List[str]:
        """获取当前模型支持的检查项"""
        compatible = []
        for name, check in self._checks.items():
            if check.check_provider_capability(self.provider):
                compatible.append(name)
        return compatible

    def run(self, dataset: List[str]) -> "ClinicReport":
        """执行体检"""
        start_time = time.time()

        # 确定要执行的检查项
        if self.config.preset == "custom" and self.config.checks:
            check_names = self.config.checks
        else:
            check_names = self.PRESETS.get(self.config.preset, self.PRESETS["quick"])

        # 过滤不兼容的检查项
        compatible_checks = self.get_compatible_checks()
        checks_to_run = [n for n in check_names if n in compatible_checks]
        checks_skipped = [n for n in check_names if n not in compatible_checks]

        # 执行检查
        if self.config.parallel:
            results = self._run_parallel(checks_to_run, dataset)
        else:
            results = self._run_sequential(checks_to_run, dataset)

        self._results = results

        # 生成报告
        report = self._generate_report(results, checks_skipped, time.time() - start_time)

        return report

    def _run_parallel(self, check_names: List[str], dataset: List[str]) -> List[CheckResult]:
        """并行执行检查"""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            for name in check_names:
                check = self._checks[name]
                futures.append(executor.submit(check.run, self.provider, dataset, self.cache))

            results = [f.result() for f in futures]
        return results

    def _run_sequential(self, check_names: List[str], dataset: List[str]) -> List[CheckResult]:
        """串行执行检查"""
        results = []
        for name in check_names:
            check = self._checks[name]
            result = check.run(self.provider, dataset, self.cache)
            results.append(result)
        return results

    def _generate_report(self, results: List[CheckResult], skipped: List[str], duration: float) -> "ClinicReport":
        """生成体检报告"""
        # 计算总体健康分数
        scores = [r.health_score for r in results if r.health_status != HealthStatus.SKIP]
        overall_score = sum(scores) / len(scores) if scores else 0

        # 确定总体健康状态
        if any(r.health_status == HealthStatus.RED for r in results):
            overall_status = HealthStatus.RED
        elif any(r.health_status == HealthStatus.YELLOW for r in results):
            overall_status = HealthStatus.YELLOW
        else:
            overall_status = HealthStatus.GREEN

        # 生成建议
        recommendations = self._generate_recommendations(results)

        report = ClinicReport(
            report_id=f"clinic_{int(time.time())}",
            created_at=datetime.now(),
            model_info={"model_id": self.provider.model_id},
            check_preset=self.config.preset,
            checks_executed=[r.check_name for r in results],
            checks_skipped=skipped,
            overall_health_score=overall_score,
            overall_health_status=overall_status,
            check_results=results,
            recommendations=recommendations,
        )

        return report

    def _generate_recommendations(self, results: List[CheckResult]) -> List[str]:
        """根据检查结果生成建议"""
        recommendations = []

        for result in results:
            if result.health_status == HealthStatus.RED:
                if result.check_name == "vocabulary_utilization":
                    recommendations.append("建议检查训练数据多样性和分词器配置")
                elif result.check_name == "representation_collapse":
                    recommendations.append("建议调整学习率或增加正则化项")
                # ... 其他建议

        return recommendations
```

#### 3.5.2 报告生成器

```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
from io import BytesIO
import base64

class ReportGenerator:
    """体检报告生成器"""

    def __init__(self, template: str = "default"):
        self.template = template

    def generate_markdown(self, report: "ClinicReport") -> str:
        """生成Markdown格式报告"""
        md = f"""# LLM Clinic 体检报告

## 基本信息

- **报告ID**: {report.report_id}
- **生成时间**: {report.created_at.strftime("%Y-%m-%d %H:%M:%S")}
- **模型**: {report.model_info.get("model_id", "Unknown")}
- **体检套餐**: {report.check_preset}

## 总体评估

**健康分数**: {report.overall_health_score:.1f}/100

**健康状态**: {self._status_emoji(report.overall_health_status)} {report.overall_health_status.value.upper()}

---

## 详细检查结果

"""
        # 按类别分组显示
        categories = {"blood": "概率与分布", "neural": "敏感度与鲁棒性", "bone": "表征与权重", "immune": "校准与置信度"}

        for cat_key, cat_name in categories.items():
            cat_results = [r for r in report.check_results if r.check_category == cat_key]
            if cat_results:
                md += f"### {cat_name}\n\n"
                for result in cat_results:
                    md += self._format_check_result(result)

        # 建议
        if report.recommendations:
            md += """---

## 建议与改进方向

"""
            for i, rec in enumerate(report.recommendations, 1):
                md += f"{i}. {rec}\n"

        # 跳过的检查项
        if report.checks_skipped:
            md += f"""

---

**未检测项**: {', '.join(report.checks_skipped)}（因模型能力限制）
"""

        return md

    def _format_check_result(self, result: CheckResult) -> str:
        """格式化单个检查结果"""
        status_emoji = self._status_emoji(result.health_status)
        lines = [
            f"#### {status_emoji} {result.check_name.replace('_', ' ').title()}",
            "",
            f"- **状态**: {result.health_status.value}",
            f"- **分数**: {result.health_score:.1f}/100",
        ]

        # 原始指标
        lines.append("\n**指标详情**:")
        for key, value in result.raw_metrics.items():
            if isinstance(value, dict):
                continue  # 跳过复杂结构
            lines.append(f"- {key}: {value}")

        lines.append("")
        return "\n".join(lines)

    def _status_emoji(self, status: HealthStatus) -> str:
        """状态对应的emoji"""
        return {
            HealthStatus.GREEN: "🟢",
            HealthStatus.YELLOW: "🟡",
            HealthStatus.RED: "🔴",
            HealthStatus.SKIP: "⚪",
        }.get(status, "⚪")

    def generate_chart(self, result: CheckResult) -> str:
        """为检查结果生成图表，返回base64编码"""
        if not result.visualization_data:
            return None

        data = result.visualization_data

        fig, ax = plt.subplots(figsize=(10, 6))

        if data["type"] == "line_chart":
            ax.plot(data["x"], data["y"], marker="o")
            ax.set_xlabel(data.get("xlabel", "X"))
            ax.set_ylabel(data.get("ylabel", "Y"))
            ax.set_title(result.check_name)

        plt.tight_layout()

        # 转换为base64
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close()

        return img_base64
```

---

## 4. 关键技术细节

### 4.1 显存管理策略

对于大模型（70B+），中间层激活值提取是显存瓶颈。采用以下策略：

```python
class MemoryEfficientExtraction:
    """显存友好的特征提取"""

    @staticmethod
    def extract_with_offload(model, prompts, layers, output_dir):
        """
        逐层提取并offload到磁盘

        1. 注册hook获取当前层输出
        2. 立即转移到CPU
        3. 写入磁盘
        4. 释放GPU显存
        5. 处理下一层
        """
        import torch

        for layer_idx in layers:
            # 注册forward hook
            hook = model.layers[layer_idx].register_forward_hook(
                lambda m, i, o: save_and_offload(o, layer_idx, output_dir)
            )

            # 执行前向传播
            with torch.no_grad():
                _ = model(prompts)

            # 移除hook
            hook.remove()

            # 清理显存
            torch.cuda.empty_cache()

def save_and_offload(tensor, layer_idx, output_dir):
    """保存到磁盘并释放显存"""
    import numpy as np
    tensor_cpu = tensor.cpu()
    np.save(f"{output_dir}/layer_{layer_idx}.npy", tensor_cpu.numpy())
    del tensor_cpu
```

### 4.2 缓存键设计

```python
def compute_cache_key(model_id: str, prompts: List[str], config: dict) -> str:
    """
    计算缓存键

    考虑因素：
    1. 模型ID（唯一标识模型）
    2. 数据集哈希（相同输入相同缓存）
    3. 提取配置（不同配置不同缓存）
    """
    import hashlib
    import json

    # 数据集哈希
    data_str = "".join(prompts)
    data_hash = hashlib.sha256(data_str.encode()).hexdigest()[:16]

    # 配置哈希
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

    # 组合
    key = f"{model_id}|{data_hash}|{config_hash}"
    return hashlib.sha256(key.encode()).hexdigest()
```

### 4.3 基线对比机制

```python
# 预置基线数据
BASELINES = {
    "llama-3-8b": {
        "vocabulary_utilization": {"utilization_ratio": 0.35},
        "representation_collapse": {"min_rank_ratio": 0.85},
        # ...
    },
    "mistral-7b": {
        "vocabulary_utilization": {"utilization_ratio": 0.38},
        # ...
    },
}

class BaselineComparator:
    """基线对比器"""

    def __init__(self, baseline_name: str = "llama-3-8b"):
        self.baseline = BASELINES.get(baseline_name)

    def compare(self, result: CheckResult) -> float:
        """计算相对基线的偏差"""
        if not self.baseline or result.check_name not in self.baseline:
            return None

        baseline_metrics = self.baseline[result.check_name]
        deviations = {}

        for metric_name, baseline_value in baseline_metrics.items():
            actual_value = result.raw_metrics.get(metric_name)
            if actual_value is not None:
                deviation = (actual_value - baseline_value) / baseline_value
                deviations[metric_name] = deviation

        return deviations
```

---

## 5. API设计

### 5.1 Python API

```python
from llm_clinic import ClinicRunner, HFProvider

# 创建模型提供者
provider = HFProvider("meta-llama/Llama-3-8b")

# 创建体检运行器
runner = ClinicRunner(provider, preset="full")

# 准备测试数据
dataset = ["What is machine learning?", "Explain quantum computing."]

# 执行体检
report = runner.run(dataset)

# 生成报告
from llm_clinic import ReportGenerator
generator = ReportGenerator()
markdown_report = generator.generate_markdown(report)
print(markdown_report)
```

### 5.2 命令行接口

```bash
# 快速体检
llm-clinic check --model meta-llama/Llama-3-8b --preset quick

# 全套体检并保存报告
llm-clinic check --model ./my_model --preset full --output report.md

# 自定义检查项
llm-clinic check --model ./my_model --checks vocab_utilization,representation_collapse

# 指定数据集
llm-clinic check --model ./my_model --dataset ./test_prompts.txt --preset full
```

---

## 6. 测试策略

### 6.1 单元测试

- 每个计算函数的边界条件测试
- 适配器的mock测试
- 缓存读写测试

### 6.2 集成测试

- 端到端体检流程测试
- 不同模型提供者的兼容性测试

### 6.3 性能测试

- 大模型显存占用测试
- 缓存效率测试
- 并行执行性能测试

---

## 7. 部署考虑

### 7.1 依赖管理

```
# requirements.txt
torch>=2.0
transformers>=4.35
numpy>=1.24
scipy>=1.11
zarr>=2.16
matplotlib>=3.8
```

### 7.2 可选依赖

```
# requirements-optional.txt
vllm>=0.3  # vLLM支持
openai>=1.0  # OpenAI API
anthropic>=0.8  # Anthropic API
```

---

## 8. 扩展指南

### 8.1 添加新的模型适配器

1. 继承 `BaseProvider`
2. 实现 `generate()` 方法
3. 声明 `capabilities`
4. 注册到适配器工厂

### 8.2 添加新的检查项

1. 继承 `BaseCheck`
2. 实现 `run()` 方法
3. 定义健康阈值
4. 添加到预设套餐

### 8.3 添加新的计算能力

1. 在 `core/` 下添加模块
2. 保持纯函数设计
3. 添加类型注解和文档
