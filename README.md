#### 1. 代码结构
```
optimizer/
├── core/                    # 核心流程模块
│   ├── __init__.py
│   ├── pipeline.py
│   └── main.py
├── knowledge/              # 知识管理模块
│   ├── __init__.py
│   ├── mechanism.py
│   ├── llm_extractor.py
│   ├── knowledge_fusion.py
│   └── knowledge_tree.py
├── optimization/           # 优化器模块
│   ├── __init__.py
│   ├── leaf_mobo.py
│   └── evaluator.py
├── analysis/               # 分析与报告模块
│   ├── __init__.py
│   ├── metrics.py
│   └── reporter.py
├── utils/                  # 工具模块
│   ├── __init__.py
│   └── common.py (原utils.py)
└── __init__.py             # 顶层初始化
```

### 2. 模块分类

#### Core 模块（核心流程）
- **文件**: `pipeline.py`, `main.py`
- **职责**: 主优化流程、命令行入口
- **关键类/函数**: `run_pipeline()`, `load_history_csv()`, `main()`

#### Knowledge 模块（知识管理）
- **文件**: `mechanism.py`, `llm_extractor.py`, `knowledge_fusion.py`, `knowledge_tree.py`
- **职责**: 机理配置、LLM提取、知识融合、知识树构建
- **关键类**: `Mechanism`, `LLMExtractor`, `KnowledgeFusion`, `KnowledgeTree`, `Node`

#### Optimization 模块（优化器）
- **文件**: `leaf_mobo.py`, `evaluator.py`
- **职责**: 贝叶斯优化、评估函数
- **关键类**: `LeafMOBO`, `make_evaluator()`

#### Analysis 模块（分析与报告）
- **文件**: `metrics.py`, `reporter.py`
- **职责**: 指标计算、报告生成、可视化
- **关键函数**: `aggregate_reward_from_batch()`, `nondominated_filter()`, `build_report_and_export()`

#### Utils 模块（工具）
- **文件**: `common.py` (原`utils.py`)
- **职责**: 通用工具函数
- **关键函数**: `set_seed()`, `safe_eval_expr()`, `short()`
