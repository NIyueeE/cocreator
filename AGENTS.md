# Project Instructions

This file provides context for AI assistants working on this project.

## Project Overview

CoCreator 是一个驾驶场景事件检测与因果推理工具。通过多阶段流水线对驾驶数据进行处理：

1. **事件检测 (detect)** — 从位置/速度数据中识别异常事件（急刹车、加速、转向）
2. **因果推理 (reason)** — 利用 VLM 进行两阶段分析（历史帧预测 → 未来帧验证），生成因果链
3. **报告生成 (review)** — 从因果链输出结构化 Markdown 报告

## Project Type: Python (≥3.9)

### Core Dependencies
- **CLI**: `typer` (via click)
- **Data models**: `pydantic` v2
- **VLM client**: `openai` (async) + `httpx`
- **Templating**: `jinja2` for VLM prompt templates
- **Config**: `pyyaml` + `python-dotenv`
- **Media**: `pillow` (image handling)
- **Math**: `numpy` (velocity/acceleration/direction computation)

### Commands
- Install: `uv sync`
- Test: `pytest` (unit); `pytest --run-smoke` (integration tests)
- Format: `ruff format .`
- Lint: `ruff check .`

## Project Structure

```
cocreator/
├── main.py                          # Entry: `python main.py`
├── pyproject.toml                   # Package config, CLI entry: cocreator
├── config.yaml.example              # YAML config template with env var ${...} support
├── .python-version                  # Python 3.9.21
│
├── src/cocreator/
│   ├── cli.py                       # 3 CLI commands: detect, reason, review
│   ├── config.py                    # YAML loading + recursive ${ENV_VAR} substitution
│   ├── schemas.py                   # All pydantic models (AppConfig, DetectedEvent, CausalChain, etc.)
│   │
│   ├── pipeline/                    # Core business logic
│   │   ├── detector.py              # EventDetector: velocity/acceleration/steering anomaly detection
│   │   ├── extractor.py             # VideoFrameExtractor: temporal frame extraction (Protocol pattern)
│   │   ├── reasoner.py              # CausalReasoner: 2-stage VLM analysis (predict → verify)
│   │   └── progress_tracker.py      # ProgressTracker: JSON-based resume for crash recovery
│   │
│   ├── providers/
│   │   └── openai_compatible.py     # OpenAICompatibleProvider: async VLM client with retry + semaphore
│   │
│   └── prompts/
│       ├── history_analysis.j2      # Stage 1 prompt: analyze frames BEFORE event
│       └── future_confirmation.j2   # Stage 2 prompt: verify frames AFTER event
│
└── tests/
    ├── conftest.py                  # --run-smoke flag registration
    ├── test_schemas.py              # Schema validation & serialization
    ├── test_detector.py             # Event detector unit tests
    ├── test_extractor.py            # Frame extractor unit tests
    ├── test_reasoner.py             # Causal reasoner unit tests
    ├── test_progress.py             # Progress tracker unit tests
    ├── test_provider.py             # VLM provider unit tests
    └── test_integration.py          # Smoke tests: detector, tracker, extractor, config
```

## Key Design Decisions

### 1. Strict Temporal Data Isolation
History analysis (`_analyze_history`) 只能看到事件前的帧，未来验证 (`_confirm_future`) 只能看到事件后的帧。两阶段之间**不允许**共享帧数据。`_validate_no_leakage()` 通过 assert 进行内部不变性检查。

### 2. Two-Stage VLM Reasoning
- **Stage 1 (History)**: 分析事件前的帧 → 预测 ego vehicle 会采取什么动作
- **Stage 2 (Future)**: 分析事件后的帧 → 验证预测是否准确
- 两阶段使用独立的 prompt 模板，Stage 2 只接收 Stage 1 的文本摘要（不含帧）

### 3. Async Concurrency
- `OpenAICompatibleProvider` 使用 `asyncio.Semaphore` 控制并发数
- `retry_with_backoff` 装饰器实现指数退避重试（默认 3 次）
- VLM 调用全部使用 `async/await`

### 4. Resume Capability
- `ProgressTracker` 将已处理事件的 key (`episode_id:frame_id`) 持久化到 `progress.json`
- 写入使用原子替换（先写 .tmp 再 rename），避免写中断损坏

### 5. Configuration
- YAML 配置支持 `${ENV_VAR}` 递归替换（`_substitute_env_vars`）
- 支持任意 OpenAI 兼容 API（SiliconFlow、Ollama、Azure OpenAI 等）

## CLI Commands

| Command | Description | Required Args |
|---------|-------------|---------------|
| `cocreator detect` | 事件检测：从 position 数据检测异常事件 | `-c <config.yaml>`, `-o <events.jsonl>` |
| `cocreator reason` | 因果推理：对检测到的事件生成因果链 | `-c <config.yaml>`, `-e <events.jsonl>`, `-o <chains.jsonl>` |
| `cocreator review` | 报告生成：从因果链输出 Markdown 报告 | `-i <chains.jsonl>`, `-n <count>`, `-o <report.md>` |

## Output Formats

- **事件**: JSONL, 每行一个 `DetectedEvent` (episode_id, frame_id, action_type, confidence)
- **因果链**: JSONL, 每行一个 `CausalChain` (含 historical_analysis + future_confirmation + causal_link)
- **报告**: Markdown 文档，按置信度排列因果链

## Testing

### Test markers
- `pytest` — 运行单元测试（不包含集成测试）
- `pytest --run-smoke` — 包含集成冒烟测试

### Key test coverage
- Schema validation (extra fields forbidden, serialization roundtrip)
- Detector anomaly detection with mock position data
- Frame extractor temporal isolation (history < event, future > event)
- Progress tracker save/load/atomic write
- Config loading with env var substitution
- CausalChain JSON roundtrip

## Important Notes

- **数据隔离是核心约束**：任何时候都不能将历史帧传给未来阶段，反之亦然
- `DetectedEvent` 和 `CausalChain` 的 `model_config = {"extra": "forbid"}`，不允许额外字段
- `_validate_no_leakage()` 使用 `assert`（非异常）——这是内部不变量检查，不要移除
- `HistoryAnalysis` 的 `reasoning` 和 `FutureConfirmation` 的 `verification_notes` 默认空字符串
- VLM 返回的 JSON 可能包裹在 markdown 代码块中，`_extract_json_from_response` 处理多种格式
- Prompt 模板使用 Jinja2，路径相对于 `src/cocreator/prompts/`
