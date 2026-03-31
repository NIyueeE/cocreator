# CoCreator

驾驶场景事件检测与因果推理工具。

## 项目简介

CoCreator 是一款基于视觉语言模型（VLM）的驾驶场景分析工具，通过多阶段流水线对驾驶数据进行分析：

1. **事件检测**：从驾驶片段中识别异常事件
2. **因果推理**：分析事件之间的因果关系链
3. **报告生成**：输出结构化的分析报告

## 功能特性

- **自动化事件检测**：基于统计异常算法的驾驶事件识别
- **因果链分析**：利用 VLM 生成事件间的因果关系
- **灵活配置**：支持 YAML 配置文件和环境变量
- **批量处理**：支持多 episode 并行处理
- **Markdown 报告**：自动生成可读性高的分析报告

## 技术架构

```
cocreator/
├── src/cocreator/
│   ├── cli.py              # CLI 命令入口
│   ├── config.py           # 配置管理
│   ├── schemas.py          # 数据模型
│   ├── pipeline/
│   │   ├── detector.py     # 事件检测模块
│   │   ├── reasoner.py     # 因果推理模块
│   │   └── extractor.py    # 数据提取模块
│   └── providers/
│       └── openai_compatible.py  # VLM 提供者
├── config.yaml.example     # 配置示例
└── tests/
```

### 核心模块

| 模块 | 功能 |
|------|------|
| `detector` | 事件检测，基于阈值判断异常 |
| `reasoner` | 因果推理，调用 VLM 分析因果 |
| `extractor` | 数据提取，从驾驶数据中获取信息 |

## 快速开始

### 安装依赖

```bash
uv sync
```

### 配置环境

1. 复制配置示例文件：

```bash
cp config.yaml.example config.yaml
```

2. 编辑 `config.yaml`，设置必要的参数

3. 设置环境变量（如需）：

```bash
export COCREATOR_VLM__API_KEY="your-api-key"
```

### 运行检测

```bash
cocreator detect -c config.yaml -o output/events.jsonl
```

## 使用指南

CoCreator 提供三个主要命令：

### detect - 事件检测

检测驾驶数据中的异常事件。

```bash
cocreator detect -c <配置文件> -o <输出文件>
```

**参数说明：**

| 参数 | 必填 | 说明 |
|------|------|------|
| `-c, --config` | 是 | YAML 配置文件路径 |
| `-o, --output` | 是 | 输出 JSONL 文件路径 |

**示例：**

```bash
cocreator detect -c config.yaml -o output/events.jsonl
```

### reason - 因果推理

为检测到的事件生成因果链。

```bash
cocreator reason -c <配置文件> -e <事件文件> -o <输出文件>
```

**参数说明：**

| 参数 | 必填 | 说明 |
|------|------|------|
| `-c, --config` | 是 | YAML 配置文件路径 |
| `-e, --events` | 是 | 输入的 JSONL 事件文件 |
| `-o, --output` | 是 | 输出 JSONL 因果链文件 |

**示例：**

```bash
cocreator reason -c config.yaml -e output/events.jsonl -o output/chains.jsonl
```

### review - 报告生成

从因果链生成 Markdown 报告。

```bash
cocreator review -i <因果链文件> -n <数量> -o <输出文件>
```

**参数说明：**

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `-i, --input` | 是 | - | 输入的 JSONL 因果链文件 |
| `-n, --count` | 否 | 10 | 报告中包含的因果链数量 |
| `-o, --output` | 否 | 标准输出 | 输出 Markdown 文件路径 |

**示例：**

```bash
cocreator review -i output/chains.jsonl -n 20 -o output/report.md
```

## 配置说明

### config.yaml 示例

```yaml
vlm:
  base_url: "https://api.siliconflow.cn"
  api_key: "${SILICONFLOW_API_KEY}"
  model: "Qwen/Qwen3.5-397B-A17B"

rate_limit:
  rpm: 500
  tpm: 2000000
  concurrency: 20

pipeline:
  dataset_path: "/data/CoCreator/dataset/drivingdojo_mini/action_info"
  videos_path: "/data/CoCreator/dataset/drivingdojo_mini/videos"
  output_dir: "./output"
  history_segments: 3
  future_segments: 2
  frames_per_segment: 5
  anomaly_threshold: 2.0
  retry_max_attempts: 3
  retry_backoff_factor: 2.0
```

### 配置项详解

#### VLM 配置 (`vlm`)

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `base_url` | `https://api.siliconflow.cn` | API 地址（注意：不包含 /v1） |
| `api_key` | 空 | API 密钥，支持 `${ENV_VAR}` 语法 |
| `model` | `Qwen/Qwen3.5-397B-A17B` | 模型名称 |

#### 限流配置 (`rate_limit`)

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `rpm` | 500 | 每分钟请求数限制 |
| `tpm` | 100000 | 每分钟 token 数限制 |
| `concurrency` | 20 | 最大并发请求数 |

#### 流水线配置 (`pipeline`)

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `dataset_path` | 空 | action_info 目录路径（用于事件检测） |
| `videos_path` | 空 | videos 目录路径（用于帧提取） |
| `output_dir` | 空 | 输出目录路径 |
| `history_segments` | 3 | 历史片段数（事件前） |
| `future_segments` | 2 | 未来片段数（事件后） |
| `frames_per_segment` | 5 | 每个片段的帧数 |
| `anomaly_threshold` | 2.0 | 异常检测的标准差倍数 |
| `retry_max_attempts` | 3 | API 调用最大重试次数 |
| `retry_backoff_factor` | 2.0 | 指数退避因子 |

### 环境变量

配置文件中支持 `${ENV_VAR}` 语法引用环境变量：

```yaml
api_key: "${SILICONFLOW_API_KEY}"
```

直接使用环境变量名作为引用。

## 输出格式

### 事件格式 (JSONL)

```json
{
  "episode_id": "episode_001",
  "frame_id": "frame_150",
  "action_type": "hard_brake",
  "confidence": 0.95
}
```

### 因果链格式 (JSONL)

```json
{
  "episode_id": "episode_001",
  "event_frame_id": "frame_150",
  "confidence": 0.85,
  "historical_analysis": {
    "ego_status": "cruising",
    "key_objects": [
      {"type": "pedestrian", "location": "crosswalk", "threat_level": "high"}
    ],
    "most_critical_object": {"type": "pedestrian", "location": "crosswalk", "threat_level": "high"},
    "predicted_action": "brake"
  },
  "future_confirmation": {
    "actual_action": "brake",
    "action_status": "completed",
    "related_to_history": true
  },
  "causal_link": "行人在人行横道上突然出现，导致驾驶员紧急制动。"
}
```

### 报告格式 (Markdown)

```markdown
# Causal Chain Report

Generated from 10 causal chains.

## Chain 1: Episode episode_001 - Frame frame_150

**Confidence:** 85.00%

### Historical Analysis
- Ego Status: cruising
- Predicted Action: brake
- Critical Object: pedestrian at crosswalk

### Future Confirmation
- Actual Action: brake
- Action Status: completed
- Related to History: true

### Causal Link
行人在人行横道上突然出现，导致驾驶员紧急制动。

---
```

## 抽检报告

本工具生成的抽检报告适用于以下场景：

- 驾驶行为分析研究
- 异常事件统计与归因
- 事故致因分析
- 自动驾驶系统评估

报告中的置信度得分反映模型对因果链的判断准确度，建议优先关注高置信度（>80%）的案例。

## 常见问题

**Q: 配置文件找不到怎么办？**

确保使用正确路径，或使用绝对路径：

```bash
cocreator detect -c /absolute/path/config.yaml -o output.jsonl
```

**Q: API 请求失败怎么排查？**

1. 检查 `api_key` 是否正确设置
2. 确认 `base_url` 可访问
3. 查看网络连接和代理设置

**Q: 如何调整检测灵敏度？**

修改 `pipeline.anomaly_threshold`：
- 降低值（如 1.5）：更敏感，检测更多事件
- 提高值（如 3.0）：更严格，减少误报

**Q: 输出文件太大怎么办？**

使用 `review` 命令的 `-n` 参数限制报告中的因果链数量：

```bash
cocreator review -i chains.jsonl -n 50 -o report.md
```

**Q: 支持批量处理吗？**

是的，通过 `detect` 命令自动处理 `dataset_path` 下的所有 episode。
