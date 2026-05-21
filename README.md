# CoCreator

驾驶场景因果关系数据集构建框架。

## 项目简介

CoCreator 是一个面向驾驶场景的**因果关系数据集构建框架**，通过三阶段流水线从原始驾驶数据自动构建高质量因果推理数据集：

1. **detect** — 从位置/速度数据中检测异常事件（急刹、加速、转向）
2. **reason** — 两阶段 VLM 分析（历史帧预测 → 未来帧验证）生成因果链
3. **pack** — 将因果链打包为 HuggingFace 兼容的训练数据集 + HTML 报告

## 快速开始

### 安装依赖

```bash
uv sync
```

### 配置环境

创建 `config.yaml`（参考 `config.example.yaml`，或直接从环境变量配置）：

```yaml
vlm:
  base_url: "https://api.siliconflow.cn"
  api_key: "${SILICONFLOW_API_KEY}"
  model: "Qwen/Qwen3.5-397B-A17B"
  timeout: 120.0

rate_limit:
  rpm: 500
  tpm: 2000000
  concurrency: 20

pipeline:
  dataset_path: "/data/cocreator/action_info"
  videos_path: "/data/cocreator/videos"
  output_dir: "./output"
  history_frames: 7
  future_frames: 11
  anomaly_threshold: 2.0
  steering_threshold: 15.0
  min_event_interval: 5
  merge_adjacent_events: true
  retry_max_attempts: 3
  retry_backoff_factor: 2.0
```

环境变量 `${ENV_VAR}` 语法在配置文件中所有字段均支持。

### 完整流水线

```bash
# 1. 事件检测
cocreator detect -c config.yaml

# 2. 因果推理
cocreator reason -c config.yaml

# 3. 打包数据集 + 浏览报告
cocreator pack -c config.yaml
cocreator pack review -c config.yaml   # 单独重新生成 review.html
```

## 使用指南

### detect — 事件检测

从驾驶数据中检测异常事件，输出 JSON 到 `{output_dir}/events/`。

```bash
cocreator detect -c config.yaml
cocreator detect -c config.yaml --episode-id ep001 --episode-id ep002
```

| 参数 | 说明 |
|------|------|
| `-c, --config` | YAML 配置文件路径 |
| `--episode-id` | 仅处理指定 episode（可重复） |

检测算法：基于速度和加速度的统计异常检测（自适应阈值），以及方向变化的转向检测。相邻事件可合并去重。

### reason — 因果推理

对检测到的事件进行两阶段 VLM 因果推理，输出 JSON 到 `{output_dir}/chains/`。

```bash
cocreator reason -c config.yaml
cocreator reason -c config.yaml --event-id ep001:0037 --event-id ep002:0051
cocreator reason -c config.yaml --no-resume   # 从头重新处理
```

| 参数 | 说明 |
|------|------|
| `-c, --config` | YAML 配置文件路径 |
| `--event-id` | 仅处理指定事件（`episode_id:frame_id` 格式，可重复） |
| `--resume` | 从上次进度继续（默认 true） |

两阶段分析：

1. **历史分析**：分析事件前的帧（含事件帧），预测自车行为
2. **未来确认**：分析事件后的帧，验证预测并生成因果描述

两阶段间严格数据隔离：第二阶段只看事件后帧，只接收第一阶段的文本摘要。

### pack — 打包数据集

将因果链打包为 HuggingFace 格式的训练数据集，输出到 `{output_dir}/dataset/`。

```bash
cocreator pack -c config.yaml
cocreator pack review -c config.yaml   # 重新生成 review.html
```

| 子命令 | 说明 |
|--------|------|
| `pack`（默认） | 打包：拷贝帧图像、生成因果文本、生成 HuggingFace 加载器、生成 review.html |
| `pack review` | 单独重新生成 review.html（需已运行过 pack） |

输出结构：

```
{output_dir}/dataset/
├── videos/{sample_id}/   # 按顺序编号的帧图像（01.jpg, 02.jpg, ...）
├── causal/{sample_id}.txt  # 因果描述文本
├── meta.json              # 样本元数据（事件帧位置、action_type 等）
├── review.html            # 可浏览的 HTML 报告
├── cocreator-dataset.py   # HuggingFace datasets 加载器
└── README.md
```

## 配置说明

### VLM 配置 (`vlm`)

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `base_url` | `https://api.siliconflow.cn` | API 地址（/v1 后缀自动补全） |
| `api_key` | 空 | API 密钥，支持 `${ENV_VAR}` |
| `model` | `Qwen/Qwen3.5-397B-A17B` | 模型名称 |
| `timeout` | 120.0 | 请求超时（秒） |

兼容任何 OpenAI 兼容 API（SiliconFlow、Ollama、Azure OpenAI 等）。

### 限流配置 (`rate_limit`)

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `rpm` | 500 | 每分钟请求数限制 |
| `tpm` | 2000000 | 每分钟 token 数限制 |
| `concurrency` | 20 | 最大并发请求数 |

### 流水线配置 (`pipeline`)

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `dataset_path` | 空 | action_info 目录（位置数据，用于事件检测） |
| `videos_path` | 空 | 视频帧目录（JPEG 图像） |
| `output_dir` | `./output` | 所有输出文件的根目录 |
| `history_frames` | 7 | 事件前（含事件帧）提取的帧数 |
| `future_frames` | 11 | 事件后提取的帧数 |
| `anomaly_threshold` | 2.0 | 异常检测的标准差倍数 |
| `steering_threshold` | 15.0 | 转向检测角度阈值（度） |
| `min_event_interval` | 5 | 事件间最小帧间隔（去重） |
| `merge_adjacent_events` | true | 是否合并相邻事件 |
| `retry_max_attempts` | 3 | API 调用最大重试次数 |
| `retry_backoff_factor` | 2.0 | 指数退避因子 |

## 输出格式

### 事件格式 (JSON)

`{output_dir}/events/{episode_id}_{frame_id}.json`：

```json
{
  "episode_id": "ep001",
  "frame_id": "0037_position_at_current_camera",
  "action_type": "hard_brake"
}
```

### 因果链格式 (JSON)

`{output_dir}/chains/{episode_id}_{event_frame_id}.json`：

```json
{
  "episode_id": "ep001",
  "event_frame_id": "0037_position_at_current_camera",
  "frame_ids": ["0032", "0034", "0036", "0038", "0040", "0042"],
  "causal_text": "The ego vehicle was cruising at a moderate speed when..."
}
```

`frame_ids` 按时间序排列：历史帧（含事件帧）在前，未来帧在后。

## 技术架构

```
src/cocreator/
├── cli.py                       # 3 CLI 命令：detect, reason, pack
├── config.py                    # YAML 加载 + ${ENV_VAR} 递归替换
├── schemas.py                   # Pydantic 数据模型
├── pipeline/
│   ├── detector.py              # 速度/加速度/转向异常检测
│   ├── extractor.py             # 视频帧提取（严格时间隔离）
│   ├── reasoner.py              # 两阶段 VLM 因果推理
│   └── progress_tracker.py      # 断点续跑（原子写入）
├── providers/
│   └── openai_compatible.py     # 异步 VLM 客户端（信号量 + 重试）
└── prompts/
    └── __init__.py              # Jinja2 模板（遗留，reasoner 未使用）
```
