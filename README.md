# CoCreator

驾驶场景事件检测与因果推理工具。

## 项目简介

CoCreator 是一款基于视觉语言模型（VLM）的驾驶场景分析工具，通过三阶段流水线对驾驶数据进行分析：

1. **detect** — 从驾驶片段中识别异常事件（急刹车、加速、转向）
2. **reason** — 两阶段 VLM 分析（历史帧预测 → 未来帧验证），生成因果链
3. **pack** — 将因果链打包为 HuggingFace 兼容的训练数据集，附带 HTML 浏览报告

## 技术架构

```
cocreator/
├── src/cocreator/
│   ├── cli.py              # CLI 入口：detect, reason, pack
│   ├── config.py           # YAML 配置加载 + ${ENV_VAR} 替换
│   ├── schemas.py          # Pydantic 数据模型
│   ├── pipeline/
│   │   ├── detector.py     # 事件检测（速度/加速度/转向异常）
│   │   ├── reasoner.py     # 两阶段 VLM 因果推理（严格数据隔离）
│   │   ├── extractor.py    # 视频帧提取（时序隔离）
│   │   └── progress_tracker.py  # 断点续跑（原子写入）
│   ├── prompts/
│   │   └── __init__.py     # Jinja2 模板（遗留，reasoner 未使用）
│   └── providers/
│       └── openai_compatible.py  # 异步 VLM 客户端（信号量 + 指数退避重试）
└── tests/
```

## 快速开始

### 安装

```bash
uv sync
```

### 配置

编辑 `config.yaml`，关键参数：

```yaml
vlm:
  base_url: "https://api.siliconflow.cn"
  api_key: "${API_KEY}"          # 支持环境变量引用
  model: "Qwen/Qwen3.5-397B-A17B"

rate_limit:
  concurrency: 20                # VLM 请求并发数

pipeline:
  dataset_path: "/data/action_info"  # 位置数据目录
  videos_path: "/data/videos"        # 视频帧目录
  output_dir: "./output"
  history_frames: 7                  # 事件前历史帧数
  future_frames: 11                  # 事件后未来帧数
  anomaly_threshold: 2.0
  steering_threshold: 15.0
```

### 运行

```bash
# 1. 事件检测
cocreator detect -c config.yaml

# 2. 因果推理
cocreator reason -c config.yaml

# 3. 打包数据集
cocreator pack -c config.yaml

# 4. （可选）重新生成浏览报告
cocreator pack review -c config.yaml
```

## CLI 命令

### detect — 事件检测

从位置数据中检测异常驾驶事件（急刹车、加速、转向）。

```bash
cocreator detect -c config.yaml
cocreator detect -c config.yaml --episode-id ep001 --episode-id ep002
```

| 参数 | 说明 |
|------|------|
| `-c, --config` | YAML 配置文件路径 |
| `--episode-id` | 指定处理的 episode（可重复） |

输出：`{output_dir}/events/{episode_id}_{frame_id}.json`，每事件一个 JSON。

### reason — 因果推理

两阶段 VLM 分析：
1. **历史分析**：事件前帧 → 预测自车行为
2. **未来确认**：事件后帧 → 验证预测，生成因果描述

两阶段严格数据隔离：历史分析只能看到事件前帧（含事件帧），未来确认只能看到事件后帧。

```bash
cocreator reason -c config.yaml
cocreator reason -c config.yaml --event-id ep001:0037 --event-id ep002:0042
cocreator reason -c config.yaml --no-resume  # 从头开始
```

| 参数 | 说明 |
|------|------|
| `-c, --config` | YAML 配置文件路径 |
| `--event-id` | 指定处理的 event（`episode_id:frame_id`，可重复） |
| `--resume` | 从上次进度继续（默认 true） |

输出：`{output_dir}/chains/{episode_id}_{event_frame_id}.json`。

### pack — 数据集打包

将因果链打包为 HuggingFace `datasets` 兼容的训练数据集：

```bash
cocreator pack -c config.yaml              # 全量打包
cocreator pack review -c config.yaml        # 仅重新生成 review.html
```

从 `{output_dir}/chains/` 读取链数据，输出到 `{output_dir}/dataset/`：
- `videos/{sample_id:04d}/*.jpg` — 按时间序排列的帧（含事件帧）
- `causal/{sample_id:04d}.txt` — 因果描述文本
- `review.html` — 嵌入式图片的浏览报告（蓝色边框=历史帧，绿色边框=未来帧，EVENT 标记事件帧）
- `cocreator-dataset.py` — HuggingFace 数据集加载器

## 数据模型

### DetectedEvent

```json
{
  "episode_id": "ep001",
  "frame_id": "0037_position_at_current_camera",
  "action_type": "hard_brake"
}
```

### CausalChain

```json
{
  "episode_id": "ep001",
  "event_frame_id": "0037_position_at_current_camera",
  "frame_ids": ["0032", "0034", "0036", "0038", "0040"],
  "causal_text": "The ego vehicle was cruising... I predicted I would brake..."
}
```

## 关键设计

- **事件帧归属历史**：`get_history_frames` 包含事件帧本身（`num <= event_num`），VLM 能看到事件发生瞬间的上下文
- **严格时序隔离**：历史分析只能访问事件前帧（含事件帧），未来分析只能访问事件后帧。`_validate_no_leakage` 断言保证
- **异步并发**：`OpenAICompatibleProvider` 使用 `asyncio.Semaphore` 控制并发，`retry_with_backoff` 指数退避重试
- **原子写入**：所有输出文件先写 `.tmp` 再 `rename`，防止写入中断导致数据损坏
- **`history_frames`/`future_frames`**：PipelineConfig 使用直接帧数配置（非 segment 模式）
