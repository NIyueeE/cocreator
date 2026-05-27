# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CoCreator is a driving scene event detection and causal inference tool. Three-stage pipeline:

1. **detect** — anomaly detection on position/velocity data (hard brake, acceleration, steering)
2. **reason** — two-stage VLM analysis (history predict → future verify) producing causal chains
3. **pack** — Pack causal chains into a structured training dataset (frame images + causal text + HTML report view)

## Commands

| Command | Description |
|---------|-------------|
| `uv sync` | Install dependencies |
| `pytest` | Run unit tests (skips smoke/integration) |
| `pytest tests/test_foo.py::test_bar -xvs` | Run a single test verbosely |
| `pytest --run-smoke` | Include integration smoke tests |
| `ruff format .` | Format code |
| `ruff check .` | Lint code |
| `ruff check --fix .` | Lint and auto-fix |
| `ty check src/cocreator/` | Type-check the package |
| `cocreator --help` | Show CLI help (entry point: `cocreator = "cocreator.cli:app"`) |
| `cocreator detect -c config.yaml` | Run event detection |
| `cocreator reason -c config.yaml` | Run causal reasoning |
| `cocreator pack -c config.yaml` | Pack chains into training dataset |
| `cocreator pack review -c config.yaml` | Regenerate review HTML report from packed dataset |
| `cocreator pack convert -c config.yaml` | Convert packed dataset to Hugging Face Parquet format |

*Note: Prefix commands with `uv run` if the virtual environment is not active (e.g., `uv run pytest`).*

## Project Structure

```
src/cocreator/
├── __init__.py
├── cli.py                       # 3 CLI commands: detect, reason, pack
├── config.py                    # YAML loading + recursive ${ENV_VAR} substitution
├── schemas.py                   # All pydantic models (extra="forbid" on output schemas)
├── pipeline/
│   ├── detector.py              # EventDetector: velocity/acceleration/steering anomaly detection
│   ├── extractor.py             # VideoFrameExtractor + FrameExtractor Protocol
│   ├── reasoner.py              # CausalReasoner: 2-stage VLM analysis with strict data isolation
│   └── progress_tracker.py      # ProgressTracker: JSON resume (atomic write via .tmp+rename)
├── providers/
│   └── openai_compatible.py     # Async VLM client with semaphore + retry_with_backoff decorator
tests/
├── conftest.py                  # --run-smoke marker: skip smoke tests by default
├── test_detector.py             # EventDetector unit tests (tmp_path fixtures)
├── test_extractor.py            # VideoFrameExtractor tests
├── test_reasoner.py             # CausalReasoner tests (mock provider)
├── test_schemas.py              # Pydantic model validation tests
├── test_progress.py             # ProgressTracker tests
├── test_provider.py             # OpenAICompatibleProvider tests
├── test_config_loading.py       # Config loading tests
└── test_integration.py          # Full pipeline smoke tests (marked @pytest.mark.smoke)
```

## Key Design Decisions

### Strict Temporal Data Isolation
History analysis gets only **pre-event** frames; future verification gets only **post-event** frames. `_validate_no_leakage()` uses `assert` (internal invariant, never remove). Stage 2 receives only the text summary from Stage 1, never frames.

### Two-Stage VLM Reasoning
1. Analyze history frames → predict ego action
2. Analyze future frames → verify prediction

### Async Concurrency
`OpenAICompatibleProvider` uses `asyncio.Semaphore` for concurrency control. `retry_with_backoff` decorator (3 attempts, exponential backoff). All VLM calls are `async/await`. CLI wraps the async pipeline with `asyncio.run()` since the entry point is synchronous.

### Config
YAML supports `${ENV_VAR}` substitution. Works with any OpenAI-compatible API (SiliconFlow, Ollama, Azure OpenAI, etc.). Config provider appends `/v1` suffix automatically if missing.

## Important Notes

- PipelineConfig uses `history_frames`/`future_frames`
- `DetectedEvent` and `CausalChain` have `extra="forbid"` — no extra fields allowed
- Response format uses OpenAI SDK `json_schema` structured outputs (no manual parsing)
- Provider disables HTTP connection pooling (`max_keepalive_connections=0`) to avoid pool-related hangs
- Progress tracker writes to `.tmp` then renames for crash-safe persistence. Same pattern used for event/chain file output in CLI
- CLI event/chain output uses atomic `.tmp` + rename writes for crash safety (same pattern as progress tracker)
- Detector reads position data from `*_position_*.txt` files in `{dataset_path}/{episode_id}/`, parsing 3D vectors from bracket-delimited format like `[x y z]`
- Extractor reads JPEG frames from `{videos_path}/{episode_id}/` and matches by numeric prefix in filename
- Detect command uses `concurrent.futures.ThreadPoolExecutor` (CPU-bound); reason command uses `asyncio` with semaphore (I/O-bound VLM calls)
- `pack` command copies frame images + causal text + generates `review.html` for browsing (no HuggingFace loader script — the prebuilt dataset is on HuggingFace at `NIyueeE/cocreator-driving-scene`)
- `httpx[socks]` dependency enables proxy support for API calls
