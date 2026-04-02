# Pipeline Module

**Parent:** `src/cocreator/`

## OVERVIEW
Event detection + causal reasoning. Core business logic with strict data isolation.

## STRUCTURE
```
pipeline/
├── detector.py        # Velocity anomaly detection
├── extractor.py       # Frame extraction (Protocol + impl)
├── reasoner.py        # 2-stage VLM causal reasoning
└── progress_tracker.py  # JSON resume capability
```

## WHERE TO LOOK
| Task | Location | Key Function |
|------|----------|-------------|
| Anomaly detection | `detector.py:26` | `EventDetector.detect()` |
| History frames | `extractor.py:92` | `get_history_frames()` |
| Future frames | `extractor.py:136` | `get_future_frames()` |
| VLM reasoning | `reasoner.py:49` | `CausalReasoner.reason()` |
| Data isolation | `extractor.py:180` | `_validate_no_leakage()` |

## CONVENTIONS
- **Data isolation STRICT**: `_analyze_history()` only sees past frames; `_confirm_future()` only sees future frames
- **No event frame in history**: `get_history_frames()` excludes event frame
- **No event frame in future**: `get_future_frames()` excludes event frame  
- **Async**: All VLM calls use `async/await`
- **Semaphore concurrency**: `OpenAICompatibleProvider` uses `asyncio.Semaphore`

## ANTI-PATTERNS (THIS MODULE)
- **NEVER** pass history frames to `_confirm_future()` — data isolation violation
- **NEVER** include event frame itself in history or future analysis
- **NEVER** remove or bypass `_validate_no_leakage()` assertions

## UNIQUE STYLES
- **Protocol pattern**: `FrameExtractor` is a Protocol (extractor.py:13) — abstraction for different implementations
- **Assertion-based validation**: `_validate_no_leakage()` uses assert statements (not exceptions) for internal invariants
- **Two-stage VLM**: History analysis (predict) → Future confirmation (verify) — stages CANNOT share context
- **JSONL progress**: `ProgressTracker` persists to JSON for resume after interruption
