# Changelog

## Unreleased

### Core Infrastructure
- **Async Utilities** (`ai_factory/core/async_utils.py`): Added async primitives for concurrent operations, rate limiting, and backpressure.
- **Cache Layer** (`ai_factory/core/cache/`): Implemented memory-backed caching with LRU eviction and disk persistence.
- **Distributed Compute** (`ai_factory/core/distributed/`): Added worker registration, job dispatch, and result aggregation primitives.
- **Security Module** (`ai_factory/core/security/`): Implemented credential rotation, encrypted artifact storage, and access controls.

### Orchestration Enhancements
- **Service Layer** (`ai_factory/core/orchestration/service.py`): Added circuit breaker patterns, lease-based task ownership, retry policies, and real-time heartbeat monitoring.
- **Metrics Collection** (`ai_factory/platform/monitoring/metrics.py`): Enhanced with system-level, training job, and inference metrics plus historical queries.

### Platform Updates
- **Monitoring Alerts** (`ai_factory/platform/monitoring/alerts.py`): Added alert routing and anomaly detection.
- **Hardware Monitoring** (`ai_factory/platform/monitoring/hardware.py`): GPU/VRAM utilization and temperature tracking.

### Configuration & Tooling
- Updated `.gitignore` with explicit cache directory exclusions (`__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `.claude`)
- Added Docker build exclusions for cache directories
- Consolidated Python tooling config in `pyproject.toml`
- Added GitHub Actions checks for pytest plus changed-file Ruff and mypy validation

### Repository Cleanup
- Cache directories now properly ignored in CI/CD
- `data/processed/*.jsonl` and `data/processed/*.json` tracked but excluded from caches
- `.claude/` session data excluded from version control

### Breaking Changes
- None. All new modules are additive.

