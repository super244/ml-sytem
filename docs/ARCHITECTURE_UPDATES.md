# AI-Factory Architecture Updates (April 2026)

This document summarizes the changes made to the platform to improve stability, responsiveness, and hardware support.

## 1. Resolved Blocking GPU Metrics Collection
- **Issue**: The `_gpu_snapshot()` function in `ai_factory/core/monitoring/collectors.py` was calling `nvidia-smi` without a timeout and without checking if it exists. On non-NVIDIA systems (like Apple Silicon), this caused the entire API (instances listing) to hang.
- **Fix**:
    - Added `shutil.which("nvidia-smi")` as a lazily-determined availability check.
    - Implemented a **3-layer defense**:
        1. **Pre-check**: Skip `subprocess` entirely if `nvidia-smi` is missing.
        2. **TTL Cache**: Results are cached for 30 seconds to prevent redundant subprocess spawns.
        3. **Strict Timeout**: A 2.0s timeout ensures the API remains responsive even if `nvidia-smi` hangs.

## 2. Integrated `ai_factory_titan` Telemetry
- **Enhancement**: Added a native telemetry bridge to the `ai-factory-titan` Rust core for hardware status on non-NVIDIA systems.
- **Implementation**:
    - Added `_titan_snapshot()` to `collectors.py` as a fallback when `nvidia-smi` is unavailable.
    - Captures silicon type (e.g., Apple M1/M2/M3), unified memory status, and backend capabilities (Metal/CUDA/CPU).

## 3. Strict Typing Compliance (Mypy)
- **Status**: Reached 100% strict compliance in `ai_factory/core/execution/`.
- **Changes**:
    - Fixed 8 `mypy --strict` errors in `ssh.py` and `local.py`.
    - Standardized type annotations for all internal execution runners and pipe-streaming loops.

## 4. Frontend Workspace Refinements
- **Overview**: Verified and polished the `/solve` and `/compare` workspaces in the Next.js frontend.
- **Features**:
    - Support for workspace density switching (Compact/Balanced/Expanded).
    - Integrated model comparison in the control rail.
    - Real-time candidate inspection and verifier-aware reasoning visibility.

## 5. Bootstrap-First Training Workflow
- **Overview**: Added platform-specific Linux cloud and macOS local start scripts so the training workflow can bootstrap dependencies, tokenizer assets, and runtime checks automatically.
- **Data Path**: The corpus preparation and tokenization flow is tuned to reduce repeated work and shorten the wait before a training run.
- **Runtime**: Titan hardware reporting now makes CUDA, Metal, and CPU fallback capability easier to verify before launch.

---
**Status**: The platform is now fully responsive across heterogeneous hardware and maintains an immaculate codebase.
