#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "[cloud-start] failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

timestamp() {
  date +"%Y%m%d-%H%M%S"
}

log() {
  printf '[cloud-start] %s\n' "$*"
}

warn() {
  printf '[cloud-start] WARN: %s\n' "$*" >&2
}

die() {
  printf '[cloud-start] ERROR: %s\n' "$*" >&2
  exit 1
}

run() {
  log "Running: $*"
  "$@"
}

retry() {
  local attempts="$1"
  shift
  local delay=3
  local n=1
  until "$@"; do
    if (( n >= attempts )); then
      return 1
    fi
    warn "Command failed (attempt ${n}/${attempts}). Retrying in ${delay}s: $*"
    sleep "${delay}"
    n=$((n + 1))
    delay=$((delay * 2))
  done
}

usage() {
  cat <<'EOF'
Usage: scripts/start-linux.sh [options]

Options:
  --config PATH          Training profile to launch
  --prepare-data         Force dataset preparation
  --skip-data            Skip dataset preparation
  --prepare-tokenizer    Force tokenizer build
  --skip-tokenizer       Skip tokenizer build
  --dry-run              Run training dry-run before launch
  --no-dry-run           Skip the dry-run gate
  --train                Launch training after setup
  --no-train             Stop after setup/preflight
  --frontend             Install frontend deps if frontend/package.json exists
  --skip-frontend        Skip frontend deps
  --force-data           Rebuild data even if processed artifacts exist
  --force-tokenizer      Rebuild tokenizer even if the target files exist
  --force-torch-cuda     Reinstall torch from the CUDA wheel index if needed
  --train-args "..."     Extra arguments appended to the training command
  --help                 Show this help

Environment defaults:
  AI_FACTORY_PROFILE=training/configs/profiles/pretraining.yaml
  AI_FACTORY_PREPARE_DATA=1
  AI_FACTORY_PREPARE_TOKENIZER=auto
  AI_FACTORY_RUN_DRY_RUN=1
  AI_FACTORY_LAUNCH_TRAINING=1
  AI_FACTORY_INSTALL_DEV=1
  AI_FACTORY_INSTALL_FRONTEND=0
  AI_FACTORY_FORCE_REBUILD_DATA=0
  AI_FACTORY_FORCE_REBUILD_TOKENIZER=0
  AI_FACTORY_FORCE_TORCH_CUDA=auto
EOF
}

TRAIN_CONFIG="${AI_FACTORY_PROFILE:-training/configs/profiles/pretraining.yaml}"
PREPARE_DATA="${AI_FACTORY_PREPARE_DATA:-1}"
PREPARE_TOKENIZER="${AI_FACTORY_PREPARE_TOKENIZER:-auto}"
RUN_DRY_RUN="${AI_FACTORY_RUN_DRY_RUN:-1}"
LAUNCH_TRAINING="${AI_FACTORY_LAUNCH_TRAINING:-1}"
INSTALL_DEV="${AI_FACTORY_INSTALL_DEV:-1}"
INSTALL_FRONTEND="${AI_FACTORY_INSTALL_FRONTEND:-0}"
FORCE_REBUILD_DATA="${AI_FACTORY_FORCE_REBUILD_DATA:-0}"
FORCE_REBUILD_TOKENIZER="${AI_FACTORY_FORCE_REBUILD_TOKENIZER:-0}"
FORCE_TORCH_CUDA="${AI_FACTORY_FORCE_TORCH_CUDA:-auto}"
TRAIN_ARGS="${AI_FACTORY_TRAIN_ARGS:-}"
SYSTEM_DEPS="${AI_FACTORY_INSTALL_SYSTEM_DEPS:-auto}"
VENV_DIR="${AI_FACTORY_VENV_DIR:-${ROOT_DIR}/.venv}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${ROOT_DIR}/artifacts}"
export AI_FACTORY_REPO_ROOT="${AI_FACTORY_REPO_ROOT:-${ROOT_DIR}}"
export ARTIFACTS_DIR

while (($#)); do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || die "--config requires a value"
      TRAIN_CONFIG="$2"
      shift 2
      ;;
    --prepare-data)
      PREPARE_DATA=1
      shift
      ;;
    --skip-data)
      PREPARE_DATA=0
      shift
      ;;
    --prepare-tokenizer)
      PREPARE_TOKENIZER=1
      shift
      ;;
    --skip-tokenizer)
      PREPARE_TOKENIZER=0
      shift
      ;;
    --dry-run)
      RUN_DRY_RUN=1
      shift
      ;;
    --no-dry-run)
      RUN_DRY_RUN=0
      shift
      ;;
    --train)
      LAUNCH_TRAINING=1
      shift
      ;;
    --no-train)
      LAUNCH_TRAINING=0
      shift
      ;;
    --frontend)
      INSTALL_FRONTEND=1
      shift
      ;;
    --skip-frontend)
      INSTALL_FRONTEND=0
      shift
      ;;
    --force-data)
      FORCE_REBUILD_DATA=1
      shift
      ;;
    --force-tokenizer)
      FORCE_REBUILD_TOKENIZER=1
      shift
      ;;
    --force-torch-cuda)
      FORCE_TORCH_CUDA=1
      shift
      ;;
    --train-args)
      [[ $# -ge 2 ]] || die "--train-args requires a value"
      TRAIN_ARGS="${TRAIN_ARGS} ${2}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/.cache}"
export TMPDIR="${TMPDIR:-${ARTIFACTS_DIR}/tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"

mkdir -p "${ARTIFACTS_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${TMPDIR}"
LOG_DIR="${ARTIFACTS_DIR}/bootstrap-logs/${RUN_ID:-$(timestamp)}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/cloud-start.log") 2>&1

log "Repo root: ${ROOT_DIR}"
log "Training config: ${TRAIN_CONFIG}"
log "Artifacts dir: ${ARTIFACTS_DIR}"
log "Virtualenv: ${VENV_DIR}"

[[ -f "${TRAIN_CONFIG}" ]] || die "Training config not found: ${TRAIN_CONFIG}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    die "Python 3 is required but was not found."
  fi
fi

install_system_dependencies() {
  if [[ "${SYSTEM_DEPS}" != "auto" && "${SYSTEM_DEPS}" != "1" ]]; then
    log "Skipping system package installation by request."
    return 0
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    warn "apt-get not available; skipping system package installation."
    return 0
  fi

  local apt_prefix=()
  if [[ "$(id -u)" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      apt_prefix=(sudo)
    else
      warn "No sudo found; skipping system package installation."
      return 0
    fi
  fi

  export DEBIAN_FRONTEND=noninteractive
  run "${apt_prefix[@]}" apt-get update
  run "${apt_prefix[@]}" apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    git \
    git-lfs \
    ninja-build \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-wheel
}

ensure_git_lfs() {
  if command -v git >/dev/null 2>&1 && git lfs version >/dev/null 2>&1; then
    run git lfs install --local
    retry 3 git lfs pull
  else
    warn "git-lfs not available yet; skipping LFS download."
  fi
}

ensure_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    log "Creating virtual environment at ${VENV_DIR}"
    run "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  export VIRTUAL_ENV="${VENV_DIR}"
  export PATH="${VENV_DIR}/bin:${PATH}"
}

install_python_deps() {
  run python -m pip install --upgrade pip setuptools wheel
  local pip_args=(-e ".")
  if [[ "${INSTALL_DEV}" == "1" ]]; then
    pip_args=(-e ".[dev,train-cuda]")
  else
    pip_args=(-e ".[train-cuda]")
  fi
  if [[ "${FORCE_TORCH_CUDA}" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
      export AI_FACTORY_USE_CUDA="${AI_FACTORY_USE_CUDA:-1}"
    else
      export AI_FACTORY_USE_CUDA="${AI_FACTORY_USE_CUDA:-0}"
    fi
  elif [[ "${FORCE_TORCH_CUDA}" == "1" ]]; then
    export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
    export AI_FACTORY_USE_CUDA=1
  fi
  run python -m pip install "${pip_args[@]}"
  run python -m pip check
}

ensure_cuda_health() {
  local torch_probe
  torch_probe="$("${PYTHON_BIN}" - <<'PY'
import json
try:
    import torch
    payload = {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
    }
except Exception as exc:
    payload = {"installed": False, "error": str(exc)}
print(json.dumps(payload))
PY
)"
  log "Torch probe: ${torch_probe}"

  if command -v nvidia-smi >/dev/null 2>&1; then
    run nvidia-smi
  elif [[ "${AI_FACTORY_USE_CUDA:-0}" == "1" ]]; then
    warn "CUDA was requested, but nvidia-smi was not found."
  fi

  if [[ "${FORCE_TORCH_CUDA}" == "1" ]]; then
    log "CUDA torch wheel was explicitly requested; reinstalling torch from the CUDA index."
    run python -m pip install --force-reinstall --no-cache-dir --extra-index-url "${PIP_EXTRA_INDEX_URL}" "torch>=2.5.1"
  fi
}

maybe_install_frontend() {
  if [[ "${INSTALL_FRONTEND}" != "1" ]]; then
    return 0
  fi
  if [[ ! -f "frontend/package.json" ]]; then
    warn "frontend/package.json not found; skipping frontend install."
    return 0
  fi
  if [[ -d "frontend/node_modules" ]]; then
    log "Frontend dependencies already present; skipping npm install."
    return 0
  fi
  if ! command -v npm >/dev/null 2>&1; then
    warn "npm not found; skipping frontend install."
    return 0
  fi
  run bash -lc 'cd frontend && npm install'
}

maybe_probe_titan() {
  if command -v cargo >/dev/null 2>&1; then
    log "Building Titan engine with ultimate optimization features."
    # Build with ultimate feature that includes metal, cuda, and cpp
    cargo build --manifest-path ai_factory_titan/Cargo.toml --features ultimate --release 2>/dev/null || \
      cargo build --manifest-path ai_factory_titan/Cargo.toml --features cuda,cpp --release 2>/dev/null || \
      cargo build --manifest-path ai_factory_titan/Cargo.toml --features cuda,cpp 2>/dev/null || true
  fi
  
  log "Running Titan hardware probe with ultimate optimization detection."
  run python -m ai_factory.cli titan status --write-hardware-doc || true
  
  # Run the new hardware detection and optimization layer
  log "Detecting hardware capabilities for ultimate optimization."
  run python -m training.src.optimization || true
}

maybe_prepare_data() {
  if [[ "${PREPARE_DATA}" != "1" ]]; then
    return 0
  fi
  if [[ "${FORCE_REBUILD_DATA}" != "1" ]] && [[ -f "data/processed/manifest.json" ]] && [[ -f "data/processed/corpus.sqlite" ]]; then
    log "Processed corpus already exists; skipping dataset rebuild."
    return 0
  fi
  if [[ ! -f "data/configs/processing.yaml" ]]; then
    warn "data/configs/processing.yaml not found; skipping dataset preparation."
    return 0
  fi
  run python data/prepare_dataset.py \
    --config data/configs/processing.yaml \
    --source-load-workers "${AI_FACTORY_SOURCE_LOAD_WORKERS:-6}"
}

read_training_config_value() {
  local expression="$1"
  "${PYTHON_BIN}" - <<PY
from pathlib import Path
import yaml
payload = yaml.safe_load(Path("${TRAIN_CONFIG}").read_text())
value = ${expression}
print("" if value is None else value)
PY
}

maybe_prepare_tokenizer() {
  local prepare_mode="${PREPARE_TOKENIZER}"
  if [[ "${prepare_mode}" == "auto" ]]; then
    local init_mode
    init_mode="$(read_training_config_value "payload.get('model', {}).get('initialization')")"
    if [[ "${init_mode}" == "scratch" ]]; then
      prepare_mode=1
    else
      prepare_mode=0
    fi
  fi

  if [[ "${prepare_mode}" != "1" ]]; then
    return 0
  fi

  local tokenizer_path
  tokenizer_path="$(read_training_config_value "payload.get('model', {}).get('tokenizer_path') or ''")"
  if [[ -z "${tokenizer_path}" ]]; then
    warn "Tokenizer path was not set in ${TRAIN_CONFIG}; skipping tokenizer build."
    return 0
  fi
  if [[ -f "${tokenizer_path}/tokenizer.json" && -f "${tokenizer_path}/tokenizer_config.json" && "${FORCE_REBUILD_TOKENIZER}" != "1" ]]; then
    log "Tokenizer already exists at ${tokenizer_path}; skipping tokenizer build."
    return 0
  fi
  run python training/scripts/train_tokenizer.py --config "${TRAIN_CONFIG}" --output-dir "${tokenizer_path}"
}

run_quality_checks() {
  run python -m ai_factory.cli ready --root "${ROOT_DIR}"
  run python scripts/doctor.py
  run python -m ai_factory.cli train-preflight --config "${TRAIN_CONFIG}"
}

launch_training() {
  # Check if using ultimate optimization profile and set appropriate env vars
  if [[ "${TRAIN_CONFIG}" == *"ultimate"* ]]; then
    log "Ultimate optimization profile detected."
    export AI_FACTORY_ULTIMATE_OPTIMIZATION=1
    
    # Run quick benchmark to verify optimization is working
    log "Running performance benchmark."
    run python -c "from training.src.ultimate_harness import quick_benchmark; quick_benchmark()" || true
  fi

  local train_cmd=(python -m training.train --config "${TRAIN_CONFIG}")
  if [[ "${RUN_DRY_RUN}" == "1" ]]; then
    run python -m training.train --config "${TRAIN_CONFIG}" --dry-run --validate-model-load
  fi
  if [[ -n "${TRAIN_ARGS// }" ]]; then
    # shellcheck disable=SC2206
    local extra_args=(${TRAIN_ARGS})
    train_cmd+=("${extra_args[@]}")
  fi
  run "${train_cmd[@]}"
}

install_system_dependencies
ensure_venv
ensure_git_lfs
install_python_deps
ensure_cuda_health
maybe_probe_titan
maybe_install_frontend
maybe_prepare_data
maybe_prepare_tokenizer
run_quality_checks
launch_training

log "Bootstrap and launch finished successfully."
