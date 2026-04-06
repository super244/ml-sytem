#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_CONFIG="${REPO_ROOT}/training/configs/profiles/local_metal.yaml"
DEFAULT_DATA_CONFIG="${REPO_ROOT}/data/configs/processing.yaml"
DEFAULT_VENV_DIR="${REPO_ROOT}/.venv-macos"
DEFAULT_ARTIFACTS_DIR="${REPO_ROOT}/artifacts"
DEFAULT_HF_HOME_DIR="${REPO_ROOT}/.cache/huggingface"

CONFIG_PATH="${DEFAULT_CONFIG}"
DATA_CONFIG_PATH="${DEFAULT_DATA_CONFIG}"
VENV_DIR="${DEFAULT_VENV_DIR}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${DEFAULT_ARTIFACTS_DIR}}"
HF_HOME_DIR="${HF_HOME:-${DEFAULT_HF_HOME_DIR}}"
TOKENIZER_OUTPUT_DIR=""
DO_PREPARE_DATA=auto
DO_TRAIN_TOKENIZER=auto
DO_DOCTOR=1
DO_PREFLIGHT=1
DO_DRY_RUN=1
DO_REAL_TRAIN=0
VALIDATE_MODEL_LOAD=0
INSTALL_FRONTEND=0
FORCE_REAL_TRAIN=0

log() {
  printf '[macos-start] %s\n' "$*"
}

warn() {
  printf '[macos-start] warning: %s\n' "$*" >&2
}

die() {
  printf '[macos-start] error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: scripts/start-mac.sh [options]

macOS Apple Silicon bootstrap for AI-Factory local setup and dry-run training.

Options:
  --config PATH              Training config to use (default: ${DEFAULT_CONFIG})
  --data-config PATH         Dataset processing config (default: ${DEFAULT_DATA_CONFIG})
  --venv-dir PATH            Local virtualenv path (default: ${DEFAULT_VENV_DIR})
  --prepare-data             Run dataset processing first
  --train-tokenizer          Train the tokenizer before launching training
  --tokenizer-output-dir DIR  Override tokenizer output directory
  --train                    Attempt a real training run instead of dry-run
  --dry-run                  Force dry-run mode (default)
  --validate-model-load      Load the model during dry-run validation
  --install-frontend         Install frontend Node dependencies if the frontend exists
  --skip-doctor              Skip scripts/doctor.py after setup
  --skip-preflight           Skip ai-factory train-preflight
  --force-real-train         Allow real training even on macOS
  --help                     Show this help text
EOF
}

have() {
  command -v "$1" >/dev/null 2>&1
}

python_version_ok() {
  local python_bin="$1"
  "$python_bin" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
}

get_config_value() {
  local query="$1"
  "$PYTHON_BIN" - "$CONFIG_PATH" "$query" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
query = sys.argv[2]
payload = yaml.safe_load(config_path.read_text()) or {}
current = payload
for part in query.split("."):
    if not isinstance(current, dict):
        current = None
        break
    current = current.get(part)
if isinstance(current, bool):
    print("true" if current else "false")
elif current is None:
    print("")
else:
    print(current)
PY
}

ensure_macos() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    die "This launcher is macOS-only."
  fi
}

ensure_system_tools() {
  if have xcode-select; then
    if ! xcode-select -p >/dev/null 2>&1; then
      warn "Xcode Command Line Tools are not installed. Some packages may fail to build."
      warn "Install them with: xcode-select --install"
    fi
  fi

  if ! have brew; then
    warn "Homebrew is not installed. Skipping automatic system package bootstrap."
    return
  fi

  if ! have git-lfs; then
    log "Installing git-lfs via Homebrew."
    brew install git-lfs
  fi

  if ! have python3 || ! python_version_ok "$(command -v python3)"; then
    log "Installing Python 3.12 via Homebrew."
    brew install python@3.12
  fi
}

resolve_python() {
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    PYTHON_BIN="${VENV_DIR}/bin/python"
    return
  fi

  if have python3 && python_version_ok "$(command -v python3)"; then
    PYTHON_BIN="$(command -v python3)"
    return
  fi

  if have python && python_version_ok "$(command -v python)"; then
    PYTHON_BIN="$(command -v python)"
    return
  fi

  if have brew; then
    local brew_python
    brew_python="$(brew --prefix python@3.12 2>/dev/null || brew --prefix python@3.11 2>/dev/null || true)/bin/python3"
    if [[ -x "${brew_python}" ]]; then
      PYTHON_BIN="${brew_python}"
      return
    fi
  fi

  die "Could not find a Python 3.11+ interpreter."
}

bootstrap_venv() {
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    log "Creating virtual environment at ${VENV_DIR}."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
  PYTHON_BIN="${VENV_DIR}/bin/python"
  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
}

install_python_deps() {
  log "Installing macOS-compatible repo dependencies."
  if ! "${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}[dev,macos]"; then
    warn "The macOS extras install did not complete cleanly. Retrying with the base editable install."
    "${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}[dev]"
  fi
}

install_frontend_deps() {
  if [[ ! -f "${REPO_ROOT}/frontend/package.json" ]]; then
    return
  fi
  if ! have npm; then
    warn "npm is not installed; skipping frontend dependency installation."
    return
  fi
  log "Installing frontend dependencies."
  npm install --prefix "${REPO_ROOT}/frontend"
}

maybe_print_runtime_summary() {
  local model_init
  model_init="$(get_config_value "model.initialization")"
  local use_4bit use_8bit use_full_precision
  use_4bit="$(get_config_value "model.use_4bit")"
  use_8bit="$(get_config_value "model.use_8bit")"
  use_full_precision="$(get_config_value "model.use_full_precision")"
  log "Config summary: initialization=${model_init:-unknown}, 4bit=${use_4bit:-false}, 8bit=${use_8bit:-false}, full_precision=${use_full_precision:-false}"
}

maybe_train_tokenizer() {
  local tokenizer_path
  tokenizer_path="${TOKENIZER_OUTPUT_DIR}"
  if [[ -z "${tokenizer_path}" ]]; then
    tokenizer_path="$(get_config_value "model.tokenizer_path")"
  fi

  if [[ -z "${tokenizer_path}" ]]; then
    warn "No tokenizer path was configured or supplied; skipping tokenizer training."
    return
  fi

  if [[ -f "${tokenizer_path}/tokenizer.json" ]]; then
    log "Tokenizer already exists at ${tokenizer_path}; skipping tokenizer training."
    return
  fi

  log "Training tokenizer into ${tokenizer_path}."
  "${PYTHON_BIN}" "${REPO_ROOT}/training/scripts/train_tokenizer.py" \
    --config "${CONFIG_PATH}" \
    --output-dir "${tokenizer_path}"
}

maybe_prepare_data() {
  log "Preparing processed datasets from ${DATA_CONFIG_PATH}."
  "${PYTHON_BIN}" "${REPO_ROOT}/data/prepare_dataset.py" \
    --config "${DATA_CONFIG_PATH}" \
    --source-load-workers "${SOURCE_LOAD_WORKERS:-4}"
}

run_doctor() {
  log "Running repo health checks."
  "${PYTHON_BIN}" -m ai_factory.cli ready --root "${REPO_ROOT}"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/doctor.py"
}

run_preflight() {
  log "Running training preflight."
  "${PYTHON_BIN}" -m ai_factory.cli train-preflight --config "${CONFIG_PATH}"
}

probe_titan() {
  if have cargo; then
    log "Building Titan engine with ultimate optimization features."
    # Build with ultimate feature that includes metal, cuda, and cpp
    cargo build --manifest-path "${REPO_ROOT}/ai_factory_titan/Cargo.toml" --features ultimate --release 2>/dev/null || \
      cargo build --manifest-path "${REPO_ROOT}/ai_factory_titan/Cargo.toml" --features metal,cpp --release 2>/dev/null || \
      cargo build --manifest-path "${REPO_ROOT}/ai_factory_titan/Cargo.toml" --features metal,cpp 2>/dev/null || true
  fi
  log "Running Titan hardware probe with ultimate optimization detection."
  "${PYTHON_BIN}" -m ai_factory.cli titan status --write-hardware-doc || true
  
  # Run the new hardware detection and optimization layer
  log "Detecting hardware capabilities for ultimate optimization."
  "${PYTHON_BIN}" -m training.src.optimization || true
}

run_training() {
  local -a args=("${PYTHON_BIN}" -m training.train --config "${CONFIG_PATH}")
  if [[ "${DO_DRY_RUN}" -eq 1 ]]; then
    args+=(--dry-run)
  fi
  if [[ "${VALIDATE_MODEL_LOAD}" -eq 1 ]]; then
    args+=(--validate-model-load)
  fi

  local model_init use_4bit use_8bit use_full_precision
  model_init="$(get_config_value "model.initialization")"
  use_4bit="$(get_config_value "model.use_4bit")"
  use_8bit="$(get_config_value "model.use_8bit")"
  use_full_precision="$(get_config_value "model.use_full_precision")"

  if [[ "$(uname -s)" == "Darwin" && "${DO_REAL_TRAIN}" -eq 1 ]]; then
    if [[ "${FORCE_REAL_TRAIN}" -ne 1 ]]; then
      if [[ "${use_4bit}" == "true" || "${use_8bit}" == "true" ]]; then
        warn "macOS does not support the repo's CUDA-only quantized training path."
        warn "Falling back to dry-run. Use --force-real-train only if you have a custom compatible config."
        args=( "${PYTHON_BIN}" -m training.train --config "${CONFIG_PATH}" --dry-run )
        if [[ "${VALIDATE_MODEL_LOAD}" -eq 1 ]]; then
          args+=(--validate-model-load)
        fi
        DO_DRY_RUN=1
      elif [[ "${model_init}" == "scratch" ]]; then
        warn "Scratch training on macOS is not wired to an MLX backend in this repo."
        warn "Falling back to dry-run. Use --force-real-train only if you have a custom compatible config."
        args=( "${PYTHON_BIN}" -m training.train --config "${CONFIG_PATH}" --dry-run )
        if [[ "${VALIDATE_MODEL_LOAD}" -eq 1 ]]; then
          args+=(--validate-model-load)
        fi
        DO_DRY_RUN=1
      fi
    fi
  fi

  if [[ "${DO_DRY_RUN}" -eq 1 ]]; then
    log "Launching training in dry-run mode."
  else
    log "Launching real training."
  fi
  "${args[@]}"
}

main() {
  for arg in "$@"; do
    case "$arg" in
      --help|-h)
        usage
        exit 0
        ;;
    esac
  done

  ensure_macos

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config)
        CONFIG_PATH="${2:?missing value for --config}"
        shift 2
        ;;
      --data-config)
        DATA_CONFIG_PATH="${2:?missing value for --data-config}"
        shift 2
        ;;
      --venv-dir)
        VENV_DIR="${2:?missing value for --venv-dir}"
        shift 2
        ;;
      --prepare-data)
        DO_PREPARE_DATA=1
        shift
        ;;
      --train-tokenizer)
        DO_TRAIN_TOKENIZER=1
        shift
        ;;
      --tokenizer-output-dir)
        TOKENIZER_OUTPUT_DIR="${2:?missing value for --tokenizer-output-dir}"
        shift 2
        ;;
      --train)
        DO_REAL_TRAIN=1
        DO_DRY_RUN=0
        shift
        ;;
      --dry-run)
        DO_DRY_RUN=1
        DO_REAL_TRAIN=0
        shift
        ;;
      --validate-model-load)
        VALIDATE_MODEL_LOAD=1
        shift
        ;;
      --install-frontend)
        INSTALL_FRONTEND=1
        shift
        ;;
      --skip-doctor)
        DO_DOCTOR=0
        shift
        ;;
      --skip-preflight)
        DO_PREFLIGHT=0
        shift
        ;;
      --force-real-train)
        FORCE_REAL_TRAIN=1
        shift
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done

  ensure_system_tools
  resolve_python
  bootstrap_venv
  install_python_deps

  export AI_FACTORY_REPO_ROOT="${REPO_ROOT}"
  export ARTIFACTS_DIR="${ARTIFACTS_DIR}"
  export HF_HOME="${HF_HOME_DIR}"
  export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.85}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

  mkdir -p "${ARTIFACTS_DIR}" "${HF_HOME_DIR}"

  if [[ "${INSTALL_FRONTEND}" -eq 1 ]]; then
    install_frontend_deps
  fi

  maybe_print_runtime_summary
  probe_titan
  
  # Check if using ultimate optimization profile
  if [[ "${CONFIG_PATH}" == *"ultimate"* ]]; then
    log "Ultimate optimization profile detected. Running performance benchmark."
    "${PYTHON_BIN}" -c "from training.src.ultimate_harness import quick_benchmark; quick_benchmark()" || true
  fi

  if [[ "${DO_PREPARE_DATA}" == "auto" ]]; then
    if [[ ! -f "${REPO_ROOT}/data/processed/manifest.json" || ! -f "${REPO_ROOT}/data/processed/corpus.sqlite" ]]; then
      DO_PREPARE_DATA=1
    else
      DO_PREPARE_DATA=0
    fi
  fi

  if [[ "${DO_PREPARE_DATA}" -eq 1 ]]; then
    maybe_prepare_data
  fi

  if [[ "${DO_TRAIN_TOKENIZER}" == "auto" ]]; then
    if [[ "$(get_config_value "model.initialization")" == "scratch" ]]; then
      DO_TRAIN_TOKENIZER=1
    else
      DO_TRAIN_TOKENIZER=0
    fi
  fi

  if [[ "${DO_TRAIN_TOKENIZER}" -eq 1 ]]; then
    maybe_train_tokenizer
  fi

  if [[ "${DO_DOCTOR}" -eq 1 ]]; then
    run_doctor
  fi

  if [[ "${DO_PREFLIGHT}" -eq 1 ]]; then
    run_preflight
  fi

  run_training

  log "Mac local bootstrap complete."
}

main "$@"
