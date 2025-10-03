#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
VENV_PATH="${PROJECT_ROOT}/.venv"

# Configuration (override via env vars)
PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_NAME="${ENV_NAME:-LunarLander-v2}"
PRESETS_CSV="${PRESETS:-vanilla,rtg,baseline,rtg_baseline}"
GAE_LAMBDA="${GAE_LAMBDA:-0.95}"
N_ITER="${N_ITER:-200}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-400}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1000}"
LEARNING_RATE="${LEARNING_RATE:-5e-3}"
DISCOUNT="${DISCOUNT:-0.99}"
NUM_AGENT_TRAIN_STEPS="${NUM_AGENT_TRAIN_STEPS:-1}"
N_LAYERS="${N_LAYERS:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
EP_LEN="${EP_LEN:-}"  # optional
ACTION_NOISE_STD="${ACTION_NOISE_STD:-0}"
SEED_START="${SEED:-1}"
EXP_NAME_PREFIX="${EXP_NAME_PREFIX:-pg}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
CUSTOM_FLAGS="${CUSTOM_FLAGS:-}"  # extra flags appended to every run

log() {
  echo "[run_hw2.sh] $*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "ERROR: Required command '$1' not found in PATH."
    exit 1
  fi
}

create_and_activate_venv() {
  if [[ ! -d "${VENV_PATH}" ]]; then
    log "Creating virtual environment at ${VENV_PATH} using ${PYTHON_BIN}."
    "${PYTHON_BIN}" -m venv "${VENV_PATH}"
  else
    log "Using existing virtual environment at ${VENV_PATH}."
  fi
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
  log "Virtual environment activated (python $(python --version 2>&1))."
}

install_dependencies() {
  if [[ "${SKIP_INSTALL}" == "1" ]]; then
    log "Skipping dependency installation (SKIP_INSTALL=1)."
    return
  fi
  log "Upgrading pip and wheel."
  pip install --upgrade pip wheel
  log "Installing homework requirements."
  pip install -r "${PROJECT_ROOT}/requirements.txt"
  log "Installing cs285 package in editable mode."
  pip install -e "${PROJECT_ROOT}"
}

build_flag_array() {
  local preset="$1"
  local -n __flags_ref=$2

  __flags_ref=()

  case "${preset}" in
    vanilla)
      :
      ;;
    rtg)
      __flags_ref+=("--reward_to_go")
      ;;
    baseline)
      __flags_ref+=("--nn_baseline")
      ;;
    rtg_baseline)
      __flags_ref+=("--reward_to_go" "--nn_baseline")
      ;;
    gae)
      __flags_ref+=("--reward_to_go" "--nn_baseline" "--gae_lambda" "${GAE_LAMBDA}")
      ;;
    no_std_adv)
      __flags_ref+=("--dont_standardize_advantages")
      ;;
    custom)
      if [[ -z "${CUSTOM_FLAGS}" ]]; then
        log "ERROR: PRESETS includes 'custom' but CUSTOM_FLAGS is empty."
        exit 1
      fi
      read -r -a __flags_ref <<< "${CUSTOM_FLAGS}"
      ;;
    *)
      log "ERROR: Unknown preset '${preset}'."
      log "       Supported presets: vanilla, rtg, baseline, rtg_baseline, gae, no_std_adv, custom."
      exit 1
      ;;
  esac
}

run_experiment() {
  local preset="$1"
  local run_idx="$2"
  local -a flag_array=()

  build_flag_array "${preset}" flag_array

  local exp_name="${EXP_NAME_PREFIX}_${preset}"
  local seed=$((SEED_START + run_idx))
  local cmd=(python "${PROJECT_ROOT}/cs285/scripts/run_hw2.py"
    --env_name "${ENV_NAME}"
    --exp_name "${exp_name}"
    --n_iter "${N_ITER}"
    --batch_size "${BATCH_SIZE}"
    --eval_batch_size "${EVAL_BATCH_SIZE}"
    --train_batch_size "${TRAIN_BATCH_SIZE}"
    --learning_rate "${LEARNING_RATE}"
    --discount "${DISCOUNT}"
    --num_agent_train_steps_per_iter "${NUM_AGENT_TRAIN_STEPS}"
    --n_layers "${N_LAYERS}"
    --size "${HIDDEN_SIZE}"
    --seed "${seed}"
    --action_noise_std "${ACTION_NOISE_STD}")

  if [[ -n "${EP_LEN}" ]]; then
    cmd+=(--ep_len "${EP_LEN}")
  fi

  cmd+=("${flag_array[@]}")

  if [[ -n "${CUSTOM_FLAGS}" && "${preset}" != "custom" ]]; then
    # Allow appending extra flags globally
    read -r -a extra_flags <<< "${CUSTOM_FLAGS}"
    cmd+=("${extra_flags[@]}")
  fi

  log "Running preset '${preset}' (seed=${seed})."
  log "Command: ${cmd[*]}"
  "${cmd[@]}"
}

main() {
  require_command "${PYTHON_BIN}"
  create_and_activate_venv
  install_dependencies

  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  log "Set MUJOCO_GL=${MUJOCO_GL}."

  IFS=',' read -r -a preset_list <<< "${PRESETS_CSV}"
  local idx=0
  for preset_raw in "${preset_list[@]}"; do
    local preset="${preset_raw// /}" # trim whitespace
    if [[ -z "${preset}" ]]; then
      continue
    fi
    run_experiment "${preset}" "${idx}"
    idx=$((idx + 1))
  done

  log "All requested experiments finished. Logs are under ${PROJECT_ROOT}/cs285/data."
}

main "$@"
