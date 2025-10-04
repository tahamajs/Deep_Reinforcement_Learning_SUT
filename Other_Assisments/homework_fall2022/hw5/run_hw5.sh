#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
VENV_PATH="${PROJECT_ROOT}/.venv"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
MUJOCO_GL="${MUJOCO_GL:-}"

RUN_EXPLORATION="${RUN_EXPLORATION:-1}"
RUN_AWAC="${RUN_AWAC:-1}"
RUN_IQL="${RUN_IQL:-1}"

# Shared defaults (exploration / AWAC / IQL)
EXPL_ENV_NAME="${EXPL_ENV_NAME:-PointmassHard-v0}"
EXPL_EXP_NAME="${EXPL_EXP_NAME:-expl_default}"
EXPL_BATCH_SIZE="${EXPL_BATCH_SIZE:-256}"
EXPL_EVAL_BATCH_SIZE="${EXPL_EVAL_BATCH_SIZE:-1000}"
EXPL_NUM_EXPLORATION_STEPS="${EXPL_NUM_EXPLORATION_STEPS:-10000}"
EXPL_SEED="${EXPL_SEED:-2}"
EXPL_USE_RND="${EXPL_USE_RND:-1}"
EXPL_UNSUPERVISED="${EXPL_UNSUPERVISED:-0}"
EXPL_OFFLINE_EXPLOITATION="${EXPL_OFFLINE_EXPLOITATION:-0}"
EXPL_USE_BOLTZMANN="${EXPL_USE_BOLTZMANN:-0}"
EXPL_CQL_ALPHA="${EXPL_CQL_ALPHA:-0.0}"
EXPL_EXPECTILE="${EXPL_EXPECTILE:-0.8}"
EXPL_EXPLOIT_REW_SHIFT="${EXPL_EXPLOIT_REW_SHIFT:-0.0}"
EXPL_EXPLOIT_REW_SCALE="${EXPL_EXPLOIT_REW_SCALE:-1.0}"
EXPL_RND_OUTPUT_SIZE="${EXPL_RND_OUTPUT_SIZE:-5}"
EXPL_RND_N_LAYERS="${EXPL_RND_N_LAYERS:-2}"
EXPL_RND_SIZE="${EXPL_RND_SIZE:-400}"
EXPL_EXTRA_FLAGS="${EXPL_EXTRA_FLAGS:-}"

# AWAC-specific overrides
AWAC_ENV_NAME="${AWAC_ENV_NAME:-$EXPL_ENV_NAME}"
AWAC_EXP_NAME="${AWAC_EXP_NAME:-awac_default}"
AWAC_BATCH_SIZE="${AWAC_BATCH_SIZE:-$EXPL_BATCH_SIZE}"
AWAC_EVAL_BATCH_SIZE="${AWAC_EVAL_BATCH_SIZE:-$EXPL_EVAL_BATCH_SIZE}"
AWAC_NUM_EXPLORATION_STEPS="${AWAC_NUM_EXPLORATION_STEPS:-$EXPL_NUM_EXPLORATION_STEPS}"
AWAC_SEED="${AWAC_SEED:-$EXPL_SEED}"
AWAC_USE_RND="${AWAC_USE_RND:-$EXPL_USE_RND}"
AWAC_UNSUPERVISED="${AWAC_UNSUPERVISED:-0}"
AWAC_OFFLINE_EXPLOITATION="${AWAC_OFFLINE_EXPLOITATION:-0}"
AWAC_USE_BOLTZMANN="${AWAC_USE_BOLTZMANN:-0}"
AWAC_CQL_ALPHA="${AWAC_CQL_ALPHA:-$EXPL_CQL_ALPHA}"
AWAC_LAMBDA="${AWAC_LAMBDA:-1.0}"
AWAC_N_LAYERS="${AWAC_N_LAYERS:-4}"
AWAC_HIDDEN_SIZE="${AWAC_HIDDEN_SIZE:-512}"
AWAC_NUM_ACTIONS="${AWAC_NUM_ACTIONS:-10}"
AWAC_EXPLOIT_REW_SHIFT="${AWAC_EXPLOIT_REW_SHIFT:-$EXPL_EXPLOIT_REW_SHIFT}"
AWAC_EXPLOIT_REW_SCALE="${AWAC_EXPLOIT_REW_SCALE:-$EXPL_EXPLOIT_REW_SCALE}"
AWAC_RND_OUTPUT_SIZE="${AWAC_RND_OUTPUT_SIZE:-$EXPL_RND_OUTPUT_SIZE}"
AWAC_RND_N_LAYERS="${AWAC_RND_N_LAYERS:-$EXPL_RND_N_LAYERS}"
AWAC_RND_SIZE="${AWAC_RND_SIZE:-$EXPL_RND_SIZE}"
AWAC_EXTRA_FLAGS="${AWAC_EXTRA_FLAGS:-}"

# IQL-specific overrides
IQL_ENV_NAME="${IQL_ENV_NAME:-$EXPL_ENV_NAME}"
IQL_EXP_NAME="${IQL_EXP_NAME:-iql_default}"
IQL_BATCH_SIZE="${IQL_BATCH_SIZE:-$EXPL_BATCH_SIZE}"
IQL_EVAL_BATCH_SIZE="${IQL_EVAL_BATCH_SIZE:-$EXPL_EVAL_BATCH_SIZE}"
IQL_NUM_EXPLORATION_STEPS="${IQL_NUM_EXPLORATION_STEPS:-$EXPL_NUM_EXPLORATION_STEPS}"
IQL_SEED="${IQL_SEED:-$EXPL_SEED}"
IQL_USE_RND="${IQL_USE_RND:-$EXPL_USE_RND}"
IQL_UNSUPERVISED="${IQL_UNSUPERVISED:-0}"
IQL_OFFLINE_EXPLOITATION="${IQL_OFFLINE_EXPLOITATION:-0}"
IQL_USE_BOLTZMANN="${IQL_USE_BOLTZMANN:-0}"
IQL_CQL_ALPHA="${IQL_CQL_ALPHA:-$EXPL_CQL_ALPHA}"
IQL_EXPECTILE="${IQL_EXPECTILE:-$EXPL_EXPECTILE}"
IQL_N_LAYERS="${IQL_N_LAYERS:-4}"
IQL_HIDDEN_SIZE="${IQL_HIDDEN_SIZE:-512}"
IQL_NUM_ACTIONS="${IQL_NUM_ACTIONS:-10}"
IQL_EXPLOIT_REW_SHIFT="${IQL_EXPLOIT_REW_SHIFT:-$EXPL_EXPLOIT_REW_SHIFT}"
IQL_EXPLOIT_REW_SCALE="${IQL_EXPLOIT_REW_SCALE:-$EXPL_EXPLOIT_REW_SCALE}"
IQL_RND_OUTPUT_SIZE="${IQL_RND_OUTPUT_SIZE:-$EXPL_RND_OUTPUT_SIZE}"
IQL_RND_N_LAYERS="${IQL_RND_N_LAYERS:-$EXPL_RND_N_LAYERS}"
IQL_RND_SIZE="${IQL_RND_SIZE:-$EXPL_RND_SIZE}"
IQL_EXTRA_FLAGS="${IQL_EXTRA_FLAGS:-}"

log() {
  echo "[run_hw5.sh] $*"
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

append_common_flags() {
  local flags_to_add=()
  local use_rnd="$1"
  local unsupervised="$2"
  local offline_exploit="$3"
  local use_boltzmann="$4"

  if [[ "${use_rnd}" == "1" ]]; then
    flags_to_add+=(--use_rnd)
  fi
  if [[ "${unsupervised}" == "1" ]]; then
    flags_to_add+=(--unsupervised_exploration)
  fi
  if [[ "${offline_exploit}" == "1" ]]; then
    flags_to_add+=(--offline_exploitation)
  fi
  if [[ "${use_boltzmann}" == "1" ]]; then
    flags_to_add+=(--use_boltzmann)
  fi
  echo "${flags_to_add[@]}"
}

run_exploration() {
  if [[ "${RUN_EXPLORATION}" != "1" ]]; then
    log "Skipping exploration/CQL run (RUN_EXPLORATION=${RUN_EXPLORATION})."
    return
  fi

  local cmd=(python "${PROJECT_ROOT}/cs285/scripts/run_hw5_expl.py"
    --env_name "${EXPL_ENV_NAME}"
    --exp_name "${EXPL_EXP_NAME}"
    --batch_size "${EXPL_BATCH_SIZE}"
    --eval_batch_size "${EXPL_EVAL_BATCH_SIZE}"
    --num_exploration_steps "${EXPL_NUM_EXPLORATION_STEPS}"
    --seed "${EXPL_SEED}"
    --cql_alpha "${EXPL_CQL_ALPHA}"
    --exploit_rew_shift "${EXPL_EXPLOIT_REW_SHIFT}"
    --exploit_rew_scale "${EXPL_EXPLOIT_REW_SCALE}"
    --rnd_output_size "${EXPL_RND_OUTPUT_SIZE}"
    --rnd_n_layers "${EXPL_RND_N_LAYERS}"
    --rnd_size "${EXPL_RND_SIZE}")

  # shellcheck disable=SC2207
  local common_flags=( $(append_common_flags "${EXPL_USE_RND}" "${EXPL_UNSUPERVISED}" "${EXPL_OFFLINE_EXPLOITATION}" "${EXPL_USE_BOLTZMANN}") )
  if [[ ${#common_flags[@]} -gt 0 ]]; then
    cmd+=("${common_flags[@]}")
  fi

  if [[ -n "${EXPL_EXTRA_FLAGS}" ]]; then
    # shellcheck disable=SC2206
    local extra=( ${EXPL_EXTRA_FLAGS} )
    cmd+=("${extra[@]}")
  fi

  log "Running exploration/CQL experiment (${EXPL_ENV_NAME})."
  log "Command: ${cmd[*]}"
  "${cmd[@]}"
}

run_awac() {
  if [[ "${RUN_AWAC}" != "1" ]]; then
    log "Skipping AWAC run (RUN_AWAC=${RUN_AWAC})."
    return
  fi

  local cmd=(python "${PROJECT_ROOT}/cs285/scripts/run_hw5_awac.py"
    --env_name "${AWAC_ENV_NAME}"
    --exp_name "${AWAC_EXP_NAME}"
    --batch_size "${AWAC_BATCH_SIZE}"
    --eval_batch_size "${AWAC_EVAL_BATCH_SIZE}"
    --num_exploration_steps "${AWAC_NUM_EXPLORATION_STEPS}"
    --seed "${AWAC_SEED}"
    --cql_alpha "${AWAC_CQL_ALPHA}"
    --awac_lambda "${AWAC_LAMBDA}"
    --n_layers "${AWAC_N_LAYERS}"
    --size "${AWAC_HIDDEN_SIZE}"
    --n_actions "${AWAC_NUM_ACTIONS}"
    --exploit_rew_shift "${AWAC_EXPLOIT_REW_SHIFT}"
    --exploit_rew_scale "${AWAC_EXPLOIT_REW_SCALE}"
    --rnd_output_size "${AWAC_RND_OUTPUT_SIZE}"
    --rnd_n_layers "${AWAC_RND_N_LAYERS}"
    --rnd_size "${AWAC_RND_SIZE}")

  # shellcheck disable=SC2207
  local common_flags=( $(append_common_flags "${AWAC_USE_RND}" "${AWAC_UNSUPERVISED}" "${AWAC_OFFLINE_EXPLOITATION}" "${AWAC_USE_BOLTZMANN}") )
  if [[ ${#common_flags[@]} -gt 0 ]]; then
    cmd+=("${common_flags[@]}")
  fi

  if [[ -n "${AWAC_EXTRA_FLAGS}" ]]; then
    # shellcheck disable=SC2206
    local extra=( ${AWAC_EXTRA_FLAGS} )
    cmd+=("${extra[@]}")
  fi

  log "Running AWAC experiment (${AWAC_ENV_NAME})."
  log "Command: ${cmd[*]}"
  "${cmd[@]}"
}

run_iql() {
  if [[ "${RUN_IQL}" != "1" ]]; then
    log "Skipping IQL run (RUN_IQL=${RUN_IQL})."
    return
  fi

  local cmd=(python "${PROJECT_ROOT}/cs285/scripts/run_hw5_iql.py"
    --env_name "${IQL_ENV_NAME}"
    --exp_name "${IQL_EXP_NAME}"
    --batch_size "${IQL_BATCH_SIZE}"
    --eval_batch_size "${IQL_EVAL_BATCH_SIZE}"
    --num_exploration_steps "${IQL_NUM_EXPLORATION_STEPS}"
    --seed "${IQL_SEED}"
    --cql_alpha "${IQL_CQL_ALPHA}"
    --iql_expectile "${IQL_EXPECTILE}"
    --n_layers "${IQL_N_LAYERS}"
    --size "${IQL_HIDDEN_SIZE}"
    --n_actions "${IQL_NUM_ACTIONS}"
    --exploit_rew_shift "${IQL_EXPLOIT_REW_SHIFT}"
    --exploit_rew_scale "${IQL_EXPLOIT_REW_SCALE}"
    --rnd_output_size "${IQL_RND_OUTPUT_SIZE}"
    --rnd_n_layers "${IQL_RND_N_LAYERS}"
    --rnd_size "${IQL_RND_SIZE}")

  # shellcheck disable=SC2207
  local common_flags=( $(append_common_flags "${IQL_USE_RND}" "${IQL_UNSUPERVISED}" "${IQL_OFFLINE_EXPLOITATION}" "${IQL_USE_BOLTZMANN}") )
  if [[ ${#common_flags[@]} -gt 0 ]]; then
    cmd+=("${common_flags[@]}")
  fi

  if [[ -n "${IQL_EXTRA_FLAGS}" ]]; then
    # shellcheck disable=SC2206
    local extra=( ${IQL_EXTRA_FLAGS} )
    cmd+=("${extra[@]}")
  fi

  log "Running IQL experiment (${IQL_ENV_NAME})."
  log "Command: ${cmd[*]}"
  "${cmd[@]}"
}

main() {
  require_command "${PYTHON_BIN}"
  
  # Uninstall any existing cs285 package and install hw5's version
  log "Ensuring hw5's cs285 package is installed..."
  pip uninstall -y cs285 2>/dev/null || true
  
  # Clear Python cache to avoid stale bytecode
  find "${PROJECT_ROOT}/cs285" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
  find "${PROJECT_ROOT}/cs285" -type f -name "*.pyc" -delete 2>/dev/null || true
  
  pip install -e "${PROJECT_ROOT}"

  if [[ -n "${MUJOCO_GL}" ]]; then
    export MUJOCO_GL
    log "Set MUJOCO_GL=${MUJOCO_GL}."
  else
    log "Using MuJoCo's auto-detected rendering backend."
  fi

  run_exploration
  run_awac
  run_iql

  log "All requested experiments finished. Logs are under ${PROJECT_ROOT}/cs285/data."
}

main "$@"
