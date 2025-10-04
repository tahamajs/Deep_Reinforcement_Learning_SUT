#!/usr/bin/env bash
set -euo pipefail

# Determine project root relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
VENV_PATH="${PROJECT_ROOT}/.venv"

# Configuration (can be overridden via environment variables)
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_BEHAVIOR_CLONING="${RUN_BEHAVIOR_CLONING:-1}"
RUN_DAGGER="${RUN_DAGGER:-1}"
ENV_NAME="${ENV_NAME:-HalfCheetah-v4}"
BC_EXPNAME="${BC_EXPNAME:-bc_${ENV_NAME}}"
DAGGER_EXPNAME="${DAGGER_EXPNAME:-dagger_${ENV_NAME}}"
DAGGER_ITERS="${DAGGER_ITERS:-5}"
MAX_REPLAY_BUFFER_SIZE="${MAX_REPLAY_BUFFER_SIZE:-1000000}"
NUM_AGENT_TRAIN_STEPS="${NUM_AGENT_TRAIN_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-100}"
LEARNING_RATE="${LEARNING_RATE:-5e-3}"
N_LAYERS="${N_LAYERS:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
SEED="${SEED:-1}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

# Helper to echo status messages
log() {
  echo "[run_hw1.sh] $*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "ERROR: Required command '$1' not found."
    exit 1
  fi
}

resolve_paths() {
  case "${ENV_NAME}" in
    Ant-v4)
      EXPERT_POLICY_FILE="${PROJECT_ROOT}/cs285/policies/experts/Ant.pkl"
      EXPERT_DATA_FILE="${PROJECT_ROOT}/cs285/expert_data/expert_data_Ant-v4.pkl"
      ;;
    HalfCheetah-v4)
      EXPERT_POLICY_FILE="${PROJECT_ROOT}/cs285/policies/experts/HalfCheetah.pkl"
      EXPERT_DATA_FILE="${PROJECT_ROOT}/cs285/expert_data/expert_data_HalfCheetah-v4.pkl"
      ;;
    Hopper-v4)
      EXPERT_POLICY_FILE="${PROJECT_ROOT}/cs285/policies/experts/Hopper.pkl"
      EXPERT_DATA_FILE="${PROJECT_ROOT}/cs285/expert_data/expert_data_Hopper-v4.pkl"
      ;;
    Walker2d-v4)
      EXPERT_POLICY_FILE="${PROJECT_ROOT}/cs285/policies/experts/Walker2d.pkl"
      EXPERT_DATA_FILE="${PROJECT_ROOT}/cs285/expert_data/expert_data_Walker2d-v4.pkl"
      ;;
    Humanoid-v4)
      EXPERT_POLICY_FILE="${PROJECT_ROOT}/cs285/policies/experts/Humanoid.pkl"
      EXPERT_DATA_FILE="${PROJECT_ROOT}/cs285/expert_data/expert_data_Humanoid-v4.pkl"
      if [[ ! -f "${EXPERT_DATA_FILE}" ]]; then
        log "ERROR: Humanoid expert data not bundled. Please provide ${EXPERT_DATA_FILE}."
        exit 1
      fi
      ;;
    *)
      log "ERROR: Unsupported ENV_NAME '${ENV_NAME}'."
      log "       Supported values: Ant-v4, HalfCheetah-v4, Hopper-v4, Walker2d-v4, Humanoid-v4 (requires user-supplied expert data)."
      exit 1
      ;;
  esac

  if [[ ! -f "${EXPERT_POLICY_FILE}" ]]; then
    log "ERROR: Missing expert policy file ${EXPERT_POLICY_FILE}."
    exit 1
  fi

  if [[ ! -f "${EXPERT_DATA_FILE}" ]]; then
    log "ERROR: Missing expert dataset ${EXPERT_DATA_FILE}."
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

  # shellcheck source=/dev/null
  source "${VENV_PATH}/bin/activate"
  log "Virtual environment activated (python: $(python --version))."
}

install_dependencies() {
  if [[ "${SKIP_INSTALL}" == "1" ]]; then
    log "Skipping dependency installation (SKIP_INSTALL=1)."
    return
  fi

  log "Upgrading pip and wheel."
  pip install --upgrade pip wheel

  log "Installing project requirements."
  pip install -r "${PROJECT_ROOT}/requirements.txt"

  log "Installing cs285 as editable package."
  pip install -e "${PROJECT_ROOT}"
}

run_behavior_cloning() {
  if [[ "${RUN_BEHAVIOR_CLONING}" != "1" ]]; then
    log "Skipping behavior cloning run (RUN_BEHAVIOR_CLONING=${RUN_BEHAVIOR_CLONING})."
    return
  fi

  log "Starting Behavior Cloning experiment for ${ENV_NAME}."
  python "${PROJECT_ROOT}/cs285/scripts/run_hw1.py" \
    --env_name "${ENV_NAME}" \
    --expert_policy_file "${EXPERT_POLICY_FILE}" \
    --expert_data "${EXPERT_DATA_FILE}" \
    --exp_name "${BC_EXPNAME}" \
    --n_iter 1 \
    --batch_size "${BATCH_SIZE}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --n_layers "${N_LAYERS}" \
    --size "${HIDDEN_SIZE}" \
    --max_replay_buffer_size "${MAX_REPLAY_BUFFER_SIZE}" \
    --num_agent_train_steps_per_iter "${NUM_AGENT_TRAIN_STEPS}" \
    --seed "${SEED}"
}

run_dagger() {
  if [[ "${RUN_DAGGER}" != "1" ]]; then
    log "Skipping DAgger run (RUN_DAGGER=${RUN_DAGGER})."
    return
  fi

  log "Starting DAgger experiment for ${ENV_NAME}."
  python "${PROJECT_ROOT}/cs285/scripts/run_hw1.py" \
    --env_name "${ENV_NAME}" \
    --expert_policy_file "${EXPERT_POLICY_FILE}" \
    --expert_data "${EXPERT_DATA_FILE}" \
    --exp_name "${DAGGER_EXPNAME}" \
    --n_iter "${DAGGER_ITERS}" \
    --do_dagger \
    --batch_size "${BATCH_SIZE}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --n_layers "${N_LAYERS}" \
    --size "${HIDDEN_SIZE}" \
    --max_replay_buffer_size "${MAX_REPLAY_BUFFER_SIZE}" \
    --num_agent_train_steps_per_iter "${NUM_AGENT_TRAIN_STEPS}" \
    --seed "${SEED}"
}

main() {
  require_command "${PYTHON_BIN}"

  resolve_paths
  
  # Uninstall any existing cs285 package and install hw1's version
  log "Ensuring hw1's cs285 package is installed..."
  pip uninstall -y cs285 2>/dev/null || true

  # Clear Python cache to avoid stale bytecode
  find "${PROJECT_ROOT}/cs285" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
  find "${PROJECT_ROOT}/cs285" -type f -name "*.pyc" -delete 2>/dev/null || true

#   if [[ "${SKIP_INSTALL}" == "1" ]]; then
#     log "Skipping dependency installation (SKIP_INSTALL=1)."
#   else
#     log "Installing hw1 requirements and package."
#     pip install --upgrade pip wheel
#     pip install -r "${PROJECT_ROOT}/requirements.txt"
#     pip install -e "${PROJECT_ROOT}"
#   fi

  MUJOCO_GL="${MUJOCO_GL:-}"
  if [[ -n "${MUJOCO_GL}" ]]; then
    export MUJOCO_GL
    log "Set MUJOCO_GL=${MUJOCO_GL}."
  else
    log "Using MuJoCo's auto-detected rendering backend."
  fi

  run_behavior_cloning
  run_dagger

  log "All requested experiments finished. Logs are under ${PROJECT_ROOT}/cs285/data."
}

main "$@"
