#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
VENV_PATH="${PROJECT_ROOT}/.venv"

# General configuration
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
MUJOCO_GL="${MUJOCO_GL:-egl}"

RUN_MB="${RUN_MB:-1}"
RUN_MBPO="${RUN_MBPO:-1}"

# Model-Based (MB) defaults
MB_ENV_NAME="${MB_ENV_NAME:-cheetah-cs285-v0}"
MB_EXP_NAME="${MB_EXP_NAME:-mb_default}"
MB_N_ITER="${MB_N_ITER:-20}"
MB_ENSEMBLE_SIZE="${MB_ENSEMBLE_SIZE:-3}"
MB_MPC_HORIZON="${MB_MPC_HORIZON:-10}"
MB_MPC_ACTION_SEQUENCES="${MB_MPC_ACTION_SEQUENCES:-1000}"
MB_MPC_SAMPLING="${MB_MPC_SAMPLING:-random}"
MB_CEM_ITERATIONS="${MB_CEM_ITERATIONS:-4}"
MB_CEM_NUM_ELITES="${MB_CEM_NUM_ELITES:-5}"
MB_CEM_ALPHA="${MB_CEM_ALPHA:-1}"
MB_ADD_SL_NOISE="${MB_ADD_SL_NOISE:-0}"
MB_BATCH_SIZE="${MB_BATCH_SIZE:-8000}"
MB_BATCH_SIZE_INITIAL="${MB_BATCH_SIZE_INITIAL:-20000}"
MB_TRAIN_BATCH_SIZE="${MB_TRAIN_BATCH_SIZE:-512}"
MB_EVAL_BATCH_SIZE="${MB_EVAL_BATCH_SIZE:-400}"
MB_LEARNING_RATE="${MB_LEARNING_RATE:-0.001}"
MB_N_LAYERS="${MB_N_LAYERS:-2}"
MB_HIDDEN_SIZE="${MB_HIDDEN_SIZE:-250}"
MB_NUM_AGENT_TRAIN_STEPS="${MB_NUM_AGENT_TRAIN_STEPS:-1000}"
MB_SEED="${MB_SEED:-1}"
MB_EXTRA_FLAGS="${MB_EXTRA_FLAGS:-}"

# MBPO defaults
MBPO_ENV_NAME="${MBPO_ENV_NAME:-cheetah-cs285-v0}"
MBPO_EXP_NAME="${MBPO_EXP_NAME:-mbpo_default}"
MBPO_N_ITER="${MBPO_N_ITER:-20}"
MBPO_ENSEMBLE_SIZE="${MBPO_ENSEMBLE_SIZE:-3}"
MBPO_MPC_HORIZON="${MBPO_MPC_HORIZON:-10}"
MBPO_MPC_ACTION_SEQUENCES="${MBPO_MPC_ACTION_SEQUENCES:-1000}"
MBPO_MPC_SAMPLING="${MBPO_MPC_SAMPLING:-random}"
MBPO_CEM_ITERATIONS="${MBPO_CEM_ITERATIONS:-4}"
MBPO_CEM_NUM_ELITES="${MBPO_CEM_NUM_ELITES:-5}"
MBPO_CEM_ALPHA="${MBPO_CEM_ALPHA:-1}"
MBPO_ADD_SL_NOISE="${MBPO_ADD_SL_NOISE:-0}"
MBPO_BATCH_SIZE_INITIAL="${MBPO_BATCH_SIZE_INITIAL:-20000}"
MBPO_BATCH_SIZE="${MBPO_BATCH_SIZE:-8000}"
MBPO_TRAIN_BATCH_SIZE="${MBPO_TRAIN_BATCH_SIZE:-512}"
MBPO_EVAL_BATCH_SIZE="${MBPO_EVAL_BATCH_SIZE:-400}"
MBPO_LEARNING_RATE="${MBPO_LEARNING_RATE:-0.001}"
MBPO_N_LAYERS="${MBPO_N_LAYERS:-2}"
MBPO_HIDDEN_SIZE="${MBPO_HIDDEN_SIZE:-250}"
MBPO_NUM_AGENT_TRAIN_STEPS="${MBPO_NUM_AGENT_TRAIN_STEPS:-1000}"
MBPO_ROLLOUT_LENGTH="${MBPO_ROLLOUT_LENGTH:-1}"
MBPO_SEED="${MBPO_SEED:-1}"
MBPO_EXTRA_FLAGS="${MBPO_EXTRA_FLAGS:-}"

SAC_N_ITER="${SAC_N_ITER:-200}"
SAC_TRAIN_BATCH_SIZE="${SAC_TRAIN_BATCH_SIZE:-256}"
SAC_BATCH_SIZE="${SAC_BATCH_SIZE:-1000}"
SAC_DISCOUNT="${SAC_DISCOUNT:-0.99}"
SAC_INIT_TEMPERATURE="${SAC_INIT_TEMPERATURE:-1.0}"
SAC_LEARNING_RATE="${SAC_LEARNING_RATE:-3e-4}"
SAC_N_LAYERS="${SAC_N_LAYERS:-2}"
SAC_HIDDEN_SIZE="${SAC_HIDDEN_SIZE:-64}"
SAC_NUM_AGENT_TRAIN_STEPS="${SAC_NUM_AGENT_TRAIN_STEPS:-1}"
SAC_NUM_CRITIC_UPDATES="${SAC_NUM_CRITIC_UPDATES:-1}"
SAC_NUM_ACTOR_UPDATES="${SAC_NUM_ACTOR_UPDATES:-1}"
SAC_ACTOR_UPDATE_FREQ="${SAC_ACTOR_UPDATE_FREQ:-1}"
SAC_CRITIC_TARGET_UPDATE_FREQ="${SAC_CRITIC_TARGET_UPDATE_FREQ:-1}"

log() {
  echo "[run_hw4.sh] $*"
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

run_mb() {
  if [[ "${RUN_MB}" != "1" ]]; then
    log "Skipping model-based MPC experiment (RUN_MB=${RUN_MB})."
    return
  fi

  local cmd=(python "${PROJECT_ROOT}/cs285/scripts/run_hw4_mb.py"
    --env_name "${MB_ENV_NAME}"
    --exp_name "${MB_EXP_NAME}"
    --n_iter "${MB_N_ITER}"
    --ensemble_size "${MB_ENSEMBLE_SIZE}"
    --mpc_horizon "${MB_MPC_HORIZON}"
    --mpc_num_action_sequences "${MB_MPC_ACTION_SEQUENCES}"
    --mpc_action_sampling_strategy "${MB_MPC_SAMPLING}"
    --cem_iterations "${MB_CEM_ITERATIONS}"
    --cem_num_elites "${MB_CEM_NUM_ELITES}"
    --cem_alpha "${MB_CEM_ALPHA}"
    --batch_size_initial "${MB_BATCH_SIZE_INITIAL}"
    --batch_size "${MB_BATCH_SIZE}"
    --train_batch_size "${MB_TRAIN_BATCH_SIZE}"
    --eval_batch_size "${MB_EVAL_BATCH_SIZE}"
    --learning_rate "${MB_LEARNING_RATE}"
    --n_layers "${MB_N_LAYERS}"
    --size "${MB_HIDDEN_SIZE}"
    --num_agent_train_steps_per_iter "${MB_NUM_AGENT_TRAIN_STEPS}"
    --seed "${MB_SEED}")

  if [[ "${MB_ADD_SL_NOISE}" == "1" ]]; then
    cmd+=(--add_sl_noise)
  fi

  if [[ -n "${MB_EXTRA_FLAGS}" ]]; then
    # shellcheck disable=SC2206
    local extra=( ${MB_EXTRA_FLAGS} )
    cmd+=("${extra[@]}")
  fi

  log "Running model-based RL experiment (${MB_ENV_NAME})."
  log "Command: ${cmd[*]}"
  "${cmd[@]}"
}

run_mbpo() {
  if [[ "${RUN_MBPO}" != "1" ]]; then
    log "Skipping MBPO experiment (RUN_MBPO=${RUN_MBPO})."
    return
  fi

  local cmd=(python "${PROJECT_ROOT}/cs285/scripts/run_hw4_mbpo.py"
    --env_name "${MBPO_ENV_NAME}"
    --exp_name "${MBPO_EXP_NAME}"
    --n_iter "${MBPO_N_ITER}"
    --ensemble_size "${MBPO_ENSEMBLE_SIZE}"
    --mpc_horizon "${MBPO_MPC_HORIZON}"
    --mpc_num_action_sequences "${MBPO_MPC_ACTION_SEQUENCES}"
    --mpc_action_sampling_strategy "${MBPO_MPC_SAMPLING}"
    --cem_iterations "${MBPO_CEM_ITERATIONS}"
    --cem_num_elites "${MBPO_CEM_NUM_ELITES}"
    --cem_alpha "${MBPO_CEM_ALPHA}"
    --batch_size_initial "${MBPO_BATCH_SIZE_INITIAL}"
    --batch_size "${MBPO_BATCH_SIZE}"
    --train_batch_size "${MBPO_TRAIN_BATCH_SIZE}"
    --eval_batch_size "${MBPO_EVAL_BATCH_SIZE}"
    --learning_rate "${MBPO_LEARNING_RATE}"
    --n_layers "${MBPO_N_LAYERS}"
    --size "${MBPO_HIDDEN_SIZE}"
    --num_agent_train_steps_per_iter "${MBPO_NUM_AGENT_TRAIN_STEPS}"
    --sac_num_agent_train_steps_per_iter "${SAC_NUM_AGENT_TRAIN_STEPS}"
    --sac_num_critic_updates_per_agent_update "${SAC_NUM_CRITIC_UPDATES}"
    --sac_num_actor_updates_per_agent_update "${SAC_NUM_ACTOR_UPDATES}"
    --sac_actor_update_frequency "${SAC_ACTOR_UPDATE_FREQ}"
    --sac_critic_target_update_frequency "${SAC_CRITIC_TARGET_UPDATE_FREQ}"
    --sac_train_batch_size "${SAC_TRAIN_BATCH_SIZE}"
    --sac_batch_size "${SAC_BATCH_SIZE}"
    --sac_discount "${SAC_DISCOUNT}"
    --sac_init_temperature "${SAC_INIT_TEMPERATURE}"
    --sac_learning_rate "${SAC_LEARNING_RATE}"
    --sac_n_layers "${SAC_N_LAYERS}"
    --sac_size "${SAC_HIDDEN_SIZE}"
    --sac_n_iter "${SAC_N_ITER}"
    --mbpo_rollout_length "${MBPO_ROLLOUT_LENGTH}"
    --seed "${MBPO_SEED}")

  if [[ "${MBPO_ADD_SL_NOISE}" == "1" ]]; then
    cmd+=(--add_sl_noise)
  fi

  if [[ -n "${MBPO_EXTRA_FLAGS}" ]]; then
    # shellcheck disable=SC2206
    local extra=( ${MBPO_EXTRA_FLAGS} )
    cmd+=("${extra[@]}")
  fi

  log "Running MBPO experiment (${MBPO_ENV_NAME})."
  log "Command: ${cmd[*]}"
  "${cmd[@]}"
}

main() {
  require_command "${PYTHON_BIN}"
  create_and_activate_venv
  install_dependencies

  export MUJOCO_GL
  log "Set MUJOCO_GL=${MUJOCO_GL}."

  run_mb
  run_mbpo

  log "All requested experiments finished. Logs are under ${PROJECT_ROOT}/cs285/data."
}

main "$@"
