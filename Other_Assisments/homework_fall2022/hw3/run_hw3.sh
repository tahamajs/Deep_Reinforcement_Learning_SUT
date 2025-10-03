#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
VENV_PATH="${PROJECT_ROOT}/.venv"

# General configuration
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
MUJOCO_GL="${MUJOCO_GL:-egl}"

RUN_DQN="${RUN_DQN:-1}"
RUN_ACTOR_CRITIC="${RUN_ACTOR_CRITIC:-1}"
RUN_SAC="${RUN_SAC:-1}"

# DQN defaults
DQN_ENV_NAME="${DQN_ENV_NAME:-LunarLander-v3}"
DQN_EXP_NAME="${DQN_EXP_NAME:-dqn_default}"
DQN_BATCH_SIZE="${DQN_BATCH_SIZE:-32}"
DQN_TRAIN_STEPS_PER_ITER="${DQN_TRAIN_STEPS_PER_ITER:-1}"
DQN_CRITIC_UPDATES="${DQN_CRITIC_UPDATES:-1}"
DQN_DOUBLE_Q="${DQN_DOUBLE_Q:-0}"
DQN_EVAL_BATCH_SIZE="${DQN_EVAL_BATCH_SIZE:-1000}"
DQN_EP_LEN="${DQN_EP_LEN:-200}"
DQN_SEED="${DQN_SEED:-1}"
DQN_EXTRA_FLAGS="${DQN_EXTRA_FLAGS:-}"

# Actor-Critic defaults
AC_ENV_NAME="${AC_ENV_NAME:-CartPole-v0}"
AC_EXP_NAME="${AC_EXP_NAME:-ac_default}"
AC_N_ITER="${AC_N_ITER:-200}"
AC_BATCH_SIZE="${AC_BATCH_SIZE:-1000}"
AC_EVAL_BATCH_SIZE="${AC_EVAL_BATCH_SIZE:-400}"
AC_TRAIN_BATCH_SIZE="${AC_TRAIN_BATCH_SIZE:-1000}"
AC_DISCOUNT="${AC_DISCOUNT:-1.0}"
AC_LEARNING_RATE="${AC_LEARNING_RATE:-5e-3}"
AC_NUM_AGENT_TRAIN_STEPS="${AC_NUM_AGENT_TRAIN_STEPS:-1}"
AC_NUM_CRITIC_UPDATES="${AC_NUM_CRITIC_UPDATES:-1}"
AC_NUM_ACTOR_UPDATES="${AC_NUM_ACTOR_UPDATES:-1}"
AC_NUM_TARGET_UPDATES="${AC_NUM_TARGET_UPDATES:-10}"
AC_NUM_GRAD_STEPS_PER_TARGET_UPDATE="${AC_NUM_GRAD_STEPS_PER_TARGET_UPDATE:-10}"
AC_N_LAYERS="${AC_N_LAYERS:-2}"
AC_HIDDEN_SIZE="${AC_HIDDEN_SIZE:-64}"
AC_EP_LEN="${AC_EP_LEN:-200}"
AC_SEED="${AC_SEED:-1}"
AC_STANDARDIZE_ADV="${AC_STANDARDIZE_ADV:-1}"
AC_EXTRA_FLAGS="${AC_EXTRA_FLAGS:-}"

# SAC defaults
SAC_ENV_NAME="${SAC_ENV_NAME:-CartPole-v0}"
SAC_EXP_NAME="${SAC_EXP_NAME:-sac_default}"
SAC_N_ITER="${SAC_N_ITER:-200}"
SAC_BATCH_SIZE="${SAC_BATCH_SIZE:-1000}"
SAC_EVAL_BATCH_SIZE="${SAC_EVAL_BATCH_SIZE:-400}"
SAC_TRAIN_BATCH_SIZE="${SAC_TRAIN_BATCH_SIZE:-256}"
SAC_DISCOUNT="${SAC_DISCOUNT:-0.99}"
SAC_LEARNING_RATE="${SAC_LEARNING_RATE:-3e-4}"
SAC_INIT_TEMPERATURE="${SAC_INIT_TEMPERATURE:-1.0}"
SAC_N_LAYERS="${SAC_N_LAYERS:-2}"
SAC_HIDDEN_SIZE="${SAC_HIDDEN_SIZE:-64}"
SAC_NUM_AGENT_TRAIN_STEPS="${SAC_NUM_AGENT_TRAIN_STEPS:-1}"
SAC_NUM_CRITIC_UPDATES="${SAC_NUM_CRITIC_UPDATES:-1}"
SAC_NUM_ACTOR_UPDATES="${SAC_NUM_ACTOR_UPDATES:-1}"
SAC_ACTOR_UPDATE_FREQ="${SAC_ACTOR_UPDATE_FREQ:-1}"
SAC_CRITIC_TARGET_UPDATE_FREQ="${SAC_CRITIC_TARGET_UPDATE_FREQ:-1}"
SAC_EP_LEN="${SAC_EP_LEN:-200}"
SAC_SEED="${SAC_SEED:-1}"
SAC_EXTRA_FLAGS="${SAC_EXTRA_FLAGS:-}"

log() {
  echo "[run_hw3.sh] $*"
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

run_dqn() {
  if [[ "${RUN_DQN}" != "1" ]]; then
    log "Skipping DQN experiments (RUN_DQN=${RUN_DQN})."
    return
  fi

  if [[ "${DQN_ENV_NAME}" != "PongNoFrameskip-v4" && "${DQN_ENV_NAME}" != "LunarLander-v3" && "${DQN_ENV_NAME}" != "MsPacman-v0" ]]; then
    log "WARNING: ${DQN_ENV_NAME} is not in the default choices for run_hw3_dqn.py; attempting anyway."
  fi

  local cmd=(python "${PROJECT_ROOT}/cs285/scripts/run_hw3_dqn.py"
    --env_name "${DQN_ENV_NAME}"
    --exp_name "${DQN_EXP_NAME}"
    --batch_size "${DQN_BATCH_SIZE}"
    --num_agent_train_steps_per_iter "${DQN_TRAIN_STEPS_PER_ITER}"
    --num_critic_updates_per_agent_update "${DQN_CRITIC_UPDATES}"
    --eval_batch_size "${DQN_EVAL_BATCH_SIZE}"
    --ep_len "${DQN_EP_LEN}"
    --seed "${DQN_SEED}")

  if [[ "${DQN_DOUBLE_Q}" == "1" ]]; then
    cmd+=(--double_q)
  fi

  if [[ -n "${DQN_EXTRA_FLAGS}" ]]; then
    read -r -a extra <<< "${DQN_EXTRA_FLAGS}"
    cmd+=("${extra[@]}")
  fi

  log "Running DQN experiment (${DQN_ENV_NAME})."
  log "Command: ${cmd[*]}"
  "${cmd[@]}"
}

run_actor_critic() {
  if [[ "${RUN_ACTOR_CRITIC}" != "1" ]]; then
    log "Skipping Actor-Critic experiments (RUN_ACTOR_CRITIC=${RUN_ACTOR_CRITIC})."
    return
  fi

  local cmd=(python "${PROJECT_ROOT}/cs285/scripts/run_hw3_actor_critic.py"
    --env_name "${AC_ENV_NAME}"
    --exp_name "${AC_EXP_NAME}"
    --n_iter "${AC_N_ITER}"
    --batch_size "${AC_BATCH_SIZE}"
    --eval_batch_size "${AC_EVAL_BATCH_SIZE}"
    --train_batch_size "${AC_TRAIN_BATCH_SIZE}"
    --discount "${AC_DISCOUNT}"
    --learning_rate "${AC_LEARNING_RATE}"
    --num_agent_train_steps_per_iter "${AC_NUM_AGENT_TRAIN_STEPS}"
    --num_critic_updates_per_agent_update "${AC_NUM_CRITIC_UPDATES}"
    --num_actor_updates_per_agent_update "${AC_NUM_ACTOR_UPDATES}"
    --num_target_updates "${AC_NUM_TARGET_UPDATES}"
    --num_grad_steps_per_target_update "${AC_NUM_GRAD_STEPS_PER_TARGET_UPDATE}"
    --n_layers "${AC_N_LAYERS}"
    --size "${AC_HIDDEN_SIZE}"
    --ep_len "${AC_EP_LEN}"
    --seed "${AC_SEED}")

  if [[ "${AC_STANDARDIZE_ADV}" == "0" ]]; then
    cmd+=(--dont_standardize_advantages)
  fi

  if [[ -n "${AC_EXTRA_FLAGS}" ]]; then
    read -r -a extra <<< "${AC_EXTRA_FLAGS}"
    cmd+=("${extra[@]}")
  fi

  log "Running Actor-Critic experiment (${AC_ENV_NAME})."
  log "Command: ${cmd[*]}"
  "${cmd[@]}"
}

run_sac() {
  if [[ "${RUN_SAC}" != "1" ]]; then
    log "Skipping SAC experiments (RUN_SAC=${RUN_SAC})."
    return
  fi

  local cmd=(python "${PROJECT_ROOT}/cs285/scripts/run_hw3_sac.py"
    --env_name "${SAC_ENV_NAME}"
    --exp_name "${SAC_EXP_NAME}"
    --n_iter "${SAC_N_ITER}"
    --batch_size "${SAC_BATCH_SIZE}"
    --eval_batch_size "${SAC_EVAL_BATCH_SIZE}"
    --train_batch_size "${SAC_TRAIN_BATCH_SIZE}"
    --discount "${SAC_DISCOUNT}"
    --learning_rate "${SAC_LEARNING_RATE}"
    --init_temperature "${SAC_INIT_TEMPERATURE}"
    --n_layers "${SAC_N_LAYERS}"
    --size "${SAC_HIDDEN_SIZE}"
    --num_agent_train_steps_per_iter "${SAC_NUM_AGENT_TRAIN_STEPS}"
    --num_critic_updates_per_agent_update "${SAC_NUM_CRITIC_UPDATES}"
    --num_actor_updates_per_agent_update "${SAC_NUM_ACTOR_UPDATES}"
    --actor_update_frequency "${SAC_ACTOR_UPDATE_FREQ}"
    --critic_target_update_frequency "${SAC_CRITIC_TARGET_UPDATE_FREQ}"
    --ep_len "${SAC_EP_LEN}"
    --seed "${SAC_SEED}")

  if [[ -n "${SAC_EXTRA_FLAGS}" ]]; then
    read -r -a extra <<< "${SAC_EXTRA_FLAGS}"
    cmd+=("${extra[@]}")
  fi

  log "Running SAC experiment (${SAC_ENV_NAME})."
  log "Command: ${cmd[*]}"
  "${cmd[@]}"
}

main() {
  require_command "${PYTHON_BIN}"
  create_and_activate_venv
  install_dependencies

  export MUJOCO_GL
  log "Set MUJOCO_GL=${MUJOCO_GL}."

  run_dqn
  run_actor_critic
  run_sac

  log "All requested experiments finished. Logs are under ${PROJECT_ROOT}/cs285/data."
}

main "$@"
