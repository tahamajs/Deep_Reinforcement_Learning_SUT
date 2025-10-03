#!/usr/bin/env bash
set -e

echo "🔍 Checking MuJoCo availability for HW4..."
MUJOCO_AVAILABLE=false
python -c "import mujoco_py" >/dev/null 2>&1 && MUJOCO_AVAILABLE=true || true

if [ "$MUJOCO_AVAILABLE" != "true" ]; then
  echo "⚠️  MuJoCo not available — running with Pendulum-v0 instead."
  echo "   (Results may differ from HalfCheetah. Install MuJoCo for full experiments.)"
  echo ""
  ENV_NAME="Pendulum"
else
  echo "✅ MuJoCo is available — using HalfCheetah-v2"
  echo ""
  ENV_NAME="HalfCheetah"
fi

##########
### Q1 ###
##########

python main.py q1 --exp_name exp --env "$ENV_NAME" --overwrite

##########
### Q2 ###
##########

python main.py q2 --exp_name exp --env "$ENV_NAME" --overwrite

###########
### Q3a ###
###########

python main.py q3 --exp_name default --env "$ENV_NAME" --overwrite
python plot.py --exps ${ENV_NAME}_q3_default --save ${ENV_NAME}_q3_default || echo "⚠️ Plot skipped: no data for ${ENV_NAME}_q3_default"

###########
### Q3b ###
###########

python main.py q3 --exp_name action128 --num_random_action_selection 128 --env "$ENV_NAME" --overwrite
python main.py q3 --exp_name action4096 --num_random_action_selection 4096 --env "$ENV_NAME" --overwrite
python main.py q3 --exp_name action16384 --num_random_action_selection 16384 --env "$ENV_NAME" --overwrite
python plot.py --exps ${ENV_NAME}_q3_action128 ${ENV_NAME}_q3_action4096 ${ENV_NAME}_q3_action16384 --save ${ENV_NAME}_q3_actions || echo "⚠️ Plot skipped: some/none logs missing for actions"

python main.py q3 --exp_name horizon10 --mpc_horizon 10 --env "$ENV_NAME" --overwrite
python main.py q3 --exp_name horizon15 --mpc_horizon 15 --env "$ENV_NAME" --overwrite
python main.py q3 --exp_name horizon20 --mpc_horizon 20 --env "$ENV_NAME" --overwrite
python plot.py --exps ${ENV_NAME}_q3_horizon10 ${ENV_NAME}_q3_horizon15 ${ENV_NAME}_q3_horizon20 --save ${ENV_NAME}_q3_mpc_horizon || echo "⚠️ Plot skipped: some/none logs missing for horizons"

python main.py q3 --exp_name layers1 --nn_layers 1 --env "$ENV_NAME" --overwrite
python main.py q3 --exp_name layers2 --nn_layers 2 --env "$ENV_NAME" --overwrite
python main.py q3 --exp_name layers3 --nn_layers 3 --env "$ENV_NAME" --overwrite
python plot.py --exps ${ENV_NAME}_q3_layers1 ${ENV_NAME}_q3_layers2 ${ENV_NAME}_q3_layers3 --save ${ENV_NAME}_q3_nn_layers || echo "⚠️ Plot skipped: some/none logs missing for layers"

echo ""
echo "✅ HW4 experiments completed using $ENV_NAME environment!"
echo "   Results: data/${ENV_NAME}_*"
echo "   Plots: plots/${ENV_NAME}_*.jpg"