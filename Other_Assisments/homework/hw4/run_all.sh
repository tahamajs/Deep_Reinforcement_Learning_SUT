#!/usr/bin/env bash
set -e

echo "🔍 Checking MuJoCo availability for HW4..."
MUJOCO_AVAILABLE=false
python -c "import mujoco_py" >/dev/null 2>&1 && MUJOCO_AVAILABLE=true || true

if [ "$MUJOCO_AVAILABLE" != "true" ]; then
	echo "🚫 Skipping all HW4 runs (all require MuJoCo)."
	echo "   To enable HW4, install MuJoCo and mujoco-py. See MUJOCO_COMPATIBILITY.md"
	exit 0
fi

##########
### Q1 ###
##########

python main.py q1 --exp_name exp

##########
### Q2 ###
##########

python main.py q2 --exp_name exp

###########
### Q3a ###
###########

python main.py q3 --exp_name default
python plot.py --exps HalfCheetah_q3_default --save HalfCheetah_q3_default || echo "⚠️ Plot skipped: no data for HalfCheetah_q3_default"

###########
### Q3b ###
###########

python main.py q3 --exp_name action128 --num_random_action_selection 128
python main.py q3 --exp_name action4096 --num_random_action_selection 4096
python main.py q3 --exp_name action16384 --num_random_action_selection 16384
python plot.py --exps HalfCheetah_q3_action128 HalfCheetah_q3_action4096 HalfCheetah_q3_action16384 --save HalfCheetah_q3_actions || echo "⚠️ Plot skipped: some/none logs missing for actions"

python main.py q3 --exp_name horizon10 --mpc_horizon 10
python main.py q3 --exp_name horizon15 --mpc_horizon 15
python main.py q3 --exp_name horizon20 --mpc_horizon 20
python plot.py --exps HalfCheetah_q3_horizon10 HalfCheetah_q3_horizon15 HalfCheetah_q3_horizon20 --save HalfCheetah_q3_mpc_horizon || echo "⚠️ Plot skipped: some/none logs missing for horizons"

python main.py q3 --exp_name layers1 --nn_layers 1
python main.py q3 --exp_name layers2 --nn_layers 2
python main.py q3 --exp_name layers3 --nn_layers 3
python plot.py --exps HalfCheetah_q3_layers1 HalfCheetah_q3_layers2 HalfCheetah_q3_layers3 --save HalfCheetah_q3_nn_layers || echo "⚠️ Plot skipped: some/none logs missing for layers"
