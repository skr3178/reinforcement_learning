#!/bin/bash

python3 -u -m run_sb3_ppo \
  --level 11_vs_11_easy_stochastic \
  --reward_experiment scoring \
  --policy CnnPolicy \
  --cliprange 0.115 \
  --gamma 0.997 \
  --ent_coef 0.00155 \
  --num_timesteps 500000 \
  --max_grad_norm 0.76 \
  --lr 0.00011879 \
  --num_envs 16 \
  --state extracted_stacked \
  --save_path ./ppo_sb3_gfootball_impala_params \
  "$@" 