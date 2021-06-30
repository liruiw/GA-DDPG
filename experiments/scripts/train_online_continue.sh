#!/bin/bash
set -x
set -e

DIR="$( cd "$( dirname -- "$0" )" && pwd )"
export PYTHONUNBUFFERED="True"
LOG_NAME="agent"

SCRIPT_NAME=${1-"ddpg_finetune.yaml"}
POLICY_NAME=${2-"DDPG"}
PRETRAINED=${3-"" }
MODEL_NAME=${4-"`date +'%d_%m_%Y_%H:%M:%S'`"}

LOG="../output/${MODEL_NAME}/log.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

python -m core.train_online  --save_model --config_file ${SCRIPT_NAME}   \
		 --policy ${POLICY_NAME} --log --pretrained output/${PRETRAINED} \
		 --fix_output_time ${MODEL_NAME} --visdom

bash ./experiments/scripts/test_ycb.sh  ${MODEL_NAME}
