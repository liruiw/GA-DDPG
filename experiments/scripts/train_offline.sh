#!/bin/bash
set -x
set -e

DIR="$( cd "$( dirname -- "$0" )" && pwd )"
export PYTHONUNBUFFERED="True"
LOG_NAME="agent"


exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
SCRIPT_NAME=${1-"ddpg_finetune.yaml"}
POLICY_NAME=${2-"DDPG"}
MODEL_NAME=${4-"`date +'%d_%m_%Y_%H:%M:%S'`"}
LOG="output/${MODEL_NAME}/log.txt"


time python -m core.train_test_offline  --save_model   \
		 --config_file ${SCRIPT_NAME} --policy ${POLICY_NAME} --log  --fix_output_time ${MODEL_NAME}  \
		    --finetune

bash ./experiments/scripts/test_ycb.sh  ${MODEL_NAME}
