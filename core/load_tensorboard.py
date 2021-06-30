# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import time
import sys
import IPython
import argparse
import glob
parser = argparse.ArgumentParser(description="utility script for loading tensor board models")

parser.add_argument( '--model',
                help='model path',
                type=str,
                default='02_07_2020_12:12:51')
parser.add_argument( '--root',
                help='Root dir for data',
                type=str,
                default='.')
parser.add_argument( '--copy',
                help='copy log',
                action='store_true')
parser.add_argument( '--sleep',
                help='copy log',
                type=int,
                default=1)
parser.add_argument( '--output',
                help='output dir',
                default='output')

args = parser.parse_args()
time.sleep(args.sleep) # 1 h
print('tensorboard model', args.model)
test_models = [item.strip() for item in args.model.split(',')]
model_num = len(test_models)
output_root_dir = os.path.join(args.root, args.output)
run_root_dir = args.root + '/output'
run_models = []
tensorboard_format_str = []

for model in test_models:
	root_dir = '{}/{}'.format(output_root_dir, model)
	if not os.path.exists(root_dir) :
		root_dir = os.path.join('temp_model_output', model)

	config_file = glob.glob(root_dir + '/*.yaml')
	print('model:', model)
	config_file = [f for f in config_file if 'td3' in f or 'bc' in f or 'ddpg' in f or 'dqn' in f][0] 
	tfboard_name = config_file[len(root_dir)+1:]
	name_cnt = 1

	while tfboard_name in tensorboard_format_str:
		tfboard_name = tfboard_name + '_{}'.format(name_cnt) 
		name_cnt += 1
	
	tensorboard_format_str.append(tfboard_name)
	print('{}: {}'.format(output_root_dir, model, tfboard_name))
	run_file = glob.glob('{}/*{}*'.format(run_root_dir, model))[0]
	run_models.append(run_file)
	tensorboard_format_str.append(run_file[len(run_root_dir)+1:])

tensorboard_scripts = 'python -m tensorboard.main --bind_all --reload_interval 5 --logdir_spec='+','.join(
					  ['{}:' + run_root_dir + '/{}'] * model_num)
tensorboard_scripts = tensorboard_scripts.format(*tensorboard_format_str)
print('tensorboard_scripts:', tensorboard_scripts)	
os.system(tensorboard_scripts)