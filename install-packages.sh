#!/usr/bin/bash
is_conda_env_active="$(conda info --envs | grep -F "*" | wc -l)"
if [ $is_conda_env_active -lt 1 ]; then
	echo "Please activate your conda environment before invoking this script"
	exit 125
fi
current_dir="$(cd $(dirname $0) && pwd)"
pip install -e $current_dir/utils
pip install -e $current_dir/data_prep
pip install -e $current_dir/data_analysis
