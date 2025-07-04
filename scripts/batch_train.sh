#!/bin/sh
# PBS -q rt_HF
# PBS -l select=1
# PBS -l walltime=30:00:00
# PBS -P gch51615

cd ${PBS_O_WORKDIR}

# source /etc/profile.d/modules.sh
# module load cuda/12.6/12.6.1

source ~/anomar/bin/activate
bash scripts/train_mc.sh