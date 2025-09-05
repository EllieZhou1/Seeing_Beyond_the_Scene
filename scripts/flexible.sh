#!/bin/sh
#SBATCH --exclude neu306,neu301,neu327
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=20     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:2         # the number of GPUs requested
#SBATCH --mem=50G           # memory
#SBATCH -o slurm/outfile_new        # send stdout to outfile
#SBATCH -e slurm/errfile_new  # send stderr to errfile
#SBATCH -t 168:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=ez9517@princeton.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ellie_env2

python flexible_train_test.py --config $1