#!/bin/sh
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=36     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:1         # the number of GPUs requested
#SBATCH --mem=50G             # memory
#SBATCH -o /n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/slurm/outfile_minikinetics        # send stdout to outfile
#SBATCH -e /n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/slurm/errfile_minikinetics  # send stderr to errfile
#SBATCH -t 168:00:00           # time requested in hour:minute:second
#SBATCH --mail-type=all       # choice between begin, end, all to notify you via email
#SBATCH --mail-user=ez9517@princeton.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ellie_env
python slowfast.py --config $1
