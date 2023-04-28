#!/bin/bash

#SBATCH --partition=long                                                # Ask for unkillable job
#SBATCH -o /network/scratch/a/aliakbar.ghayouri-sales/slurm-%j.out      # Write the log on scratch
#SBATCH --cpus-per-task=2                                               # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                                    # Ask for 1 GPU
#SBATCH --mem=8G                                                       # Ask for 10 GB of RAM

module load python/3.8

source $HOME/wsi/bin/activate

cp -r $HOME/scratch/movasaghi_files/xray_classify/chest_xray $SLURM_TMPDIR/

python -u code/server_train.py $HOME/scratch/movasaghi_files/xray_classify/output