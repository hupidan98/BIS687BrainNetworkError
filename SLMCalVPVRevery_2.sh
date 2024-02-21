#!/bin/bash
#SBATCH --partition=day       # specify the partition or queue to use
#SBATCH --nodes=1                 # request one node
#SBATCH --ntasks-per-node=1       # use one task (process) per node
#SBATCH --cpus-per-task=63        # use 36 CPUs per task
#SBATCH --time=15:00:00            # set the walltime to 15 hours
#SBATCH --mem-per-cpu=5G

#SBATCH --mail-user=ruize.han@yale.edu
#SBATCH --mail-type=begin,end,fail

# load any necessary modules

module load miniconda
conda activate py3_env


# run the Python script with 36 CPUs
python ZCal_Cal_VPVRevery_2.py

conda deavtivate
