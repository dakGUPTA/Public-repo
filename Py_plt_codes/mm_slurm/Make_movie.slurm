#!/bin/bash
#SBATCH --job-name=VM_plots
#SBATCH --time=200:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --oversubscribe
#SBATCH --output=/export/home/ms22resch11003/CUDA_codes/KKSMV_VM/bin/prof/plot_out.txt

module load anaconda/24.1.2
cd /export/home/ms22resch11003/CUDA_codes/KKSMV_VM/bin/
conda activate pyenv
python ~/Make_movie5.py

export CUDA_VISIBLE_DEVICES=0

echo "SLURM_JOBID="$SLURM_JOBID                                             
echo "SLURMTMPDIR="$SLURMTMPDIR                                             
echo "Date              = $(date)"                                          
echo "Hostname          = $(hostname -s)"                                   
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"                         
echo "Number of Nodes Allocated         = $SLURM_JOB_NUM_NODES"             
echo "Number of Tasks Allocated         = $SLURM_NTASKS"                    
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"                
echo "Working Directory = $(pwd)"                                           
echo "working directory = "$SLURM_SUBMIT_DIR       

module unload anaconda/24.1.2 
