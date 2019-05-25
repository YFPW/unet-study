#!/bin/bash
#SBATCH --account=punim0619
#SBATCH --mail-user=fany10@student.unimelb.edu.au
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=unet

echo '---------------------------- loading module ------------------------------------'
module load Keras/2.1.5-intel-2017.u2-GCC-6.2.0-CUDA9-Python-3.6.4-GPU

# Launch 
time python main.py
