#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=40GB
#SBATCH --mail-user=jacob.k.johnson@utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o assignment_1-%j

#conda create -n salASR python=3.8
#pip install -r requirements.txt
conda activate salASR

python proof-of-concept.py