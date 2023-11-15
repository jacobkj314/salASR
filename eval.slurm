#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --mem=40GB
#SBATCH --mail-user=####
#SBATCH --mail-type=FAIL,END
#SBATCH -o op-%j
#SBATCH --export=ALL


# REPLACE any string with #### with your own values/paths, etc.,
# In short, search for #### in this file and replace all of them

#UID=####
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ####

mkdir -p /scratch/general/vast/$UID/huggingface_cache
export TRANSFORMERS_CACHE=/scratch/general/vast/$UID/huggingface_cache
export HF_DATASETS_CACHE=/scratch/general/vast/$UID/huggingface_cache

OUT_DIR=/scratch/general/vast/$UID/#### salASR/masked_outputs/
mkdir -p $OUT_DIR
python main.py -s 0 -n 100 -m large -o $OUT_DIR --mode remove --what random -r 0.9
