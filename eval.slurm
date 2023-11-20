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

U_ID=u0403624
source ~/miniconda3/etc/profile.d/conda.sh
conda activate finetuneWhisperEnv

mkdir -p /scratch/general/vast/$U_ID/huggingface_cache
export TRANSFORMERS_CACHE=/scratch/general/vast/$U_ID/huggingface_cache
export HF_DATASETS_CACHE=/scratch/general/vast/$U_ID/huggingface_cache

OUT_DIR=/scratch/general/vast/$U_ID/salASR/masked_outputs/
mkdir -p $OUT_DIR
python main.py -s 0 -n 100 -o $OUT_DIR --mode remove --what random -r 1 -c /scratch/general/vast/u0403624/cs6966/salASR/models/current/checkpoint-1440
