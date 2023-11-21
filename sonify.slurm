#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:30:00
#SBATCH --mem=40GB
#SBATCH --mail-user=guru@cs.utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o op-%j
#SBATCH --export=ALL


# REPLACE any string with #### with your own values/paths, etc.,
# In short, search for #### in this file and replace all of them

U_ID=u1419540
source ~/miniconda3/etc/profile.d/conda.sh
conda activate whisper

mkdir -p /scratch/general/vast/$U_ID/huggingface_cache
export TRANSFORMERS_CACHE=/scratch/general/vast/$U_ID/huggingface_cache
export HF_DATASETS_CACHE=/scratch/general/vast/$U_ID/huggingface_cache

OUT_DIR=/scratch/general/vast/$U_ID/salASR/dummy_audio_outputs_0_30/
mkdir -p $OUT_DIR
python sonify.py -s 0 -n 2 -o $OUT_DIR/dummy_op1/
