#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --mem=40GB
#SBATCH --mail-user=guru@cs.utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o op-%j
#SBATCH --export=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate whisper

mkdir -p /scratch/general/vast/u1419540/huggingface_cache
export TRANSFORMERS_CACHE="/scratch/general/vast/u1419540/huggingface_cache"
export HF_DATASETS_CACHE="/scratch/general/vast/u1419540/huggingface_cache"

OUT_DIR=/scratch/general/vast/u1419540/salASR/streaming_outputs/output_r_0.8
mkdir -p $OUT_DIR
python main.py -n 100 -m tiny -o $OUT_DIR -r 0.9
