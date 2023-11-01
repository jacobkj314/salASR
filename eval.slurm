#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --mem=40GB
#SBATCH --mail-user=
#SBATCH --mail-type=FAIL,END
#SBATCH -o op-%j
#SBATCH --export=ALL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate salASR

mkdir -p /scratch/general/vast/u0403624/huggingface_cache
export TRANSFORMERS_CACHE="/scratch/general/vast/u0403624/huggingface_cache"
export HF_DATASETS_CACHE="/scratch/general/vast/u0403624/huggingface_cache"

OUT_DIR=/scratch/general/vast/u0403624/salASR/streaming_outputs/output
mkdir -p $OUT_DIR
python streaming_main.py -s 200 -n 100 -m large -o $OUT_DIR -r 0.9
