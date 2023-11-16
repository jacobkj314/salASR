#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --mem=40GB
#SBATCH -o outputs-%j
#SBATCH --export=ALL

export PYTHONPATH=/scratch/general/vast/<UID_HERE>/miniconda3/envs/finetuneWhisperEnv/bin/python
source /scratch/general/vast/<UID_HERE>/miniconda3/etc/profile.d/conda.sh
conda activate finetuneWhisperEnv

mkdir -p /scratch/general/vast/<UID_HERE>/huggingface_cache
export TRANSFORMERS_CACHE="/scratch/general/vast/<UID_HERE>/huggingface_cache"
export HF_DATASETS_CACHE="/scratch/general/vast/<UID_HERE>/huggingface_cache"

NUM_SAMPLES=100
MODEL_SIZE="large"
NUM_EPOCHS=10
BATCH_SIZE=16
LEARNING_RATE=2e-5
R=-1
while getopts 'b:e:l:n:s:r:'  opt; do
  case "$opt" in
    b) BATCH_SIZE="$OPTARG" ;; 
    e) NUM_EPOCHS="$OPTARG" ;; 
    l) LEARNING_RATE="$OPTARG" ;; 
    n) NUM_SAMPLES="$OPTARG" ;;
    s) MODEL_SIZE="$OPTARG" ;;
    r) R="$OPTARG" ;;
    *) echo "Unexpected option: $1 - this should not happen."
       usage ;;
  esac
done

OUT_DIR=/scratch/general/vast/<UID_HERE>/cs6966/salASR/models/$R/
mkdir -p $OUT_DIR

python3 finetuneWhisper.py -info -out $OUT_DIR -cacheDir $TRANSFORMERS_CACHE -size $MODEL_SIZE -numSamples $NUM_SAMPLES -numEpochs $NUM_EPOCHS -batchSize $BATCH_SIZE -learningRate $LEARNING_RATE -r $R