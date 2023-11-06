#!/bin/bash
#SBATCH --account <ACCOUNT>
#SBATCH --partition <PARTITION>
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --mem=40GB
#SBATCH -o outputs-%j
#SBATCH --export=ALL

export PYTHONPATH=<PATH_TO_MINICONDA>/envs/finetuneWhisperEnv/bin/python
conda activate finetuneWhisperEnv

mkdir -p <HF_CACHE_PATH>
export TRANSFORMERS_CACHE=<HF_CACHE_PATH>
export HF_DATASETS_CACHE=<HF_CACHE_PATH>

OUT_DIR=<OUT_DIR>
mkdir -p $OUT_DIR

NUM_SAMPLES=100
MODEL_SIZE="tiny"
NUM_EPOCHS=10
BATCH_SIZE=16
LEARNING_RATE=2e-5
while getopts 'b:e:l:n:s:'  opt; do
  case "$opt" in
    b) BATCH_SIZE="$OPTARG" ;; 
    e) NUM_EPOCHS="$OPTARG" ;; 
    l) LEARNING_RATE="$OPTARG" ;; 
    n) NUM_SAMPLES="$OPTARG" ;;
    s) MODEL_SIZE="$OPTARG" ;;
    *) echo "Unexpected option: $1 - this should not happen."
       usage ;;
  esac
done

python3 finetuneWhisper.py \
    -info \
    -out $OUT_DIR \
    -cacheDir $TRANSFORMERS_CACHE \
    -size $MODEL_SIZE \    
    -numSamples $NUM_SAMPLES \
    -numEpochs $NUM_EPOCHS \ 
    -batchSize $BATCH_SIZE \
    -learningRate $LEARNING_RATE 