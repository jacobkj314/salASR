from EvalWhisper import *
import soundfile as sf

import argparse
import json
import pathlib
from pathlib import Path
import statistics

from EvalWhisper import EvalWhisper

from datasets import load_dataset
from tqdm import tqdm


def get_audio(output_path, whisper_evaluator, inputs, mask_ratios, mode_value, what_to_mask_list, sampling_rate=16000):
    output_audio_list = []
    
    
    for idx, input in tqdm(enumerate(inputs)):
        masked_spectrograms = []
        original_spectrogram = whisper_evaluator.get_spectrogram(input)
        # print(f"os:{original_spectrogram.shape}")
        sf.write(f"{output_path}/{idx}_original_all.wav", input["audio"]["array"], sampling_rate) #whisper_evaluator.top_r_features(input, r=1.0, mode="retain", where="top").detach().numpy()
        sf.write(f"{output_path}/{idx}_{10}_all.wav", whisper_evaluator.sonify(original_spectrogram), sampling_rate) #whisper_evaluator.top_r_features(input, r=1.0, mode="retain", where="top").detach().numpy()
        with open(f"{output_path}/{idx}_original_transcription.txt", "w") as op_file:
            op_file.write(input["text"] + "\n")
        for mask_ratio in tqdm(mask_ratios):
            mask_list = []
            for mask in what_to_mask_list:
                masked_spectrogram = whisper_evaluator.top_r_features(input, r=mask_ratio, mode=mode_value, where=mask)
                masked_audio = whisper_evaluator.sonify(masked_spectrogram.detach().numpy())
                mask_list.append(masked_audio)
                sf.write(f"{output_path}/{idx}_{int(mask_ratio * 10)}_{mask[0]}.wav", masked_audio, sampling_rate)
            masked_spectrograms.append(mask_list)

    
def main(args):
    num_skipped = args.num_skipped
    num_samples = args.num_samples
    output_path = Path(args.output_dir)
    
    model_size = "large"
    model_checkpoint = f"openai/whisper-{model_size}"
    processor_checkpoint = [f"openai/whisper-{model_size}"]
    #load processor and model
    print(f"Loading model . . . ({model_checkpoint})")
    whisper_evaluator = EvalWhisper(model_checkpoint, *processor_checkpoint)
    print(f"Loaded model")
    
    print(f"Loading data . . . .")
    ds = load_dataset("librispeech_asr", split="validation.clean", streaming=True)
    print(f"Loaded data")
    
    pathlib.Path.mkdir(output_path, exist_ok=True)
    mask_ratios = [0.8, 0.5, 0.2]
    mode_value = "retain"
    what_to_mask_list = ["top", "bottom", "random"]
    # num_samples = 3
    skip_to_index = num_skipped
    inputs = []
    for sample in tqdm(ds.skip(skip_to_index).take(num_samples)):
        inputs.append(sample)
    get_audio(output_path, whisper_evaluator, inputs, mask_ratios, mode_value, what_to_mask_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--num_skipped", type=int, default=0)
    parser.add_argument("-n", "--num_samples", type=int, default=30)
    parser.add_argument("-o", "--output_dir", type=str, default="./")
    args = parser.parse_args()
    main(args)
