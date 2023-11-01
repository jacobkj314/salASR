import argparse
from pathlib import Path
from EvalWhisper import EvalWhisper

from datasets import load_dataset

def main(args):
    # parse args
    args = parser.parse_args()
    num_samples = args.num_samples
    model_size = args.model_size
    output_dir = Path(args.output_path)
    #load processor and model
    whisper_evaluator = EvalWhisper("openai/whisper-{model_size}")
    ds = load_dataset("librispeech_asr", split="validation.clean", streaming=True)
    print(f"Loaded data")
    scores_list = []
    for sample in ds.take(num_samples):
        scores_list.append(whisper_evaluator.evaluate(sample, whisper_evaluator.top_r_features(sample, r=1.0)))
    print(f"scores_list:{scores_list}")
    with open(output_dir / "output.txt", "w") as op_file:
        for score in scores_list:
            op_file.write(str(score)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_samples", type=int, default=10)
    parser.add_argument("-m", "--model_size", type=str, default="tiny")
    parser.add_argument("-o", "--output_dir", type=str, default="./")
    args = parser.parse_args()
    main(args)