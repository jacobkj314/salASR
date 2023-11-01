import argparse
import json
from pathlib import Path
import statistics

from EvalWhisper import EvalWhisper

from datasets import load_dataset
from tqdm import tqdm

def calculate_stats(scores_list):
    return statistics.mean(scores_list), statistics.stdev(scores_list)

def dump_output(scores_list, mean, standard_deviation, output_path):
    output_dict = {}
    output_dict["mean"] = mean
    output_dict["standard_deviation"] = standard_deviation
    output_dict["scores"] = scores_list
    with open(output_path, "w") as output_file:
        json.dump(output_dict, output_file)

def main(args):
    # parse args
    args = parser.parse_args()
    num_samples = args.num_samples
    model_size = args.model_size
    output_dir = Path(args.output_dir)
    r_value = args.r_value
    #load processor and model
    print(f"Loading model . . . ")
    whisper_evaluator = EvalWhisper(f"openai/whisper-{model_size}")
    print(f"Loaded model")
    ds = load_dataset("librispeech_asr", split="validation.clean", streaming=True)
    print(f"Loaded data")
    scores_list = []
    for sample in tqdm(ds.take(num_samples)):
        scores_list.append(whisper_evaluator.evaluate(sample, whisper_evaluator.top_r_features(sample, r=r_value)))
    print(f"scores_list:{scores_list}")
    output_path = output_dir / "output.json"
    mean, standard_deviation = calculate_stats(scores_list)
    dump_output(scores_list, mean, standard_deviation, output_path)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_samples", type=int, default=10)
    parser.add_argument("-m", "--model_size", type=str, default="tiny")
    parser.add_argument("-o", "--output_dir", type=str, default="./")
    parser.add_argument("-r", "--r_value", type=float, default=1.0)
    args = parser.parse_args()
    main(args)