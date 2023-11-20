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
    num_skipped = args.num_skipped
    num_samples = args.num_samples
    model_size = args.model_size
    r_value = args.r_value
    mode_value = args.mode
    what_to_mask = args.what
    output_dir = Path(args.output_dir)
    output_file = f"r{r_value}_mode{mode_value}_mask{what_to_mask}" + "_" + args.output_file

    model_checkpoint = f"openai/whisper-{model_size}" if args.model_checkpoint == "" else args.model_checkpoint
    
    #load processor and model
    print(f"Loading model . . . ({model_checkpoint})")
    whisper_evaluator = EvalWhisper(model_checkpoint)
    print(f"Loaded model")
    ds = load_dataset("librispeech_asr", split="validation.clean", streaming=True)
    print(f"Loaded data")
    scores_list = []
    with open(output_dir / output_file, "a") as score_writer:
        for sample in tqdm(ds.take(num_samples)):
            score = whisper_evaluator.evaluate(sample, whisper_evaluator.top_r_features(sample, r=r_value, mode=mode_value, where=what_to_mask))
            scores_list.append(score)
            score_writer.write(str(score) + "\n")
    print(f"scores_list:{scores_list}")
    output_path = output_dir / f"output_r{r_value}_mode{mode_value}_mask{what_to_mask}.json"
    mean, standard_deviation = calculate_stats(scores_list)
    dump_output(scores_list, mean, standard_deviation, output_path)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--num_skipped", type=int, default=0)
    parser.add_argument("-n", "--num_samples", type=int, default=10)
    parser.add_argument("-c", "--model_checkpoint", type=str, default="")
    parser.add_argument("-m", "--model_size", type=str, default="tiny")
    parser.add_argument("-o", "--output_dir", type=str, default="./")
    parser.add_argument("-f", "--output_file", type=str, default="output.txt")
    parser.add_argument("-r", "--r_value", type=float, default=1.0)
    parser.add_argument('--mode',
                    default='retain',
                    const='retain',
                    nargs='?',
                    choices=["retain", "remove"],
                    help='arg for retaining/removing top_r feats (default: %(default)s)')
    parser.add_argument('--what',
                    default='top',
                    const='top',
                    nargs='?',
                    choices=["top", "bottom", "random"],
                    help='Specifies which part of the feature space the mask acts on to retin/remove the features (default: %(default)s)')

    args = parser.parse_args()
    main(args)
