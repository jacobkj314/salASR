from EvalWhisper import EvalWhisper

from datasets import load_dataset

def main():
    #load processor and model
    whisper_evaluator = EvalWhisper("openai/whisper-tiny")
    ds = load_dataset("librispeech_asr", split="validation.clean", streaming=True)
    print(f"Loaded data")
    tiny_scores = []
    for sample in ds.take(10):
        tiny_scores.append(whisper_evaluator.evaluate(sample, whisper_evaluator.top_r_features(sample, r=0.1)))
    print(f"tiny_scores:{tiny_scores}")
    with open("tiny_output.txt", "w") as op_file:
        for score in tiny_scores:
            op_file.write(str(score)+"\n")


if __name__ == "__main__":
    main()