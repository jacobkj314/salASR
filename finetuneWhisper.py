import logging 
import argparse
from pathlib import Path
from os.path import exists
import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import numpy as np

MODELS = {
    "whisper": {
        "tiny": "openai/whisper-tiny",
        "large": "openai/whisper-large"
    }
}

TRAIN_SPLIT="train.clean.100"
VAL_SPLIT="validation.clean"
MAX_LENGTH=256

parser = argparse.ArgumentParser()

parser.add_argument(
    "-info",
    action="store_true",
    help="Boolean flag to enable info mode"
)

parser.add_argument(
    "-log",
    "--logFile",
    type=str,
    help="Path to file to print logging information",
    default=None
)

parser.add_argument(
    "-cacheDir",
    help="Path to cache location for Huggingface",
    default="/scratch/general/vast/u1419542/huggingface_cache/"
)

parser.add_argument(
    "-dataset",
    choices = [
        "librispeech_asr",
    ],
    help="Name of HF dataset to use",
    default="librispeech_asr",
)

parser.add_argument(
    "-model",
    choices=["whisper"],
    help="Name of HF model to use",
    default="whisper"
)

parser.add_argument(
    "-size",
    choices=["tiny", "large"],
    help="Size of model to use",
    default="tiny"
)

parser.add_argument(
    "-numSamples",
    type=int,
    help="No. of samples to get from dataset",
    default=10
)

parser.add_argument(
    "-numEpochs",
    type=int,
    help="Number of epochs to train model for",
    default=1
)

parser.add_argument(
    "-batchSize",
    type=int,
    help="Batch size of dataloader",
    default=16
)

parser.add_argument(
    "-learningRate",
    type=float,
    help="Learning rate for optimizer",
    default=2e-5
)

parser.add_argument(
    "-out",
    "--outputDir",
    help="Path to output directory where trained model is to be saved",
    default="/scratch/general/vast/u1419542/cs6966/salASR/"
)

parser.add_argument(
    '-seed', 
    type=int, 
    help='Random seed', 
    default=13
)

parser.add_argument(
    "-r",
    type=float,
    help="Percentage of input spectogram features to retain (Balanced by default) (set to -1 for random r for every instance)",
    default=-1
)
#----------------------------------------------------------------------
def checkIfExists(path, isDir=False, createIfNotExists=False): 
    if isDir and not path.endswith("/"):
        raise ValueError("Directory path should end with '/'")
    pathExists = exists(path)
    if not pathExists:
        if createIfNotExists:
            os.makedirs(path) 
        else:
            raise ValueError(f"{path} is an invalid path!")
    if not isDir:
        filePath = Path(path)
        if not filePath.is_file():
            raise ValueError(f"{path} is not a file!")
#----------------------------------------------------------------------
def checkFile(fileName, fileExtension=None):
    if fileExtension:
        if not fileName.endswith(fileExtension):
            raise ValueError(f"[checkFile] {fileName} does not have expected file extension {fileExtension}!")
    file_exists = exists(fileName)
    if not file_exists:
        raise RuntimeError(f"[checkFile] {fileName} is an invalid file path!")
    path = Path(fileName)
    if not path.is_file():
        raise RuntimeError(f"[checkFile] {fileName} is not a file!")
#----------------------------------------------------------------------
class PrepareDataset():
    def __init__(self, processor, r=None, balanced=True):
        self.processor = processor 
        self.r = r 
        self.balanced = balanced
    
    def mask_unsalient_features(self, features: torch.Tensor, mask):
        ambient_intensity = features.min()
        return  (
                    (features - ambient_intensity)  # Shift ambient intensity to 0 ...
                    * mask                          # ... then mask ...
                    + ambient_intensity             # ... then shift back
                )
    
    def build_saliency_mask(self, saliency: torch.Tensor, r=.5, balanced=True):
        k = int(r * saliency.numel())
        saliency_abs : torch.Tensor = saliency.abs()
        if balanced:
            return (saliency_abs >= saliency_abs.T.topk(k //(saliency.shape[-1])).values.min(dim=-1).values)
        return (saliency_abs >= saliency_abs.flatten().topk(k).values.min())

    def top_r_features(self, instance, r=.25, balanced=True):
        if r == -1:
            r = min(100, max((int(torch.rand(1).item()*100), 10)))/100
        features : torch.Tensor = self.get_spectrogram(instance)
        saliency_map = torch.rand(features.shape)
        mask = self.build_saliency_mask(saliency=saliency_map, r=r, balanced=balanced)
        return self.mask_unsalient_features(features, mask)

    def get_spectrogram(self, instance):
        assert self.processor != None
        array = instance['audio']["array"]
        samples_per_second = instance['audio']["sampling_rate"]
        num_audio_samples = array.shape[0]
        spectrogram_frames_per_second = 100 # whisper constant
        num_spectrogram_frames = num_audio_samples * spectrogram_frames_per_second // samples_per_second +1
        return  self.processor(array, sampling_rate=samples_per_second, return_tensors="pt").input_features[0,:,:num_spectrogram_frames]
    
    def __call__(self, instance):
        if self.r == None or self.r == 1:
            instance.update({
                "input_features": self.get_spectrogram(instance),
                "labels": self.processor.tokenizer(instance["text"]).input_ids
            })
        else: 
            instance.update({
                "input_features": self.top_r_features(instance, self.r, self.balanced),
                "labels": self.processor.tokenizer(instance["text"]).input_ids
            })
        return instance
#----------------------------------------------------------------------
class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def pad_for_whisper(self, features):
        '''
        Whisper only accepts tensors of shape [B, 80, 3000]
        '''
        desired_shape = (80, 3000)
        ambient_intensity = features.min()
        padding = [max(0, desired_shape[i] - features.shape[i]) for i in range(2)]
        padded_tensor = torch.nn.functional.pad(features, (0, padding[1], 0, 0), mode='constant', value=float(ambient_intensity))
        return padded_tensor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        
        input_features = [{"input_features": self.pad_for_whisper(feature["input_features"])} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
#----------------------------------------------------------------------
class computeMetrics():
    def __init__(self, tokenizer, cacheDir):
        self.tokenizer = tokenizer
        self.cacheDir = cacheDir
        self.metric = evaluate.load("wer", cache_dir=cacheDir)

    def __call__(self, pred):    
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
#----------------------------------------------------------------------
def main(errTrace="main"):
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.logFile:
        checkFile(args.logFile)
        logging.basicConfig(filename=args.logFile, filemode='w', level=logging.INFO)
    elif args.info:
        logging.basicConfig(filemode='w', level=logging.INFO)
    else:
        # logging.basicConfig(filemode='w', level=logging.ERROR)
        logging.basicConfig(filemode='w', level=logging.INFO)

    if torch.cuda.is_available():
        logging.info("Using GPU: cuda")
        device = "cuda"
    else: 
        logging.info("Using CPU")
        device = "cpu"

    if args.batchSize <= 0:
        raise ValueError("[{}] Batch Size has to be a positive number!".format(errTrace))
    
    if args.numEpochs <= 0:
        raise ValueError("[{}] No. of epochs has to be a positive number!".format(errTrace))

    if not ((0 <= args.r <= 1.0) or args.r == -1):
        raise ValueError("[{}] r has to be a value in [0, 1] U {-1}!".format(errTrace))

    logging.info(args)

    checkIfExists(args.outputDir, isDir=True, createIfNotExists=True)

    model = WhisperForConditionalGeneration.from_pretrained(MODELS[args.model][args.size], cache_dir=args.cacheDir)
    processor = WhisperProcessor.from_pretrained(MODELS[args.model][args.size], cache_dir=args.cacheDir)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.to(device)

    trainDS = load_dataset(args.dataset, split=TRAIN_SPLIT, streaming=True, cache_dir=args.cacheDir)
    valDS = load_dataset(args.dataset, split=VAL_SPLIT, streaming=True, cache_dir=args.cacheDir)
    trainDS = trainDS.take(args.numSamples)
    valDS = valDS.take(args.numSamples)
    trainDS = trainDS.map(PrepareDataset(processor, args.r))
    valDS = valDS.map(PrepareDataset(processor, args.r))

    dataCollator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.outputDir, 
        per_device_train_batch_size=args.batchSize,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=args.learningRate,
        warmup_steps=0,
        max_steps=(args.numEpochs)*args.numSamples,
        gradient_checkpointing=True,
        fp16=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_eval_batch_size=args.batchSize,
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        save_steps=1,
        # eval_steps=1,
        logging_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=trainDS,
        eval_dataset=valDS,
        data_collator=dataCollator,
        compute_metrics=computeMetrics(processor.tokenizer, args.cacheDir),
        tokenizer=processor.tokenizer,
    )

    trainer.train()
#----------------------------------------------------------------------
if __name__=="__main__":
    main()