import argparse, logging

import transformers


import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

from ExplainableWhisper import ExplainableWhisper # # # 

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = ExplainableWhisper.from_pretrained("openai/whisper-tiny") # # # model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
model.config.forced_decoder_ids = None

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
#sample = ds[0]["audio"]

#Make spectrogram ?
#input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

# # # # # predicted_ids = model.generate(input_features)
#model_output = model.forward(input_features, decoder_input_ids=torch.LongTensor([[50258, 1, 2, 3]]))

#logits = model_output.logits
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

# print(transcription)



#load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train[:1%]+validation[:1%]+test[:1%]")

def build_saliency_mask(saliency: torch.Tensor, r=.5, k=None):
    if k is None:
        k = int(r * saliency.numel())
    saliency_abs = saliency.abs()
    return (saliency_abs >= saliency_abs.flatten().topk(k).values.min())



def build_saliency_map(sample):
    input_features = processor(sample['audio']["array"], sampling_rate=sample['audio']["sampling_rate"], return_tensors="pt").input_features

    input_features.requires_grad = True

    sample['input_features'] = input_features

    decoder_input_ids = torch.LongTensor([processor.tokenizer(sample['text']).input_ids])

    model_output = model.forward(input_features, decoder_input_ids=decoder_input_ids)
    logits = model_output.logits[0]

    total = logits.gather(1, logits.argmax(dim=1)[None].T).sum()

    total.backward(retain_graph=True)

    return input_features.grad


def top_r_features(sample):
    saliency_map = build_saliency_map(sample)

    saliency_mask = build_saliency_mask(saliency_map)

    return sample['input_features'] * saliency_mask

