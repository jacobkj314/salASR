import argparse, logging

import transformers



from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

from ExplainableWhisper import ExplainableWhisper # # # 

processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = ExplainableWhisper.from_pretrained("openai/whisper-large") # # # model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
model.config.forced_decoder_ids = None

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]

#Make spectrogram ?
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

input_features.requires_grad = True


predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(transcription)
