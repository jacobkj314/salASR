import torch
import numpy as np
import librosa
from utils import pad_for_whisper, build_saliency_mask, mask_unsalient_features

from transformers import WhisperProcessor, WhisperForConditionalGeneration


class EvalWhisper:

    def __init__(self, model_checkpoint):
        self.processor = WhisperProcessor.from_pretrained(model_checkpoint)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_checkpoint)
        self.model.config.forced_decoder_ids = None

    #get spectrograms from dataset instance
    def get_spectrogram(self, instance):
        array = instance['audio']["array"]
        samples_per_second = instance['audio']["sampling_rate"]
        num_audio_samples = array.shape[0]
        spectrogram_frames_per_second = 100 # whisper constant
        num_spectrogram_frames = num_audio_samples * spectrogram_frames_per_second // samples_per_second +1
        return  self.processor(array, sampling_rate=samples_per_second, return_tensors="pt").input_features[0,:,:num_spectrogram_frames]
    #get tokens from dataset instance
    def get_decoder_input_ids(self, instance, use_gold=False):
        if use_gold:
            return torch.LongTensor([self.processor.tokenizer(instance['text']).input_ids])
        return torch.LongTensor(self.model.generate(pad_for_whisper(self.get_spectrogram(instance))))


    def build_saliency_map(self, features : torch.Tensor, decoder_input_ids):
        '''
        '''
        #Get inputs for model
        features.requires_grad = True
        input_features = pad_for_whisper(features)
        #Get output logits at each time step
        model_output = self.model.forward(input_features, decoder_input_ids=decoder_input_ids)
        logits = model_output.logits[0]
        #Sum up the logits for the predicted tokens at each step
        total = logits.gather(1, logits.argmax(dim=1)[None].T).sum() # # # # # total = logits.gather(1, decoder_input_ids.T).sum() # # # # # using predicted instead of gold tokens
        #backward pass
        total.backward(retain_graph=True)
        return features.grad

    def transcribe(self, features):
        return self.processor.batch_decode(self.model.generate(pad_for_whisper(features)), skip_special_tokens=True)[0]

    def top_r_features(self, instance, r=.25, balanced=True):
        features : torch.Tensor = self.get_spectrogram(instance)
        decoder_input_ids = self.get_decoder_input_ids(instance)
        saliency_map = self.build_saliency_map(features, decoder_input_ids)
        mask = build_saliency_mask(saliency_map, r=r, balanced=balanced)
        return mask_unsalient_features(features, mask)



    def evaluate(self, instance, features=None):
        #get the inputs to the model
        features : torch.Tensor = features if features is not None else self.get_spectrogram(instance)
        input_features = pad_for_whisper(features)
        decoder_input_ids : torch.LongTensor = self.get_decoder_input_ids(instance)
        #get the outputs from the model
        decoder_output_ids = self.model.forward(input_features, decoder_input_ids=decoder_input_ids).logits.argmax(dim=-1)
        #count up how many of them match to give a score to the model
        return float((decoder_input_ids[0,1:] == decoder_output_ids[0,:-1]).sum() / (decoder_input_ids.numel() -1))



    def ablate(self, instance):
        for i in range(10):
            print(((10-i)/10), self.evaluate(instance, self.top_r_features(instance, r=((10-i)/10))))



    def sonify(self, spectrogram):
        return librosa.istft(
                                np.abs  (
                                            np.dot  (
                                                        np.linalg.pinv(self.processor.feature_extractor.mel_filters).T,
                                                        10**(4 * spectrogram - 4)
                                                    )
                                        ) ** 0.5
                            )
