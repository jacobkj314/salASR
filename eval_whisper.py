import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset



class EvalWhisper:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    #get spectrograms from dataset instance
    def get_spectrogram(self, instance):
        array = instance['audio']["array"]
        samples_per_second = instance['audio']["sampling_rate"]
        num_audio_samples = array.shape[0]
        spectrogram_frames_per_second = 100 # whisper constant
        num_spectrogram_frames = num_audio_samples * spectrogram_frames_per_second // samples_per_second +1
        return self.processor(array, sampling_rate=samples_per_second, return_tensors="pt").input_features[0,:,:num_spectrogram_frames]
    
    def pad_for_whisper(features):
        '''
        Whisper only accepts tensors of shape [batch_size, 80, 3000]
        '''
        desired_shape = (80, 3000)
        ambient_intensity = features.min()
        padding = [max(0, desired_shape[i] - features.shape[i]) for i in range(2)]
        padded_tensor = torch.nn.functional.pad(features, (0, padding[1], 0, 0), mode='constant', value=float(ambient_intensity))
        return padded_tensor[None]
    
    def build_saliency_mask(saliency: torch.Tensor, r=.5, balanced=True, translucent=True):
        k = int(r * saliency.numel())
        saliency_abs : torch.Tensor = saliency.abs()
        if translucent:
            if balanced:
                return saliency_abs.softmax(dim=0)
            return saliency_abs.flatten().softmax(dim=0).reshape(saliency_abs.shape)
        if balanced:
            return (saliency_abs >= saliency_abs.T.topk(k //(saliency.shape[-1])).values.min(dim=-1).values)
        return (saliency_abs >= saliency_abs.flatten().topk(k).values.min())

    def mask_unsalient_features(features: torch.Tensor, mask):
        '''
        For some reason, the ambient intensity of whisper spectrograms is slightly negative and differs slightly between instances
        This method applies a mask that matches the ambient intensity
        '''
        ambient_intensity = features.min()
        return  (
                    (features - ambient_intensity)  # Shift ambient intensity to 0 ...
                    * mask                          # ... then mask ...
                    + ambient_intensity             # ... then shift back
                )

    #get tokens from dataset instance
    def get_decoder_input_ids(self, instance, use_gold=False):
        if use_gold:
            return torch.LongTensor([self.processor.tokenizer(instance['text']).input_ids])
        return torch.LongTensor(self.model.generate(EvalWhisper.pad_for_whisper(self.get_spectrogram(instance))))
        
    def build_saliency_map(self, features : torch.Tensor, decoder_input_ids):
        '''
        '''
        #Get inputs for model
        features.requires_grad = True
        input_features = EvalWhisper.pad_for_whisper(features)
        #Get output logits at each time step
        model_output = self.model.forward(input_features, decoder_input_ids=decoder_input_ids)
        logits = model_output.logits[0]
        #Sum up the logits for the predicted tokens at each step
        total = logits.gather(1, logits.argmax(dim=1)[None].T).sum() # # # # # total = logits.gather(1, decoder_input_ids.T).sum() # # # # # using predicted instead of gold tokens
        #backward pass
        total.backward(retain_graph=True)
        return features.grad

    def transcribe(self, features):
        return self.processor.batch_decode(self.model.generate(self.pad_for_whisper(features)), skip_special_tokens=True)[0]

    def top_r_features(self, instance, r=.25, balanced=True):
        features : torch.Tensor = self.get_spectrogram(instance)
        decoder_input_ids = self.get_decoder_input_ids(instance)
        saliency_map = self.build_saliency_map(features, decoder_input_ids)
        mask = EvalWhisper.build_saliency_mask(saliency_map, r=r, balanced=balanced)
        return EvalWhisper.mask_unsalient_features(features, mask)



    def evaluate(self, instance, features=None):
        #get the inputs to the model
        features : torch.Tensor = features if features is not None else self.get_spectrogram(instance)
        input_features = EvalWhisper.pad_for_whisper(features)
        decoder_input_ids : torch.LongTensor = self.get_decoder_input_ids(instance)
        #get the outputs from the model
        decoder_output_ids = self.model.forward(input_features, decoder_input_ids=decoder_input_ids).logits.argmax(dim=-1)
        #count up how many of them match to give a score to the model
        return float((decoder_input_ids[0,1:] == decoder_output_ids[0,:-1]).sum() / (decoder_input_ids.numel() -1))


        
    def ablate(self, instance):
        for i in range(10):
            print(((10-i)/10), self.evaluate(instance, self.top_r_features(instance, r=((10-i)/10))))


def main():
    #load processor and model
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny") # # # model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
    model.config.forced_decoder_ids = None
    whisper_evaluator = EvalWhisper(model, processor)
    ds = load_dataset("librispeech_asr", split="validation.clean")
    print(f"Loaded data")
    tiny_scores = []
    for sample in ds.select(list(range(10))):
        tiny_scores.append(whisper_evaluator.evaluate(sample, whisper_evaluator.top_r_features(sample, r=0.1)))
    print(f"tiny_scores:{tiny_scores}")
    with open("tiny_output.txt", "w") as op_file:
        for score in tiny_scores:
            op_file.write(str(score)+"\n")


if __name__ == "__main__":
    main()