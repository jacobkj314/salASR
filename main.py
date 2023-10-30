import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

#load processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny") # # # model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
model.config.forced_decoder_ids = None

#load dataset
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

#get spectrograms from dataset instance
def get_spectrogram(instance):
    array = instance['audio']["array"]
    samples_per_second = instance['audio']["sampling_rate"]
    num_audio_samples = array.shape[0]
    spectrogram_frames_per_second = 100 # whisper constant
    num_spectrogram_frames = num_audio_samples * spectrogram_frames_per_second // samples_per_second +1
    return  processor(array, sampling_rate=samples_per_second, return_tensors="pt").input_features[0,:,:num_spectrogram_frames]
#get tokens from dataset instance
def get_decoder_input_ids(instance, use_gold=False):
    if use_gold:
        return torch.LongTensor([processor.tokenizer(instance['text']).input_ids])
    return torch.LongTensor(model(pad_for_whisper(get_spectrogram(instance))))


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

def pad_for_whisper(features):
    '''
    Whisper only accepts tensors of shape [B, 80, 3000]
    '''
    desired_shape = (80, 3000)
    ambient_intensity = features.min()
    padding = [max(0, desired_shape[i] - features.shape[i]) for i in range(2)]
    padded_tensor = torch.nn.functional.pad(features, (0, padding[1], 0, 0), mode='constant', value=float(ambient_intensity))
    return padded_tensor[None]

def build_saliency_map(features : torch.Tensor, decoder_input_ids):
    '''
    '''
    #Get inputs for model
    features.requires_grad = True
    input_features = pad_for_whisper(features)
    #Get output logits at each time step
    model_output = model.forward(input_features, decoder_input_ids=decoder_input_ids)
    logits = model_output.logits[0]
    #Sum up the logits for the predicted tokens at each step
    total = logits.gather(1, logits.argmax(dim=1)[None].T).sum() # # # # # total = logits.gather(1, decoder_input_ids.T).sum() # # # # # using predicted instead of gold tokens
    #backward pass
    total.backward(retain_graph=True)
    return features.grad

def transcribe(features):
    return processor.batch_decode(model.generate(pad_for_whisper(features)), skip_special_tokens=True)[0]

def top_r_features(instance, r=.25, balanced=True):
    features : torch.Tensor = get_spectrogram(instance)
    decoder_input_ids = get_decoder_input_ids(instance)
    saliency_map = build_saliency_map(features, decoder_input_ids)
    mask = build_saliency_mask(saliency_map, r=r, balanced=balanced)
    return mask_unsalient_features(features, mask)



def evaluate(instance, features=None):
    #get the inputs to the model
    features : torch.Tensor = features if features is not None else get_spectrogram(instance)
    input_features = pad_for_whisper(features)
    decoder_input_ids : torch.LongTensor = get_decoder_input_ids(instance)
    #get the outputs from the model
    decoder_output_ids = model.forward(input_features, decoder_input_ids=decoder_input_ids).logits.argmax(dim=-1)
    #count up how many of them match to give a score to the model
    return float((decoder_input_ids[0,1:] == decoder_output_ids[0,:-1]).sum() / (decoder_input_ids.numel() -1))


    
def ablate(instance):
    for i in range(10):
        print(((10-i)/10), evaluate(instance, top_r_features(instance, r=((10-i)/10))))