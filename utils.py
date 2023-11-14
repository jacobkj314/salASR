import torch

import matplotlib.pyplot as plt

def pad_for_whisper(features):
    '''
    Whisper only accepts tensors of shape [B, 80, 3000]
    '''
    desired_shape = (80, 3000)
    ambient_intensity = features.min()
    padding = [max(0, desired_shape[i] - features.shape[i]) for i in range(2)]
    padded_tensor = torch.nn.functional.pad(features, (0, padding[1], 0, 0), mode='constant', value=float(ambient_intensity))
    return padded_tensor[None]



def build_saliency_mask(saliency: torch.Tensor, r=.5, balanced=True, translucent=False, mode="retain", where="top"):
    """
    translucent=True is not working, don't use
    """
    if balanced == False:
        print("balanced=False for build_saliency_mask is no longer supported, ignoring this and using balanced=True")

    assert where in {"top", "bottom", "random"}
    assert mode in {"retain", "remove"}
    remove_top = (mode == "remove")
    
    if where == "bottom":
        r = 1-r
        remove_top = not remove_top

    k = int(r * saliency.numel())

    if where == "random":
        mask = torch.rand_like(saliency)
        mask = (mask >= mask.T.topk(k //(mask.shape[-1])).values.min(dim=-1).values)
    else: #where in {"top", "bottom"} #if "bottom" then 
        saliency_abs : torch.Tensor = saliency.abs()
        mask = (saliency_abs >= saliency_abs.T.topk(k //(saliency.shape[-1])).values.min(dim=-1).values)

    if remove_top:
        mask = ~mask

    return mask

    # # # # # Dead code
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

def visualize(spectrogram, filename):
    plt.cla()
    plt.figure(figsize=(300,8),dpi=100)
    plt.imshow(spectrogram.flip([0]).detach().numpy()) # .flip() reverses the order of the rows, so that, in the visualization, lower pitches appear lower on the y-axis
    plt.colorbar()
    plt.savefig(filename)