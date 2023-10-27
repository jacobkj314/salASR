import torch

import matplotlib.pyplot as plt




def visualize(spectrogram, filename):
    plt.cla()
    plt.figure(figsize=(300,8),dpi=100)
    plt.imshow(spectrogram.flip([0]).detach().numpy()) # .flip() reverses the order of the rows, so that, in the visualization, lower pitches appear lower on the y-axis
    plt.colorbar()
    plt.savefig(filename)