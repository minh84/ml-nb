import matplotlib.pyplot as plt
import numpy as np

# Function for displaying a training image by it's index in the MNIST set
def show_digit(img, label, ax = None):

    # Reshape 784 array into 28x28 image
    img = img.reshape([28,28])

    if ax is None:
        fig, ax = plt.figure()

    # Draw image
    ax.set_title('Training label: {}'.format(label))
    ax.imshow(img, cmap='gray_r')

    return ax

def grid_digits(dataset, grid_size = 3):
    fig, axes = plt.subplots(figsize=(8, 8), nrows=grid_size, ncols=grid_size, sharey=True, sharex=True)
    N = dataset.num_examples
    for ax in axes.flatten():
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        index = np.random.randint(N)
        img = dataset.images[index]

        label = dataset.labels[index].argmax(axis=0)
        show_digit(img, label, ax = ax)