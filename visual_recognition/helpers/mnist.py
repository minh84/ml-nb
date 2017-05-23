import matplotlib.pyplot as plt
import numpy as np

# Function for displaying a training image by it's index in the MNIST set
def show_digit(dataset, labels, index, ax = None):
    label = labels[index].argmax(axis=0)

    # Reshape 784 array into 28x28 image
    image = dataset[index].reshape([28,28])

    if ax is None:
        fig, ax = plt.figure()

    # Draw image
    ax.set_title('Training label: {}'.format(label))
    ax.imshow(image, cmap='gray_r')

    return ax

def grid_digits(dataset, labels, grid_size = 3):
    fig, axes = plt.subplots(figsize=(8, 8), nrows=grid_size, ncols=grid_size, sharey=True, sharex=True)
    N = dataset.shape[0]
    for ax in axes.flatten():
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        index = np.random.randint(N)
        show_digit(dataset, labels, index, ax = ax)