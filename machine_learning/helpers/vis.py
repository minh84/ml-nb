import matplotlib.pyplot as plt
import numpy as np

def draw(x, y, fig_ax = None, title = None, xy_labels = None, legends=None, plot_type = "scatter", colors = None):
    '''
    Utility function to do scatter plot
    :param x: 1d array
    :param y: 1d or 2d array
    :param title: title of the graph
    :param xy_labels: labels for x & y axis
    :param legends: legends for each series
    :param plot_type: can be a string of an array of string
    :return: 
        None, it draws to screen a scatter plot
    '''
    x = np.array(x)
    y = np.array(y)

    if isinstance(plot_type, list):
        plot_types = np.array(plot_type)
    else:
        plot_types = np.array([plot_type])

    # check input
    assert (len(x.shape) == 1), "input x must be 1D array"
    assert (len(y.shape) == 1 or len(y.shape) == 2), "input y must be 1D or 2D array"
    assert np.all(np.in1d(plot_types, ['scatter', 'plot'])), "draw only support scatter or plot"

    # convert y to 2d-array
    y = np.reshape(y, [y.shape[0], -1])

    # if y is 2D array, we only support each column is a plot
    assert (y.shape[0] == x.shape[0]), "input x and y's column must have same size"
    assert (legends is None or len(legends) == y.shape[1]), "legends must have same size as y.shape[1]"
    assert (colors is None or len(colors) == y.shape[1]), "colors must be empty or have same size as y.shape[1]"

    if fig_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig_ax = (fig,ax)
    else:
        fig, ax = fig_ax

    for i in range(y.shape[1]):
        labeli = None if legends is None else legends[i]
        ci = None if colors is None else colors[i]
        ptypei = plot_types[min(i, plot_types.shape[0]-1)]
        if ptypei == "scatter":
            ax.scatter(x, y[:, i], label = labeli, c=ci)
        elif ptypei == "plot":
            ax.plot(x, y[:, i], label=labeli, c=ci)

    if legends is not None:
        ax.legend()

    if title is not None:
        ax.set_title(title)

    if xy_labels is not None:
        assert (len(xy_labels) == 2), 'input xy_labels must have len 2'
        ax.set_xlabel(xy_labels[0])
        ax.set_ylabel(xy_labels[1])

    return fig_ax
