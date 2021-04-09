import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
try:
    from . import classify_data as cd
    from . import fit_lorentz as fl
    from . import utilities as util
except BaseException:
    import classify_data as cd
    import fit_lorentz as fl
    import utilities as util

# Let's you 'spy' into the data you're working with to see how the models
# are working.


def plot_count(i, count_predictions, count_label, count_data):
    """
    Visualizes Lorentzian counting predictions.
    """
    true_label = int(count_label[i][0])
    predictions_array = count_predictions[i]
    v = count_data[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    f = np.linspace(0, 1, 1024)
    plt.plot(f, v)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    elif np.abs(predicted_label - true_label) == 1:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel(
        "{} {:2.0f}% ({})".format(
            str(predicted_label) +
            ' lorentzian cluster:',
            100 *
            np.max(predictions_array),
            str(true_label)),
        color=color)


def plot_value_array(i, count_predictions, count_label):
    """
    Shows the predicted value for a Lorentzian counting prediction.
    """
    predictions_array = count_predictions[i]
    true_label = int(count_label[i][0])
    plt.grid(False)
    plt.xticks([])
    plt.yticks([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    thisplot = plt.bar(range(len(predictions_array)), predictions_array)
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def plot_count_result(i, count_predictions, count_label, count_data):
    """
    Plots the results after counting Lorentzians within given regions.
    """
    plt.figure(figsize=(20, 6))
    plt.subplot(1, 2, 1)
    plot_count(i, count_predictions, count_label, count_data)
    plt.subplot(1, 2, 2)
    plot_value_array(i, count_predictions, count_label)


def preview_data(i, data_set):
    """
    Plots and prints some basic stats about a generated data set.
    """
    lorentz_params = data_set[1][i]
    data = data_set[2][i]
    (f, v) = cd.separate_data(data)
    f = cd.normalize_1d(f, (0, 1, 1024))
    v = cd.normalize_1d(v, (0, 1, 1024))
    plt.plot(f, v)
    print(cd.disect_lorentz_params_array(lorentz_params)[0])


def scatter_data_files(data_files):
    for i in range(0, len(data_files)):
        scale = (0, 1, len(data_files[i].r))
        f = data_files[i].f
        r = cd.normalize_1d(data_files[i].r, scale)
        just_plotted = plt.plot(f, r + i, alpha=0.5)
        color = just_plotted[0].get_color()
        if not data_files[i].params is None:
            pts_f = data_files[i].params[..., 1]
            pts_r = util.scatter_pts(pts_f, f, r)
            plt.scatter(pts_f, pts_r + i, color=color)
        plt.text(f[0], r[0] + i, str(data_files[i].T[0]) +
                 ' K ', color=color, ha='right')
        plt.text(f[-1], r[-1] + i, ' ' + str(data_files[i].T[-1]) +
                 ' K', color=color, ha='left')


def spider_plot(data_file, params=None, bg=True, color_1=None, color_2=None):
    if bg:
        plt.plot(data_file.x, data_file.y, color=color_1)
    if params is None:
        params = data_file.params
    if params is not None:
        regions = fl.regions_from_parameters(data_file.f, params)
        for region in regions:
            plt.plot(data_file.x[int(region[0]):int(region[1])], data_file.y[int(
                region[0]):int(region[1])], color=color_2)


def plot_all_params(params, index):
    gs = GridSpec(3, 2)
    fig = plt.figure()
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, :])
    ax0.set_title('Amplitude')
    ax1.set_title('Position')
    ax2.set_title('Full Width at Half Maximum')
    ax3.set_title('Phase')
    ax0.plot(params[:, index, 0], picker=True)
    ax1.plot(params[:, index, 1], picker=True)
    ax2.plot(params[:, index, 2], picker=True)
    ax3.plot(params[:, index, 3], picker=True)
    ax4.plot(np.linspace(0, 1, 100))

    def on_pick(event):
        print(event.artist.get_xdata())
        print(event.artist.get_ydata())

    fig.canvas.mpl_connect('pick_event', on_pick)
