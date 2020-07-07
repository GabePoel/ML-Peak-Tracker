import numpy as np
import matplotlib.pyplot as plt
try:
    from . import classify_data as cd
except:
    import classify_data as cd

# Let's you 'spy' into the data you're working with to see how the models are working.

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
    plt.xlabel("{} {:2.0f}% ({})".format(str(predicted_label) + ' lorentzian cluster:', 100*np.max(predictions_array), str(true_label)), color=color)

def plot_value_array(i, count_predictions, count_label):
    """
    Shows the predicted value for a Lorentzian counting prediction.
    """
    predictions_array = count_predictions[i]
    true_label = int(count_label[i][0])
    plt.grid(False)
    plt.xticks([])
    plt.yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
    thisplot = plt.bar(range(len(predictions_array)), predictions_array)
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def plot_count_result(i, count_predictions, count_label, count_data):
    """
    Plots the results after counting Lorentzians within given regions.
    """
    plt.figure(figsize=(20,6))
    plt.subplot(1,2,1)
    plot_count(i, count_predictions, count_label, count_data)
    plt.subplot(1,2,2)
    plot_value_array(i, count_predictions, count_label)

def preview_data(i, data_set):
    """
    Plots and prints some basic stats about a generated data set.
    """
    lorentz_params = data_set[1][i]
    data = data_set[2][i]
    (f, v) = cd.separate_data(data)
    f = cd.normalize_1d(f, (0,1,1024))
    v = cd.normalize_1d(v, (0,1,1024))
    plt.plot(f, v)
    print(cd.disect_lorentz_params_array(lorentz_params)[0])