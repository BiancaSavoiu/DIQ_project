import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from library.E_plot_results import mean, distance, speed, your_path


def pollute_data_mcar(data, percent_incomplete=0.2, seed=2023):
    """
    Generate a random mask for incompleteness (MCAR) and introduce NaN values in the DataFrame according to the
    generated mask.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame.
    percent_incomplete : float
        The percentage of missing values to be introduced in the DataFrame.
    seed : int
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with NaN values introduced according to the generated mask.
    """
    # Set random seed for consistent behavior
    np.random.seed(seed)

    # Check if the input is a DataFrame, otherwise raise an error
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas.DataFrame")

    # Generate a random mask for incompleteness (MCAR)
    mask = np.random.rand(*data.shape) < percent_incomplete

    # MCAR experiment: introduce NaN values in the DataFrame according to the generated mask
    data[mask] = np.nan

    return data


def zoomed_plot(x_axis_values, x_label, results, title, algorithms, plot_type, zoom):
    """
    This function generates the plots for the results of the experiments. It is possible to zoom the y-axis in order to
    have a better visualization of the results.

    Parameters
    ----------
    x_axis_values : list
        The list of x-axis values.
    x_label : str
        The label of the x-axis.
    results : list
        The list of results.
    title : str
        The title of the plot.
    algorithms : list
        The list of algorithms.
    plot_type : str
        The type of plot to be generated.
    zoom : dict
        The dictionary containing the zoom values for the y-axis.

    Returns
    -------
    None
    """
    title = str(title)

    if plot_type == "performance":
        if algorithms[0] == "LinearRegressor":
            generate_figure_performance(x_axis_values, results, algorithms, "RMSE")
        else:
            generate_figure_performance(x_axis_values, results, algorithms, "silhouette")

    elif plot_type == "distance train-test":
        generate_figure_distance(x_axis_values, results, algorithms, "RMSE_test - RMSE_train")

    else:
        generate_figure_speed(x_axis_values, results, algorithms, "speed")

    plt.title(title)
    plt.xlabel(x_label)
    plt.legend()
    plt.ylim(zoom['min'], zoom['max'])  # if you want to fix a limit for the y_axis
    plt.savefig(your_path + title + ".pdf", bbox_inches='tight')  # if you want to save the figure
    plt.show()


def generate_figure_performance(x_axis, results_all, legend, score):
    """
    This function generates the performance plot.

    Parameters
    ----------
    x_axis : list
        The list of x-axis values.
    results_all : list
        The list of results.
    legend : list
        The list of legend names.
    score : str
        The score to be plotted.

    Returns
    -------
    None
    """
    for i in range(0, len(results_all)):
        plt.plot(x_axis, mean(results_all[i]), marker='o', label=legend[i], markersize=3)
    plt.ylabel(score)


def generate_figure_distance(x_axis, results_all, legend, score):
    """
    This function generates the distance plot.

    Parameters
    ----------
    x_axis : list
        The list of x-axis values.
    results_all : list
        The list of results.
    legend : list
        The list of legend names.
    score : str
        The score to be plotted.

    Returns
    -------
    None
    """
    for i in range(0, len(results_all)):
        plt.plot(x_axis, distance(results_all[i]), marker='o', label=legend[i], markersize=3)
    plt.ylabel(score)


def generate_figure_speed(x_axis, results_all, legend, score):
    """
    This function generates the speed plot.

    Parameters
    ----------
    x_axis : list
        The list of x-axis values.
    results_all : list
        The list of results.
    legend : list
        The list of legend names.
    score : str
        The score to be plotted.

    Returns
    -------
    None
    """
    for i in range(0, len(results_all)):
        plt.plot(x_axis, speed(results_all[i]), marker='o', label=legend[i], markersize=3)
    plt.ylabel(score)


def zoom_data(results, attribute):
    """
    This function zooms the y-axis of the plots. It removes the values from attributes far away the median of the values
    inside attributes. It returns a dictionary containing the maximum and minimum values of the zoomed y-axis.

    Parameters
    ----------
    results : list
        The list of results.
    attribute : str
        The attribute to be zoomed.

    Returns
    -------
    dict
        The dictionary containing the maximum and minimum values of the zoomed y-axis.
    """
    # Check if the input is a list, otherwise raise an error
    if not isinstance(results, list):
        raise TypeError("Input results must be a list")

    # Check if the input is a string, otherwise raise an error
    if not isinstance(attribute, str):
        raise TypeError("Input attribute must be a string")

    max_values = []
    min_values = []

    for algorithm in results:
        max_results_list = []
        min_results_list = []

        for result in algorithm:
            max_results_list.append(result[attribute])
            min_results_list.append(result[attribute])

        max_values.append(max(max_results_list))
        min_values.append(min(min_results_list))

    # Remove the values from attributes far away the median of the values inside attributes
    max_values = [x for x in max_values if (x < np.median(max_values) + 1.5 * np.std(max_values))]

    return {"max": np.max(max_values) + np.std(max_values), "min": np.min(min_values) - np.std(max_values)}
