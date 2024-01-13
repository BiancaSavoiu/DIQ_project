import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from E_plot_results import your_path, mean, distance, speed


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
    title = str(title)

    if plot_type == "performance":
        if algorithms[0] == "LinearRegressor":
            generate_figure_performance(x_axis_values, x_label, results, title, algorithms, "RMSE", zoom)
        else:
            generate_figure_performance(x_axis_values, x_label, results, title, algorithms, "silhouette", zoom)

    elif plot_type == "distance train-test":
        generate_figure_distance(x_axis_values, x_label, results, title, algorithms, "RMSE_test - RMSE_train", zoom)

    else:
        generate_figure_speed(x_axis_values, x_label, results, title, algorithms, "speed", zoom)


def generate_figure_performance(x_axis, xlabel, results_all, title, legend, score, zoom):
    plt.title(title)
    for i in range(0, len(results_all)):
        mean_perf = mean(results_all[i])

        plt.plot(x_axis, mean_perf, marker='o', label=legend[i], markersize=3)

    plt.xlabel(xlabel)
    plt.ylabel(score)
    plt.legend()
    plt.ylim(-0.1, zoom)
    plt.savefig(your_path + title + ".pdf", bbox_inches='tight')  # if you want to save the figure
    plt.show()


def generate_figure_distance(x_axis, xlabel, results_all, title, legend, score, zoom):
    plt.title(title)
    for i in range(0, len(results_all)):
        distance_perf = distance(results_all[i])

        plt.plot(x_axis, distance_perf, marker='o', label=legend[i], markersize=3)

    plt.xlabel(xlabel)
    plt.ylabel(score)
    plt.legend()
    plt.ylim(-0.1, zoom)
    plt.savefig(your_path + title + ".pdf", bbox_inches='tight')  # if you want to save the figure
    plt.show()


def generate_figure_speed(x_axis, xlabel, results_all, title, legend, score, zoom):
    plt.title(title)
    for i in range(0, len(results_all)):
        speed_perf = speed(results_all[i])

        plt.plot(x_axis, speed_perf, marker='o', label=legend[i], markersize=3)

    plt.xlabel(xlabel)
    plt.ylabel(score)
    plt.legend()
    plt.ylim(-0.1, zoom)
    plt.savefig(your_path + title + ".pdf", bbox_inches='tight')  # if you want to save the figure
    plt.show()


def zoom_data(results, attribute):
    attributes = []

    for algorithm in results:
        p_list = []

        for performance in algorithm:
            p_list.append(performance[attribute])

        attributes.append(max(p_list))

    attributes.remove(np.max(attributes))

    return np.max(attributes) + (np.max(attributes) - np.min(attributes)) / 10
