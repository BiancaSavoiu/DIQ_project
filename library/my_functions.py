import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from library.E_plot_results import plot


def pollute_data_with_constant_feature(data, percentage, seed=2023):
    """
    Take a random column and copy its first value in a percentage of the remaining rows, in a random way.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame.
    percentage : float
        The percentage of constant feature to be introduced in the DataFrame.
    seed : int
        Determines random number generation for dataset creation. Pass an int for reproducible output across multiple
        function calls.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with a constant feature.
    """
    # Set random seed for consistent behavior
    np.random.seed(seed)

    # Check if the input is a DataFrame, otherwise raise an error
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas.DataFrame")

    random_col = np.random.randint(data.shape[1])

    # Take a random column and copy its first value in a percentage of the remaining rows, in a random way
    data.iloc[np.random.choice(data.index, int(data.shape[0] * percentage)), random_col] = data.iloc[0, random_col]

    return data


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


def pollute_data_mcar_for_each_feature_in_incremental_way(data, seed=2023):
    """
    Generate a random mask for incompleteness (MCAR) and introduce NaN values in the DataFrame according to the
    generated mask. The NaN values are introduced for each feature in an incremental way. For example, for a DataFrame
    with 10 features, the first feature will have 10% of missing values, the second feature will have 20% of missing
    values, and so on.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame.
    seed : int
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
    """

    # Set random seed for consistent behavior
    np.random.seed(seed)

    # Check if the input is a DataFrame, otherwise raise an error
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas.DataFrame")

    # Generate a random mask for incompleteness (MCAR)
    for col in data.columns:
        pollute_data_mcar_for_one_feature(data=data, feature=col, percent_incomplete=(col * 10))


def pollute_data_mcar_for_one_feature(data, feature, percent_incomplete=0.2, seed=2023):
    """
    Generate a random mask for incompleteness (MCAR) and introduce NaN values in the DataFrame according to the
    generated mask. The NaN values are introduced only for the chosen feature.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame.
    feature : int
        The feature for which NaN values are introduced, identified by the number of the column.
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

    # Generate a random mask for incompleteness (MCAR), introducing NaN values only for one feature
    mask = np.random.rand(*data[feature].shape) < percent_incomplete

    # MCAR experiment: introduce NaN values in the DataFrame according to the generated mask
    data[feature][mask] = np.nan

    return data


def pollute_data_mnar(data, feature_referenced, feature_dependent, seed=2023):
    """
    Generate a random mask for incompleteness (MNAR) and introduce NaN values in the DataFrame according to the
    generated mask.
    Introduce missing values in a way that depends on the values of specific features.
    Simulate a non-random incompleteness pattern based on the values of certain features.
    For example only if a column has a certain range of values for a value then modify the correspondent completeness
    of another column referring to the value.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame.
    feature_referenced : int
        The feature for which we check a certain condition, to modify the completeness of another feature.
    feature_dependent : int
        The feature for which NaN values are introduced, identified by the number of the column.
    seed : int
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
    """

    # Set random seed for consistent behavior
    np.random.seed(seed)

    # Check if the input is a DataFrame, otherwise raise an error
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas.DataFrame")

    # Simulate a non-random incompleteness pattern based on the values of certain features
    for element in data.index:
        if data[feature_referenced][element] > 100:
            data[feature_dependent][element] = np.nan


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
    plt.ylim(zoom['min'], zoom['max'])  # if you want to fix a limit for the y_axis
    plot(x_axis_values, x_label, results, title, algorithms, plot_type)


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
