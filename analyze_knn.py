import numpy as np
import matplotlib.pyplot as plt
from nearest_neighbour import predictknn, learnknn, gensmallm
import time

ROUND_DIGITS = 5

def test_sample_size(sample_size: int = 200, k: int = 1):
    data = np.load('mnist_all.npz')

    labels = [2, 3, 5, 6]

    train_set = [data[f'train{i}'] for i in labels]
    test_set = [data[f'test{i}'] for i in labels]

    x_train, y_train = gensmallm(train_set, labels, sample_size)
    x_test, y_test = gensmallm(test_set, labels, 50)

    classifer = learnknn(k, x_train, y_train)
    y_preds = predictknn(classifer, x_test)

    error = np.mean(np.vstack(y_test) != np.vstack(y_preds))

    return error

def analyze_sample_sizes():
    sample_sizes = [1] + list(range(10, 101, 10))
    test_rep = 10

    averages = []
    errors = []
    for size in sample_sizes:
        sample_results = [test_sample_size(size) for _ in range(test_rep)]
        low = round(min(sample_results), ROUND_DIGITS)
        high = round(max(sample_results), ROUND_DIGITS)
        average = round(sum(sample_results) / test_rep, ROUND_DIGITS)
        averages.append(average)
        errors.append((average - low, high - average))  # error below and above the average

    # Convert errors to a format suitable for error bars (separate positive and negative)
    errors_below, errors_above = zip(*errors)
    yerr = [errors_below, errors_above]
    title = 'Error Range vs. Sample Size'
    x_label = 'Sample Size'
    plot_error_bar_graph(sample_sizes, averages, yerr, title, x_label, errors_below, errors_above)

def analyze_k_values():
    sample_size = 200
    k_values = list(range(1, 11))
    test_rep = 1000

    averages = []
    errors = []
    for k in k_values:
        sample_results = [test_sample_size(sample_size, k) for _ in range(test_rep)]
        low = round(min(sample_results), ROUND_DIGITS)
        high = round(max(sample_results), ROUND_DIGITS)
        average = round(sum(sample_results) / test_rep, ROUND_DIGITS)
        averages.append(average)
        errors.append((average - low, high - average))  # error below and above the average

    # Convert errors to a format suitable for error bars (separate positive and negative)
    errors_below, errors_above = zip(*errors)
    yerr = [errors_below, errors_above]
    title = 'Error Range vs. K Value'
    x_label = 'K Value'
    plot_error_bar_graph(k_values, averages, yerr, title, x_label, errors_below, errors_above)

def plot_error_bar_graph(x_values, averages, yerr, title, x_label, errors_below, errors_above):
    # Plotting the graph with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(x_values, averages,
                 yerr=yerr, fmt='o',
                 color='b', ecolor='gray',
                 capsize=5, label='Average with Error',
                 )

    # Annotating each data point
    for i, size in enumerate(x_values):
        plt.annotate(f'High: {round(errors_above[i] + averages[i], ROUND_DIGITS)}\n'
                    f'Avg: {round(averages[i], ROUND_DIGITS)}\n'
                    f'Low: {round(averages[i] - errors_below[i], ROUND_DIGITS)}',
                    (size, averages[i]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=12)

    plt.title(title, fontsize=26)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel('Error', fontsize=16)
    plt.xticks(x_values, fontsize=12)
    plt.grid(True)
    plt.show(block=True)

if __name__ == '__main__':
    # analyze_sample_sizes()
    start = time.time()
    print(f'start time {start}')
    analyze_k_values()
    end = time.time()
    print(f'end time {end}')
    time_to_process = end - start
    print(f'time_to_process in seconds {time_to_process}')
    