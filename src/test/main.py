import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def convert_history_format(history):
    result = {}

    for key, values in history.items():
        for entry in values:
            for subkey, subvalue in entry.items():
                if subkey not in result:
                    result[subkey] = {}
                if key not in result[subkey]:
                    result[subkey][key] = []
                result[subkey][key].append(subvalue)

    return result


def plot_metric_over_epochs(data, metric, ylabel):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, data[metric]['training'], label='Training', marker='o')
    plt.plot(epochs, data[metric]['validation'], label='Validation', marker='o')

    plt.title(f'{metric.capitalize()} Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


# Example usage
history = {'training': [{'loss': 123, 'accuracy': 0.7, 'abc': 0.1}, {'loss': 0.3, 'accuracy': 0.8, 'abc': 0.1}],
           'validation': [{'loss': 0.9, 'accuracy': 0.72, 'abc': 0.1}, {'loss': 0.2, 'accuracy': 0.21, 'abc': 0.1}]}

data = convert_history_format(history)

sns.set_context("notebook")
sns.set_theme(style="whitegrid")

# Iterate over metrics
for metric in data.keys():
    epochs = np.arange(1, len(data[metric]['training']) + 1)
    plot_metric_over_epochs(data, metric, f'{metric.capitalize()} Value')
