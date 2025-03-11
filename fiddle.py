import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy(accuracy_results):
    emotions = list(next(iter(accuracy_results.values())).keys())  # Get emotion order from the first condition
    x = np.arange(len(emotions))  # X-axis positions

    width = 0.25  # Bar width
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (condition, acc_dict) in enumerate(accuracy_results.items()):
        accuracies = [acc_dict.get(emotion, 0) for emotion in emotions]  # Use original order
        ax.bar(x + i * width, accuracies, width=width, label=condition)

    ax.set_xticks(x + width)
    ax.set_xticklabels(emotions)
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy")
    plt.xlabel("Emotion")
    plt.title("Emotion Accuracy per Condition")
    plt.legend(title="Condition")
    plt.tight_layout()
    plt.show()

# Example JSON accuracy results
accuracy_results = {
    "original_video_": {1: 0.75, 2: 0.66, 3: 0.5},
    "metahuman_video_": {1: 0.60, 2: 0.50, 3: 0.40},
    "furhat_video_": {1: 0.55, 2: 0.45, 3: 0.35}
}

plot_accuracy(accuracy_results)