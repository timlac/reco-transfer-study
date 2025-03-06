from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_conf_mat(cm, labels=None, condition=None):
    print("labels", labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d')  # 'd' for integer values

    # Rotate x-axis labels
    plt.xticks(rotation=90)

    plt.subplots_adjust(bottom=.25, left=.25)

    plt.title(condition)

    plt.tight_layout()
    plt.show()