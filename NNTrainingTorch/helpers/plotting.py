from pathlib import Path

import matplotlib.pyplot as plt

def plot_performance(train_batch_losses: list[float],
                     train_accuracies: list[float],
                     test_accuracies: list[float],
                     epoch_times: list[float],
                     savePath: Path = None, datsetName: str = ""
                     ) -> None:
    """
    Erstellt einen umfassenden Performance-Plot
    Args:
        train_batch_losses: Liste der Loss-Werte pro Batch-Update
        train_accuracies: Liste der Trainingsgenauigkeiten pro Epoche
        test_accuracies: Liste der Testgenauigkeiten pro Epoche
        epoch_times: Optional, Liste der Epochen-Zeiten
    """
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Performance - ' + str(datsetName), fontsize=16, fontweight='bold')
    epochs = createXaxis(train_accuracies)

    plot_Training_Loss(train_batch_losses)

    plot_Training_Test_Acc(epochs, test_accuracies, train_accuracies)

    plot_Epoch_Times(epoch_times, epochs)

    plot_compare_Train_and_Test_ACC(test_accuracies, train_accuracies)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    plt.savefig(savePath / "training_metrics.png")


def plot_compare_Train_and_Test_ACC(test_accuracies: list[float], train_accuracies: list[float]):
    # 4. Final Accuracy Vergleich (unten rechts)
    ax4 = plt.subplot(2, 2, 4)
    categories = ['Training', 'Test']
    final_accuracies = [train_accuracies[-1], test_accuracies[-1]]

    bars = ax4.bar(categories, final_accuracies, color=['blue', 'red'], alpha=0.7)
    ax4.set_title('Final Accuracy Comparison', fontweight='bold')
    ax4.set_ylabel('Accuracy')
    ax4.set_ylim(0, 1.0)

    # Werte auf den Bars anzeigen
    for bar, accuracy in zip(bars, final_accuracies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{accuracy:.3f}', ha='center', va='bottom')


def plot_Epoch_Times(epoch_times: list[float], epochs: range):
    ax3 = plt.subplot(2, 2, 3)
    if epoch_times is not None:
        ax3.bar(epochs, epoch_times, alpha=0.7, color='green')
        ax3.set_title('Epoch Duration', fontweight='bold')
        ax3.set_xlabel('# Epochs')
        ax3.set_ylabel('Time (seconds)')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No epoch time data',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Epoch Duration', fontweight='bold')


def plot_Training_Test_Acc(x_axis: range, test_accuracies: list[float], train_accuracies: list[float]):
    # 2. Accuracy Plot (oben rechts)
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(x_axis, train_accuracies, 'b-', label='Training', marker='o', markersize=4)
    ax2.plot(x_axis, test_accuracies, 'r-', label='Test', marker='s', markersize=4)
    ax2.set_title('Accuracy', fontweight='bold')
    ax2.set_xlabel('# Epochs')
    ax2.set_ylabel('Prediction Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)


def createXaxis(train_accuracies: list[float]) -> range:
    epochs = range(1, len(train_accuracies) + 1)
    return epochs


def plot_Training_Loss(batch_losses: list[float]):
    # 1. Batch Loss Plot (oben links)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(batch_losses, 'b-', alpha=0.7, linewidth=0.8)
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.set_xlabel('# Batch Updates')
    ax1.set_ylabel('Batch Loss')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.5)