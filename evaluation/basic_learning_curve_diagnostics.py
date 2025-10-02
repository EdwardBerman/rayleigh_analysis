import matplotlib.pyplot as plt

def plot_learning_curve(train_loss, val_loss, test_loss, output_dir):

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', color='orange')
    plt.plot(epochs, test_loss, label='Test Loss', color='green')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/learning_curve.png")

    plt.yscale('log')
    plt.savefig(f"{output_dir}/learning_curve_log_scale.png")

