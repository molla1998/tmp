import matplotlib.pyplot as plt

def plot_precision(models, values):
    colors = ['blue', 'green', 'red', 'purple', 'orange']  # Different colors for each bar
    
    bars = plt.bar(models, values, color=colors)
    plt.xlabel("Models")
    plt.ylabel("Magnitude")
    plt.title("Precision")
    
    # Add value labels on top of each bar
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(value), ha='center', va='bottom')
    
    plt.show()

# Example usage
models = ["Model A", "Model B", "Model C", "Model D", "Model E"]  # Replace with your model names
values = [10, 20, 15, 30, 25]  # Replace with your five values
plot_precision(models, values)
