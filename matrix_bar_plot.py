import matplotlib.pyplot as plt

def plot_precision(models, values):
    colors = ['blue', 'green', 'red', 'purple', 'orange']  # Different colors for each bar
    
    plt.bar(models, values, color=colors)
    plt.xlabel("Models")
    plt.ylabel("Magnitude")
    plt.title("Precision")
    
    plt.show()

# Example usage
models = ["Model A", "Model B", "Model C", "Model D", "Model E"]  # Replace with your model names
values = [10, 20, 15, 30, 25]  # Replace with your five values
plot_precision(models, values)
