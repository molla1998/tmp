import pandas as pd
import matplotlib.pyplot as plt

# Function to load CSV and plot label counts
def plot_label_distribution(csv_path):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Ensure "preds" column exists
    if "preds" not in df.columns:
        raise ValueError("The input CSV must contain a 'preds' column.")

    # Count label occurrences
    label_counts = df["preds"].value_counts()

    # Plot bar graph
    plt.figure(figsize=(6, 4))
    label_counts.plot(kind="bar", color=["blue", "green"])
    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.title("KH vs NLQ Label Distribution")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show the plot
    plt.show()

# Example usage
if __name__ == "__main__":
    input_path = input("Enter the path to the input CSV file: ").strip()
    
    try:
        plot_label_distribution(input_path)
    except Exception as e:
        print(f"Error: {e}")
