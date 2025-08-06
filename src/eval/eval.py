import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker


def eval(dir):
    result_cat = []
    total_correct = 0
    total_q = 0
    for i in range(1, 106):
        with open(f"{dir}/{i}_results.json", "r") as f:
            data = f.read()
        data = json.loads(data)
        acc = float(data[-1]["accuracy"].strip("%"))
        num_correct = round(acc * (len(data)-1) / 100)
        total_correct += num_correct
        total_q += len(data)-1
        result_cat.append(acc)
    
    # print(total_correct, total_q)
    return result_cat, (total_correct / total_q * 100)


def main():
    directory = ["naive_rag", "KGs", "CDRAG"]
    category_acc = []
    total_acc = []
    print("----------RESULTS----------")
    for dir in directory:
        cat, tot = eval(f"./data/{dir}")
        category_acc.append(cat)
        total_acc.append(tot)
        print(f"{dir}: {tot:.4f}")
    print("---------------------------")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    bins = np.linspace(0, 100, 16)
    bar_colors = ['#a9d1e8', '#cce5c8', '#f4c4c4'] 
    line_colors = ['#4c72b0', '#55a868', '#c44e52']
    ax.hist(
        category_acc,
        bins=bins,
        label=directory,
        color=bar_colors,
    )

    bin_width = bins[1] - bins[0]
    x_grid = np.linspace(0, 100, 1000)
    for data, color in zip(category_acc, line_colors):
        # Create the KDE
        kde = gaussian_kde(data)
        
        # Calculate density values on the grid
        density = kde(x_grid)
        
        # Scale the density to match the histogram's frequency count
        # The scaling factor is (number of data points * bin width)
        scaled_density = density * len(data) * bin_width
        
        ax.plot(x_grid, scaled_density, color=color, linewidth=2.5)

    ax.axvline(total_acc[0], color=line_colors[0], linestyle='--', linewidth=2, label=f'Naive RAG Mean: {total_acc[0]:.2f}%')
    ax.axvline(total_acc[1], color=line_colors[1], linestyle='--', linewidth=2, label=f'KGGen Mean: {total_acc[1]:.2f}%')
    ax.axvline(total_acc[2], color=line_colors[2], linestyle='--', linewidth=2, label=f'CDRAG Mean: {total_acc[2]:.2f}%')

    ax.set_title('Distribution of Accuracy for Each Method', fontsize=18, pad=20)
    ax.set_xlabel('Accuracy (%)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)

    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    plt.xticks(rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Set axis limits
    ax.set_xlim(0, 105)
    ax.set_ylim(bottom=0)

    mean_line_handles = [
        Line2D([0], [0], color=line_colors[0], lw=2, linestyle='--', label=f'Naive RAG: {total_acc[0]:.2f}%'),
        Line2D([0], [0], color=line_colors[1], lw=2, linestyle='--', label=f'KGGen: {total_acc[1]:.2f}%'),
        Line2D([0], [0], color=line_colors[2], lw=2, linestyle='--', label=f'CDRAG: {total_acc[2]:.2f}%')
    ]

    hist_handles, hist_labels = ax.get_legend_handles_labels()
    combined_handles = mean_line_handles + [hist_handles[0], hist_handles[1], hist_handles[2]]

    ax.legend(handles=combined_handles, ncol=2, fontsize=12, loc='upper left', frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig("./src/eval/image.svg")
    plt.show()

if __name__ == "__main__":
    main()