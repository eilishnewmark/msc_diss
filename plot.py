import matplotlib.pyplot as plt
import csv

# plot proportions of wordid/homograph counts in two different datasets (e.g. WHD_train and WHD_train_inlex)
# TODO: edit plot so that x labels appear when proportional representation is <0.005 (5%) of the dataset or >0.007
def calculate_proportions(data):
    total_count = sum(data)
    return [count / total_count for count in data]

def plot_proportional_representation(csv1, csv2):
    with open(csv1, "r") as csv_file1:
        with open(csv2, "r") as csv_file2:
            data1 = csv.reader(csv_file1)
            data2 = csv.reader(csv_file2)

            homographs1 = []
            homographs2 = []
            counts1 = []
            counts2 = []
            for entry1, entry2 in zip(data1, data2):
                homographs1.append(entry1[0])
                homographs2.append(entry2[0])
                counts1.append(int(entry1[1]))
                counts2.append(int(entry2[1]))

    labels = list(set(homographs1 + homographs2))
    proportions1 = calculate_proportions(counts1)
    proportions2 = calculate_proportions(counts2)
    average1 = sum(counts1)/len(counts1)
    average2 = sum(counts2)/len(counts2)

    # width = 0.35  # Width of the bars
    x = range(len(labels))

    fig, ax = plt.subplots()

    ax.bar(x, proportions1, label=csv1, alpha=0.5)
    ax.bar([i for i in x], proportions2, label=csv2, alpha=0.5)

    ax.set_xlabel('Homographs')
    ax.set_ylabel('Proportional Representation')
    ax.set_title('Proportional Representation of Homographs in Two Datasets')
    # ax.set_xticks([i/2 for i in x])
    # ax.set_xticklabels(labels)
    ax.legend()

    plt.show()

# Example usage:
plot_proportional_representation("WHD_data/train_homograph_counts.csv", "WHD_data/train_inlex_homograph_counts.csv")