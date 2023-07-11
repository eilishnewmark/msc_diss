# Get statistics of homograph distribution in LibriSpeech data and their Festival prons
import pickle
from collections import defaultdict
import csv
import pandas as pd

def get_homograph_dist(WHD_df, src_fn, tgt_fn):
    with open(src_fn, "r") as f:
        src_lines = f.readlines()

    with open(tgt_fn, "r") as f:
        tgt_lines = f.readlines()

    with open(WHD_df, "rb") as df:
        WHD = pickle.load(df)

    homographs = list(set(WHD['homograph'].tolist()))
    homographs = list(map(str.upper, homographs))

    homograph_count = defaultdict(lambda: 0)
    homograph_lines = []
    homograph_lines_prons = []

    for i, (line, pron) in enumerate(zip(src_lines, tgt_lines)):
        this_homograph_line = []
        words = line.split()
        word_prons = pron.replace("_B", "+").split("+")[:-1]
        for word in words:
            if word in homographs:
                homograph_idx = words.index(word)
                homograph_count[word.lower()] += 1
                homograph_lines.append(i)
                if len(words) == len(word_prons):
                    homograph_pron = word_prons[homograph_idx].rstrip()
                    this_homograph_line.append(i)
                    this_homograph_line.append(line.lower().rstrip())
                    this_homograph_line.append(word)
                    this_homograph_line.append(homograph_pron)
                break
        if this_homograph_line != []:
            homograph_lines_prons.append(this_homograph_line)

    homograph_lines = list(set(homograph_lines))

    with open('libri_analysis/libri_homograph_counts.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in homograph_count.items():
            writer.writerow([key, value])

    with open('libri_analysis/libri_homograph_analysis.csv', 'w') as word:
        # using csv.writer method from CSV package
        write = csv.writer(word)
        write.writerows(homograph_lines_prons)

    print("raw homograph counts: ", homograph_count)
    print("no. of lines with homographs: ", len(homograph_lines))
    print("no. of unique homographs in data: ", len(homograph_count.keys()))
    print("no. of total homographs in data: ", sum(homograph_count.values()))
    print("no. of word-pron aligned sentences: ", len(homograph_lines_prons))


def get_sample(csv_file):
    data = pd.read_csv(csv_file)
    data.columns = ['index', 'sentence', 'homograph', 'pronunciation']

    for i in list(data['pronunciation'].unique()):
        data[data['pronunciation'] == i].sample(frac=0.01).to_csv("libri_homograph_analysis_SAMPLE.csv", mode="a", header=False, index=False, columns=['index', 'sentence', 'homograph', 'pronunciation'])


def get_statistics(csv_file):
    data = pd.read_csv(csv_file)
    data.columns = ['t/f', 'index', 'sentence', 'homograph', 'pronunciation']

    truefalse = data['t/f'].tolist()
    homographs = list(set(data['homograph'].tolist()))

    true_count = truefalse.count("t")
    false_count = truefalse.count("f")

    print("no. of homographs in sample: ", len(homographs))
    print("percentage correct: ", true_count/len(truefalse))
    print("percentage incorrect: ", false_count/len(truefalse))

    data['freq'] = data.groupby('homograph')['homograph'].transform('count')

    homograph_accuracies = defaultdict(list)
    for homograph in homographs:
        total = data.loc[data.homograph == homograph, 'freq'].tolist()[0]
        true = len(data[(data['t/f'] == 't') & (data['homograph'] == homograph)])
        homograph_accuracies[homograph] = [true/total, total]

    print("homograph accuracies: ", homograph_accuracies)

    with open('libri_analysis/libri_homograph_accuracies.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in homograph_accuracies.items():
            row = [key]
            finished_row = row + value
            writer.writerow(finished_row)

    # with open('libri_analysis/libri_homograph_incorrect.csv', 'w') as csv_file:
    #     writer = csv.writer(csv_file)
    #     data_false = data.loc[data['t/f'] == 'f']
    #     sentences = data_false['sentence'].tolist()
    #     homographs = data_false['homograph'].tolist()
    #     homograph_pron = data_false['pronunciation'].tolist()
    #     for sentence, homograph, pron in zip(sentences, homographs, homograph_pron):
    #         writer.writerow([sentence, homograph, pron])

get_statistics("libri_analysis/libri_homograph_analysis_SAMPLE.csv")

