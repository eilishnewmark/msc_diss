# Get statistics of homograph distribution in LibriSpeech data and their Festival prons
import pickle
from collections import defaultdict
import csv
import pandas as pd
from matplotlib_venn import venn3
from matplotlib import pyplot as plt


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
        #         if len(words) == len(word_prons):
        #             homograph_pron = word_prons[homograph_idx].rstrip()
        #             this_homograph_line.append(i)
        #             this_homograph_line.append(line.lower().rstrip())
        #             this_homograph_line.append(word)
        #             this_homograph_line.append(homograph_pron)
        #         break
        # if this_homograph_line != []:
        #     homograph_lines_prons.append(this_homograph_line)

    homograph_lines = list(set(homograph_lines))

    # with open('libri_analysis/libri_homograph_counts.csv', 'w') as csv_file:
    #     writer = csv.writer(csv_file)
    #     for key, value in homograph_count.items():
    #         writer.writerow([key, value])
    #
    # with open('libri_analysis/libri_homograph_analysis.csv', 'w') as word:
    #     # using csv.writer method from CSV package
    #     write = csv.writer(word)
    #     write.writerows(homograph_lines_prons)

    # print("raw homograph counts: ", homograph_count)
    # print("no. of lines with homographs: ", len(homograph_lines))
    # print("no. of unique homographs in data: ", len(homograph_count.keys()))
    # print("no. of total homographs in data: ", sum(homograph_count.values()))
    # print("no. of word-pron aligned sentences: ", len(homograph_lines_prons))

    return homograph_lines


def get_libri_train_no_homo(src_file, tgt_file, POS_file):

    homograph_line_ids = get_homograph_dist("WHD_full.pkl", src_file + ".txt", tgt_file + ".txt")

    with open(src_file + ".txt", "r") as src:
        with open(tgt_file + ".txt", "r") as tgt:
            with open(POS_file + ".txt", "r") as POS:
                src_lines = src.readlines()
                tgt_lines = tgt.readlines()
                POS_lines = POS.readlines()

    assert len(src_lines) == len(tgt_lines) == len(POS_lines)

    # src_train = [src for i, src in enumerate(src_lines) if i not in homograph_line_ids]
    # tgt_train = [tgt for i, tgt in enumerate(tgt_lines) if i not in homograph_line_ids]
    # POS_train = [POS for i, POS in enumerate(POS_lines) if i not in homograph_line_ids]

    src_train = []
    tgt_train = []
    POS_train = []
    for i, (src, tgt, POS) in enumerate(zip(src_lines, tgt_lines, POS_lines)):
        if i not in homograph_line_ids:
            src_train.append(src)
            tgt_train.append(tgt)
            POS_train.append(POS)

    assert len(src_train) == len(tgt_train) == len(POS_train)

    with open(src_file + "-aug.txt", "w") as src:
        with open(tgt_file + "-aug.txt", "w") as tgt:
            with open(POS_file + "-aug.txt", "w") as POS:
                for src_line, tgt_line, POS_line in zip(src_train, tgt_train, POS_train):
                    src.write(src_line)
                    tgt.write(tgt_line)
                    POS.write(POS_line)


# get_libri_train_no_homo("libri960_data/data/src-train", "libri960_data/data/tgt-train", "libri960_data/data/src-POS-train")


def get_word_types_tokens(libri_src_file, WHD_src_file):
    with open(libri_src_file, "r") as f:
        lines = f.readlines()

    with open(WHD_src_file, "r") as f:
        WHD_lines = f.readlines()

    homograph_line_ids = get_homograph_dist("WHD_full.pkl", libri_src_file, "libri960_data/data/tgt-train.txt")

    homograph_lines = [line for i, line in enumerate(lines) if i in homograph_line_ids]
    non_homograph_lines = [line for i, line in enumerate(lines) if i not in homograph_line_ids]

    assert len(homograph_lines) == len(homograph_line_ids)

    homograph_line_tokens = []
    for line in homograph_lines:
        words = line.rstrip().split()
        for word in words:
            homograph_line_tokens.append(word)

    non_homograph_line_tokens = []
    for line in non_homograph_lines:
        words = line.rstrip().split()
        for word in words:
            non_homograph_line_tokens.append(word)

    all_tokens = []
    for line in lines:
        words = line.rstrip().split()
        for word in words:
            all_tokens.append(word)

    all_WHD_tokens = []
    for line in WHD_lines:
        words = line.rstrip().split()
        for word in words:
            all_WHD_tokens.append(word)

    # print("no. of tokens in LibriSpeech training: ", len(all_tokens))
    # print("no. of types in LibriSpeech training: ", len(list(set(all_tokens))))
    # print("no. of tokens in homograph lines: ", len(homograph_line_tokens))
    # print('no. of types in homograph lines: ', len(set(homograph_line_tokens)))
    # print("no. of tokens in non-homograph lines: ", len(non_homograph_line_tokens))
    # print('no. of types in non-homograph lines: ', len(set(non_homograph_line_tokens)))
    # # get any type in homograph lines that isn't in training data without homograph lines
    # print('no. of types that would be deleted from LibriSpeech training:', len(set(homograph_line_tokens) - set(non_homograph_line_tokens)))
    # print('no. of tokens in WHD training data: ', len(all_WHD_tokens))
    # print('no. of types in WHD training data: ', len(set(all_WHD_tokens)))
    # print('no. of types that would be added to LibriSpeech training: ', len(set(all_WHD_tokens) - set(non_homograph_line_tokens)))
    # print('no. of types that would be added that had been removed: ', len(set.intersection(set(all_WHD_tokens), set(homograph_line_tokens))))

    homograph_line_types = set(homograph_line_tokens)
    non_homograph_line_types = set(non_homograph_line_tokens)
    WHD_types = set(all_WHD_tokens)

    venn3([homograph_line_types, non_homograph_line_types, WHD_types], ('LS homograph line types', 'LS non-homograph line types', 'WHD types'))

    plt.show()


# get_word_types_tokens("libri960_data/data/src-train.txt", "WHD_data/data/WHD_src_train.txt")


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


# get_statistics("libri_analysis/libri_homograph_analysis_SAMPLE.csv")

