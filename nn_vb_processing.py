import csv
import re
from collections import defaultdict


def find_indices(search_list, search_item):
    return [index for (index, item) in enumerate(search_list) if item == search_item]


def get_sentences_wordids_and_homographs(csv_fpath, sentence_outfile, homograph_outfile, wordid_outfile):
    with open(csv_fpath, "r", encoding='latin-1') as csv_file:
        csv_reader = csv.reader(csv_file)

        data = list(csv_reader)

    # get lists
    wordids = []
    homographs = []
    sentences = []
    for line in data:
        wordid, homograph, sentence = line[0], line[1], line[2]
        wordids.append(wordid)
        homographs.append(homograph)
        sentences.append(sentence)

    assert len(wordids) == len(homographs) == len(sentences)
    print("No. of sentences in NN/VB eval file: ", len(sentences))

    # preliminary clean of sentences to remove instances of [1] numbers
    no_digit_sents = []
    for sentence in sentences:
        no_weird_e = re.sub(r"Ê", " ", sentence)
        no_digits = re.sub(r"\[\d+]", "", no_weird_e)
        no_digit_sents.append(no_digits)


    # write lists to files
    with open(sentence_outfile, "w") as f:
        with open(homograph_outfile, "w") as f2:
            with open(wordid_outfile, "w") as f3:
                for wid, homograph, sent in zip(wordids, homographs, no_digit_sents):
                    f.write(sent + "\n")
                    f2.write(homograph + "\n")
                    f3.write(wid + "\n")

    return wordids, homographs, no_digit_sents


# get_sentences_wordids_and_homographs("WHD_nnvb_analysis/nn_vb_data.csv", "WHD_nnvb_analysis/sentences.txt", "WHD_nnvb_analysis/homographs.txt", "WHD_nnvb_analysis/wordids.txt")


def correct_homograph_prons(word_seqs, homographs, word_ids, gt_prons, tgt_file):
    with open(word_seqs, "r") as words:
        with open(homographs, "r") as homographs:
            with open(word_ids, "r") as wids:
                with open(tgt_file, "r") as tgts:
                    with open(gt_prons, "r") as gts:
                        words = words.readlines()
                        homographs = homographs.readlines()
                        wordids = wids.readlines()
                        tgts = tgts.readlines()
                        gts = gts.readlines()

    wordids = list(map(str.rstrip, wordids))
    homographs = list(map(str.rstrip, homographs))

    gt_prons_long = []
    wid_prons = defaultdict(str)
    for line in gts:
        line = line.rstrip().split(',')
        wid = line[0]
        pron = line[1]
        wid_prons[wid] = pron

    for id in wordids:
        gt_pron = wid_prons[id]
        gt_prons_long.append(gt_pron)

    new_tgts = []
    for i, (wordseq, homograph, wid, gt_pron, tgt) in enumerate(zip(words, homographs, wordids, gt_prons_long, tgts)):
        wordseq = wordseq.rstrip().split(",")
        tgt = tgt.rstrip().replace("_B", "+").split("+")[:-1]
        if len(wordseq) == len(tgt):
            homograph_ids = find_indices(wordseq, homograph)
            for id in homograph_ids:
                tgt[id] = f" {gt_pron} "
            new_tgts.append(tgt)
        else:
            print(i)
            new_tgts.append(tgt)

    with open("WHD_nnvb_analysis/gt_homograph_prons.txt", "w") as f:
        for pron in gt_prons_long:
            f.write(pron + "\n")

    with open("WHD_nnvb_analysis/tgt_nnvb_correct.txt", "w") as f:
        for tgt in new_tgts:
            tgt = "+".join(tgt) + "_B"
            f.write(tgt + "\n")


correct_homograph_prons("WHD_nnvb_analysis/word_seqs", "WHD_nnvb_analysis/homographs.txt", "WHD_nnvb_analysis/wordids.txt", "WHD_nnvb_analysis/gt_homographs.txt", "WHD_nnvb_analysis/tgt_nnvb_incorrect.txt")