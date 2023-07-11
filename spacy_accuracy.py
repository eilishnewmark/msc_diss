import pickle
from collections import defaultdict
import csv
# We collected counts from the training corpus of the form <word, POS tag, word_sense, Count>.
# These counts show how many times a given word got assigned to a certain word sense when it has a certain POS tag.


def get_spacy_preds(pos_seqs, token_seqs, WHD_df):
    with open(token_seqs, "r") as f:
        token_lines = f.readlines()
    with open(pos_seqs, "r") as f2:
        pos_lines = f2.readlines()
    with open(WHD_df, "rb") as df:
        WHD = pickle.load(df)

    homographs = WHD['homograph'].tolist()

    spacy_preds = []
    for tokens, pos, homograph in zip(token_lines, pos_lines, homographs):
        tokens = tokens.rstrip().split(",")
        pos = pos.rstrip().split(",")
        homograph_idx = tokens.index(homograph)
        pos_tag = pos[homograph_idx]
        spacy_preds.append(pos_tag)

    return spacy_preds


def get_predicted_wids(WHD_df):
    with open(WHD_df, "rb") as df:
        WHD = pickle.load(df)

    train_spacy_preds = WHD['spacy_preds'].tolist()
    train_homographs = WHD['homograph'].tolist()
    train_wordids = WHD['wordid'].tolist()

    counts = defaultdict(lambda: 0)
    for spacy_pred, homograph, wid in zip(train_spacy_preds, train_homographs, train_wordids):
        counts[(homograph, spacy_pred, wid)] += 1

    # test_spacy_preds = WHD['spacy_preds'].tolist()[14487:]
    # test_homographs = WHD['homograph'].tolist()[14487:]
    # test_wordids = WHD['wordid'].tolist()[14487:]

    predicted_wordids = []
    for spacy_pred, homograph in zip(train_spacy_preds, train_homographs):
        potential_wid_counts = []
        for info, count in counts.items():
            word, tag, wordid = info
            if word == homograph and tag == spacy_pred:
                potential_wid_counts.append((wordid, count))
        try:
            highest_wid_count = sorted(potential_wid_counts, key=lambda x: x[1], reverse=True)[0]
            predicted_wordids.append(highest_wid_count[0])
        except IndexError:
            predicted_wordids.append(potential_wid_counts)

    return train_homographs, train_wordids, predicted_wordids


def find_indices(list_to_check, item_to_find):
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices


def get_macro_acc():
    # the arithmetic mean of the per-homograph accuracies
    test_homographs, test_wids, predicted_wids = get_predicted_wids("WHD_full.pkl")

    homograph_set = list(set(test_homographs))

    homograph_accs = []
    for homograph in homograph_set:
        correct = 0
        homograph_ids = find_indices(test_homographs, homograph)
        homograph_total = len(homograph_ids)
        relevant_test_wids = test_wids[homograph_ids[0]:homograph_ids[-1] + 1]
        relevant_pred_wids = predicted_wids[homograph_ids[0]:homograph_ids[-1] + 1]
        for test_wid, pred_wid in zip(relevant_test_wids, relevant_pred_wids):
            if test_wid == pred_wid:
                correct += 1
        homograph_acc = correct/homograph_total
        homograph_accs.append(homograph_acc)

    total_accs = 0
    for acc in homograph_accs:
        total_accs += acc

    macro_acc = total_accs/len(homograph_set)
    print("macro acc: ", macro_acc)


def get_micro_acc():
    # the percentage of examples correctly classified across all homographs
    _, test_wids, predicted_wids = get_predicted_wids("WHD_full.pkl")
    assert len(test_wids) == len(predicted_wids)

    total_wids = len(test_wids)

    correct_count = 0
    incorrect = []
    for i, (test, pred) in enumerate(zip(test_wids, predicted_wids)):
        if test == pred:
            correct_count += 1
        else:
            incorrect.append([i, test, pred])

    micro_acc = correct_count/total_wids

    print("micro acc: ", micro_acc)

    incorrect_counts = defaultdict(lambda: 0)
    for pred in incorrect:
        incorrect_counts[pred[1]] += 1

    # sorted_incorrect_counts = {k: v for k, v in sorted(incorrect_counts.items(), key=lambda item: item[1])}
    #
    # with open('spacy_analysis/spacy_incorrect_tag_counts.csv', 'w') as csv_file:
    #     writer = csv.writer(csv_file)
    #     for key, value in sorted_incorrect_counts.items():
    #         writer.writerow([key, value])


def get_corrected_pos_seqs(pos_seqs, token_seqs, WHD_df, spacy_preds):
    with open(token_seqs, "r") as f:
        token_lines = f.readlines()
    with open(pos_seqs, "r") as f2:
        pos_lines = f2.readlines()
    with open(WHD_df, "rb") as df:
        WHD = pickle.load(df)

    gt_pos = [gt[0].split()[0] for gt in WHD['gt_homograph_pron'].tolist()]
    wordids = WHD['wordid'].tolist()
    labels = WHD['label'].tolist()
    sentences = WHD['sentence'].tolist()

    incorrect = []
    for i, (gt, pred, wid, label, sentence) in enumerate(zip(gt_pos, spacy_preds, wordids, labels, sentences)):
        if label == "adjective-noun" or "adj-nou" in wid and gt == "jj" or gt == "nn":
            continue
        if "vb" in gt and "VB" in pred and "read" not in wid:
            continue
        if "nn" in gt and "NN" in pred:
            continue
        if wid == "row_1" or wid == "upset_vrb" or wid == "bow_nou-ship" or wid == "laminate_vrb" or wid == "sake_jp":
            continue

        elif gt.upper() not in pred:
            print(gt, pred, wid, label, sentence)
            incorrect.append(i)

    print(len(incorrect))


get_macro_acc()
get_micro_acc()




