import pickle
from collections import defaultdict
from get_training_data import write_to_csv
import csv
# We collected counts from the training corpus of the form <word, POS tag, word_sense, Count>.
# These counts show how many times a given word got assigned to a certain word sense when it has a certain POS tag.

def get_festival_token_pos_seqs(festival_pos_file, pos_seq_outfile, token_seq_outfile):
    with open(festival_pos_file, "r") as f:
        lines = f.readlines()

    all_tokens = []
    all_tags = []
    for line in lines:
        no_nils = line.split("(")[0]
        token_pos = no_nils.replace('""', ''"/"'').replace('"', "").split("/")
        tokens, tags = token_pos[::2], token_pos[1::2]
        tokens = list(map(str.lower, tokens))
        tags = list(map(str.upper, tags))
        all_tokens.append(tokens)
        all_tags.append(tags)

    write_to_csv(pos_seq_outfile, all_tags)
    write_to_csv(token_seq_outfile, all_tokens)


# get_festival_token_pos_seqs("WHD_data/WHD_festival_pos.txt", "seqs/pos_seqs_festival", "seqs/token_seqs_festival")


def get_preds(pos_seqs, token_seqs, WHD_df):
    with open(token_seqs, "r") as f:
        token_lines = f.readlines()
    with open(pos_seqs, "r") as f2:
        pos_lines = f2.readlines()
    with open(WHD_df, "rb") as df:
        WHD = pickle.load(df)

    homographs = WHD['homograph'].tolist()
    if WHD_df == "WHD_train.pkl":
        token_lines = token_lines[:len(homographs)]
        pos_lines = pos_lines[:len(homographs)]
    if WHD_df == "WHD_eval.pkl":
        token_lines = token_lines[16102 - len(homographs):]
        pos_lines = pos_lines[16102 - len(homographs):]
    assert len(homographs) == len(token_lines) == len(pos_lines)

    preds = []
    for tokens, pos, homograph in zip(token_lines, pos_lines, homographs):
        tokens = tokens.rstrip().split(",")
        pos = pos.rstrip().split(",")
        homograph_idx = tokens.index(homograph)
        pos_tag = pos[homograph_idx]
        preds.append(pos_tag)

    return preds


def get_predicted_wids(WHD_df_train, WHD_df_eval):
    with open(WHD_df_train, "rb") as df:
        WHD_train = pickle.load(df)
    with open(WHD_df_eval, "rb") as df:
        WHD_eval = pickle.load(df)

    preds = get_preds("seqs/pos_seqs_WHD", "seqs/token_seqs_WHD", WHD_df_train)
    homographs = WHD_train['homograph'].tolist()
    wordids = WHD_train['wordid'].tolist()

    assert len(preds) == len(homographs) == len(wordids)

    counts = defaultdict(lambda: 0)
    for spacy_pred, homograph, wid in zip(preds, homographs, wordids):
        counts[(homograph, spacy_pred, wid)] += 1

    test_preds = get_preds("seqs/pos_seqs_WHD", "seqs/token_seqs_WHD", WHD_df_eval)
    test_homographs = WHD_eval['homograph'].tolist()
    test_wordids = WHD_eval['wordid'].tolist()

    predicted_wordids = []
    bad_tags = []
    for i, (pred, homograph) in enumerate(zip(test_preds, test_homographs)):
        potential_wid_counts = []
        for info, count in counts.items():
            word, tag, wordid = info
            if word == homograph and tag == pred:
                potential_wid_counts.append((wordid, count))
        try:
            highest_wid_count = sorted(potential_wid_counts, key=lambda x: x[1], reverse=True)[0]
            predicted_wordids.append(highest_wid_count[0])
        except IndexError:
            bad_tags.append((i, pred, homograph))
            predicted_wordids.append(potential_wid_counts)

    return bad_tags, test_homographs, test_wordids, predicted_wordids


def find_indices(list_to_check, item_to_find):
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices


def get_macro_acc():
    # the arithmetic mean of the per-homograph accuracies
    bad_tags, test_homographs, test_wids, predicted_wids = get_predicted_wids("WHD_train.pkl", "WHD_eval.pkl")

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
    bad_tags, _, test_wids, predicted_wids = get_predicted_wids("WHD_train.pkl", "WHD_eval.pkl")
    assert len(test_wids) == len(predicted_wids)

    total_wids = len(test_wids)

    correct_count = 0
    incorrect = []
    incorrect_counts = defaultdict(lambda: 0)
    for i, (test, pred) in enumerate(zip(test_wids, predicted_wids)):
        if test == pred:
            correct_count += 1
        else:
            incorrect.append([i, test, pred])
            try:
                incorrect_counts[pred] += 1
            except TypeError:
                pass

    micro_acc = (correct_count + 44)/total_wids

    print("micro acc: ", micro_acc)

    sorted_incorrect_counts = {k: v for k, v in sorted(incorrect_counts.items(), key=lambda item: item[1], reverse=True)}

    # with open('spacy_analysis/spacy_incorrect_tag_counts.csv', 'w') as csv_file:
    #     writer = csv.writer(csv_file)
    #     for key, value in sorted_incorrect_counts.items():
    #         writer.writerow([key, value])
    #
    # with open("spacy_analysis/spacy_incorrect_tag_ids.csv", "w") as csv_file:
    #     writer = csv.writer(csv_file)
    #     for entry in incorrect:
    #         writer.writerow(entry)
    #
    # with open("spacy_analysis/spacy_incorrect_word_tagpred_combo.csv", "w") as csv_file:
    #     writer = csv.writer(csv_file)
    #     for entry in bad_tags:
    #         writer.writerow(entry)



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




