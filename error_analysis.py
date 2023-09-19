import pickle
from collections import defaultdict
import csv
from get_training_data import write_to_csv, get_csv_as_list
import itertools


def get_libri_homograph_ref_preds(src_file, ref_file, pred_file, WHD_df, outfile):
    with open(src_file, "r") as f:
        src_lines = f.readlines()

    with open(ref_file, "r") as ref:
        ref_lines = ref.readlines()

    with open(pred_file, "r") as pred:
        pred_lines = pred.readlines()

    with open(WHD_df, "rb") as df:
        WHD = pickle.load(df)

    homographs = list(set(WHD['homograph'].tolist()))

    all_homograph_info = []
    all_mismatching_info = []
    homograph_count = 0
    for i, (src, ref, pred) in enumerate(zip(src_lines, ref_lines, pred_lines)):
        homograph_info = []
        mismatching_homograph_info = []
        words = src.rstrip().lower().split()
        ref_prons = ref.rstrip().replace("_B", "+").split("+")
        pred_prons = pred.rstrip().replace("_B", "+").split("+")
        found_homos = [(j, word) for j, word in enumerate(words) if word in homographs]
        homo_ids = [item[0] for item in found_homos]
        found_words = [item[1] for item in found_homos]
        homograph_count += len(found_words)
        if len(ref_prons) == len(pred_prons):
            homo_refs = [ref_prons[homo_id] for homo_id in homo_ids]
            homo_preds = [pred_prons[homo_id] for homo_id in homo_ids]
            if homo_refs != []:
                homograph_info.append(i)
                homograph_info.append(src.lower().strip())
                homograph_info.append(",".join(homo_refs))
                homograph_info.append(",".join(homo_preds))
                homograph_info.append(",".join(found_words))
                all_homograph_info.append(homograph_info)
            for z, (r, p) in enumerate(zip(homo_refs, homo_preds)):
                if r != p:
                    mismatching_homograph_info.append(i)
                    mismatching_homograph_info.append(src.lower().strip())
                    mismatching_homograph_info.append(r)
                    mismatching_homograph_info.append(p)
                    mismatching_homograph_info.append(homo_ids[z])
                    mismatching_homograph_info.append(found_words[z])
                    all_mismatching_info.append(mismatching_homograph_info)

    print("no. of homographs in src_file: ", homograph_count)
    print("no. of mismatching homograph references and predictions: ", len(all_mismatching_info))

    write_to_csv(outfile + ".csv", all_homograph_info)
    write_to_csv(outfile + "_incorrect.csv", all_mismatching_info)


def get_WHD_homograph_gt_preds(csv_file, pred_file, WHD_df):
    csv_preds = get_csv_as_list(csv_file + ".csv")
    pred_ids = []
    for pred in csv_preds:
        pred_ids.append(int(pred[0]))

    with open(WHD_df, "rb") as df:
        WHD = pickle.load(df)

    with open(pred_file, "r") as f:
        all_preds = f.readlines()

    preds = [pred for i, pred in enumerate(all_preds) if i in pred_ids]

    assert len(pred_ids) == len(preds)

    data = WHD.iloc[pred_ids]

    gt_prons = data['gt_homograph_pron'].tolist()
    homographs = data['homograph'].tolist()
    words = data['words'].tolist()
    morpho_lexical = data['homograph_type'].to_list()

    match_word_pron_length = 0
    all_info = []
    for i, (homograph, word, pred, gt, ml) in enumerate(zip(homographs, words, preds, gt_prons, morpho_lexical)):
        line_info = []
        pred = pred.rstrip().replace("_B", "+").split("+")[:-1]
        if len(word) == len(pred):
            match_word_pron_length += 1
            try:
                homograph_idx = word.index(homograph)
                pred_pron = pred[homograph_idx]
            except ValueError:
                print(homograph, word)
            line_info.append(i)
            line_info.append(" ". join(word))
            line_info.append(gt)
            line_info.append(pred_pron)
            line_info.append(homograph_idx)
            line_info.append(homograph)
            line_info.append(ml)
        if line_info != []:
            all_info.append(line_info)

    print("no. of matching lines: ", match_word_pron_length)

    write_to_csv(csv_file + "_DATA.csv", all_info)


def get_iliv_ilov_homograph_accs(fpath, homograph_file, nnvb=False):
    with open(homograph_file, "r") as f:
        all_homographs = f.readlines()

    homographs = list(map(str.upper, set(all_homographs)))
    homographs = list(map(str.strip, homographs))
    if nnvb:
        with open("WHD_nnvb_analysis/nnvb_homographs.csv", "r", newline="", encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            homographs = list(map(str.upper, itertools.chain.from_iterable(reader)))
            print(homographs)
            print(len(homographs))

    with open(fpath + ".txt", "r") as f:
        lines = f.readlines()

    homograph_word_acc_counts = defaultdict(lambda:0)
    homograph_phone_acc_counts = defaultdict(lambda:0)
    homograph_stress_acc_counts = defaultdict(lambda:0)
    homograph_syl_acc_counts = defaultdict(lambda:0)
    homograph_count = defaultdict(lambda: 0)

    refs = []
    preds = []
    for line in lines:
        split_line = line.rstrip().split("|")
        word, word_match, phone_match, stress_match, syl_match, ref, pred = split_line[0], split_line[1], split_line[2], split_line[3], split_line[4], split_line[-2], split_line[-1]
        if word in homographs:
            homograph_count[word] += 1
            if word_match == "True":
                homograph_word_acc_counts[word] += 1
            if phone_match == "True":
                homograph_phone_acc_counts[word] += 1
            if stress_match == "True":
                homograph_stress_acc_counts[word] += 1
            if syl_match == "True":
                homograph_syl_acc_counts[word] += 1
            refs.append(ref)
            preds.append(pred)

    homograph_accs = defaultdict(list)
    for homograph in homographs:
        if homograph in homograph_count.keys():
            count = homograph_count[homograph]
            word_acc = homograph_word_acc_counts[homograph]/count
            phone_acc = homograph_phone_acc_counts[homograph]/count
            stress_acc = homograph_stress_acc_counts[homograph]/count
            syl_acc = homograph_syl_acc_counts[homograph]/count
            homograph_accs[homograph].append(word_acc)
            homograph_accs[homograph].append(phone_acc)
            homograph_accs[homograph].append(stress_acc)
            homograph_accs[homograph].append(syl_acc)
            homograph_accs[homograph].append(count)

    if nnvb:
        with open("WHD_nnvb_analysis/" + fpath[21:] + "NNVB.csv", "w") as f:
            writer = csv.writer(f)
            header = ["homograph", "word acc", "phone acc", "stress acc", "syl acc", "count"]
            writer.writerow(header)
            for key, value in homograph_accs.items():
                word_acc, phone_acc, stress_acc, syl_acc, count = value[0], value[1], value[2], value[3], value[4]
                writer.writerow([key, word_acc, phone_acc, stress_acc, syl_acc, count])
    else:
        with open(fpath + "_homograph_ACCs.csv", "w") as f:
            writer = csv.writer(f)
            header = ["homograph", "word acc", "phone acc", "stress acc", "syl acc", "count"]
            writer.writerow(header)
            for key, value in homograph_accs.items():
                word_acc, phone_acc, stress_acc, syl_acc, count = value[0], value[1], value[2], value[3], value[4]
                writer.writerow([key, word_acc, phone_acc, stress_acc, syl_acc, count])

        assert len(refs) == len(preds)

        with open(fpath + "_refs_preds.csv", "w") as f:
            writer = csv.writer(f)
            header = ["ref", "pred"]
            writer.writerow(header)
            for ref, pred in zip(refs, preds):
                writer.writerow([ref, pred])


# models = ["FE", "FE_POS", "FE_POS_MTL", "FE_DATA", "FE_POS_DATA", "FE_POS_MTL_DATA"]
#
# for model in models:
#     get_iliv_ilov_homograph_accs(f"WHD_seq2seq_analysis/{model}/WHD_eval/in_lex_in_vocab",
#                                  "clean_eval_homographs_NEW.txt", nnvb=False)
#     # get_iliv_ilov_homograph_accs(f"WHD_seq2seq_analysis/{model}/WHD_eval/in_lex_out_vocab",
#     #                              "clean_eval_homographs_NEW.txt", nnvb=False)
#     print(model + " : done!")

# get_iliv_ilov_homograph_accs(f"WHD_seq2seq_analysis/FE_POS_MTL_DATA/in_lex_in_vocab",
#                                  "clean_eval_homographs_NEW.txt", nnvb=False)

get_iliv_ilov_homograph_accs("WHD_seq2seq_analysis/FESTIVAL/WHD_eval/in_lex_out_vocab", "clean_eval_homographs_NEW.txt", nnvb=False)
