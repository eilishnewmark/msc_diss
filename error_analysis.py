import pickle
from get_training_data import write_to_csv, get_csv_as_list


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


# get_libri_homograph_ref_preds("libri960_data/data/src-test.txt", "libri960_data/data/tgt-test.txt",
#                               "libri960_data/model_outputs/FE_src-test.ph.txt", "WHD_full_clean.pkl", "libri_analysis/error_analysis/fe_test")


# TO DO: change so that preds come from model_outputs folder/WHD_pron_clean
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


get_WHD_homograph_gt_preds("WHD_seq2seq_analysis/FE_POS/just_pronunciation_correct", "WHD_data/model_outputs/FE_POS_WHD_src.ph.txt", "WHD_full_clean.pkl")