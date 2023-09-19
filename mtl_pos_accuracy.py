
def get_pos_ref_preds(tgt_POS_fpath, pred_POS_fpath):
    with open(tgt_POS_fpath, "r") as tgt_POS_file:
        with open(pred_POS_fpath, "r") as pred_POS_file:
            tgts = tgt_POS_file.readlines()
            preds = pred_POS_file.readlines()

    stripped_tgts = list(map(str.rstrip, tgts))
    stripped_preds = list(map(str.rstrip, preds))

    long_tgts = [seq.split(" + ")[:-1] for seq in stripped_tgts]
    long_preds = [seq.split(" # ")[:-1] for seq in stripped_preds]

    tgts = []
    preds = []
    for tgt_seq, pred_seq in zip(long_tgts, long_preds):
        short_tgts = [word.split()[0] for word in tgt_seq]
        short_preds = [word.split()[0] for word in pred_seq]
        tgts.append(short_tgts)
        preds.append(short_preds)

    assert len(tgts) == len(preds)

    return tgts, preds


def get_pos_acccuracy(micro=True):
    tgts, preds = get_pos_ref_preds("WHD_data/data/WHD_POS_eval.txt", "AUG_WHD_src_eval.POS_eos.txt")

    if micro:
        total_seqs = 0
        seq_accuracy = 0
        for tgt, pred in zip(tgts, preds):
            seq_accuracy += sum(1 for x, y in zip(tgt, pred) if x == y) / len(tgt)
            total_seqs += 1

        accuracy = seq_accuracy/total_seqs

        print("No. of seqs: ", total_seqs)
        print("Micro accuracy: ", accuracy)

    else:
        total_pos_tags = 0
        correct_count = 0
        for tgt, pred in zip(tgts, preds):
            correct_count += sum(1 for x, y in zip(tgt, pred) if x == y)
            total_pos_tags += len(tgt)

        print("No. of correct POS tags: ", correct_count)
        print("Total no. of POS tags: ", total_pos_tags)
        accuracy = correct_count/total_pos_tags
        print("Macro Acc: ", accuracy)


def get_homograph_ids(src_eval_fpath, homograph_eval_fpath):
    with open(src_eval_fpath, "r") as src_eval_file:
        with open(homograph_eval_fpath, "r") as homograph_file:
            data = src_eval_file.readlines()
            homographs = homograph_file.readlines()

    homographs = list(map(str.upper, homographs))
    homographs = list(map(str.strip, homographs))

    homograph_ids = []
    for line, homograph in zip(data, homographs):
        line = line.rstrip().split()
        homograph_idx = line.index(homograph)
        homograph_ids.append(homograph_idx)

    assert len(homograph_ids) == len(homographs) == len(data)

    return homograph_ids


def get_homograph_pos_accuracy():
    homograph_ids = get_homograph_ids("WHD_data/data/WHD_src_eval.txt", "clean_eval_homographs_NEW.txt")
    tgts, preds = get_pos_ref_preds("WHD_data/data/WHD_POS_eval.txt", "WHD_src_eval.POS_eos.txt")

    total_pos_tags = 0
    correct_count = 0
    error_count = 0
    for tgt, pred, id in zip(tgts, preds, homograph_ids):
        try:
            homograph_tgt = tgt[id]
            homograph_pred = pred[id]
            total_pos_tags += 1
            if homograph_pred == homograph_tgt:
                correct_count += 1
        except IndexError:
            error_count += 1

    print("No. of correct POS tags: ", correct_count)
    print("Total no. of POS tags: ", total_pos_tags)
    accuracy = correct_count / total_pos_tags
    print("Macro Acc: ", accuracy)
    print("Alignment error: ", error_count)

get_homograph_pos_accuracy()