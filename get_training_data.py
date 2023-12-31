from tqdm import tqdm
import csv
import re
import pickle
from unidecode import unidecode
import spacy

# # LOAD IN SPACY MODEL
# print("Loading model...")
# nlp = spacy.load("en_core_web_trf")
# print("Model loaded!")
# import en_core_web_trf
# print("Loading again...")
# nlp = en_core_web_trf.load()
# print("Ready!")


def write_to_csv(fpath, list):
    with open(fpath, 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(list)


def spacy_pos_tagger(input_file):
    """
        input_file: text file of \n delimited sentences
    """
    word_seqs = []
    token_seqs = []
    pos_seqs = []
    with open(input_file, "r") as file:
        sentences = file.readlines()
        sentences = [sentence.strip().lower() for sentence in sentences]
        for sentence in tqdm(sentences):
            doc = nlp(sentence)
            whitespace_split = sentence.split()
            doc_pos_seq = []
            doc_token_seq = []
            word_seq = []
            for w in doc:
                doc_pos_seq.append(w.tag_)
                doc_token_seq.append(w.text)
            for word in whitespace_split:
                word_seq.append(word)
            word_seqs.append(word_seq)
            token_seqs.append(doc_token_seq)
            pos_seqs.append(doc_pos_seq)

    write_to_csv("WHD_nnvb_analysis/word_seqs", word_seqs)
    write_to_csv("WHD_nnvb_analysis/pos_seqs", pos_seqs)
    write_to_csv("WHD_nnvb_analysis/token_seqs", token_seqs)

    return word_seqs, pos_seqs, token_seqs


def get_target_expanded(word_seqs, token_seqs, pos_seqs):
    cont_poss_list = ["'s", "n't", "'ll", "'ve", "'m", "'re", "'d"]

    expanded_tgts = []
    j = 0
    for token_seq, unexpanded_seq in zip(token_seqs, pos_seqs):
        expanded_sentence = []
        for i, token in enumerate(token_seq):
            count = len(token)
            expanded_POS = [unexpanded_seq[i]] * count
            if token in cont_poss_list and token not in word_seqs[j]:
                expanded_sentence[-1] = expanded_sentence[-1] + expanded_POS
            elif token == "not" and token_seq[i-1] == "can" and "cannot" in word_seqs[j]:
                expanded_sentence[-1] = expanded_sentence[-1] + expanded_POS
            elif token == "d" and token_seq[i-1] == "i" and "id" in word_seqs[j]:
                expanded_sentence[-1] = expanded_sentence[-1] + expanded_POS
            elif token == "nt" and token_seq[i-1] == "wo" and "wont" in word_seqs[j]:
                expanded_sentence[-1] = expanded_sentence[-1] + expanded_POS
            elif token == "d" and token_seq[i-1] == "we" and "wed" in word_seqs[j]:
                expanded_sentence[-1] = expanded_sentence[-1] + expanded_POS
            elif token == "nt" and token_seq[i-1] == "ca" and "cant" in word_seqs[j]:
                expanded_sentence[-1] = expanded_sentence[-1] + expanded_POS
            elif token == "ta" and token_seq[i-1] == "got" and "gotta" in word_seqs[j]:
                expanded_sentence[-1] = expanded_sentence[-1] + expanded_POS
            elif token == "'" and token_seq[i-1] == "o" and "o'" in word_seqs[j]:
                expanded_sentence[-1] = expanded_sentence[-1] + expanded_POS
            else:
                expanded_sentence.append(expanded_POS)
        expanded_tgts.append(expanded_sentence)
        j += 1
    return expanded_tgts


def get_csv_as_list(fpath):
    with open(fpath, 'r') as read_obj:
        # Return a reader object which will
        # iterate over lines in the given csvfile
        csv_reader = csv.reader(read_obj)

        # convert string to list
        list_of_csv = list(csv_reader)

    return list_of_csv


def get_src_data(WHD_df, outfile):
    with open(WHD_df, "rb") as df:
        WHD = pickle.load(df)

    word_seqs_list = WHD['words'].tolist()

    sentences = []
    for seq in word_seqs_list:
        sentence = " ".join(seq)
        sentence = sentence.upper()
        sentences.append(sentence)
    print(len(sentences))
    input()

    with open(outfile, "w") as f:
        for sentence in sentences:
            f.write(sentence + "\n")


# get_src_data("WHD_eval.pkl", "WHD_data/data/WHD_src_eval_UNCLEAN.txt")


def unidecode_src_file(src_file, outfile):
    with open(src_file, "r") as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        line = line.strip()
        no_e_line = line.replace("É", "E")
        no_foreign_chars = unidecode(no_e_line)
        modified_lines.append(no_foreign_chars)

    assert len(modified_lines) == len(lines)

    with open(outfile, "w") as f:
        for line in modified_lines:
            f.write(line + "\n")


# unidecode_src_file("WHD_data/data/WHD_src_eval_UNCLEAN.txt", "WHD_data/data/WHD_src_eval_DECODED.txt")


def remove_non_ascii(src_file, outfile):
    with open(src_file, "r") as f:
        lines = f.readlines()

    modified_lines = []
    for i, line in enumerate(lines):
        line = line.strip()
        no_e_line = line.replace("É", "E")
        modified_lines.append(no_e_line)

    with open(outfile, "w") as f:
        for line in modified_lines:
            f.write(line + "\n")


# remove_non_ascii("WHD_data/data/WHD_src_eval.txt", "WHD_data/data/WHD_src_eval.txt")


def get_non_ascii_lines(src_file):
    with open(src_file, "r") as f:
        lines = f.readlines()

    no_ascii_idxs = []
    for i, line in enumerate(lines):
        line = line.strip()
        # no_e_line = line.replace("É", "E")
        foreign_chars = re.search(r"[^a-zA-Z' ]", line)
        if foreign_chars == None:
            no_ascii_idxs.append(i)

    return no_ascii_idxs


def get_no_e_lines(src_file):
    with open(src_file, "r") as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        no_e_line = line.replace("É", "E")
        modified_lines.append(no_e_line)

    return modified_lines


# get_non_ascii_lines("WHD_data/data/WHD_src_train.txt")


def get_libri_WHD_data(filenames, output_fpath):
    with open(output_fpath, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)


# get_libri_WHD_data(["libri960_data/data/src-POS-train-aug.txt", "WHD_data/data/WHD_POS_train.txt"], "libri_WHD_data/libri_WHD_POS_train.txt")


def clean_POS_tags(POS_file, out_file):
    with open(POS_file, "r") as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        no_dollar = line.replace(" $", " XX")
        modified_lines.append(no_dollar)

    with open(out_file, "w") as f:
        for line in modified_lines:
            f.write(line)


def fix_seqs_sentencesWHD(seqs_list, sentences_fpath):
    with open(sentences_fpath, 'r') as f:
        sentences = f.readlines()
    all_word_seqs_fixed = []
    for seq in seqs_list:
        word_seqs_fixed = []
        for word in seq:
            starting_quote = re.match("^'([a-z]+\'?[a-z]+?)$", word)
            ending_quote = re.match("^([a-z]+\'?[a-z]+?)'$", word)
            enquote = re.match("^'([a-z]+\'?[a-z]+?)'$", word)
            if starting_quote is not None:
                word_seqs_fixed.append("'")
                word_seqs_fixed.append(starting_quote.group(1))
            elif ending_quote is not None:
                word_seqs_fixed.append(ending_quote.group(1))
                word_seqs_fixed.append("'")
            elif enquote is not None:
                word_seqs_fixed.append("'")
                word_seqs_fixed.append(enquote.group(1))
                word_seqs_fixed.append("'")
            else:
                word_seqs_fixed.append(word)
        all_word_seqs_fixed.append(word_seqs_fixed)

    with open(sentences_fpath, "w") as f:
        for sentence in sentences:
            sentence = sentence.lower().strip()
            sentence = re.sub(r"(^| )\'([a-z]+\'?[a-z]+?)($| )"," \' \g<2> ", sentence)
            sentence = re.sub(r"(^| )([a-z]+\'?[a-z]+?)\'($| )"," \g<2> \' ", sentence)
            sentence = re.sub(r"(^| )\'([a-z]+\'?[a-z]+?)\'($| )"," \' \g<2> \' ", sentence)
            f.write(sentence.upper() + "\n")
    return all_word_seqs_fixed


# spacy_pos_tagger("WHD_nnvb_analysis/no_punc_sents.txt")
#
# word_seqs_WHD = get_csv_as_list("WHD_nnvb_analysis/word_seqs")
# token_seqs_WHD = get_csv_as_list("WHD_nnvb_analysis/token_seqs")
# pos_seqs_WHD = get_csv_as_list("WHD_nnvb_analysis/pos_seqs")
#
# expanded_tgts = get_target_expanded(word_seqs_WHD, token_seqs_WHD, pos_seqs_WHD)
#
# expanded_POSseqs = []
# for sentence in expanded_tgts:
#     expanded_sentence = []
#     for word in sentence:
#         for POS_tag in word:
#             expanded_sentence.append(POS_tag)
#         expanded_sentence.append("+")
#     expanded_POSseqs.append(expanded_sentence)
#
# with open("WHD_nnvb_analysis/src_POS_nnvb.txt", "w") as f:
#     for seq in expanded_POSseqs:
#         f.write(" ".join(seq) + "\n")