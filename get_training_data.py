from tqdm import tqdm
import spacy
import csv
import re
from unidecode import unidecode

# LOAD IN SPACY MODEL
# print("Loading model...")
# nlp = spacy.load("en_core_web_trf")
# print("Model loaded!")
# import en_core_web_trf
# print("Loading again...")
# nlp = en_core_web_trf.load()
# print("Ready!")


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
    return word_seqs, pos_seqs, token_seqs


def get_target_expanded(word_seqs, token_seqs, pos_seqs):
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


def write_seqs_to_file(fpath_word, fpath_pos, fpath_token):
    with open(fpath_word, 'w') as word:
        # using csv.writer method from CSV package
        write = csv.writer(word)
        write.writerows(word_seqs)

    with open(fpath_pos, 'w') as pos:
        # using csv.writer method from CSV package
        write = csv.writer(pos)
        write.writerows(pos_seqs)

    with open(fpath_token, 'w') as token:
        # using csv.writer method from CSV package
        write = csv.writer(token)
        write.writerows(token_seqs)


def get_csv_as_list(fpath):
    with open(fpath, 'r') as read_obj:
        # Return a reader object which will
        # iterate over lines in the given csvfile
        csv_reader = csv.reader(read_obj)

        # convert string to list
        list_of_csv = list(csv_reader)

    return list_of_csv


def get_src_data(word_seqs_list):
    sentences = []
    for seq in word_seqs_list:
        sentence = " ".join(seq)
        sentence = sentence.upper()
        sentences.append(sentence)
    print(len(sentences))

    with open("libri960_data/test_data/src-train-test.txt", "w") as f:
        for sentence in sentences:
            f.write(sentence + "\n")


def remove_non_ascii(src_file, out_file):
    with open(src_file) as f:
        lines = f.readlines()

    modified_lines = []
    for i, line in enumerate(lines):
        line = line.strip()
        no_e_line = line.replace("É", "E")
        no_foreign_chars = re.sub(r'[^a-zA-Z\' ]', 'X', no_e_line)
        modified_lines.append(no_foreign_chars)

    with open(out_file, "w") as f:
        for line in modified_lines:
            f.write(line + "\n")


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


clean_POS_tags("WHD_data/data/WHD_POS.txt", "WHD_data/data/WHD_POS_clean.txt")


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


# cont_poss_list = ["'s", "n't", "'ll", "'ve", "'m", "'re", "'d"]
#
# word_seqs_WHD = get_csv_as_list("seqs/word_seqs_WHD")
# token_seqs_WHD = get_csv_as_list("seqs/token_seqs_WHD")
# pos_seqs_WHD = get_csv_as_list("seqs/pos_seqs_WHD")
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
# with open("WHD_data/data/WHD_POS.txt", "w") as f:
#     for seq in expanded_POSseqs:
#         f.write(" ".join(seq) + "\n")