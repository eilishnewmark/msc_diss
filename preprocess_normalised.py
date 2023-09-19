import spacy
from tqdm import tqdm
import re
from unidecode import unidecode
# import pandas as pd
# import pickle
import string

def process_file(input_file):
    sentences = []
    with open(input_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            # Replace punctuation (excluding apostrophes) with whitespace
            modified_sentence = re.sub(r'[^\w\s\']', ' ', line)
            # Remove double whitespaces
            modified_sentence = re.sub(r'\s+', ' ', modified_sentence)
            no_e_line = modified_sentence.replace("é", "e")
            no_foreign_chars = unidecode(no_e_line)
            modified_sentence = no_foreign_chars.upper()
            # Add modified sentence to the list
            sentences.append(modified_sentence.strip())
    return sentences


def remove_punc_from_sentences(input_file, output_file):
    sentences = process_file(input_file)
    with open(output_file, "w") as file:
        for sentence in sentences:
            file.write(sentence + "\n\n")


# remove_punc_from_sentences("WHD_data/data/WHD_src_eval.txt",
#                            "WHD_data/data/WHD_src_eval_festinput.txt")


def postprocess_festival_output(input_file):
    festival_pos_seqs = []
    festival_homograph_pairs = []
    no_nils = []
    uh_oh = []
    with open(input_file, encoding="latin1") as file:
        lines = file.readlines()
    # get rid of nils
    for line in lines:
        no_nils.append(line.split(" (nil")[0])
    print(len(no_nils))
    print(no_nils[0])
    with open("test/festival_sentences.txt", "r") as sentences:
        sentences = sentences.readlines()
    print(len(sentences))
    for line in no_nils:
        for i, sentence in enumerate(sentences):
            if line[1] != sentence[0]:
                uh_oh.append(i)
    print(len(uh_oh))


def clean_festival_output(fest_fpath, ling_fpath):
    ling_f = open(ling_fpath, 'w')

    with open(fest_fpath) as fest_f:
        for line in fest_f:
            ling = line.rstrip()
            ling = ling.split(' ', 1)[-1]
            ling = ling.replace('{}', '').replace('<{(', '').replace(')}>', '').replace(')(', '- ').replace(')}{(', '+ ')
            ling = ' '.join(ling.split())

            ling_f.write(ling + '\n')

    ling_f.close()


clean_festival_output("WHD_data/data/WHD_src_eval_festoutput.txt", "WHD_data/data/WHD_src_eval_incorrect.txt")



