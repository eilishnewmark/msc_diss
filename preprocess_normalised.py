import spacy
from tqdm import tqdm
import re
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
            modified_sentence = modified_sentence.upper()
            # Add modified sentence to the list
            sentences.append(modified_sentence.strip())
    return sentences


def remove_punc_from_sentences(input_file, output_file):
    sentences = process_file(input_file)
    with open(output_file, "w") as file:
        for sentence in sentences:
            file.write(sentence + "\n\n")


remove_punc_from_sentences("WHD_data/WHD_normalised.txt", "WHD_data/WHD_nopunc_for_festival.txt")


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





