from nltk.corpus import treebank
import spacy
from spacy.tokens import Doc
from spacy.training import Alignment
from tqdm import tqdm

# # LOAD IN SPACY MODEL
print("Loading model...")
nlp = spacy.load("en_core_web_trf", disable="tokenizer")
print("Ready!")

def find_indices(list_to_check, item_to_find):
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices

fileids = treebank.fileids()

correct_count = 0
total_pos_tags = 0
sent_count = 0
for fileid in tqdm(fileids):
    spacy_tags = []
    treebank_sents = treebank.sents(fileid)
    # treebank_sents = [[ele for ele in sub if "*" not in ele] for sub in treebank_sents]
    treebank_tags = treebank.tagged_words(fileid)
    treebank_tags = [token_tag[1] for token_tag in treebank_tags]
    for sent in treebank_sents:
        # sent = list(map(str.lower, sent))
        sent_count += 1
        doc = nlp(Doc(nlp.vocab, words=sent))
        tags = [t.tag_ for t in doc]
        spacy_tags.extend(tags)
    # none_ids = find_indices(treebank_tags, "-NONE-")
    # spacy_tags = [x for i, x in enumerate(spacy_tags) if i not in none_ids]
    # treebank_tags = [x for i, x in enumerate(treebank_tags) if i not in none_ids]
    assert len(spacy_tags) == len(treebank_tags), print(fileid, len(spacy_tags), "\n", len(treebank_tags))
    correct_count += sum(1 for x, y in zip(spacy_tags, treebank_tags) if x == y)
    total_pos_tags += len(spacy_tags)

print(sent_count)
print(correct_count)
print(total_pos_tags)
print(correct_count/total_pos_tags)