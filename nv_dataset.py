import pyconll

my_conll_file_location = 'nv_data/train.conll'
train = pyconll.load.iter_from_file(my_conll_file_location)

all_sentences = []
for sentence in train:
    words = []
    for word in sentence:
        words.append(word.form)
    all_sentences.append(words)

for sentence in all_sentences:
    print(sentence)
