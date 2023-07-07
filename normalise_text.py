import nemo_text_processing
import os
from nemo_text_processing.text_normalization.normalize import Normalizer
normalizer = Normalizer(input_case='cased', lang='en')

data = []
with open("testing/test_sentences.txt", 'r') as fp:
    for line in fp:
        data.append(line.strip())

normalised = normalizer.normalize_list(data, punct_post_process=True)

for sent in normalised:
    print(sent + "/n")